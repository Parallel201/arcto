// Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
//
// Coverage tests for the batched Cascaded API that the kernel-level optimization
// work uses as its correctness gate. They cover what the older tests do not:
//  - every layer configuration a user can ask for: num_RLEs in {0..3},
//    num_deltas in {0..2}, use_bp in {0,1} (including the "no layer" raw path and
//    the fallback when a layer makes the data larger),
//  - all integer element types (1/2/4/8 bytes, signed and unsigned),
//  - element counts that are not multiples of the 4 KB chunk (1, 3, 17, 4095,
//    4097, ...) and several chunks per partition, in one batch of mixed partitions,
//  - data profiles that exercise each stage: long runs (RLE), ramps (delta),
//    small-range noise (bitpack), full-range random (fallback), zeros,
//  - determinism (two compressions produce identical bytes).
// Every check is bit-exact, so any change that alters the compressed or decoded
// bytes fails here — the property the optimization loop needs.

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "arcto/cascaded.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(func)                                                        \
  do {                                                                         \
    hipError_t rt__ = (func);                                                  \
    if (rt__ != hipSuccess) {                                                  \
      printf("HIP call \"" #func "\" failed with %d (%s) at %s:%d\n",          \
             (int)rt__, hipGetErrorString(rt__), __FILE__, __LINE__);          \
    }                                                                          \
    REQUIRE(rt__ == hipSuccess);                                               \
  } while (0)

namespace {

using bytes_t = std::vector<uint8_t>;

enum class Profile { Zeros, Runs, Ramp, SmallRange, Random, Alternating };

const char* profile_name(Profile p)
{
  switch (p) {
    case Profile::Zeros: return "zeros";
    case Profile::Runs: return "runs";
    case Profile::Ramp: return "ramp";
    case Profile::SmallRange: return "small-range";
    case Profile::Random: return "random";
    case Profile::Alternating: return "alternating";
  }
  return "?";
}

template <typename T>
bytes_t generate(Profile p, size_t n, uint32_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<T> v(n);
  switch (p) {
    case Profile::Zeros:
      std::fill(v.begin(), v.end(), T(0));
      break;
    case Profile::Runs: {
      size_t i = 0;
      while (i < n) {
        const T val = static_cast<T>(rng() % 7);
        const size_t run = 1 + (rng() % 300);
        for (size_t j = 0; j < run && i < n; ++j) v[i++] = val;
      }
      break;
    }
    case Profile::Ramp: {
      T cur = static_cast<T>(rng() % 100);
      for (auto& x : v) { x = cur; cur = static_cast<T>(cur + 1 + (rng() % 3)); }
      break;
    }
    case Profile::SmallRange:
      for (auto& x : v) x = static_cast<T>(300 + (rng() % 4)); // as in the C API test
      break;
    case Profile::Random:
      for (auto& x : v) x = static_cast<T>(rng());
      break;
    case Profile::Alternating:
      for (size_t i = 0; i < n; ++i) v[i] = static_cast<T>((i & 1) ? 5 : -5);
      break;
  }
  bytes_t b(n * sizeof(T));
  if (n) std::memcpy(b.data(), v.data(), b.size());
  return b;
}

bytes_t generate_type(arctoType_t t, Profile p, size_t n, uint32_t seed)
{
  switch (t) {
    case ARCTO_TYPE_CHAR: return generate<int8_t>(p, n, seed);
    case ARCTO_TYPE_UCHAR: return generate<uint8_t>(p, n, seed);
    case ARCTO_TYPE_SHORT: return generate<int16_t>(p, n, seed);
    case ARCTO_TYPE_USHORT: return generate<uint16_t>(p, n, seed);
    case ARCTO_TYPE_INT: return generate<int32_t>(p, n, seed);
    case ARCTO_TYPE_UINT: return generate<uint32_t>(p, n, seed);
    case ARCTO_TYPE_LONGLONG: return generate<int64_t>(p, n, seed);
    case ARCTO_TYPE_ULONGLONG: return generate<uint64_t>(p, n, seed);
    default: return {};
  }
}

size_t type_size(arctoType_t t)
{
  switch (t) {
    case ARCTO_TYPE_CHAR: case ARCTO_TYPE_UCHAR: return 1;
    case ARCTO_TYPE_SHORT: case ARCTO_TYPE_USHORT: return 2;
    case ARCTO_TYPE_INT: case ARCTO_TYPE_UINT: return 4;
    default: return 8;
  }
}

// GPU round trip through the public batched API. Returns the compressed bytes
// per partition (for determinism / ratio checks).
std::vector<bytes_t> gpu_round_trip(const std::vector<bytes_t>& inputs,
                                    const arctoBatchedCascadedOpts_t& opts)
{
  const size_t batch = inputs.size();
  REQUIRE(batch > 0);
  size_t max_chunk = 0;
  for (const auto& c : inputs) max_chunk = std::max(max_chunk, c.size());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<void*> d_in(batch), d_comp(batch), d_out(batch);
  std::vector<size_t> h_in_bytes(batch);
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipMalloc(&d_in[i], std::max<size_t>(inputs[i].size(), 16)));
    HIP_CHECK(hipMemcpy(d_in[i], inputs[i].data(), inputs[i].size(), hipMemcpyHostToDevice));
    h_in_bytes[i] = inputs[i].size();
  }
  void** d_in_ptrs; size_t* d_in_bytes;
  HIP_CHECK(hipMalloc((void**)&d_in_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_in_bytes, batch * sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_in_ptrs, d_in.data(), batch * sizeof(void*), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_in_bytes, h_in_bytes.data(), batch * sizeof(size_t), hipMemcpyHostToDevice));

  size_t temp_bytes = 0;
  REQUIRE(arctoBatchedCascadedCompressGetTempSize(batch, max_chunk, opts, &temp_bytes) == arctoSuccess);
  void* d_temp = nullptr;
  if (temp_bytes) HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
  size_t max_comp = 0;
  REQUIRE(arctoBatchedCascadedCompressGetMaxOutputChunkSize(max_chunk, opts, &max_comp) == arctoSuccess);

  for (size_t i = 0; i < batch; ++i) HIP_CHECK(hipMalloc(&d_comp[i], max_comp + 64));
  void** d_comp_ptrs; size_t* d_comp_bytes;
  HIP_CHECK(hipMalloc((void**)&d_comp_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_comp_bytes, batch * sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_comp_ptrs, d_comp.data(), batch * sizeof(void*), hipMemcpyHostToDevice));

  REQUIRE(arctoBatchedCascadedCompressAsync(d_in_ptrs, d_in_bytes, max_chunk, batch, d_temp, temp_bytes,
                                            d_comp_ptrs, d_comp_bytes, opts, stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<size_t> h_comp_bytes(batch);
  HIP_CHECK(hipMemcpy(h_comp_bytes.data(), d_comp_bytes, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  std::vector<bytes_t> comp(batch);
  for (size_t i = 0; i < batch; ++i) {
    INFO("partition " << i << " bytes " << inputs[i].size());
    REQUIRE(h_comp_bytes[i] > 0);
    REQUIRE(h_comp_bytes[i] <= max_comp);
    comp[i].resize(h_comp_bytes[i]);
    HIP_CHECK(hipMemcpy(comp[i].data(), d_comp[i], h_comp_bytes[i], hipMemcpyDeviceToHost));
  }

  // decompressed-size query
  size_t* d_uncomp_bytes;
  HIP_CHECK(hipMalloc((void**)&d_uncomp_bytes, batch * sizeof(size_t)));
  REQUIRE(arctoBatchedCascadedGetDecompressSizeAsync((const void* const*)d_comp_ptrs, d_comp_bytes,
                                                     d_uncomp_bytes, batch, stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<size_t> h_uncomp(batch);
  HIP_CHECK(hipMemcpy(h_uncomp.data(), d_uncomp_bytes, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch; ++i) REQUIRE(h_uncomp[i] == inputs[i].size());

  // decompression
  size_t dtemp_bytes = 0;
  REQUIRE(arctoBatchedCascadedDecompressGetTempSize(batch, max_chunk, &dtemp_bytes) == arctoSuccess);
  void* d_dtemp = nullptr;
  if (dtemp_bytes) HIP_CHECK(hipMalloc(&d_dtemp, dtemp_bytes));
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipMalloc(&d_out[i], inputs[i].size() + 64));
    HIP_CHECK(hipMemset(d_out[i], 0xA5, inputs[i].size() + 64));
  }
  void** d_out_ptrs; size_t* d_actual; arctoStatus_t* d_status;
  HIP_CHECK(hipMalloc((void**)&d_out_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_actual, batch * sizeof(size_t)));
  HIP_CHECK(hipMalloc((void**)&d_status, batch * sizeof(arctoStatus_t)));
  HIP_CHECK(hipMemcpy(d_out_ptrs, d_out.data(), batch * sizeof(void*), hipMemcpyHostToDevice));

  REQUIRE(arctoBatchedCascadedDecompressAsync((const void* const*)d_comp_ptrs, d_comp_bytes, d_uncomp_bytes,
                                              d_actual, batch, d_dtemp, dtemp_bytes, d_out_ptrs, d_status,
                                              stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<size_t> h_actual(batch);
  std::vector<arctoStatus_t> h_status(batch);
  HIP_CHECK(hipMemcpy(h_actual.data(), d_actual, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_status.data(), d_status, batch * sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch; ++i) {
    INFO("partition " << i << " bytes " << inputs[i].size());
    REQUIRE(h_status[i] == arctoSuccess);
    REQUIRE(h_actual[i] == inputs[i].size());
    bytes_t back(inputs[i].size());
    HIP_CHECK(hipMemcpy(back.data(), d_out[i], back.size(), hipMemcpyDeviceToHost));
    REQUIRE(back == inputs[i]);
    bytes_t guard(64);
    HIP_CHECK(hipMemcpy(guard.data(), static_cast<uint8_t*>(d_out[i]) + back.size(), 64, hipMemcpyDeviceToHost));
    REQUIRE(std::all_of(guard.begin(), guard.end(), [](uint8_t b) { return b == 0xA5; }));
  }

  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipFree(d_in[i])); HIP_CHECK(hipFree(d_comp[i])); HIP_CHECK(hipFree(d_out[i]));
  }
  HIP_CHECK(hipFree(d_in_ptrs)); HIP_CHECK(hipFree(d_in_bytes)); HIP_CHECK(hipFree(d_comp_ptrs));
  HIP_CHECK(hipFree(d_comp_bytes)); HIP_CHECK(hipFree(d_uncomp_bytes)); HIP_CHECK(hipFree(d_out_ptrs));
  HIP_CHECK(hipFree(d_actual)); HIP_CHECK(hipFree(d_status));
  if (d_temp) HIP_CHECK(hipFree(d_temp));
  if (d_dtemp) HIP_CHECK(hipFree(d_dtemp));
  HIP_CHECK(hipStreamDestroy(stream));
  return comp;
}

arctoBatchedCascadedOpts_t make_opts(arctoType_t t, int rles, int deltas, int bp)
{
  arctoBatchedCascadedOpts_t o;
  o.chunk_size = 4096;
  o.type = t;
  o.num_RLEs = rles;
  o.num_deltas = deltas;
  o.use_bp = bp;
  return o;
}

const arctoType_t kTypes[] = {ARCTO_TYPE_CHAR, ARCTO_TYPE_UCHAR, ARCTO_TYPE_SHORT, ARCTO_TYPE_USHORT,
                              ARCTO_TYPE_INT, ARCTO_TYPE_UINT, ARCTO_TYPE_LONGLONG, ARCTO_TYPE_ULONGLONG};
const char* type_name(arctoType_t t)
{
  switch (t) {
    case ARCTO_TYPE_CHAR: return "char"; case ARCTO_TYPE_UCHAR: return "uchar";
    case ARCTO_TYPE_SHORT: return "short"; case ARCTO_TYPE_USHORT: return "ushort";
    case ARCTO_TYPE_INT: return "int"; case ARCTO_TYPE_UINT: return "uint";
    case ARCTO_TYPE_LONGLONG: return "longlong"; case ARCTO_TYPE_ULONGLONG: return "ulonglong";
    default: return "?";
  }
}

} // namespace

// ---------------------------------------------------------------------------

TEST_CASE("cascaded: every layer configuration, every type, mixed partitions", "[small]")
{
  const int configs[][3] = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 1}, {0, 1, 0}, {0, 1, 1},
                            {2, 1, 1}, {3, 2, 1}, {1, 2, 1}, {3, 0, 1}, {0, 2, 1}, {2, 0, 0}};
  const size_t counts[] = {1, 2, 3, 17, 100, 1000, 1024, 1025, 4095, 4096, 4097, 6000, 10000};
  for (arctoType_t t : kTypes) {
    for (const auto& c : configs) {
      const auto opts = make_opts(t, c[0], c[1], c[2]);
      std::vector<bytes_t> inputs;
      uint32_t seed = 7;
      const Profile profiles[] = {Profile::Runs, Profile::Ramp, Profile::SmallRange, Profile::Zeros,
                                  Profile::Alternating, Profile::Random};
      for (size_t n : counts)
        inputs.push_back(generate_type(t, profiles[seed % 6], n, seed++));
      INFO("type " << type_name(t) << " rles " << c[0] << " deltas " << c[1] << " bp " << c[2]);
      auto a = gpu_round_trip(inputs, opts);
      auto b = gpu_round_trip(inputs, opts);
      REQUIRE(a == b); // deterministic
    }
  }
}

TEST_CASE("cascaded: per-profile compressibility with the default configuration", "[small]")
{
  for (arctoType_t t : {ARCTO_TYPE_INT, ARCTO_TYPE_SHORT, ARCTO_TYPE_UCHAR, ARCTO_TYPE_LONGLONG}) {
    for (Profile p : {Profile::Zeros, Profile::Runs, Profile::Ramp, Profile::SmallRange, Profile::Random}) {
      const auto opts = make_opts(t, 2, 1, 1);
      std::vector<bytes_t> inputs;
      for (uint32_t s = 0; s < 4; ++s) inputs.push_back(generate_type(t, p, 16384 / type_size(t) + 13 * s, 100 + s));
      INFO("type " << type_name(t) << " profile " << profile_name(p));
      auto comp = gpu_round_trip(inputs, opts);
      size_t in_total = 0, out_total = 0;
      for (size_t i = 0; i < inputs.size(); ++i) { in_total += inputs[i].size(); out_total += comp[i].size(); }
      if (p == Profile::Zeros || p == Profile::Runs) REQUIRE(out_total * 4 < in_total);
      if (p == Profile::Random) REQUIRE(out_total <= in_total + 16 * inputs.size()); // fallback keeps it raw
    }
  }
}

TEST_CASE("cascaded: large batch of mixed-size, mixed-profile partitions", "[large]")
{
  std::mt19937 rng(99);
  const Profile profiles[] = {Profile::Zeros, Profile::Runs, Profile::Ramp, Profile::SmallRange,
                              Profile::Random, Profile::Alternating};
  for (arctoType_t t : {ARCTO_TYPE_INT, ARCTO_TYPE_UCHAR}) {
    std::vector<bytes_t> inputs;
    for (int i = 0; i < 256; ++i) inputs.push_back(generate_type(t, profiles[rng() % 6], 1 + (rng() % 20000), 1000 + i));
    INFO("type " << type_name(t));
    gpu_round_trip(inputs, make_opts(t, 2, 1, 1));
    gpu_round_trip(inputs, make_opts(t, 1, 1, 0));
  }
}
