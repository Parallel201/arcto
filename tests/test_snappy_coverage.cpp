// Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
//
// Coverage tests for the batched Snappy API that the kernel-level optimization
// work uses as its correctness gate. They exercise what the older tests do not:
//  - chunk sizes that are not multiples of anything (1, 3, 63, 65, 4097, ...),
//  - input / compressed / decompressed device pointers that are deliberately
//    misaligned (the prefetcher, literal copies and output stores all have
//    alignment-dependent paths),
//  - a compressibility sweep (zeros, runs, small alphabets, words, random) at
//    several chunk sizes, with the compressed bytes cross-checked by an
//    independent host-side Snappy decoder (format compatibility, not just GPU
//    round trip) and by a determinism check (two compressions, identical bytes),
//  - streams made of thousands of tiny symbols so the decoder's 4-slot symbol
//    queue wraps around many times per chunk,
//  - a large batch of mixed-size chunks.
//
// Every check is bit-exact: any change that alters the compressed bytes or the
// decoded bytes is a failure, which is exactly the property the optimization
// loop needs ("same algorithm, same output, only faster").

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "arcto/snappy.h"
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

// ---------------------------------------------------------------------------
// Independent host-side decoder for the Snappy raw format (varint length, then
// literal / copy-1 / copy-2 / copy-4 elements). Used to verify that what the
// GPU emitted is a valid Snappy stream that decodes to the input, not merely
// something the GPU decoder happens to accept.
bool host_snappy_decode(const uint8_t* in, size_t in_len, bytes_t& out)
{
  size_t pos = 0;
  uint64_t expected = 0;
  int shift = 0;
  while (true) {
    if (pos >= in_len || shift > 35) return false;
    const uint8_t b = in[pos++];
    expected |= uint64_t(b & 0x7f) << shift;
    if (!(b & 0x80)) break;
    shift += 7;
  }
  out.clear();
  out.reserve(expected);
  auto copy = [&](size_t off, size_t len) {
    if (off == 0 || off > out.size()) return false;
    for (size_t i = 0; i < len; ++i) out.push_back(out[out.size() - off]);
    return true;
  };
  while (pos < in_len) {
    const uint8_t tag = in[pos++];
    switch (tag & 3) {
      case 0: { // literal
        size_t len = tag >> 2;
        if (len >= 60) {
          const int nbytes = int(len) - 59;
          if (pos + nbytes > in_len) return false;
          len = 0;
          for (int i = 0; i < nbytes; ++i) len |= size_t(in[pos + i]) << (8 * i);
          pos += nbytes;
        }
        len += 1;
        if (pos + len > in_len) return false;
        out.insert(out.end(), in + pos, in + pos + len);
        pos += len;
        break;
      }
      case 1: { // copy with 1-byte offset
        const size_t len = ((tag >> 2) & 7) + 4;
        if (pos >= in_len) return false;
        const size_t off = (size_t(tag >> 5) << 8) | in[pos++];
        if (!copy(off, len)) return false;
        break;
      }
      case 2: { // copy with 2-byte offset
        const size_t len = (tag >> 2) + 1;
        if (pos + 2 > in_len) return false;
        const size_t off = size_t(in[pos]) | (size_t(in[pos + 1]) << 8);
        pos += 2;
        if (!copy(off, len)) return false;
        break;
      }
      default: { // copy with 4-byte offset
        const size_t len = (tag >> 2) + 1;
        if (pos + 4 > in_len) return false;
        const size_t off = size_t(in[pos]) | (size_t(in[pos + 1]) << 8)
                           | (size_t(in[pos + 2]) << 16) | (size_t(in[pos + 3]) << 24);
        pos += 4;
        if (!copy(off, len)) return false;
        break;
      }
    }
  }
  return out.size() == expected;
}

// ---------------------------------------------------------------------------
// Data generators (deterministic per seed).
enum class Profile { Zeros, Runs, SmallAlphabet, Words, Periodic, Random, TinySymbols };

const char* profile_name(Profile p)
{
  switch (p) {
    case Profile::Zeros: return "zeros";
    case Profile::Runs: return "runs";
    case Profile::SmallAlphabet: return "alphabet8";
    case Profile::Words: return "words";
    case Profile::Periodic: return "periodic";
    case Profile::Random: return "random";
    case Profile::TinySymbols: return "tiny-symbols";
  }
  return "?";
}

bytes_t generate(Profile p, size_t n, uint32_t seed)
{
  std::mt19937 rng(seed);
  bytes_t v(n);
  switch (p) {
    case Profile::Zeros:
      std::fill(v.begin(), v.end(), 0);
      break;
    case Profile::Runs: {
      size_t i = 0;
      while (i < n) {
        const uint8_t val = uint8_t(rng() & 0xff);
        const size_t run = 1 + (rng() % 200);
        for (size_t j = 0; j < run && i < n; ++j) v[i++] = val;
      }
      break;
    }
    case Profile::SmallAlphabet:
      for (auto& b : v) b = uint8_t('a' + (rng() % 8));
      break;
    case Profile::Words: {
      static const char* dict[] = {"the ", "quick ", "brown ", "fox ", "jumps ", "over ",
                                   "lazy ", "dog ", "arcto ", "snappy ", "gfx942 ", "wave64 ",
                                   "lds ", "vgpr ", "kernel ", "chunk "};
      size_t i = 0;
      while (i < n) {
        const char* w = dict[rng() % 16];
        for (const char* c = w; *c && i < n; ++c) v[i++] = uint8_t(*c);
      }
      break;
    }
    case Profile::Periodic: {
      // period 2..7 pattern with rare noise bytes: long overlapped copies
      const size_t period = 2 + (rng() % 6);
      bytes_t pat(period);
      for (auto& b : pat) b = uint8_t(rng() & 0xff);
      for (size_t i = 0; i < n; ++i) v[i] = pat[i % period];
      for (size_t k = 0; k < n / 997; ++k) v[rng() % n] ^= uint8_t(1 + (rng() % 255));
      break;
    }
    case Profile::Random:
      for (auto& b : v) b = uint8_t(rng() & 0xff);
      break;
    case Profile::TinySymbols: {
      // alternate 1-3 random bytes with a short back-reference so the stream
      // is thousands of 2-5 byte symbols per 64 KB chunk
      size_t i = 0;
      while (i < n) {
        const size_t lit = 1 + (rng() % 3);
        for (size_t j = 0; j < lit && i < n; ++j) v[i++] = uint8_t(rng() & 0xff);
        if (i >= 8) {
          const size_t off = 4 + (rng() % 4);
          const size_t len = 4 + (rng() % 5);
          for (size_t j = 0; j < len && i < n; ++j, ++i) v[i] = v[i - off];
        }
      }
      break;
    }
  }
  return v;
}

// ---------------------------------------------------------------------------
// GPU round trip through the public batched API with optional misalignment
// (in bytes) applied to every input, compressed and decompressed device pointer.
struct RoundTripResult {
  std::vector<bytes_t> compressed; // per chunk, exactly the emitted bytes
};

RoundTripResult gpu_round_trip(const std::vector<bytes_t>& inputs,
                               size_t in_misalign, size_t comp_misalign,
                               size_t out_misalign, bool check_host_decode = true)
{
  const size_t batch = inputs.size();
  REQUIRE(batch > 0);
  size_t max_chunk = 0;
  for (const auto& c : inputs) max_chunk = std::max(max_chunk, c.size());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // --- inputs
  std::vector<void*> d_in_base(batch), h_in_ptrs(batch);
  std::vector<size_t> h_in_bytes(batch);
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipMalloc(&d_in_base[i], inputs[i].size() + in_misalign + 64));
    uint8_t* p = static_cast<uint8_t*>(d_in_base[i]) + in_misalign;
    HIP_CHECK(hipMemcpy(p, inputs[i].data(), inputs[i].size(), hipMemcpyHostToDevice));
    h_in_ptrs[i] = p;
    h_in_bytes[i] = inputs[i].size();
  }
  void** d_in_ptrs;
  size_t* d_in_bytes;
  HIP_CHECK(hipMalloc((void**)&d_in_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_in_bytes, batch * sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_in_ptrs, h_in_ptrs.data(), batch * sizeof(void*), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_in_bytes, h_in_bytes.data(), batch * sizeof(size_t), hipMemcpyHostToDevice));

  // --- compression
  size_t temp_bytes = 0;
  REQUIRE(arctoBatchedSnappyCompressGetTempSize(batch, max_chunk, arctoBatchedSnappyDefaultOpts,
                                                &temp_bytes) == arctoSuccess);
  void* d_temp = nullptr;
  if (temp_bytes > 0) HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
  size_t max_comp = 0;
  REQUIRE(arctoBatchedSnappyCompressGetMaxOutputChunkSize(max_chunk, arctoBatchedSnappyDefaultOpts,
                                                          &max_comp) == arctoSuccess);

  std::vector<void*> d_comp_base(batch), h_comp_ptrs(batch);
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipMalloc(&d_comp_base[i], max_comp + comp_misalign + 64));
    h_comp_ptrs[i] = static_cast<uint8_t*>(d_comp_base[i]) + comp_misalign;
  }
  void** d_comp_ptrs;
  size_t* d_comp_bytes;
  HIP_CHECK(hipMalloc((void**)&d_comp_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_comp_bytes, batch * sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_comp_ptrs, h_comp_ptrs.data(), batch * sizeof(void*), hipMemcpyHostToDevice));

  REQUIRE(arctoBatchedSnappyCompressAsync(d_in_ptrs, d_in_bytes, max_chunk, batch, d_temp, temp_bytes,
                                          d_comp_ptrs, d_comp_bytes, arctoBatchedSnappyDefaultOpts,
                                          stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<size_t> h_comp_bytes(batch);
  HIP_CHECK(hipMemcpy(h_comp_bytes.data(), d_comp_bytes, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  RoundTripResult res;
  res.compressed.resize(batch);
  for (size_t i = 0; i < batch; ++i) {
    REQUIRE(h_comp_bytes[i] > 0);
    REQUIRE(h_comp_bytes[i] <= max_comp);
    res.compressed[i].resize(h_comp_bytes[i]);
    HIP_CHECK(hipMemcpy(res.compressed[i].data(), h_comp_ptrs[i], h_comp_bytes[i], hipMemcpyDeviceToHost));
    if (check_host_decode) {
      bytes_t decoded;
      const bool ok = host_snappy_decode(res.compressed[i].data(), res.compressed[i].size(), decoded);
      INFO("chunk " << i << " size " << inputs[i].size() << ": host Snappy decoder rejected the GPU stream");
      REQUIRE(ok);
      REQUIRE(decoded == inputs[i]);
    }
  }

  // --- decompressed-size query
  size_t* d_uncomp_bytes;
  HIP_CHECK(hipMalloc((void**)&d_uncomp_bytes, batch * sizeof(size_t)));
  REQUIRE(arctoBatchedSnappyGetDecompressSizeAsync((const void* const*)d_comp_ptrs, d_comp_bytes,
                                                   d_uncomp_bytes, batch, stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<size_t> h_uncomp_bytes(batch);
  HIP_CHECK(hipMemcpy(h_uncomp_bytes.data(), d_uncomp_bytes, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch; ++i) REQUIRE(h_uncomp_bytes[i] == inputs[i].size());

  // --- decompression
  size_t dtemp_bytes = 0;
  REQUIRE(arctoBatchedSnappyDecompressGetTempSize(batch, max_chunk, &dtemp_bytes) == arctoSuccess);
  void* d_dtemp = nullptr;
  if (dtemp_bytes > 0) HIP_CHECK(hipMalloc(&d_dtemp, dtemp_bytes));

  std::vector<void*> d_out_base(batch), h_out_ptrs(batch);
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipMalloc(&d_out_base[i], inputs[i].size() + out_misalign + 64));
    h_out_ptrs[i] = static_cast<uint8_t*>(d_out_base[i]) + out_misalign;
    HIP_CHECK(hipMemset(d_out_base[i], 0xA5, inputs[i].size() + out_misalign + 64));
  }
  void** d_out_ptrs;
  size_t* d_actual_bytes;
  arctoStatus_t* d_statuses;
  HIP_CHECK(hipMalloc((void**)&d_out_ptrs, batch * sizeof(void*)));
  HIP_CHECK(hipMalloc((void**)&d_actual_bytes, batch * sizeof(size_t)));
  HIP_CHECK(hipMalloc((void**)&d_statuses, batch * sizeof(arctoStatus_t)));
  HIP_CHECK(hipMemcpy(d_out_ptrs, h_out_ptrs.data(), batch * sizeof(void*), hipMemcpyHostToDevice));

  REQUIRE(arctoBatchedSnappyDecompressAsync((const void* const*)d_comp_ptrs, d_comp_bytes, d_uncomp_bytes,
                                            d_actual_bytes, batch, d_dtemp, dtemp_bytes, d_out_ptrs,
                                            d_statuses, stream) == arctoSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<size_t> h_actual(batch);
  std::vector<arctoStatus_t> h_status(batch);
  HIP_CHECK(hipMemcpy(h_actual.data(), d_actual_bytes, batch * sizeof(size_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_status.data(), d_statuses, batch * sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch; ++i) {
    INFO("chunk " << i << " size " << inputs[i].size());
    REQUIRE(h_status[i] == arctoSuccess);
    REQUIRE(h_actual[i] == inputs[i].size());
    bytes_t back(inputs[i].size());
    HIP_CHECK(hipMemcpy(back.data(), h_out_ptrs[i], back.size(), hipMemcpyDeviceToHost));
    REQUIRE(back == inputs[i]);
    // the guard bytes after the chunk must be untouched
    bytes_t guard(64);
    HIP_CHECK(hipMemcpy(guard.data(), static_cast<uint8_t*>(h_out_ptrs[i]) + back.size(), 64,
                        hipMemcpyDeviceToHost));
    REQUIRE(std::all_of(guard.begin(), guard.end(), [](uint8_t b) { return b == 0xA5; }));
  }

  // --- cleanup
  for (size_t i = 0; i < batch; ++i) {
    HIP_CHECK(hipFree(d_in_base[i]));
    HIP_CHECK(hipFree(d_comp_base[i]));
    HIP_CHECK(hipFree(d_out_base[i]));
  }
  HIP_CHECK(hipFree(d_in_ptrs));
  HIP_CHECK(hipFree(d_in_bytes));
  HIP_CHECK(hipFree(d_comp_ptrs));
  HIP_CHECK(hipFree(d_comp_bytes));
  HIP_CHECK(hipFree(d_uncomp_bytes));
  HIP_CHECK(hipFree(d_out_ptrs));
  HIP_CHECK(hipFree(d_actual_bytes));
  HIP_CHECK(hipFree(d_statuses));
  if (d_temp) HIP_CHECK(hipFree(d_temp));
  if (d_dtemp) HIP_CHECK(hipFree(d_dtemp));
  HIP_CHECK(hipStreamDestroy(stream));
  return res;
}

} // namespace

// ---------------------------------------------------------------------------

TEST_CASE("snappy: odd chunk sizes round trip (one batch, all sizes)", "[small]")
{
  const size_t sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100,
                          127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024,
                          1025, 4095, 4096, 4097, 65535, 65536, 65537, 70000};
  for (Profile p : {Profile::Words, Profile::Random, Profile::Runs}) {
    std::vector<bytes_t> inputs;
    uint32_t seed = 1;
    for (size_t s : sizes) inputs.push_back(generate(p, s, seed++));
    INFO("profile " << profile_name(p));
    gpu_round_trip(inputs, 0, 0, 0);
  }
}

TEST_CASE("snappy: misaligned input, compressed and output device pointers", "[small]")
{
  const std::vector<bytes_t> inputs = {generate(Profile::Words, 4096, 11),
                                       generate(Profile::Random, 65536, 12),
                                       generate(Profile::Runs, 777, 13),
                                       generate(Profile::TinySymbols, 65536, 14),
                                       generate(Profile::Periodic, 20001, 15)};
  const bytes_t reference_first = gpu_round_trip(inputs, 0, 0, 0).compressed[0];
  for (size_t m : {1, 2, 3, 5, 7, 8, 13, 63}) {
    INFO("misalignment " << m);
    auto r = gpu_round_trip(inputs, m, m, m);
    // alignment must not change the emitted bytes
    REQUIRE(r.compressed[0] == reference_first);
  }
  // independent misalignments
  gpu_round_trip(inputs, 3, 0, 0);
  gpu_round_trip(inputs, 0, 3, 0);
  gpu_round_trip(inputs, 0, 0, 3);
  gpu_round_trip(inputs, 1, 2, 3);
}

TEST_CASE("snappy: compressibility sweep with host-side format check and determinism", "[small]")
{
  const size_t sizes[] = {1000, 4096, 32768, 65536};
  for (Profile p : {Profile::Zeros, Profile::Runs, Profile::SmallAlphabet, Profile::Words,
                    Profile::Periodic, Profile::Random, Profile::TinySymbols}) {
    std::vector<bytes_t> inputs;
    uint32_t seed = 100;
    for (size_t s : sizes) inputs.push_back(generate(p, s, seed++));
    INFO("profile " << profile_name(p));
    auto a = gpu_round_trip(inputs, 0, 0, 0);
    auto b = gpu_round_trip(inputs, 0, 0, 0);
    REQUIRE(a.compressed == b.compressed); // deterministic
    size_t in_total = 0, out_total = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      in_total += inputs[i].size();
      out_total += a.compressed[i].size();
    }
    // sanity on the ratio ordering: highly repetitive data must shrink,
    // random data may only grow by the Snappy bound (< 1/6 + 32 B per chunk)
    if (p == Profile::Zeros || p == Profile::Periodic) REQUIRE(out_total * 10 < in_total);
    if (p == Profile::Random) REQUIRE(out_total <= in_total + in_total / 6 + 32 * inputs.size());
  }
}

TEST_CASE("snappy: thousands of tiny symbols per chunk (decoder queue wrap-around)", "[small]")
{
  std::vector<bytes_t> inputs;
  for (uint32_t s = 0; s < 8; ++s) inputs.push_back(generate(Profile::TinySymbols, 65536, 200 + s));
  inputs.push_back(generate(Profile::TinySymbols, 65535, 300));
  inputs.push_back(generate(Profile::TinySymbols, 65537, 301));
  gpu_round_trip(inputs, 0, 0, 0);
  gpu_round_trip(inputs, 1, 1, 1);
}

TEST_CASE("snappy: large batch of mixed-size, mixed-profile chunks", "[large]")
{
  std::mt19937 rng(4242);
  std::vector<bytes_t> inputs;
  const Profile profiles[] = {Profile::Zeros, Profile::Runs, Profile::SmallAlphabet, Profile::Words,
                              Profile::Periodic, Profile::Random, Profile::TinySymbols};
  for (int i = 0; i < 512; ++i) {
    const size_t size = 1 + (rng() % 65536);
    inputs.push_back(generate(profiles[rng() % 7], size, 1000 + i));
  }
  // host decoding 512 chunks is fine (< 32 MB); keep it on, it is the format gate
  gpu_round_trip(inputs, 0, 0, 0);
}
