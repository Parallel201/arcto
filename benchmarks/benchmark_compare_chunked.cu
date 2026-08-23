// MIT License
//
// Copyright (C) 2025-2026 Cristiano Künas.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// One measurement harness, three interchangeable backends: ARCTO (its own
// batched C API, HIP on AMD and the same sources through the CUDA backend on
// NVIDIA), NVIDIA nvCOMP 2.2 (the release ARCTO was translated from) and
// NVIDIA nvCOMP 5.x. The protocol -- chunking, duplication, warmup,
// repetitions, event timing, round-trip verification and the per-repetition
// CSV -- is identical for all three, so the numbers are directly comparable.
//
// Build (ARCTO, either platform): part of the normal benchmark set.
// Build (comparison against nvCOMP), e.g.
//   nvcc -O3 -std=c++17 -x cu -DARCTO_COMPARE_NVCOMP5 \
//        -I<nvcomp>/include benchmark_compare_chunked.cu \
//        -L<nvcomp>/lib -lnvcomp -o benchmark_compare_nvcomp5
//   nvcc -O3 -std=c++17 -x cu -DARCTO_COMPARE_NVCOMP22 \
//        -I<nvcomp22>/include benchmark_compare_chunked.cu \
//        -L<nvcomp22>/build/lib -lnvcomp -o benchmark_compare_nvcomp22
//
// Usage: benchmark_compare_chunked -f <file> [-a lz4|snappy|cascaded]
//        [-p chunk_bytes] [-x duplicates] [-w warmup] [-i iterations]
//        [-g gpu] [-c] [-t tag]
//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(ARCTO_COMPARE_NVCOMP5) || defined(ARCTO_COMPARE_NVCOMP22)
#include <cuda_runtime.h>
#include "nvcomp.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/lz4.h"
#include "nvcomp/snappy.h"
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp cudaDeviceProp
#define BACKEND_SUCCESS ((int)nvcompSuccess)
#else
#include <hip/hip_runtime.h>
#include "arcto.h"
#include "arcto/cascaded.h"
#include "arcto/lz4.h"
#include "arcto/snappy.h"
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuSetDevice hipSetDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp hipDeviceProp_t
#define BACKEND_SUCCESS ((int)arctoSuccess)
#endif

#if defined(ARCTO_COMPARE_NVCOMP5)
#define BACKEND_NAME "nvcomp5"
#elif defined(ARCTO_COMPARE_NVCOMP22)
#define BACKEND_NAME "nvcomp22"
#else
#define BACKEND_NAME "arcto"
#endif

#define GPU_CHECK(expr)                                                        \
  do {                                                                         \
    gpuError_t _e = (expr);                                                    \
    if (_e != gpuSuccess) {                                                    \
      std::fprintf(stderr, "GPU error %s:%d: %s -- %s\n", __FILE__, __LINE__,   \
                   #expr, gpuGetErrorString(_e));                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define REQUIRE(cond, msg)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d %s -- %s\n", __FILE__, __LINE__, #cond, \
                   msg);                                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

using status_t = int;

// ---------------------------------------------------------------------------
// Backend bindings. Every codec exposes the same five calls; the differences
// between the API generations (option structs, the extra "max total bytes"
// argument of nvCOMP 5's temp-size queries, the per-chunk status array) live
// in these wrappers only.
// ---------------------------------------------------------------------------

struct Codec
{
  const char* name;
  status_t (*comp_temp)(size_t batch, size_t max_chunk, size_t* temp_bytes);
  status_t (*comp_max_out)(size_t max_chunk, size_t* max_out_bytes);
  status_t (*comp_async)(
      const void* const* in_ptrs, const size_t* in_bytes, size_t max_chunk,
      size_t batch, void* temp, size_t temp_bytes, void* const* out_ptrs,
      size_t* out_bytes, void* statuses, gpuStream_t stream);
  status_t (*decomp_temp)(size_t batch, size_t max_chunk, size_t* temp_bytes);
  status_t (*decomp_async)(
      const void* const* comp_ptrs, const size_t* comp_bytes,
      const size_t* uncomp_buffer_bytes, size_t* actual_bytes, size_t batch,
      void* temp, size_t temp_bytes, void* const* out_ptrs, void* statuses,
      gpuStream_t stream);
  size_t status_size; // bytes of one per-chunk status entry
};

#if defined(ARCTO_COMPARE_NVCOMP5)

// ---- nvCOMP 5.x -----------------------------------------------------------
#define DEFINE_NVCOMP5_CODEC(TAG, NAME, COMP_OPTS, DECOMP_OPTS)                \
  static status_t TAG##_comp_temp(size_t b, size_t c, size_t* t)               \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressGetTempSizeAsync(            \
        b, c, COMP_OPTS, t, b* c);                                             \
  }                                                                            \
  static status_t TAG##_comp_max_out(size_t c, size_t* o)                      \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressGetMaxOutputChunkSize(       \
        c, COMP_OPTS, o);                                                      \
  }                                                                            \
  static status_t TAG##_comp_async(                                            \
      const void* const* ip, const size_t* ib, size_t mc, size_t b, void* tp,  \
      size_t tb, void* const* op, size_t* ob, void* st, gpuStream_t s)         \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressAsync(                       \
        ip, ib, mc, b, tp, tb, op, ob, COMP_OPTS, (nvcompStatus_t*)st, s);     \
  }                                                                            \
  static status_t TAG##_decomp_temp(size_t b, size_t c, size_t* t)             \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##DecompressGetTempSizeAsync(          \
        b, c, DECOMP_OPTS, t, b* c);                                           \
  }                                                                            \
  static status_t TAG##_decomp_async(                                          \
      const void* const* cp, const size_t* cb, const size_t* ubb, size_t* ab,  \
      size_t b, void* tp, size_t tb, void* const* op, void* st, gpuStream_t s) \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##DecompressAsync(                     \
        cp, cb, ubb, ab, b, tp, tb, op, DECOMP_OPTS, (nvcompStatus_t*)st, s);  \
  }

DEFINE_NVCOMP5_CODEC(
    lz4, LZ4, nvcompBatchedLZ4CompressDefaultOpts,
    nvcompBatchedLZ4DecompressDefaultOpts)
DEFINE_NVCOMP5_CODEC(
    snappy, Snappy, nvcompBatchedSnappyCompressDefaultOpts,
    nvcompBatchedSnappyDecompressDefaultOpts)
DEFINE_NVCOMP5_CODEC(
    cascaded, Cascaded, nvcompBatchedCascadedCompressDefaultOpts,
    nvcompBatchedCascadedDecompressDefaultOpts)
constexpr size_t kStatusSize = sizeof(nvcompStatus_t);

#elif defined(ARCTO_COMPARE_NVCOMP22)

// ---- nvCOMP 2.2 -----------------------------------------------------------
// Compression takes no status array in 2.2; the wrapper ignores the argument.
#define DEFINE_NVCOMP22_CODEC(TAG, NAME, OPTS)                                 \
  static status_t TAG##_comp_temp(size_t b, size_t c, size_t* t)               \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressGetTempSize(b, c, OPTS, t);  \
  }                                                                            \
  static status_t TAG##_comp_max_out(size_t c, size_t* o)                      \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressGetMaxOutputChunkSize(       \
        c, OPTS, o);                                                           \
  }                                                                            \
  static status_t TAG##_comp_async(                                            \
      const void* const* ip, const size_t* ib, size_t mc, size_t b, void* tp,  \
      size_t tb, void* const* op, size_t* ob, void*, gpuStream_t s)            \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##CompressAsync(                       \
        ip, ib, mc, b, tp, tb, op, ob, OPTS, s);                               \
  }                                                                            \
  static status_t TAG##_decomp_temp(size_t b, size_t c, size_t* t)             \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##DecompressGetTempSize(b, c, t);      \
  }                                                                            \
  static status_t TAG##_decomp_async(                                          \
      const void* const* cp, const size_t* cb, const size_t* ubb, size_t* ab,  \
      size_t b, void* tp, size_t tb, void* const* op, void* st, gpuStream_t s) \
  {                                                                            \
    return (status_t)nvcompBatched##NAME##DecompressAsync(                     \
        cp, cb, ubb, ab, b, tp, tb, op, (nvcompStatus_t*)st, s);               \
  }

DEFINE_NVCOMP22_CODEC(lz4, LZ4, nvcompBatchedLZ4DefaultOpts)
DEFINE_NVCOMP22_CODEC(snappy, Snappy, nvcompBatchedSnappyDefaultOpts)
DEFINE_NVCOMP22_CODEC(cascaded, Cascaded, nvcompBatchedCascadedDefaultOpts)
constexpr size_t kStatusSize = sizeof(nvcompStatus_t);

#else

// ---- ARCTO ----------------------------------------------------------------
#define DEFINE_ARCTO_CODEC(TAG, NAME, OPTS)                                    \
  static status_t TAG##_comp_temp(size_t b, size_t c, size_t* t)               \
  {                                                                            \
    return (status_t)arctoBatched##NAME##CompressGetTempSize(b, c, OPTS, t);   \
  }                                                                            \
  static status_t TAG##_comp_max_out(size_t c, size_t* o)                      \
  {                                                                            \
    return (status_t)arctoBatched##NAME##CompressGetMaxOutputChunkSize(        \
        c, OPTS, o);                                                           \
  }                                                                            \
  static status_t TAG##_comp_async(                                            \
      const void* const* ip, const size_t* ib, size_t mc, size_t b, void* tp,  \
      size_t tb, void* const* op, size_t* ob, void*, gpuStream_t s)            \
  {                                                                            \
    return (status_t)arctoBatched##NAME##CompressAsync(                        \
        ip, ib, mc, b, tp, tb, op, ob, OPTS, s);                               \
  }                                                                            \
  static status_t TAG##_decomp_temp(size_t b, size_t c, size_t* t)             \
  {                                                                            \
    return (status_t)arctoBatched##NAME##DecompressGetTempSize(b, c, t);       \
  }                                                                            \
  static status_t TAG##_decomp_async(                                          \
      const void* const* cp, const size_t* cb, const size_t* ubb, size_t* ab,  \
      size_t b, void* tp, size_t tb, void* const* op, void* st, gpuStream_t s) \
  {                                                                            \
    return (status_t)arctoBatched##NAME##DecompressAsync(                      \
        cp, cb, ubb, ab, b, tp, tb, op, (arctoStatus_t*)st, s);                \
  }

DEFINE_ARCTO_CODEC(lz4, LZ4, arctoBatchedLZ4DefaultOpts)
DEFINE_ARCTO_CODEC(snappy, Snappy, arctoBatchedSnappyDefaultOpts)
DEFINE_ARCTO_CODEC(cascaded, Cascaded, arctoBatchedCascadedDefaultOpts)
constexpr size_t kStatusSize = sizeof(arctoStatus_t);

#endif

const Codec kCodecs[] = {
    {"lz4", lz4_comp_temp, lz4_comp_max_out, lz4_comp_async, lz4_decomp_temp,
     lz4_decomp_async, kStatusSize},
    {"snappy", snappy_comp_temp, snappy_comp_max_out, snappy_comp_async,
     snappy_decomp_temp, snappy_decomp_async, kStatusSize},
    {"cascaded", cascaded_comp_temp, cascaded_comp_max_out, cascaded_comp_async,
     cascaded_decomp_temp, cascaded_decomp_async, kStatusSize},
};

// ---------------------------------------------------------------------------
// Input handling: read the file, split it into chunks, duplicate the chunk
// list `duplicates` times (element-wise; never a self-range insert).
// ---------------------------------------------------------------------------

std::vector<char> read_file(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  REQUIRE(f.good(), ("cannot open " + path).c_str());
  const std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(n));
  REQUIRE(f.read(data.data(), n).good(), "read failed");
  return data;
}

std::vector<std::vector<char>>
chunk_file(const std::vector<char>& data, size_t chunk_size, size_t duplicates)
{
  std::vector<std::vector<char>> chunks;
  for (size_t off = 0; off < data.size(); off += chunk_size) {
    const size_t n = std::min(chunk_size, data.size() - off);
    chunks.emplace_back(data.begin() + off, data.begin() + off + n);
  }
  const size_t base = chunks.size();
  chunks.reserve(base * (duplicates + 1));
  for (size_t d = 0; d < duplicates; ++d)
    for (size_t c = 0; c < base; ++c)
      chunks.push_back(chunks[c]);
  return chunks;
}

struct DeviceBatch
{
  void* data = nullptr;         // one allocation holding every chunk
  void** ptrs = nullptr;        // device array of per-chunk pointers
  size_t* sizes = nullptr;      // device array of per-chunk sizes
  size_t total_bytes = 0;
  size_t count = 0;

  void free_all()
  {
    if (data) GPU_CHECK(gpuFree(data));
    if (ptrs) GPU_CHECK(gpuFree(ptrs));
    if (sizes) GPU_CHECK(gpuFree(sizes));
    data = nullptr; ptrs = nullptr; sizes = nullptr;
  }
};

// Upload the chunks into one device allocation (8-byte aligned slots).
DeviceBatch upload(const std::vector<std::vector<char>>& chunks)
{
  DeviceBatch b;
  b.count = chunks.size();
  std::vector<size_t> offsets(chunks.size());
  std::vector<size_t> sizes(chunks.size());
  size_t off = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    offsets[i] = off;
    sizes[i] = chunks[i].size();
    off += (chunks[i].size() + 7u) & ~size_t(7u);
    b.total_bytes += chunks[i].size();
  }
  GPU_CHECK(gpuMalloc(&b.data, off ? off : 8));
  std::vector<void*> ptrs(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    ptrs[i] = static_cast<char*>(b.data) + offsets[i];
    if (!chunks[i].empty())
      GPU_CHECK(gpuMemcpy(ptrs[i], chunks[i].data(), chunks[i].size(),
                          gpuMemcpyHostToDevice));
  }
  GPU_CHECK(gpuMalloc((void**)&b.ptrs, sizeof(void*) * chunks.size()));
  GPU_CHECK(gpuMalloc((void**)&b.sizes, sizeof(size_t) * chunks.size()));
  GPU_CHECK(gpuMemcpy(b.ptrs, ptrs.data(), sizeof(void*) * chunks.size(),
                      gpuMemcpyHostToDevice));
  GPU_CHECK(gpuMemcpy(b.sizes, sizes.data(), sizeof(size_t) * chunks.size(),
                      gpuMemcpyHostToDevice));
  return b;
}

// Slots of `slot_bytes` each, for compressed or decompressed output.
DeviceBatch make_slots(size_t count, size_t slot_bytes)
{
  DeviceBatch b;
  b.count = count;
  const size_t stride = (slot_bytes + 7u) & ~size_t(7u);
  GPU_CHECK(gpuMalloc(&b.data, stride * count));
  std::vector<void*> ptrs(count);
  for (size_t i = 0; i < count; ++i)
    ptrs[i] = static_cast<char*>(b.data) + i * stride;
  GPU_CHECK(gpuMalloc((void**)&b.ptrs, sizeof(void*) * count));
  GPU_CHECK(gpuMalloc((void**)&b.sizes, sizeof(size_t) * count));
  GPU_CHECK(gpuMemcpy(b.ptrs, ptrs.data(), sizeof(void*) * count,
                      gpuMemcpyHostToDevice));
  b.total_bytes = stride * count;
  return b;
}

struct Args
{
  std::string file;
  std::string codec = "lz4";
  std::string tag;
  size_t chunk_size = 65536;
  size_t duplicates = 0;
  size_t warmup = 3;
  size_t iters = 10;
  int gpu = 0;
  bool csv = false;
};

void usage(const char* name)
{
  std::fprintf(
      stderr,
      "Usage: %s -f <file> [-a lz4|snappy|cascaded] [-p chunk_bytes]\n"
      "          [-x duplicates] [-w warmup] [-i iterations] [-g gpu] [-c] [-t tag]\n"
      "Backend: %s\n",
      name, BACKEND_NAME);
}

Args parse(int argc, char** argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() {
      if (++i >= argc) { usage(argv[0]); std::exit(1); }
      return std::string(argv[i]);
    };
    if (s == "-f") a.file = next();
    else if (s == "-a") a.codec = next();
    else if (s == "-p") a.chunk_size = std::stoull(next());
    else if (s == "-x") a.duplicates = std::stoull(next());
    else if (s == "-w") a.warmup = std::stoull(next());
    else if (s == "-i") a.iters = std::stoull(next());
    else if (s == "-g") a.gpu = std::stoi(next());
    else if (s == "-t") a.tag = next();
    else if (s == "-c") a.csv = true;
    else if (s == "-h") { usage(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "unknown argument %s\n", s.c_str()); usage(argv[0]); std::exit(1); }
  }
  if (a.file.empty()) { usage(argv[0]); std::exit(1); }
  return a;
}

} // namespace

int main(int argc, char** argv)
{
  const Args args = parse(argc, argv);
  const Codec* codec = nullptr;
  for (const Codec& c : kCodecs)
    if (args.codec == c.name) codec = &c;
  REQUIRE(codec != nullptr, "unknown codec (use lz4, snappy or cascaded)");

  GPU_CHECK(gpuSetDevice(args.gpu));
  gpuDeviceProp props;
  GPU_CHECK(gpuGetDeviceProperties(&props, args.gpu));

  const std::vector<char> file_data = read_file(args.file);
  const std::vector<std::vector<char>> chunks
      = chunk_file(file_data, args.chunk_size, args.duplicates);
  const size_t batch = chunks.size();
  size_t max_chunk = 0, uncompressed_bytes = 0;
  for (const auto& c : chunks) {
    max_chunk = std::max(max_chunk, c.size());
    uncompressed_bytes += c.size();
  }

  gpuStream_t stream;
  GPU_CHECK(gpuStreamCreate(&stream));
  gpuEvent_t ev_start, ev_stop;
  GPU_CHECK(gpuEventCreate(&ev_start));
  GPU_CHECK(gpuEventCreate(&ev_stop));

  DeviceBatch input = upload(chunks);

  size_t comp_temp_bytes = 0, max_out = 0;
  REQUIRE(codec->comp_temp(batch, max_chunk, &comp_temp_bytes) == BACKEND_SUCCESS,
          "compress temp size query failed");
  REQUIRE(codec->comp_max_out(max_chunk, &max_out) == BACKEND_SUCCESS,
          "compress max output query failed");

  void* comp_temp = nullptr;
  if (comp_temp_bytes) GPU_CHECK(gpuMalloc(&comp_temp, comp_temp_bytes));
  DeviceBatch comp = make_slots(batch, max_out);
  void* statuses = nullptr;
  GPU_CHECK(gpuMalloc(&statuses, codec->status_size * batch));

  // ---- compression -------------------------------------------------------
  std::vector<double> comp_ms;
  for (size_t it = 0; it < args.warmup + args.iters; ++it) {
    GPU_CHECK(gpuEventRecord(ev_start, stream));
    const status_t st = codec->comp_async(
        (const void* const*)input.ptrs, input.sizes, max_chunk, batch,
        comp_temp, comp_temp_bytes, comp.ptrs, comp.sizes, statuses, stream);
    GPU_CHECK(gpuEventRecord(ev_stop, stream));
    GPU_CHECK(gpuStreamSynchronize(stream));
    REQUIRE(st == BACKEND_SUCCESS, "compression failed");
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, ev_start, ev_stop));
    if (it >= args.warmup) comp_ms.push_back(ms);
  }

  std::vector<size_t> h_comp_sizes(batch);
  GPU_CHECK(gpuMemcpy(h_comp_sizes.data(), comp.sizes, sizeof(size_t) * batch,
                      gpuMemcpyDeviceToHost));
  size_t compressed_bytes = 0;
  for (size_t s : h_comp_sizes) compressed_bytes += s;

  // ---- decompression -----------------------------------------------------
  size_t decomp_temp_bytes = 0;
  REQUIRE(codec->decomp_temp(batch, max_chunk, &decomp_temp_bytes) == BACKEND_SUCCESS,
          "decompress temp size query failed");
  void* decomp_temp = nullptr;
  if (decomp_temp_bytes) GPU_CHECK(gpuMalloc(&decomp_temp, decomp_temp_bytes));
  DeviceBatch out = make_slots(batch, max_chunk);
  size_t* actual_sizes = nullptr;
  GPU_CHECK(gpuMalloc((void**)&actual_sizes, sizeof(size_t) * batch));

  std::vector<double> decomp_ms;
  for (size_t it = 0; it < args.warmup + args.iters; ++it) {
    GPU_CHECK(gpuEventRecord(ev_start, stream));
    const status_t st = codec->decomp_async(
        (const void* const*)comp.ptrs, comp.sizes, input.sizes, actual_sizes,
        batch, decomp_temp, decomp_temp_bytes, out.ptrs, statuses, stream);
    GPU_CHECK(gpuEventRecord(ev_stop, stream));
    GPU_CHECK(gpuStreamSynchronize(stream));
    REQUIRE(st == BACKEND_SUCCESS, "decompression failed");
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, ev_start, ev_stop));
    if (it >= args.warmup) decomp_ms.push_back(ms);
  }

  // ---- round-trip verification (outside the timed loops) -----------------
  {
    std::vector<size_t> h_actual(batch);
    GPU_CHECK(gpuMemcpy(h_actual.data(), actual_sizes, sizeof(size_t) * batch,
                        gpuMemcpyDeviceToHost));
    std::vector<void*> h_out_ptrs(batch);
    GPU_CHECK(gpuMemcpy(h_out_ptrs.data(), out.ptrs, sizeof(void*) * batch,
                        gpuMemcpyDeviceToHost));
    const size_t n_check = std::min<size_t>(batch, 64);
    std::vector<char> host(max_chunk);
    for (size_t i = 0; i < n_check; ++i) {
      REQUIRE(h_actual[i] == chunks[i].size(), "decompressed size mismatch");
      if (chunks[i].empty()) continue;
      GPU_CHECK(gpuMemcpy(host.data(), h_out_ptrs[i], chunks[i].size(),
                          gpuMemcpyDeviceToHost));
      REQUIRE(std::memcmp(host.data(), chunks[i].data(), chunks[i].size()) == 0,
              "round-trip mismatch");
    }
  }

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0
                     : (v.size() % 2 ? v[v.size() / 2]
                                     : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
  };
  const double c_ms = median(comp_ms), d_ms = median(decomp_ms);
  const double gb = static_cast<double>(uncompressed_bytes) / 1.0e9;
  const double c_gbs = gb / (c_ms / 1.0e3), d_gbs = gb / (d_ms / 1.0e3);
  const double ratio = static_cast<double>(uncompressed_bytes)
                       / static_cast<double>(compressed_bytes);

  // Per-repetition CSV, same columns as the chunked benchmarks so that the
  // same analysis scripts read both.
  if (const char* path = std::getenv("ARCTO_PER_REP_CSV")) {
    const char* tag = std::getenv("ARCTO_PER_REP_TAG");
    const bool exists = std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (!exists)
      f << "Tag,Rep,ChunkSize,NumFiles,Duplicates,UncompressedBytes,Chunks,"
           "CompThroughputGBs,DecompThroughputGBs,CompTimeMs,DecompTimeMs\n";
    for (size_t i = 0; i < comp_ms.size(); ++i)
      f << (tag ? tag : BACKEND_NAME) << ',' << i << ',' << args.chunk_size
        << ",1," << args.duplicates << ',' << uncompressed_bytes << ',' << batch
        << ',' << gb / (comp_ms[i] / 1.0e3) << ','
        << gb / (decomp_ms[std::min(i, decomp_ms.size() - 1)] / 1.0e3) << ','
        << comp_ms[i] << ',' << decomp_ms[std::min(i, decomp_ms.size() - 1)]
        << '\n';
  }

  if (args.csv) {
    std::printf("%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                BACKEND_NAME, codec->name, args.file.c_str(), args.chunk_size,
                args.duplicates, batch, uncompressed_bytes, compressed_bytes,
                ratio, c_gbs, d_gbs, c_ms, d_ms);
  } else {
    std::printf("backend %s  codec %s  device %s\n", BACKEND_NAME, codec->name,
                props.name);
    std::printf("  chunks %zu of <= %zu B, uncompressed %.2f MB, ratio %.3fx\n",
                batch, max_chunk, uncompressed_bytes / 1048576.0, ratio);
    std::printf("  compression   %8.2f GB/s (%.3f ms, median of %zu)\n", c_gbs,
                c_ms, comp_ms.size());
    std::printf("  decompression %8.2f GB/s (%.3f ms, median of %zu)\n", d_gbs,
                d_ms, decomp_ms.size());
  }

  if (comp_temp) GPU_CHECK(gpuFree(comp_temp));
  if (decomp_temp) GPU_CHECK(gpuFree(decomp_temp));
  GPU_CHECK(gpuFree(statuses));
  GPU_CHECK(gpuFree(actual_sizes));
  input.free_all();
  comp.free_all();
  out.free_all();
  GPU_CHECK(gpuEventDestroy(ev_start));
  GPU_CHECK(gpuEventDestroy(ev_stop));
  GPU_CHECK(gpuStreamDestroy(stream));
  return 0;
}
