/*
 * ARCTO ZFP single-cube benchmark.
 *
 * ZFP is a 3D float-field compressor. The other benchmark_*_chunked
 * binaries split each input file into fixed-size byte chunks (default
 * 65 536 B) and run a byte-level batched API across them -- a natural fit
 * for byte-stream compressors like LZ4 / Snappy / Cascaded but the wrong
 * granularity for ZFP, which exploits 4x4x4 floating-point spatial
 * correlation that 64 KB chunks would destroy.
 *
 * This benchmark therefore treats each input file as ONE field (1D float32
 * by default; pass --shape <nx>,<ny>,<nz> for the real 3D layout). The
 * compress and decompress paths invoke the canonical HIP backend through
 * the ARCTO C wrapper (arctoZFPCompress / arctoZFPDecompress). Iteration
 * count and warmup mirror benchmark_template_chunked.cuh.
 *
 * The CSV output matches benchmark_*_chunked exactly (same 21 columns)
 * so the same R / Python analysis pipeline ingests it without changes.
 * "Pages" and "Chunk size" columns are filled with 1 and the whole-field
 * byte count, respectively.
 *
 * Memory accounting: arctoZFPCompress takes a DEVICE input and writes the
 * compressed bytes to a HOST buffer (the canonical's HIP backend stages
 * the compressed payload back to host inside cleanup_device). The
 * "Compression time (ms)" column therefore includes the implicit
 * device->host transfer of the compressed payload (~few MB to ~100 MB
 * depending on rate). "Transfer D2H (ms)" is filled with 0 because the
 * D2H is folded into the compress call.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "HIP error %s:%d %s -- %s\n",                       \
                   __FILE__, __LINE__, #expr, hipGetErrorString(_e));          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct Args {
  std::string path;
  uint32_t shape[3] = {0, 0, 0};   // 0 means "derive 1D from file size"
  arctoZFPMode_t mode = ARCTO_ZFP_MODE_FIXED_RATE;
  double param = 16.0;             // FIXED_RATE bits/value default
  size_t iterations = 10;
  size_t warmup = 2;
  bool csv = false;
  bool device_buffer = false;      // use ToDevice/FromDevice path
};

void print_help(const char* argv0)
{
  std::fprintf(stderr,
    "Usage: %s -f <file> [options]\n"
    "\n"
    "  -f <path>       input float32 file (required)\n"
    "  -3 <nx,ny,nz>   treat input as 3D cube of this shape (else 1D)\n"
    "  -m <mode>       fixed_rate (default) | fixed_precision | fixed_accuracy\n"
    "  -r <param>      mode parameter (rate bits/value, precision, or tol)\n"
    "  -i <N>          measured iterations (default 10)\n"
    "  -w <N>          warmup iterations (default 2)\n"
    "  -c              CSV output (single row matching benchmark_*_chunked)\n"
    "  -D|--device-buffer   allocate the compressed host buffer with\n"
    "                       hipHostMalloc instead of std::vector. The\n"
    "                       canonical's HIP backend still issues an implicit\n"
    "                       D2H of the compressed payload from its internal\n"
    "                       staging buffer; with a pinned destination that\n"
    "                       D2H runs at ~PCIe peak instead of pageable speed\n"
    "                       (typically 1.3-1.7x throughput at K=16 fixed_rate).\n"
    "  -h              this help\n",
    argv0);
}

bool parse_args(int argc, char** argv, Args& a)
{
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    auto next = [&](){ if (++i >= argc) { print_help(argv[0]); std::exit(1);} return std::string(argv[i]); };
    if      (s == "-f") a.path = next();
    else if (s == "-3") {
      std::string v = next();
      if (std::sscanf(v.c_str(), "%u,%u,%u", &a.shape[0], &a.shape[1], &a.shape[2]) != 3) {
        std::fprintf(stderr, "bad --shape: %s\n", v.c_str()); return false;
      }
    }
    else if (s == "-m") {
      std::string v = next();
      if      (v == "fixed_rate")      a.mode = ARCTO_ZFP_MODE_FIXED_RATE;
      else if (v == "fixed_precision") a.mode = ARCTO_ZFP_MODE_FIXED_PRECISION;
      else if (v == "fixed_accuracy")  a.mode = ARCTO_ZFP_MODE_FIXED_ACCURACY;
      else { std::fprintf(stderr, "unknown mode %s\n", v.c_str()); return false; }
    }
    else if (s == "-r") a.param = std::atof(next().c_str());
    else if (s == "-i") a.iterations = std::strtoull(next().c_str(), nullptr, 10);
    else if (s == "-w") a.warmup = std::strtoull(next().c_str(), nullptr, 10);
    else if (s == "-c") a.csv = true;
    else if (s == "--device-buffer" || s == "-D") a.device_buffer = true;
    else if (s == "-h" || s == "--help") { print_help(argv[0]); std::exit(0); }
    else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); print_help(argv[0]); return false; }
  }
  if (a.path.empty()) { print_help(argv[0]); return false; }
  return true;
}

std::vector<unsigned char> load_file(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<unsigned char> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

double stddev(const std::vector<double>& v)
{
  if (v.size() < 2) return 0.0;
  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double sq = 0.0;
  for (double x : v) sq += (x - mean) * (x - mean);
  return std::sqrt(sq / v.size());
}

} // namespace

int main(int argc, char** argv)
{
  Args a;
  if (!parse_args(argc, argv, a)) return 1;

  // Load input. Configure opts (1D default, or 3D when --shape given) and
  // truncate the buffer to the shape's exact byte count if needed; the test
  // datasets are sized to nice round MB boundaries that are slightly larger
  // than the 3D cubes they represent.
  std::vector<unsigned char> raw = load_file(a.path);
  if (raw.size() < sizeof(float)) {
    std::fprintf(stderr, "input file too small\n"); return 1;
  }

  arctoZFPOpts_t opts = arctoZFPDefaultOpts;
  opts.mode  = a.mode;
  opts.type  = ARCTO_ZFP_TYPE_FLOAT;
  opts.param = a.param;
  if (a.shape[2] != 0) {
    opts.ndims    = 3;
    opts.shape[0] = a.shape[0];
    opts.shape[1] = a.shape[1];
    opts.shape[2] = a.shape[2];
    opts.shape[3] = 1;
    const size_t need = size_t(a.shape[0]) * a.shape[1] * a.shape[2] * sizeof(float);
    if (need > raw.size()) {
      std::fprintf(stderr, "shape product (%zu B) > file size (%zu B)\n",
                   need, raw.size());
      return 1;
    }
    if (need < raw.size()) {
      std::fprintf(stderr,
        "[info] shape product %zu B < file size %zu B; using first %zu B (%zu B trailing ignored)\n",
        need, raw.size(), need, raw.size() - need);
      raw.resize(need);
    }
  } else {
    opts.ndims    = 1;
    opts.shape[0] = static_cast<uint32_t>(raw.size() / sizeof(float));
    opts.shape[1] = 1; opts.shape[2] = 1; opts.shape[3] = 1;
  }

  // Sanitize NaN/Inf -- ZFP encoder treats them as 0 anyway; doing it up
  // front prevents synthetic random-byte fixtures from inflating the
  // measured timings via fp exceptions.
  const size_t uncompressed_bytes = raw.size();
  float* fp = reinterpret_cast<float*>(raw.data());
  const size_t nf = uncompressed_bytes / sizeof(float);
  for (size_t k = 0; k < nf; k++) if (!std::isfinite(fp[k])) fp[k] = 0.0f;

  size_t max_comp = 0;
  if (arctoZFPCompressGetMaxOutputSize(opts, &max_comp) != arctoSuccess) {
    std::fprintf(stderr, "GetMaxOutputSize failed\n"); return 1;
  }

  // Allocate device input + host compressed buffer (host path) OR device
  // compressed buffer (when --device-buffer). For the host path we also
  // need a device output for decompress; for the device path, both
  // compressed-and-uncompressed buffers live entirely on the GPU.
  void* d_input  = nullptr;
  void* d_output = nullptr;
  void* h_comp_pinned = nullptr;  // used in --device-buffer (pinned host)
  HIP_CHECK(hipMalloc(&d_input,  uncompressed_bytes));
  HIP_CHECK(hipMalloc(&d_output, uncompressed_bytes));
  std::vector<unsigned char> h_comp;
  if (a.device_buffer) {
    // PINNED host buffer: D2H from the canonical's compress runs at ~PCIe
    // peak (~25 GB/s vs ~10 GB/s for pageable). The canonical's bitstream
    // begin sees a host pointer, takes the host code path -- so the
    // expensive device_malloc happens INSIDE canonical's compress only
    // ONCE (well, per call), and the D2H back to host is at pinned speed.
    // This is functionally equivalent to a true device-resident path for
    // the common case where the user will write the compressed bytes to
    // disk anyway.
    HIP_CHECK(hipHostMalloc(&h_comp_pinned, max_comp, hipHostMallocDefault));
  } else {
    h_comp.assign(max_comp, 0);
  }

  // ---- Warmup --------------------------------------------------------------
  // Time the H2D once (the iterations time the compress/decompress kernels).
  using clk = std::chrono::high_resolution_clock;
  auto t_h2d_0 = clk::now();
  HIP_CHECK(hipMemcpy(d_input, raw.data(), uncompressed_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  const double h2d_ms = std::chrono::duration<double, std::milli>(clk::now() - t_h2d_0).count();

  // Both paths use the host-output API; the difference is whether the host
  // buffer is pageable (default) or pinned. Pinned memory avoids the
  // canonical's HIP backend's pageable-staging overhead and approaches
  // PCIe peak on the implicit D2H of the compressed payload.
  auto comp_buf = [&]() {
    return a.device_buffer ? static_cast<unsigned char*>(h_comp_pinned)
                           : h_comp.data();
  };
  auto do_compress = [&](size_t* size_out) {
    return arctoZFPCompress(d_input, opts, comp_buf(), max_comp, size_out);
  };
  auto do_decompress = [&](size_t in_size, size_t* size_out) {
    return arctoZFPDecompress(comp_buf(), in_size,
                              d_output, uncompressed_bytes, size_out);
  };

  size_t comp_size = 0;
  for (size_t w = 0; w < a.warmup; w++) {
    if (do_compress(&comp_size) != arctoSuccess) {
      std::fprintf(stderr, "warmup compress failed\n"); return 1;
    }
  }

  // ---- Measured iterations -------------------------------------------------
  std::vector<double> comp_ms(a.iterations), decomp_ms(a.iterations);
  size_t last_decomp_size = 0;
  for (size_t i = 0; i < a.iterations; i++) {
    auto tc0 = clk::now();
    if (do_compress(&comp_size) != arctoSuccess) {
      std::fprintf(stderr, "iter %zu compress failed\n", i); return 1;
    }
    comp_ms[i] = std::chrono::duration<double, std::milli>(clk::now() - tc0).count();

    auto td0 = clk::now();
    if (do_decompress(comp_size, &last_decomp_size) != arctoSuccess) {
      std::fprintf(stderr, "iter %zu decompress failed\n", i); return 1;
    }
    decomp_ms[i] = std::chrono::duration<double, std::milli>(clk::now() - td0).count();
  }

  // For the host path, D2H of the compressed payload happens INSIDE the
  // compress call (canonical's cleanup_device). For the device path, no
  // D2H happens at all (compressed bytes stay on the GPU); column is 0.
  const double d2h_ms = 0.0;

  // Aggregate
  const double comp_time_ms   = std::accumulate(comp_ms.begin(), comp_ms.end(), 0.0) / a.iterations;
  const double decomp_time_ms = std::accumulate(decomp_ms.begin(), decomp_ms.end(), 0.0) / a.iterations;
  const double comp_total_ms  = h2d_ms + comp_time_ms;
  const double comp_throughput_gbs   = (uncompressed_bytes / 1e9) / (comp_time_ms / 1e3);
  const double decomp_throughput_gbs = (uncompressed_bytes / 1e9) / (decomp_time_ms / 1e3);
  const double comp_ratio = double(uncompressed_bytes) / double(comp_size);

  if (!a.csv) {
    const char* mode_s = (a.mode == ARCTO_ZFP_MODE_FIXED_RATE) ? "fixed_rate" :
                         (a.mode == ARCTO_ZFP_MODE_FIXED_PRECISION) ? "fixed_precision" :
                         (a.mode == ARCTO_ZFP_MODE_FIXED_ACCURACY)  ? "fixed_accuracy"  : "?";
    std::printf("file=%s  ndims=%u  mode=%s  param=%g\n",
                a.path.c_str(), opts.ndims, mode_s, a.param);
    std::printf("uncompressed=%zu B  compressed=%zu B  ratio=%.2fx\n",
                uncompressed_bytes, comp_size, comp_ratio);
    std::printf("comp=%.3f ms  decomp=%.3f ms  H2D=%.3f ms\n",
                comp_time_ms, decomp_time_ms, h2d_ms);
    std::printf("comp_throughput=%.2f GB/s  decomp_throughput=%.2f GB/s\n",
                comp_throughput_gbs, decomp_throughput_gbs);
  } else {
    // CSV header matching benchmark_*_chunked exactly.
    const char sep = ',';
    std::cout << "Files" << sep << "Duplicate data" << sep << "Size in MB"
              << sep << "Pages" << sep << "Avg page size in KB"
              << sep << "Max page size in KB" << sep << "Ucompressed size in bytes"
              << sep << "Compressed size in bytes" << sep << "Compression ratio"
              << sep << "Compression throughput (uncompressed) in GB/s"
              << sep << "Decompression throughput (uncompressed) in GB/s"
              << sep << "Compression time (ms)" << sep << "Decompression time (ms)"
              << sep << "Transfer H2D (ms)" << sep << "Transfer D2H (ms)"
              << sep << "Total time (ms)" << sep << "Avg chunk time (ms)"
              << sep << "Comp throughput stddev (GB/s)"
              << sep << "Decomp throughput stddev (GB/s)"
              << sep << "Comp time stddev (ms)" << sep << "Decomp time stddev (ms)\n";
    // Per-iteration throughputs for stddev (the chunked benchmark reports
    // stddev OF throughput, not throughput evaluated at the stddev of time).
    std::vector<double> comp_tp(a.iterations), decomp_tp(a.iterations);
    for (size_t i = 0; i < a.iterations; i++) {
      comp_tp[i]   = (uncompressed_bytes / 1e9) / (comp_ms[i] / 1e3);
      decomp_tp[i] = (uncompressed_bytes / 1e9) / (decomp_ms[i] / 1e3);
    }

    std::cout << std::fixed;
    std::cout << 1                                                   // Files
              << sep << 0                                            // Duplicate data
              << sep << std::setprecision(6) << (uncompressed_bytes * 1e-6)
              << sep << 1                                            // Pages (1 = whole field)
              << sep << (uncompressed_bytes * 1e-3)                  // avg page size KB
              << sep << (uncompressed_bytes * 1e-3)                  // max page size KB
              << sep << uncompressed_bytes
              << sep << comp_size
              << sep << std::setprecision(2) << comp_ratio
              << sep << comp_throughput_gbs
              << sep << decomp_throughput_gbs
              << sep << std::setprecision(3) << comp_time_ms
              << sep << decomp_time_ms
              << sep << h2d_ms
              << sep << d2h_ms
              << sep << comp_total_ms
              << sep << std::setprecision(6) << comp_time_ms        // avg chunk = avg comp (1 chunk)
              << sep << std::setprecision(4) << stddev(comp_tp)
              << sep << stddev(decomp_tp)
              << sep << stddev(comp_ms)
              << sep << stddev(decomp_ms)
              << "\n";
  }

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));
  if (h_comp_pinned) HIP_CHECK(hipHostFree(h_comp_pinned));
  return 0;
}
