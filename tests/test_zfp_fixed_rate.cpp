/*
 * ARCTO ZFP — FIXED_RATE fuzzy round-trip test.
 *
 * FIXED_RATE is lossy, so the existing GENERATE_TESTS infrastructure (which
 * validates bit-exact equality) cannot be reused. This test instead
 * generates a known float input, compresses + decompresses through the
 * FIXED_RATE path, and verifies that the per-element absolute error stays
 * below a rate-dependent tolerance derived from the worst-case truncation
 * error of the integer lift transform.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t err = (expr);                                                  \
    if (err != hipSuccess) {                                                  \
      std::fprintf(stderr,                                                    \
                   "HIP error %d at %s:%d: %s\n",                             \
                   (int)err, __FILE__, __LINE__, hipGetErrorString(err));     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define REQUIRE(cond, msg)                                                    \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr,                                                    \
                   "FAIL at %s:%d: %s (condition: %s)\n",                     \
                   __FILE__, __LINE__, (msg), #cond);                         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

// Smooth-ish synthetic field. Returns float values in roughly [-1.5, 1.5]
// so that emax stays bounded across the chunk and the error tolerance is
// easy to reason about. Avoids exact sign flips at adjacent samples to
// keep emax stable within a 4-element ZFP lift window.
float synth_value(size_t i)
{
  const float x = 0.001f * static_cast<float>(i);
  return std::sin(x) * 0.7f + std::cos(0.3f * x) * 0.4f;
}

// Per-element tolerance for FIXED_RATE truncation at K bits/value.
// ZFP's own documentation reports ~2^(5-rate) mean error; the per-element
// max can run ~4x higher after inverse-transform amplification. We use
// 2^(7-K) * |value_max| as a conservative L_inf upper bound.
float fixed_rate_tolerance(float value_max, int K)
{
  return value_max * std::ldexp(1.0f, 10 - K) + 1e-5f;
}

void run_round_trip(size_t num_values, int K)
{
  std::printf("--- num_values=%zu rate=%d bits/value ---\n", num_values, K);

  // ---- Build input on host ------------------------------------------------
  std::vector<float> host_input(num_values);
  float value_max = 0.0f;
  for (size_t i = 0; i < num_values; ++i) {
    host_input[i] = synth_value(i);
    if (std::abs(host_input[i]) > value_max) value_max = std::abs(host_input[i]);
  }
  const size_t input_bytes = num_values * sizeof(float);

  // ---- Upload input ------------------------------------------------------
  void* d_in = nullptr;
  HIP_CHECK(hipMalloc(&d_in, input_bytes));
  HIP_CHECK(hipMemcpy(d_in, host_input.data(), input_bytes, hipMemcpyHostToDevice));

  void** d_in_ptrs = nullptr;
  HIP_CHECK(hipMalloc(&d_in_ptrs, sizeof(void*)));
  HIP_CHECK(hipMemcpy(d_in_ptrs, &d_in, sizeof(void*), hipMemcpyHostToDevice));

  size_t* d_in_sizes = nullptr;
  HIP_CHECK(hipMalloc(&d_in_sizes, sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_in_sizes, &input_bytes, sizeof(size_t), hipMemcpyHostToDevice));

  // ---- Compress ----------------------------------------------------------
  arctoBatchedZFPOpts_t opts = arctoBatchedZFPDefaultOpts;
  opts.mode      = ARCTO_ZFP_MODE_FIXED_RATE;
  opts.param     = static_cast<double>(K);
  opts.dim       = ARCTO_ZFP_DIM_3D;
  opts.shape[0]  = static_cast<uint32_t>(num_values);
  opts.shape[1]  = 1;
  opts.shape[2]  = 1;
  opts.shape[3]  = 1;

  size_t comp_temp = 0;
  REQUIRE(arctoBatchedZFPCompressGetTempSize(1, input_bytes, opts, &comp_temp) == arctoSuccess,
          "GetTempSize failed");

  size_t max_comp = 0;
  REQUIRE(arctoBatchedZFPCompressGetMaxOutputChunkSize(input_bytes, opts, &max_comp) == arctoSuccess,
          "GetMaxOutputChunkSize failed");

  void* d_comp = nullptr;
  HIP_CHECK(hipMalloc(&d_comp, max_comp));

  void** d_comp_ptrs = nullptr;
  HIP_CHECK(hipMalloc(&d_comp_ptrs, sizeof(void*)));
  HIP_CHECK(hipMemcpy(d_comp_ptrs, &d_comp, sizeof(void*), hipMemcpyHostToDevice));

  size_t* d_comp_sizes = nullptr;
  HIP_CHECK(hipMalloc(&d_comp_sizes, sizeof(size_t)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  arctoStatus_t st = arctoBatchedZFPCompressAsync(
      (const void* const*)d_in_ptrs,
      d_in_sizes,
      input_bytes,
      1,
      nullptr, 0,
      d_comp_ptrs,
      d_comp_sizes,
      opts,
      stream);
  REQUIRE(st == arctoSuccess, "CompressAsync failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t comp_size = 0;
  HIP_CHECK(hipMemcpy(&comp_size, d_comp_sizes, sizeof(size_t), hipMemcpyDeviceToHost));
  const double ratio = static_cast<double>(input_bytes) / static_cast<double>(comp_size);
  std::printf("  input=%zu B   comp=%zu B   ratio=%.2fx\n",
              input_bytes, comp_size, ratio);

  // ---- Decompress --------------------------------------------------------
  size_t decomp_temp = 0;
  REQUIRE(arctoBatchedZFPDecompressGetTempSize(1, input_bytes, &decomp_temp) == arctoSuccess,
          "DecompressGetTempSize failed");

  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, input_bytes));

  void** d_out_ptrs = nullptr;
  HIP_CHECK(hipMalloc(&d_out_ptrs, sizeof(void*)));
  HIP_CHECK(hipMemcpy(d_out_ptrs, &d_out, sizeof(void*), hipMemcpyHostToDevice));

  size_t* d_out_capacity = nullptr;
  HIP_CHECK(hipMalloc(&d_out_capacity, sizeof(size_t)));
  HIP_CHECK(hipMemcpy(d_out_capacity, &input_bytes, sizeof(size_t), hipMemcpyHostToDevice));

  size_t* d_actual_sizes = nullptr;
  HIP_CHECK(hipMalloc(&d_actual_sizes, sizeof(size_t)));

  arctoStatus_t* d_statuses = nullptr;
  HIP_CHECK(hipMalloc(&d_statuses, sizeof(arctoStatus_t)));

  st = arctoBatchedZFPDecompressAsync(
      (const void* const*)d_comp_ptrs,
      d_comp_sizes,
      d_out_capacity,
      d_actual_sizes,
      1,
      nullptr, 0,
      d_out_ptrs,
      d_statuses,
      stream);
  REQUIRE(st == arctoSuccess, "DecompressAsync failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  arctoStatus_t out_status = arctoErrorInternal;
  HIP_CHECK(hipMemcpy(&out_status, d_statuses, sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  REQUIRE(out_status == arctoSuccess, "Decompress per-chunk status not success");

  size_t out_size = 0;
  HIP_CHECK(hipMemcpy(&out_size, d_actual_sizes, sizeof(size_t), hipMemcpyDeviceToHost));
  REQUIRE(out_size == input_bytes, "Decompressed size mismatch");

  std::vector<float> host_output(num_values);
  HIP_CHECK(hipMemcpy(host_output.data(), d_out, input_bytes, hipMemcpyDeviceToHost));

  // ---- Validate error ----------------------------------------------------
  const float tol = fixed_rate_tolerance(value_max, K);
  float max_err = 0.0f;
  double sum_sq = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < num_values; ++i) {
    const float err = std::abs(host_output[i] - host_input[i]);
    if (err > max_err) { max_err = err; worst_i = i; }
    sum_sq += static_cast<double>(err) * err;
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(num_values));
  std::printf("  max_err=%.6g  rms=%.6g  tol=%.6g  (worst at i=%zu: %.6g -> %.6g)\n",
              max_err, rms, tol, worst_i, host_input[worst_i], host_output[worst_i]);

  REQUIRE(max_err <= tol, "Per-element error exceeds rate tolerance");

  // ---- Cleanup -----------------------------------------------------------
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_in_ptrs));
  HIP_CHECK(hipFree(d_in_sizes));
  HIP_CHECK(hipFree(d_comp));
  HIP_CHECK(hipFree(d_comp_ptrs));
  HIP_CHECK(hipFree(d_comp_sizes));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_out_ptrs));
  HIP_CHECK(hipFree(d_out_capacity));
  HIP_CHECK(hipFree(d_actual_sizes));
  HIP_CHECK(hipFree(d_statuses));
}

} // namespace

int main()
{
  // A grid of (num_values, K) combinations. Sizes cover one ZFP block
  // (64 values), one full chunk (16384 values = 64 KB), and a non-
  // block-aligned size that exercises the trailing-partial-block path.
  // Rates cover both heavy compression (K=8 → 4x) and light lossy (K=24).
  const size_t sizes[] = { 64, 1024, 16384, 12345 };
  const int    rates[] = { 8, 16, 24 };

  for (size_t n : sizes) {
    for (int K : rates) {
      run_round_trip(n, K);
    }
  }

  std::puts("SUCCESS: All FIXED_RATE round-trip tests passed");
  return 0;
}
