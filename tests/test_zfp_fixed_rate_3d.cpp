/*
 * ARCTO ZFP — FIXED_RATE 3D fuzzy round-trip test (Phase 1c).
 *
 * Exercises arctoZFP3DCompressAsync / arctoZFP3DDecompressAsync on:
 *   1. Synthetic 3D fields (smooth sin/cos) at several (nx, ny, nz) shapes
 *      and rates. Validates per-element error against a rate-derived bound.
 *   2. Optionally, real TTI seismic data via the ARCTO_ZFP_TTI_DATA env var.
 *      If set, reads the binary, interprets the leading 448*448*130 floats
 *      as a 3D cube, runs the full round-trip and prints compression ratio,
 *      RMSE, and PSNR — matching the metrics surfaced by the Fletcher-IO
 *      HIP_OPT/hip_compression.c integration. The TTI path is skipped (with
 *      a printed notice) when the env var is unset, so this test always
 *      passes in CI without extra data.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t err = (expr);                                                  \
    if (err != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %d at %s:%d: %s\n",                     \
                   (int)err, __FILE__, __LINE__, hipGetErrorString(err));     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define REQUIRE(cond, msg)                                                    \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL at %s:%d: %s (condition: %s)\n",             \
                   __FILE__, __LINE__, (msg), #cond);                         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

struct ErrorStats {
  double max_abs;
  double rmse;
  double psnr;        // dB, relative to (max - min) input range
  double value_max;
};

ErrorStats compute_error(const std::vector<float>& orig,
                         const std::vector<float>& recon)
{
  ErrorStats s{0.0, 0.0, 0.0, 0.0};
  const size_t N = orig.size();
  double sum_sq = 0.0;
  double vmin = orig[0], vmax = orig[0];
  for (size_t i = 0; i < N; i++) {
    const double e = std::abs(double(recon[i]) - double(orig[i]));
    if (e > s.max_abs) s.max_abs = e;
    sum_sq += e * e;
    if (orig[i] < vmin) vmin = orig[i];
    if (orig[i] > vmax) vmax = orig[i];
  }
  s.rmse = std::sqrt(sum_sq / double(N));
  s.value_max = std::max(std::abs(vmin), std::abs(vmax));
  const double range = vmax - vmin;
  s.psnr = (range > 1e-12 && s.rmse > 0.0)
         ? 20.0 * std::log10(range / s.rmse)
         : std::numeric_limits<double>::infinity();
  return s;
}

void run_3d_round_trip(uint32_t nx, uint32_t ny, uint32_t nz, int K,
                       const std::vector<float>& input,
                       const char* label)
{
  const size_t N           = size_t(nx) * ny * nz;
  const size_t input_bytes = N * sizeof(float);

  // ---- Upload cube ------------------------------------------------------
  void* d_in = nullptr;
  HIP_CHECK(hipMalloc(&d_in, input_bytes));
  HIP_CHECK(hipMemcpy(d_in, input.data(), input_bytes, hipMemcpyHostToDevice));

  // ---- Configure opts ---------------------------------------------------
  arctoBatchedZFPOpts_t opts = arctoBatchedZFPDefaultOpts;
  opts.mode     = ARCTO_ZFP_MODE_FIXED_RATE;
  opts.dim      = ARCTO_ZFP_DIM_3D;
  opts.param    = double(K);
  opts.shape[0] = nx;
  opts.shape[1] = ny;
  opts.shape[2] = nz;
  opts.shape[3] = 1;

  size_t max_comp = 0;
  REQUIRE(arctoZFP3DCompressGetMaxOutputSize(opts, &max_comp) == arctoSuccess,
          "GetMaxOutputSize failed");

  // ---- Compress ---------------------------------------------------------
  void* d_comp = nullptr;
  HIP_CHECK(hipMalloc(&d_comp, max_comp));
  size_t* d_comp_size = nullptr;
  HIP_CHECK(hipMalloc(&d_comp_size, sizeof(size_t)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  arctoStatus_t st = arctoZFP3DCompressAsync(
      d_in, opts, d_comp, max_comp, d_comp_size, stream);
  REQUIRE(st == arctoSuccess, "3DCompressAsync failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t comp_size = 0;
  HIP_CHECK(hipMemcpy(&comp_size, d_comp_size, sizeof(size_t), hipMemcpyDeviceToHost));
  REQUIRE(comp_size > 0 && comp_size <= max_comp, "comp_size out of range");

  // ---- Decompress -------------------------------------------------------
  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, input_bytes));
  size_t* d_actual_size = nullptr;
  HIP_CHECK(hipMalloc(&d_actual_size, sizeof(size_t)));
  arctoStatus_t* d_status = nullptr;
  HIP_CHECK(hipMalloc(&d_status, sizeof(arctoStatus_t)));

  st = arctoZFP3DDecompressAsync(
      d_comp, comp_size, d_out, input_bytes, d_actual_size, d_status, stream);
  REQUIRE(st == arctoSuccess, "3DDecompressAsync failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  arctoStatus_t out_status = arctoErrorInternal;
  HIP_CHECK(hipMemcpy(&out_status, d_status, sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  REQUIRE(out_status == arctoSuccess, "Decompress per-chunk status not success");

  size_t actual = 0;
  HIP_CHECK(hipMemcpy(&actual, d_actual_size, sizeof(size_t), hipMemcpyDeviceToHost));
  REQUIRE(actual == input_bytes, "Decompressed size mismatch");

  std::vector<float> out(N);
  HIP_CHECK(hipMemcpy(out.data(), d_out, input_bytes, hipMemcpyDeviceToHost));

  // ---- Compute & report stats ------------------------------------------
  const ErrorStats s = compute_error(input, out);
  const double     ratio = double(input_bytes) / double(comp_size);

  // Generous per-element bound: |value_max| * 2^(12-K). This is ~4x the
  // theoretical lift+truncation worst case and accommodates outlier blocks
  // (notably TTI seismic data, where most of the field is much smaller
  // than the global value_max so per-block emax varies widely).
  const double tol = s.value_max * std::ldexp(1.0, 12 - K) + 1e-5;

  std::printf("  %s shape=(%u,%u,%u) K=%d  ratio=%.2fx  max_err=%.4g  rmse=%.4g  psnr=%.1fdB  tol=%.4g\n",
              label, nx, ny, nz, K,
              ratio, s.max_abs, s.rmse, s.psnr, tol);

  REQUIRE(s.max_abs <= tol, "Per-element error exceeds rate tolerance");

  // ---- Cleanup ---------------------------------------------------------
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_comp));
  HIP_CHECK(hipFree(d_comp_size));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_actual_size));
  HIP_CHECK(hipFree(d_status));
}

// Smooth 3D field — sinusoidal in all 3 axes so adjacent voxels are
// strongly correlated. Values stay in roughly [-1.7, 1.7].
std::vector<float> make_synth_3d(uint32_t nx, uint32_t ny, uint32_t nz)
{
  std::vector<float> v(size_t(nx) * ny * nz);
  for (uint32_t z = 0; z < nz; z++) {
    for (uint32_t y = 0; y < ny; y++) {
      for (uint32_t x = 0; x < nx; x++) {
        const float fx = 0.05f * float(x);
        const float fy = 0.07f * float(y);
        const float fz = 0.11f * float(z);
        v[(size_t(z) * ny + y) * nx + x] =
            std::sin(fx) * std::cos(fy) + 0.5f * std::sin(fz);
      }
    }
  }
  return v;
}

void run_synthetic_suite()
{
  std::puts("--- Synthetic 3D round-trip ---");
  struct Shape { uint32_t nx, ny, nz; };
  // Sizes cover: one minimal block, an exact 16^3, a non-4-aligned shape,
  // and a moderately large cube to exercise multi-CTA grid striding.
  const Shape shapes[] = {
      {  4,   4,   4},   // 1 block
      { 16,  16,  16},   // 64 blocks
      { 17,  19,  23},   // non-aligned, exercises padding
      { 64,  64,  32},   // 8 * 8 * 8 = 512 blocks
  };
  const int rates[] = {8, 16, 24};

  for (const Shape& sh : shapes) {
    const auto v = make_synth_3d(sh.nx, sh.ny, sh.nz);
    for (int K : rates) {
      run_3d_round_trip(sh.nx, sh.ny, sh.nz, K, v, "synth");
    }
  }
}

void run_tti_validation_if_available()
{
  const char* path = std::getenv("ARCTO_ZFP_TTI_DATA");
  if (!path) {
    std::puts("--- TTI validation: skipped (set ARCTO_ZFP_TTI_DATA to a "
              "medium_TTI_*.bin path to enable) ---");
    return;
  }

  // The Fletcher-IO testdata generator extracts a 100 MB slice from the
  // 448x448x448 simulation. The first 448*448*130 floats form a clean
  // 3D cube (the trailing floats span a partial z-slab).
  const uint32_t nx = 448, ny = 448, nz = 130;
  const size_t   N  = size_t(nx) * ny * nz;
  const size_t   bytes_needed = N * sizeof(float);

  std::printf("--- TTI validation from %s ---\n", path);

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "Could not open %s\n", path);
    std::exit(1);
  }
  std::vector<float> input(N);
  f.read(reinterpret_cast<char*>(input.data()), bytes_needed);
  REQUIRE(f.gcount() == std::streamsize(bytes_needed),
          "TTI file shorter than 448*448*130 floats");

  for (int K : {8, 16, 24}) {
    run_3d_round_trip(nx, ny, nz, K, input, "tti");
  }
}

} // namespace

int main()
{
  run_synthetic_suite();
  run_tti_validation_if_available();
  std::puts("SUCCESS: All FIXED_RATE 3D round-trip tests passed");
  return 0;
}
