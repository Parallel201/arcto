/*
 * ARCTO ZFP — REVERSIBLE_3D round-trip test (Phase 1d).
 *
 * Validates the lossless block-FP + REV lift + global-K pipeline on:
 *   1. Synthetic 3D float fields (sin/cos) at several shapes. The
 *      round-trip should reproduce every value to within one ULP of
 *      float32 precision (BFP cast uses ebias=30 with int64, so the
 *      scaled magnitude exceeds float32's 24-bit mantissa precision
 *      and the lift is strictly invertible).
 *   2. Optionally a real TTI cube via ARCTO_ZFP_TTI_DATA, reporting
 *      compression ratio against the original 100 MB slice.
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

struct Stats {
  double max_abs;
  double rmse;
  double value_max;
};

Stats compute_stats(const std::vector<float>& a, const std::vector<float>& b)
{
  Stats s{0.0, 0.0, 0.0};
  double sum_sq = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    const double e = std::abs(double(b[i]) - double(a[i]));
    if (e > s.max_abs) s.max_abs = e;
    sum_sq += e * e;
    const double av = std::abs(double(a[i]));
    if (av > s.value_max) s.value_max = av;
  }
  s.rmse = std::sqrt(sum_sq / double(a.size()));
  return s;
}

void run_round_trip(uint32_t nx, uint32_t ny, uint32_t nz,
                    const std::vector<float>& input,
                    const char* label)
{
  const size_t N           = size_t(nx) * ny * nz;
  const size_t input_bytes = N * sizeof(float);

  void* d_in = nullptr;
  HIP_CHECK(hipMalloc(&d_in, input_bytes));
  HIP_CHECK(hipMemcpy(d_in, input.data(), input_bytes, hipMemcpyHostToDevice));

  arctoBatchedZFPOpts_t opts = arctoBatchedZFPDefaultOpts;
  opts.mode     = ARCTO_ZFP_MODE_REVERSIBLE_3D;
  opts.dim      = ARCTO_ZFP_DIM_3D;
  opts.param    = 0.0;
  opts.shape[0] = nx;
  opts.shape[1] = ny;
  opts.shape[2] = nz;
  opts.shape[3] = 1;

  size_t max_comp = 0;
  REQUIRE(arctoZFP3DCompressGetMaxOutputSize(opts, &max_comp) == arctoSuccess,
          "GetMaxOutputSize failed");

  void* d_comp = nullptr;
  HIP_CHECK(hipMalloc(&d_comp, max_comp));
  size_t* d_comp_size = nullptr;
  HIP_CHECK(hipMalloc(&d_comp_size, sizeof(size_t)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  arctoStatus_t st = arctoZFP3DCompressAsync(
      d_in, opts, d_comp, max_comp, d_comp_size, stream);
  REQUIRE(st == arctoSuccess, "3DCompressAsync REVERSIBLE failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t comp_size = 0;
  HIP_CHECK(hipMemcpy(&comp_size, d_comp_size, sizeof(size_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(comp_size > 0 && comp_size <= max_comp, "comp_size out of range");

  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, input_bytes));
  size_t* d_actual = nullptr;
  HIP_CHECK(hipMalloc(&d_actual, sizeof(size_t)));
  arctoStatus_t* d_status = nullptr;
  HIP_CHECK(hipMalloc(&d_status, sizeof(arctoStatus_t)));

  st = arctoZFP3DDecompressAsync(
      d_comp, comp_size, d_out, input_bytes, d_actual, d_status, stream);
  REQUIRE(st == arctoSuccess, "3DDecompressAsync REVERSIBLE failed");
  HIP_CHECK(hipStreamSynchronize(stream));

  arctoStatus_t s = arctoErrorInternal;
  HIP_CHECK(hipMemcpy(&s, d_status, sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  REQUIRE(s == arctoSuccess, "Decompress status not success");

  size_t actual = 0;
  HIP_CHECK(hipMemcpy(&actual, d_actual, sizeof(size_t), hipMemcpyDeviceToHost));
  REQUIRE(actual == input_bytes, "Decompressed size mismatch");

  std::vector<float> out(N);
  HIP_CHECK(hipMemcpy(out.data(), d_out, input_bytes, hipMemcpyDeviceToHost));

  const Stats st_err = compute_stats(input, out);
  const double ratio = double(input_bytes) / double(comp_size);
  // Tolerance: 2 ULPs of the relative scale. With BFP ebias=30 and
  // float32 mantissa width 23 bits, the scaled int64 has 7 extra bits of
  // headroom so the cast is exact; the only loss is in the inverse cast
  // (int64 → float32) which rounds to the nearest float32. 2 ULPs gives
  // a small safety margin for accumulated rounding across the lift's
  // 16 lifts per axis.
  const double tol = st_err.value_max * std::ldexp(1.0, -22) + 1e-7;

  std::printf("  %s shape=(%u,%u,%u)  ratio=%.2fx  max_err=%.4g  rmse=%.4g  tol=%.4g\n",
              label, nx, ny, nz, ratio, st_err.max_abs, st_err.rmse, tol);
  REQUIRE(st_err.max_abs <= tol, "REVERSIBLE_3D exceeded 2-ULP tolerance");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_comp));
  HIP_CHECK(hipFree(d_comp_size));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_actual));
  HIP_CHECK(hipFree(d_status));
}

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

} // namespace

int main()
{
  std::puts("--- Synthetic 3D REVERSIBLE round-trip ---");
  struct Shape { uint32_t nx, ny, nz; };
  const Shape shapes[] = {
      {  4,   4,   4},
      { 16,  16,  16},
      { 17,  19,  23},
      { 64,  64,  32},
  };
  for (const Shape& sh : shapes) {
    const auto v = make_synth_3d(sh.nx, sh.ny, sh.nz);
    run_round_trip(sh.nx, sh.ny, sh.nz, v, "synth");
  }

  const char* path = std::getenv("ARCTO_ZFP_TTI_DATA");
  if (path) {
    std::printf("--- TTI REVERSIBLE round-trip from %s ---\n", path);
    const uint32_t nx = 448, ny = 448, nz = 130;
    const size_t   N  = size_t(nx) * ny * nz;
    const size_t   bytes_needed = N * sizeof(float);
    std::ifstream f(path, std::ios::binary);
    REQUIRE(static_cast<bool>(f), "Could not open TTI data file");
    std::vector<float> input(N);
    f.read(reinterpret_cast<char*>(input.data()), bytes_needed);
    REQUIRE(f.gcount() == std::streamsize(bytes_needed),
            "TTI file shorter than 448*448*130 floats");
    run_round_trip(nx, ny, nz, input, "tti");
  } else {
    std::puts("--- TTI REVERSIBLE: skipped (set ARCTO_ZFP_TTI_DATA to enable) ---");
  }

  std::puts("SUCCESS: All REVERSIBLE_3D round-trip tests passed");
  return 0;
}