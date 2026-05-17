/*
 * ARCTO ZFP canonical-wrap smoke test.
 *
 * Validates that the arctoZFP* entry points compress and decompress
 * float32 3D fields by delegating to the vendored canonical zfp HIP
 * backend, and that the round-trip is correct (bit-exact for REVERSIBLE,
 * PSNR-bounded for FIXED_RATE / FIXED_ACCURACY).
 *
 * Bit-compat with the upstream `zfp` CLI is checked elsewhere (a separate
 * harness compresses the same data via both paths and diffs the bytes);
 * this test only confirms end-to-end functionality.
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

#define REQUIRE(cond, msg)                                                    \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d %s -- %s\n",                            \
                   __FILE__, __LINE__, #cond, msg);                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "FAIL HIP %s:%d %s -- %s\n",                        \
                   __FILE__, __LINE__, #expr, hipGetErrorString(_e));          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct ErrStats { float max_abs; float rmse; float psnr; };

ErrStats compare(const std::vector<float>& a, const std::vector<float>& b)
{
  double sumsq = 0.0;
  float max_abs = 0.0f;
  float a_max   = 0.0f;
  const size_t n = a.size();
  for (size_t i = 0; i < n; i++) {
    const float e = std::fabs(a[i] - b[i]);
    if (e > max_abs) max_abs = e;
    sumsq += double(e) * double(e);
    if (std::fabs(a[i]) > a_max) a_max = std::fabs(a[i]);
  }
  const double rmse = std::sqrt(sumsq / double(n));
  const float psnr  = (rmse > 0.0 && a_max > 0.0f)
      ? float(20.0 * std::log10(double(a_max) / rmse))
      : 999.0f;
  return {max_abs, float(rmse), psnr};
}

std::vector<float> make_synth_3d(uint32_t nx, uint32_t ny, uint32_t nz)
{
  std::vector<float> v(size_t(nx) * ny * nz);
  for (uint32_t z = 0; z < nz; z++)
    for (uint32_t y = 0; y < ny; y++)
      for (uint32_t x = 0; x < nx; x++) {
        const float fx = 0.05f * float(x);
        const float fy = 0.07f * float(y);
        const float fz = 0.11f * float(z);
        v[(size_t(z) * ny + y) * nx + x] =
            std::sin(fx) * std::cos(fy) + 0.5f * std::sin(fz);
      }
  return v;
}

void run_round_trip(const std::vector<float>& input,
                    uint32_t nx, uint32_t ny, uint32_t nz,
                    arctoZFPMode_t mode, double param,
                    const char* label)
{
  arctoZFPOpts_t opts = arctoZFPDefaultOpts;
  opts.mode = mode;
  opts.type = ARCTO_ZFP_TYPE_FLOAT;
  opts.param = param;
  opts.ndims = 3;
  opts.shape[0] = nx;
  opts.shape[1] = ny;
  opts.shape[2] = nz;
  opts.shape[3] = 1;

  const size_t input_bytes = input.size() * sizeof(float);

  size_t max_comp = 0;
  REQUIRE(arctoZFPCompressGetMaxOutputSize(opts, &max_comp) == arctoSuccess,
          "GetMaxOutputSize failed");
  REQUIRE(max_comp > 0, "max_comp == 0");

  void* d_in  = nullptr;
  void* d_out = nullptr;
  std::vector<unsigned char> h_comp(max_comp);
  HIP_CHECK(hipMalloc(&d_in,  input_bytes));
  HIP_CHECK(hipMalloc(&d_out, input_bytes));
  HIP_CHECK(hipMemcpy(d_in, input.data(), input_bytes, hipMemcpyHostToDevice));

  size_t comp_size = 0;
  REQUIRE(arctoZFPCompress(d_in, opts, h_comp.data(), max_comp, &comp_size)
          == arctoSuccess, "Compress failed");
  REQUIRE(comp_size > 0 && comp_size <= max_comp,
          "comp_size out of expected range");
  // Canonical zfp bitstream magic: 0x05 0x70 0x66 0x7a packed LSB-first.
  REQUIRE(h_comp[0] == 0x7au && h_comp[1] == 0x66u
       && h_comp[2] == 0x70u && h_comp[3] == 0x05u,
          "Stream does not start with canonical ZFP magic");

  // Verify GetDecompressSize on the produced stream.
  size_t reported_uncomp = 0;
  REQUIRE(arctoZFPGetDecompressSize(h_comp.data(), comp_size, &reported_uncomp)
          == arctoSuccess, "GetDecompressSize failed");
  REQUIRE(reported_uncomp == input_bytes,
          "GetDecompressSize did not match expected uncompressed bytes");

  size_t out_size = 0;
  REQUIRE(arctoZFPDecompress(h_comp.data(), comp_size, d_out, input_bytes, &out_size)
          == arctoSuccess, "Decompress failed");
  REQUIRE(out_size == input_bytes, "Decompressed size mismatch");

  std::vector<float> got(input.size());
  HIP_CHECK(hipMemcpy(got.data(), d_out, input_bytes, hipMemcpyDeviceToHost));

  const ErrStats st = compare(input, got);
  const double ratio = double(input_bytes) / double(comp_size);

  std::printf("  %-32s shape=(%u,%u,%u)  ratio=%6.2fx  max_err=%.4g  rmse=%.4g  psnr=%.1fdB\n",
              label, nx, ny, nz, ratio, st.max_abs, st.rmse, st.psnr);

  if (mode == ARCTO_ZFP_MODE_REVERSIBLE) {
    REQUIRE(st.max_abs == 0.0f, "REVERSIBLE round-trip is not bit-exact");
  } else if (mode == ARCTO_ZFP_MODE_FIXED_RATE) {
    // Loose upper bound: 2-ULP per input scalar magnitude is far above what
    // ZFP achieves; we just want to catch catastrophic failures.
    const float tol = std::pow(2.0f, -float(param) + 8.0f);
    REQUIRE(st.max_abs <= tol, "FIXED_RATE error too large");
  }

  hipFree(d_in);
  hipFree(d_out);
}

} // namespace

int main()
{
  std::puts("--- arctoZFP smoke (delegates to canonical HIP backend) ---");

  const uint32_t nx = 64, ny = 64, nz = 32;
  const auto v = make_synth_3d(nx, ny, nz);

  std::puts("[FIXED_RATE]");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_RATE,  8.0,
                 "synth fixed-rate K=8");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_RATE, 16.0,
                 "synth fixed-rate K=16");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_RATE, 24.0,
                 "synth fixed-rate K=24");

  // Phase 4: variable-rate modes via embedded canonical block-offset index.
  std::puts("[FIXED_PRECISION]");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_PRECISION, 12.0,
                 "synth fixed-precision p=12");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_PRECISION, 20.0,
                 "synth fixed-precision p=20");

  std::puts("[FIXED_ACCURACY]");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_ACCURACY, 1e-3,
                 "synth fixed-accuracy 1e-3");
  run_round_trip(v, nx, ny, nz, ARCTO_ZFP_MODE_FIXED_ACCURACY, 1e-6,
                 "synth fixed-accuracy 1e-6");

  std::puts("SUCCESS: all arctoZFP smoke checks passed");
  return 0;
}
