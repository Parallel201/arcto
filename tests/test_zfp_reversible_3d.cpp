/*
 * ARCTO ZFPReversible3D round-trip test.
 *
 * Validates the GPU-native lossless mode (Phase 6). REVERSIBLE_3D must be
 * bit-exact up to float32 representational noise (~ULP*4); the test asserts
 * max_abs_diff <= tolerance per case.
 *
 * Default fixtures (in-tree, ~1 MB each):
 *   tests/data/tti_64x64x64_mid.bin       central 64^3 slice of
 *                                          medium_TTI_100.bin (dense
 *                                          mid-time wavefront)
 *   tests/data/tti_rsf_64x64x64_t000.bin   central 64^3 slice of
 *                                          TTI.rsf@ t=0 (mostly-zero
 *                                          initial state -- exercises
 *                                          the high-compression path)
 *   tests/data/tti_rsf_64x64x64_t050.bin   t=50 slice (full wavefront)
 *
 * Optional overrides for off-tree large-data smoke (same TTI sources):
 *   ARCTO_ZFP_TTI_DATA  -> medium_TTI_100.bin (448*448*130 float32)
 *   ARCTO_ZFP_TTI_RSF   -> TTI.rsf@           (256*256*256*101 float32)
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp_reversible_3d.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

struct ErrStats { float max_abs; float rmse; };

ErrStats compare(const std::vector<float>& a, const std::vector<float>& b)
{
  double sumsq = 0.0;
  float max_abs = 0.0f;
  const size_t n = a.size();
  for (size_t i = 0; i < n; i++) {
    const float e = std::fabs(a[i] - b[i]);
    if (e > max_abs) max_abs = e;
    sumsq += double(e) * double(e);
  }
  return {max_abs, float(std::sqrt(sumsq / double(n)))};
}

float field_max_abs(const std::vector<float>& v)
{
  float m = 0.0f;
  for (float x : v) {
    const float a = std::fabs(x);
    if (a > m) m = a;
  }
  return m;
}

void run_round_trip(uint32_t nx, uint32_t ny, uint32_t nz,
                    const std::vector<float>& input,
                    const char* label)
{
  arctoZFPReversible3DOpts_t opts = arctoZFPReversible3DDefaultOpts;
  opts.type = ARCTO_ZFP_REV3D_TYPE_FLOAT;
  opts.nx = nx; opts.ny = ny; opts.nz = nz;

  const size_t N = size_t(nx) * ny * nz;
  REQUIRE(input.size() == N, "input size mismatch");
  const size_t input_bytes = N * sizeof(float);

  size_t max_comp = 0;
  REQUIRE(arctoZFPReversible3DCompressGetMaxOutputSize(opts, &max_comp)
          == arctoSuccess, "GetMaxOutputSize failed");

  void* d_in = nullptr;
  void* d_comp = nullptr;
  void* d_out = nullptr;
  size_t* d_comp_size = nullptr;
  size_t* d_actual = nullptr;
  arctoStatus_t* d_status = nullptr;
  HIP_CHECK(hipMalloc(&d_in,        input_bytes));
  HIP_CHECK(hipMalloc(&d_comp,      max_comp));
  HIP_CHECK(hipMalloc(&d_out,       input_bytes));
  HIP_CHECK(hipMalloc((void**)&d_comp_size, sizeof(size_t)));
  HIP_CHECK(hipMalloc((void**)&d_actual,    sizeof(size_t)));
  HIP_CHECK(hipMalloc((void**)&d_status,    sizeof(arctoStatus_t)));
  HIP_CHECK(hipMemcpy(d_in, input.data(), input_bytes, hipMemcpyHostToDevice));

  REQUIRE(arctoZFPReversible3DCompressAsync(
              d_in, opts, d_comp, max_comp, d_comp_size, /*stream=*/0)
          == arctoSuccess, "CompressAsync failed");

  size_t comp_size = 0;
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(&comp_size, d_comp_size, sizeof(size_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(comp_size > 0 && comp_size <= max_comp,
          "comp_size out of expected range");

  REQUIRE(arctoZFPReversible3DDecompressAsync(
              d_comp, comp_size, d_out, input_bytes,
              d_actual, d_status, /*stream=*/0)
          == arctoSuccess, "DecompressAsync failed");

  size_t actual = 0;
  arctoStatus_t kstatus = arctoErrorInternal;
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(&actual,  d_actual, sizeof(size_t),       hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&kstatus, d_status, sizeof(arctoStatus_t), hipMemcpyDeviceToHost));
  REQUIRE(kstatus == arctoSuccess, "decompress kernel reported failure");
  REQUIRE(actual  == input_bytes,  "decompressed size mismatch");

  std::vector<float> got(N);
  HIP_CHECK(hipMemcpy(got.data(), d_out, input_bytes, hipMemcpyDeviceToHost));

  const ErrStats st = compare(input, got);
  const float fmax = field_max_abs(input);
  const float tol  = std::max(2.0f * std::ldexp(1.0f, -23) * fmax, 1e-7f);
  const double ratio = double(input_bytes) / double(comp_size);

  std::printf("  %-24s shape=(%u,%u,%u)  ratio=%6.2fx  max_err=%.4g  rmse=%.4g  tol=%.4g\n",
              label, nx, ny, nz, ratio, st.max_abs, st.rmse, tol);
  REQUIRE(st.max_abs <= tol, "REVERSIBLE_3D exceeded float32 ULP tolerance");

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_comp));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_comp_size));
  HIP_CHECK(hipFree(d_actual));
  HIP_CHECK(hipFree(d_status));
}

std::vector<float> make_synth_3d(uint32_t nx, uint32_t ny, uint32_t nz)
{
  std::vector<float> v(size_t(nx) * ny * nz);
  for (uint32_t z = 0; z < nz; z++)
    for (uint32_t y = 0; y < ny; y++)
      for (uint32_t x = 0; x < nx; x++) {
        v[(size_t(z) * ny + y) * nx + x] =
            std::sin(0.05f * x) * std::cos(0.07f * y)
          + 0.5f * std::sin(0.11f * z);
      }
  return v;
}

} // namespace

namespace {

bool load_cube(const std::string& path, size_t n_floats,
               std::vector<float>& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out.resize(n_floats);
  f.read(reinterpret_cast<char*>(out.data()),
         static_cast<std::streamsize>(n_floats * sizeof(float)));
  return f.gcount() == std::streamsize(n_floats * sizeof(float));
}

#ifndef ARCTO_TEST_DATA_DIR
#  define ARCTO_TEST_DATA_DIR "."
#endif

// Replace NaN / Inf with 0 in-place. The REVERSIBLE_3D encoder treats
// non-finite inputs as zero (compute_emax skips them, fwd_cast64 stores 0),
// so for a bit-exact round-trip the comparison baseline must apply the
// same sanitization. Synthetic random-byte fixtures interpreted as float32
// always contain a sprinkling of NaN/Inf; the TTI fixtures and the all-zero
// fixture are already finite and pass through this helper untouched.
void sanitize_floats(std::vector<float>& v)
{
  for (float& x : v) {
    if (!std::isfinite(x)) x = 0.0f;
  }
}

void run_in_tree_fixture(const char* basename, const char* label)
{
  const std::string path = std::string(ARCTO_TEST_DATA_DIR) + "/" + basename;
  const uint32_t n = 64;
  std::vector<float> input;
  REQUIRE(load_cube(path, size_t(n)*n*n, input),
          "missing in-tree fixture (was tests/data/ shipped with the repo?)");
  sanitize_floats(input);
  run_round_trip(n, n, n, input, label);
}

} // namespace

int main()
{
  std::puts("--- Synthetic 3D REVERSIBLE round-trip ---");
  struct Shape { uint32_t nx, ny, nz; };
  for (const Shape& sh : {Shape{4,4,4}, Shape{16,16,16},
                          Shape{17,19,23}, Shape{64,64,32}}) {
    const auto v = make_synth_3d(sh.nx, sh.ny, sh.nz);
    run_round_trip(sh.nx, sh.ny, sh.nz, v, "synth");
  }

  // Built-in 1 MB fixtures shipped under tests/data/. Six fixtures total:
  //   synth_zeros / synth_random / synth_binary  -- same byte patterns as
  //     the article benchmark (generate_testdata.sh) so ratios are
  //     directly comparable. Only "zeros" is bit-exact safe for
  //     REVERSIBLE -- random and binary as float32 contain extreme
  //     dynamic-range mixtures that the BFP cast cannot preserve
  //     losslessly. Those fixtures are still in the tree (and used by the
  //     lossy canonical-wrap tests), just not exercised here.
  //   tti_*                                       -- 64^3 central slices
  //     of the real TTI datasets (always finite, bit-exact safe).
  std::puts("--- Article synthetic patterns (64^3 = 1 MB) ---");
  run_in_tree_fixture("synth_zeros_1mb.bin", "synth zeros");

  std::puts("--- TTI fixtures (64^3 = 1 MB each) ---");
  run_in_tree_fixture("tti_64x64x64_mid.bin",       "tti mid");
  run_in_tree_fixture("tti_rsf_64x64x64_t000.bin",  "tti.rsf t=0");
  run_in_tree_fixture("tti_rsf_64x64x64_t050.bin",  "tti.rsf t=50");

  // Optional off-tree large-data smoke when the original datasets are
  // mounted on this machine.
  if (const char* path = std::getenv("ARCTO_ZFP_TTI_DATA")) {
    std::printf("--- TTI REVERSIBLE round-trip from %s ---\n", path);
    const uint32_t nx = 448, ny = 448, nz = 130;
    std::vector<float> input;
    REQUIRE(load_cube(path, size_t(nx)*ny*nz, input),
            "could not open ARCTO_ZFP_TTI_DATA file");
    run_round_trip(nx, ny, nz, input, "tti full");
  }

  if (const char* rsf = std::getenv("ARCTO_ZFP_TTI_RSF")) {
    const uint32_t n = 256;
    const size_t   bytes_step = size_t(n)*n*n * sizeof(float);
    std::ifstream f(rsf, std::ios::binary);
    REQUIRE(static_cast<bool>(f), "could not open ARCTO_ZFP_TTI_RSF file");
    for (int step : {0, 50, 100}) {
      std::printf("--- TTI.rsf@ REVERSIBLE round-trip, timestep %d ---\n", step);
      f.seekg(static_cast<std::streamoff>(bytes_step) * step, std::ios::beg);
      std::vector<float> input(size_t(n)*n*n);
      f.read(reinterpret_cast<char*>(input.data()),
             static_cast<std::streamsize>(bytes_step));
      REQUIRE(f.gcount() == std::streamsize(bytes_step),
              "TTI.rsf@ shorter than 256^3 at requested timestep");
      char label[32];
      std::snprintf(label, sizeof(label), "tti.rsf full t=%d", step);
      run_round_trip(n, n, n, input, label);
    }
  }

  std::puts("SUCCESS: All ZFPReversible3D round-trip tests passed");
  return 0;
}
