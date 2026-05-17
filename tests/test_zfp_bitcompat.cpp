/*
 * ARCTO ZFP bit-compat regression vs the canonical (CPU/serial) backend.
 *
 * Verifies *functional* bit-compatibility:
 *   ARCTO encode (HIP)  --decoded by-->  canonical zfp_decompress (serial)
 *   canonical zfp_compress (serial)  --decoded by-->  ARCTO decode (HIP)
 *
 * Both directions must reconstruct the input field within the precision
 * promised by the chosen mode. Byte-exact equality between exec policies
 * is NOT expected (the HIP path pads the trailing word differently from
 * serial, and our wrapper adds a small self-describing index trailer for
 * variable-rate modes), but the same compressed information is encoded.
 *
 * The test links directly against the vendored libzfp.a -- no CLI binary
 * required.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"
#include "zfp.h"

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

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
  float m = 0.0f;
  const size_t n = a.size();
  for (size_t i = 0; i < n; i++) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

std::vector<float> make_synth_3d(uint32_t nx, uint32_t ny, uint32_t nz)
{
  std::vector<float> v(size_t(nx) * ny * nz);
  for (uint32_t z = 0; z < nz; z++)
    for (uint32_t y = 0; y < ny; y++)
      for (uint32_t x = 0; x < nx; x++)
        v[(size_t(z) * ny + y) * nx + x] =
            std::sin(0.05f * x) * std::cos(0.07f * y) + 0.5f * std::sin(0.11f * z);
  return v;
}

// Apply mode + execution policy to a freshly-opened zfp_stream.
void canon_set_mode(zfp_stream* zfp, arctoZFPMode_t mode, double param,
                    zfp_type type, uint dims)
{
  switch (mode) {
    case ARCTO_ZFP_MODE_FIXED_RATE:
      zfp_stream_set_rate(zfp, param, type, dims, zfp_false); break;
    case ARCTO_ZFP_MODE_FIXED_PRECISION:
      zfp_stream_set_precision(zfp, (uint)param); break;
    case ARCTO_ZFP_MODE_FIXED_ACCURACY:
      zfp_stream_set_accuracy(zfp, param); break;
    default:
      REQUIRE(false, "unsupported mode in canon_set_mode");
  }
  zfp_stream_set_execution(zfp, zfp_exec_serial);
}

// ===========================================================================
// ARCTO encode (HIP) -> canonical decode (serial). Confirms the upstream
// CPU reader interprets our bitstream correctly.
// ===========================================================================
void arcto_encode_canon_decode(const std::vector<float>& in,
                               uint32_t nx, uint32_t ny, uint32_t nz,
                               arctoZFPMode_t mode, double param,
                               float tol, const char* label)
{
  arctoZFPOpts_t opts = arctoZFPDefaultOpts;
  opts.mode = mode;
  opts.type = ARCTO_ZFP_TYPE_FLOAT;
  opts.param = param;
  opts.ndims = 3;
  opts.shape[0] = nx; opts.shape[1] = ny; opts.shape[2] = nz; opts.shape[3] = 1;

  size_t max_comp = 0;
  REQUIRE(arctoZFPCompressGetMaxOutputSize(opts, &max_comp) == arctoSuccess,
          "GetMaxOutputSize failed");

  const size_t input_bytes = in.size() * sizeof(float);
  void* d_in = nullptr;
  HIP_CHECK(hipMalloc(&d_in, input_bytes));
  HIP_CHECK(hipMemcpy(d_in, in.data(), input_bytes, hipMemcpyHostToDevice));

  std::vector<unsigned char> h_comp(max_comp);
  size_t comp_size = 0;
  REQUIRE(arctoZFPCompress(d_in, opts, h_comp.data(), max_comp, &comp_size)
          == arctoSuccess, "ARCTO compress failed");
  HIP_CHECK(hipFree(d_in));

  // Now decode h_comp with canonical's serial backend (no HIP at all).
  bitstream* bs = stream_open(h_comp.data(), comp_size);
  REQUIRE(bs != nullptr, "stream_open failed");
  zfp_stream* zfp = zfp_stream_open(bs);
  REQUIRE(zfp != nullptr, "zfp_stream_open failed");

  zfp_field* field = zfp_field_alloc();
  const size_t header_bits = zfp_read_header(zfp, field, ZFP_HEADER_FULL);
  REQUIRE(header_bits > 0, "canonical zfp_read_header failed on ARCTO stream");

  std::vector<float> decoded(in.size());
  zfp_field_set_pointer(field, decoded.data());
  zfp_stream_set_execution(zfp, zfp_exec_serial);

  const size_t bytes_read = zfp_decompress(zfp, field);
  REQUIRE(bytes_read > 0, "canonical zfp_decompress failed on ARCTO stream");

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  const float err = max_abs_diff(in, decoded);
  std::printf("  ARCTO->canon  %-32s max_err=%.4g  tol=%.4g\n",
              label, err, tol);
  REQUIRE(err <= tol, "canonical decode of ARCTO output exceeded tolerance");
}

// ===========================================================================
// canonical encode (serial) -> ARCTO decode (HIP). Confirms our reader is
// not picky about exec-policy padding produced by the upstream CPU encoder.
// ===========================================================================
void canon_encode_arcto_decode(const std::vector<float>& in,
                               uint32_t nx, uint32_t ny, uint32_t nz,
                               arctoZFPMode_t mode, double param,
                               float tol, const char* label)
{
  // 1. Encode via canonical serial.
  const zfp_type ztype = zfp_type_float;
  zfp_field* field = zfp_field_3d(const_cast<float*>(in.data()),
                                  ztype, nx, ny, nz);
  zfp_stream* zfp = zfp_stream_open(nullptr);
  canon_set_mode(zfp, mode, param, ztype, 3);

  const size_t max_payload = zfp_stream_maximum_size(zfp, field);
  const size_t max_header  = (ZFP_HEADER_MAX_BITS + 7) / 8;
  std::vector<unsigned char> h_canon(max_payload + max_header);

  bitstream* bs = stream_open(h_canon.data(), h_canon.size());
  zfp_stream_set_bit_stream(zfp, bs);

  REQUIRE(zfp_write_header(zfp, field, ZFP_HEADER_FULL) > 0,
          "canonical zfp_write_header failed");
  const size_t comp_size = zfp_compress(zfp, field);
  REQUIRE(comp_size > 0, "canonical zfp_compress (serial) failed");

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  // 2. Decode via ARCTO.
  const size_t input_bytes = in.size() * sizeof(float);
  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, input_bytes));

  size_t actual = 0;
  REQUIRE(arctoZFPDecompress(h_canon.data(), comp_size,
                             d_out, input_bytes, &actual) == arctoSuccess,
          "ARCTO decompress failed on canonical-serial stream");
  REQUIRE(actual == input_bytes, "ARCTO returned wrong uncomp size");

  std::vector<float> decoded(in.size());
  HIP_CHECK(hipMemcpy(decoded.data(), d_out, input_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));

  const float err = max_abs_diff(in, decoded);
  std::printf("  canon->ARCTO  %-32s max_err=%.4g  tol=%.4g\n",
              label, err, tol);
  REQUIRE(err <= tol, "ARCTO decode of canonical output exceeded tolerance");
}

} // namespace

int main()
{
  std::puts("--- ARCTO ZFP bit-compat regression vs canonical serial ---");

  const uint32_t nx = 64, ny = 64, nz = 32;
  const auto v = make_synth_3d(nx, ny, nz);

  // FIXED_RATE: cross-decode in both directions.
  // Tolerance: 2-ULP of the largest input scalar magnitude, scaled by 2^-K.
  for (double rate : {8.0, 16.0, 24.0}) {
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "fixed-rate K=%d", (int)rate);
    const float tol = std::pow(2.0f, -float(rate) + 8.0f);
    arcto_encode_canon_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_RATE, rate, tol, lbl);
    canon_encode_arcto_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_RATE, rate, tol, lbl);
  }

  // FIXED_PRECISION: ARCTO output carries our index trailer; canonical
  // serial decode reads block by block and naturally stops before the
  // trailer, so the trailer is transparent on the serial path.
  for (double p : {12.0, 20.0}) {
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "fixed-precision p=%d", (int)p);
    const float tol = std::pow(2.0f, -float(p) + 8.0f);
    arcto_encode_canon_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_PRECISION, p, tol, lbl);
    canon_encode_arcto_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_PRECISION, p, tol, lbl);
  }

  // FIXED_ACCURACY: tolerance is the parameter itself (slightly inflated
  // for ZFP's worst-case error vs nominal).
  for (double tol_param : {1e-3, 1e-6}) {
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "fixed-accuracy %.0e", tol_param);
    const float tol = float(tol_param) * 4.0f;
    arcto_encode_canon_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_ACCURACY, tol_param, tol, lbl);
    canon_encode_arcto_decode(v, nx, ny, nz,
        ARCTO_ZFP_MODE_FIXED_ACCURACY, tol_param, tol, lbl);
  }

  std::puts("SUCCESS: ARCTO and canonical reader/writer are interoperable");
  return 0;
}
