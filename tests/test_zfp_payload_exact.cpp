/*
 * ZFP-T1: byte-exact gate for ARCTO's HIP ZFP path against the canonical
 * serial backend.
 *
 * For every (type, dimensionality, shape, mode, parameter) case:
 *   1. ARCTO compress (HIP) twice  -> identical bytes (determinism);
 *   2. the payload region of the ARCTO stream (header stripped, ARCTO's
 *      index trailer stripped) equals the canonical zfp_exec_serial payload
 *      of the same field byte for byte (sizes may differ by at most one
 *      64-bit word of flush padding, and the common prefix must match);
 *   3. ARCTO decompress (HIP) of the ARCTO stream and canonical serial
 *      decompress of the serial stream give bit-identical fields.
 *
 * Shapes with nx % 4 != 0 exercise partial blocks; the "zeros" profile
 * exercises the all-zero-block fast paths; rate 12.5 exercises word sharing
 * between blocks in fixed-rate mode. Every kernel- or host-side change to the
 * ZFP path is gated by this test.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"
#include "zfp.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define REQUIRE(cond, msg)                                                    \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d %s -- %s\n", __FILE__, __LINE__, #cond, \
                   msg);                                                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "FAIL HIP %s:%d %s -- %s\n", __FILE__, __LINE__,    \
                   #expr, hipGetErrorString(_e));                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

using bytes_t = std::vector<unsigned char>;

constexpr uint32_t kTrailerMagic = 0x50465A41u; // "AZFP"
constexpr size_t kTrailerBytes = 20u;

enum class Profile { Smooth, Zeros, Random };

const char* profile_name(Profile p)
{
  switch (p) {
    case Profile::Smooth: return "smooth";
    case Profile::Zeros: return "zeros";
    case Profile::Random: return "random";
  }
  return "?";
}

zfp_type to_zfp(arctoZFPType_t t)
{
  switch (t) {
    case ARCTO_ZFP_TYPE_FLOAT: return zfp_type_float;
    case ARCTO_ZFP_TYPE_DOUBLE: return zfp_type_double;
    case ARCTO_ZFP_TYPE_INT32: return zfp_type_int32;
    case ARCTO_ZFP_TYPE_INT64: return zfp_type_int64;
  }
  return zfp_type_none;
}

size_t elem_size(arctoZFPType_t t)
{
  switch (t) {
    case ARCTO_ZFP_TYPE_FLOAT: return 4;
    case ARCTO_ZFP_TYPE_DOUBLE: return 8;
    case ARCTO_ZFP_TYPE_INT32: return 4;
    case ARCTO_ZFP_TYPE_INT64: return 8;
  }
  return 0;
}

const char* type_name(arctoZFPType_t t)
{
  switch (t) {
    case ARCTO_ZFP_TYPE_FLOAT: return "float";
    case ARCTO_ZFP_TYPE_DOUBLE: return "double";
    case ARCTO_ZFP_TYPE_INT32: return "int32";
    case ARCTO_ZFP_TYPE_INT64: return "int64";
  }
  return "?";
}

const char* mode_name(arctoZFPMode_t m)
{
  switch (m) {
    case ARCTO_ZFP_MODE_FIXED_RATE: return "rate";
    case ARCTO_ZFP_MODE_FIXED_PRECISION: return "precision";
    case ARCTO_ZFP_MODE_FIXED_ACCURACY: return "accuracy";
    default: return "?";
  }
}

// Field contents as raw bytes of the requested type.
bytes_t make_field(arctoZFPType_t t, const uint32_t shape[3], uint32_t ndims, Profile p, uint32_t seed)
{
  const size_t nx = shape[0], ny = ndims > 1 ? shape[1] : 1, nz = ndims > 2 ? shape[2] : 1;
  const size_t n = nx * ny * nz;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ur(-1.0, 1.0);
  std::uniform_int_distribution<int64_t> ui(-100000, 100000);
  bytes_t out(n * elem_size(t));
  for (size_t z = 0; z < nz; z++)
    for (size_t y = 0; y < ny; y++)
      for (size_t x = 0; x < nx; x++) {
        const size_t i = (z * ny + y) * nx + x;
        double v;
        switch (p) {
          case Profile::Smooth:
            v = std::sin(0.05 * x) * std::cos(0.07 * y) + 0.5 * std::sin(0.11 * z) + 1e-3 * ur(rng);
            break;
          case Profile::Zeros:
            // zero except for a few scattered non-zero blocks, so most blocks
            // take the all-zero fast path and some do not
            v = ((x / 4 + y / 4 + z / 4) % 7 == 3) ? std::sin(0.3 * x + 0.2 * y) : 0.0;
            break;
          default:
            v = ur(rng);
        }
        switch (t) {
          case ARCTO_ZFP_TYPE_FLOAT: { float f = static_cast<float>(v); std::memcpy(&out[i * 4], &f, 4); break; }
          case ARCTO_ZFP_TYPE_DOUBLE: { std::memcpy(&out[i * 8], &v, 8); break; }
          case ARCTO_ZFP_TYPE_INT32: { int32_t q = (p == Profile::Random) ? static_cast<int32_t>(ui(rng)) : static_cast<int32_t>(v * 1000.0); std::memcpy(&out[i * 4], &q, 4); break; }
          case ARCTO_ZFP_TYPE_INT64: { int64_t q = (p == Profile::Random) ? ui(rng) * 100000 : static_cast<int64_t>(v * 1.0e9); std::memcpy(&out[i * 8], &q, 8); break; }
        }
      }
  return out;
}

zfp_field* canon_field(arctoZFPType_t t, const uint32_t shape[3], uint32_t ndims, void* data)
{
  const zfp_type zt = to_zfp(t);
  switch (ndims) {
    case 1: return zfp_field_1d(data, zt, shape[0]);
    case 2: return zfp_field_2d(data, zt, shape[0], shape[1]);
    default: return zfp_field_3d(data, zt, shape[0], shape[1], shape[2]);
  }
}

void canon_set_mode(zfp_stream* zfp, arctoZFPMode_t mode, double param, arctoZFPType_t t, uint32_t ndims)
{
  switch (mode) {
    case ARCTO_ZFP_MODE_FIXED_RATE: zfp_stream_set_rate(zfp, param, to_zfp(t), ndims, zfp_false); break;
    case ARCTO_ZFP_MODE_FIXED_PRECISION: zfp_stream_set_precision(zfp, static_cast<uint>(param)); break;
    case ARCTO_ZFP_MODE_FIXED_ACCURACY: zfp_stream_set_accuracy(zfp, param); break;
    default: REQUIRE(false, "unsupported mode");
  }
  REQUIRE(zfp_stream_set_execution(zfp, zfp_exec_serial), "serial execution not available");
}

// ARCTO (HIP) compression of a host field: returns the full stream.
bytes_t arcto_compress(const bytes_t& field, const arctoZFPOpts_t& opts)
{
  size_t max_comp = 0;
  REQUIRE(arctoZFPCompressGetMaxOutputSize(opts, &max_comp) == arctoSuccess, "GetMaxOutputSize");
  void* d_in = nullptr;
  HIP_CHECK(hipMalloc(&d_in, field.size()));
  HIP_CHECK(hipMemcpy(d_in, field.data(), field.size(), hipMemcpyHostToDevice));
  bytes_t comp(max_comp);
  size_t comp_size = 0;
  REQUIRE(arctoZFPCompress(d_in, opts, comp.data(), max_comp, &comp_size) == arctoSuccess, "arctoZFPCompress");
  HIP_CHECK(hipFree(d_in));
  comp.resize(comp_size);
  return comp;
}

// ARCTO (HIP) decompression into host bytes.
bytes_t arcto_decompress(const bytes_t& comp, size_t field_bytes)
{
  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, field_bytes));
  HIP_CHECK(hipMemset(d_out, 0xA5, field_bytes));
  size_t actual = 0;
  REQUIRE(arctoZFPDecompress(comp.data(), comp.size(), d_out, field_bytes, &actual) == arctoSuccess, "arctoZFPDecompress");
  REQUIRE(actual == field_bytes, "decompressed size");
  bytes_t out(field_bytes);
  HIP_CHECK(hipMemcpy(out.data(), d_out, field_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  return out;
}

// Split an ARCTO stream into header / payload / (index+trailer).
void split_arcto_stream(const bytes_t& comp, size_t* header_bytes, size_t* payload_bytes)
{
  bitstream* bs = stream_open(const_cast<unsigned char*>(comp.data()), comp.size());
  zfp_stream* zfp = zfp_stream_open(bs);
  zfp_field* f = zfp_field_alloc();
  const size_t hbits = zfp_read_header(zfp, f, ZFP_HEADER_FULL);
  zfp_field_free(f);
  zfp_stream_close(zfp);
  stream_close(bs);
  REQUIRE(hbits > 0, "header parse of ARCTO stream");
  *header_bytes = (hbits + 7) / 8;
  size_t tail_bytes = 0;
  if (comp.size() >= *header_bytes + kTrailerBytes) {
    const unsigned char* tail = comp.data() + comp.size() - kTrailerBytes;
    const uint32_t magic = uint32_t(tail[16]) | uint32_t(tail[17]) << 8 | uint32_t(tail[18]) << 16 | uint32_t(tail[19]) << 24;
    if (magic == kTrailerMagic) {
      uint64_t isz = 0;
      for (int i = 0; i < 8; i++) isz |= uint64_t(tail[8 + i]) << (8 * i);
      tail_bytes = static_cast<size_t>(isz) + kTrailerBytes;
    }
  }
  REQUIRE(comp.size() >= *header_bytes + tail_bytes, "stream layout");
  *payload_bytes = comp.size() - *header_bytes - tail_bytes;
}

// Canonical serial compression (payload only) and decompression of that payload.
void canon_round_trip(const bytes_t& field, arctoZFPType_t t, const uint32_t shape[3], uint32_t ndims,
                      arctoZFPMode_t mode, double param, bytes_t* payload, bytes_t* decoded)
{
  bytes_t in = field;
  zfp_field* f = canon_field(t, shape, ndims, in.data());
  zfp_stream* zfp = zfp_stream_open(nullptr);
  canon_set_mode(zfp, mode, param, t, ndims);
  const size_t cap = zfp_stream_maximum_size(zfp, f);
  bytes_t buf(cap + 64, 0);
  bitstream* bs = stream_open(buf.data(), buf.size());
  zfp_stream_set_bit_stream(zfp, bs);
  zfp_stream_rewind(zfp);
  const size_t n = zfp_compress(zfp, f);
  REQUIRE(n > 0, "canonical serial compress");
  payload->assign(buf.begin(), buf.begin() + n);

  // decode the serial payload with the serial decoder
  decoded->assign(field.size(), 0);
  zfp_field* g = canon_field(t, shape, ndims, decoded->data());
  zfp_stream_rewind(zfp);
  const size_t r = zfp_decompress(zfp, g);
  REQUIRE(r > 0, "canonical serial decompress");
  zfp_field_free(g);
  zfp_field_free(f);
  zfp_stream_close(zfp);
  stream_close(bs);
}

size_t first_diff(const bytes_t& a, const bytes_t& b)
{
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return i;
  return n;
}

struct Shape { uint32_t ndims; uint32_t dims[3]; };

} // namespace

int main()
{
  std::puts("--- ZFP-T1: HIP payload and decoded field byte-exact vs canonical serial ---");
  const Shape shapes[] = {{3, {61, 33, 18}}, {3, {64, 64, 32}}, {2, {70, 41, 1}}, {1, {1000, 1, 1}}};
  const arctoZFPType_t types[] = {ARCTO_ZFP_TYPE_FLOAT, ARCTO_ZFP_TYPE_DOUBLE, ARCTO_ZFP_TYPE_INT32, ARCTO_ZFP_TYPE_INT64};
  const Profile profiles[] = {Profile::Smooth, Profile::Zeros, Profile::Random};
  struct ModeParam { arctoZFPMode_t mode; double param; bool floats_only; };
  const ModeParam modes[] = {
      {ARCTO_ZFP_MODE_FIXED_RATE, 8.0, false}, {ARCTO_ZFP_MODE_FIXED_RATE, 12.5, false},
      {ARCTO_ZFP_MODE_FIXED_RATE, 16.0, false}, {ARCTO_ZFP_MODE_FIXED_RATE, 24.0, false},
      {ARCTO_ZFP_MODE_FIXED_PRECISION, 12.0, false}, {ARCTO_ZFP_MODE_FIXED_PRECISION, 20.0, false},
      {ARCTO_ZFP_MODE_FIXED_ACCURACY, 1e-3, true}, {ARCTO_ZFP_MODE_FIXED_ACCURACY, 1e-6, true}};

  size_t cases = 0;
  uint32_t seed = 1;
  for (const Shape& sh : shapes)
    for (arctoZFPType_t t : types)
      for (Profile p : profiles)
        for (const ModeParam& mp : modes) {
          const bool is_float = (t == ARCTO_ZFP_TYPE_FLOAT || t == ARCTO_ZFP_TYPE_DOUBLE);
          if (mp.floats_only && !is_float) continue;
          if (p == Profile::Random && sh.ndims == 3 && sh.dims[0] == 64) continue; // keep the run short

          arctoZFPOpts_t opts = arctoZFPDefaultOpts;
          opts.mode = mp.mode; opts.type = t; opts.param = mp.param; opts.ndims = sh.ndims;
          opts.shape[0] = sh.dims[0]; opts.shape[1] = sh.dims[1]; opts.shape[2] = sh.dims[2]; opts.shape[3] = 1;

          const bytes_t field = make_field(t, sh.dims, sh.ndims, p, seed++);
          char label[160];
          std::snprintf(label, sizeof label, "%s %ux%ux%u %s %s %s=%g", type_name(t), sh.dims[0], sh.dims[1], sh.dims[2],
                        profile_name(p), mode_name(mp.mode), mode_name(mp.mode), mp.param);

          // 1. determinism
          const bytes_t a = arcto_compress(field, opts);
          const bytes_t b = arcto_compress(field, opts);
          if (a != b) { std::fprintf(stderr, "%s: two HIP compressions differ at byte %zu of %zu\n", label, first_diff(a, b), a.size()); REQUIRE(false, "HIP compression not deterministic"); }

          // 2. payload == serial payload
          size_t header_bytes = 0, payload_bytes = 0;
          split_arcto_stream(a, &header_bytes, &payload_bytes);
          bytes_t serial_payload, serial_decoded;
          canon_round_trip(field, t, sh.dims, sh.ndims, mp.mode, mp.param, &serial_payload, &serial_decoded);
          const bytes_t hip_payload(a.begin() + header_bytes, a.begin() + header_bytes + payload_bytes);
          const size_t common = hip_payload.size() < serial_payload.size() ? hip_payload.size() : serial_payload.size();
          const size_t diff = first_diff(hip_payload, serial_payload);
          const size_t size_delta = hip_payload.size() > serial_payload.size() ? hip_payload.size() - serial_payload.size() : serial_payload.size() - hip_payload.size();
          if (diff < common || size_delta > sizeof(uint64_t)) {
            std::fprintf(stderr, "%s: HIP payload %zu B vs serial %zu B, first difference at byte %zu\n", label, hip_payload.size(), serial_payload.size(), diff);
            REQUIRE(false, "HIP payload differs from canonical serial payload");
          }

          // 3. decoded fields bit-identical
          const bytes_t hip_decoded = arcto_decompress(a, field.size());
          if (hip_decoded != serial_decoded) { std::fprintf(stderr, "%s: decoded fields differ at byte %zu\n", label, first_diff(hip_decoded, serial_decoded)); REQUIRE(false, "HIP and serial decoded fields differ"); }

          std::printf("  ok  %-52s payload %7zu B (serial %7zu B)\n", label, hip_payload.size(), serial_payload.size());
          cases++;
        }
  std::printf("SUCCESS: %zu cases byte-exact vs canonical serial\n", cases);
  return 0;
}
