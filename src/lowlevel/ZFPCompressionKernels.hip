/*
 * ARCTO ZFP — low-level kernel implementation.
 *
 * Modes supported in the current build:
 *   - REVERSIBLE (Phase 1a): lossless XOR-delta + byte-aligned bit-width
 *     packing. Round-trip is bit-exact. ZFP-inspired but not yet the
 *     canonical ZFP reversible algorithm (Phase 1c will add the integer
 *     lifting transform on top).
 *   - FIXED_RATE (Phase 1b, this commit): ZFP-style block-floating-point
 *     + integer lifting transform (3D, 4x4x4 block) + negabinary +
 *     fixed-rate truncated bit-plane. Round-trip is lossy within a bound
 *     proportional to 2^(emax - rate + 4). Validated via the dedicated
 *     test_zfp_fixed_rate fuzzy round-trip suite.
 *   - FIXED_PRECISION / FIXED_ACCURACY: still arctoErrorNotSupported.
 *
 * Compressed chunk layout (mode-independent header):
 *   [u32 actual_uncomp_size]      4 bytes
 *   [u8  mode_param]              1 byte  (K bytes per delta for REVERSIBLE,
 *                                          K bits per value for FIXED_RATE)
 *   [u8  version (=1)]            1 byte
 *   [u8  mode]                    1 byte  (0=REVERSIBLE, 1=FIXED_RATE)
 *   [u8  reserved]                1 byte
 *   [mode-specific body]
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "ZFPCompressionKernels.h"
#include "../HipUtils.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <climits>

namespace arcto
{
namespace lowlevel
{

namespace
{

// -- chunk header -----------------------------------------------------------
constexpr int      kThreadsPerChunkCTA = 256;
constexpr uint32_t kZfpHeaderBytes     = 8;
constexpr uint32_t kZfpFormatVersion   = 1;
constexpr uint8_t  kZfpModeReversible  = 0;
constexpr uint8_t  kZfpModeFixedRate   = 1;

// -- REVERSIBLE -------------------------------------------------------------
constexpr uint32_t kZfpReversiblePreambleBytes = 4;

// -- FIXED_RATE -------------------------------------------------------------
// One ZFP block in 3D = 4x4x4 = 64 float32 values = 256 bytes.
constexpr int      kZfpBlockDim3D     = 4;
constexpr int      kZfpValuesPerBlock = 64;
constexpr size_t   kZfpBytesPerBlock  = kZfpValuesPerBlock * sizeof(float);
// Bit-pattern bias for block-floating-point. Maps the largest |value| to
// roughly 2^kZfpEbias as a signed int32, leaving headroom for the integer
// lift transform's bit growth (~4-5 bits) before int32 overflow.
constexpr int      kZfpEbias          = 26;

// ===========================================================================
// Byte-stream helpers
// ===========================================================================

__device__ __forceinline__ uint32_t load_u32_padded(
    const uint8_t* p, size_t byte_offset, size_t in_size)
{
  uint32_t v = 0;
  #pragma unroll
  for (int b = 0; b < 4; b++) {
    const size_t off = byte_offset + b;
    if (off < in_size) v |= static_cast<uint32_t>(p[off]) << (8 * b);
  }
  return v;
}

__device__ __forceinline__ void store_u32_le(uint8_t* p, uint32_t v)
{
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p)
{
  return  (static_cast<uint32_t>(p[0]))
       | (static_cast<uint32_t>(p[1]) << 8)
       | (static_cast<uint32_t>(p[2]) << 16)
       | (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ int bytes_for_max(uint32_t m)
{
  if (m == 0) return 0;
  const int bits = 32 - __clz(static_cast<int>(m));
  return (bits + 7) / 8;
}

// ===========================================================================
// ZFP math: integer lift transform (lossy), block-floating-point, negabinary
// Used by FIXED_RATE / FIXED_PRECISION / FIXED_ACCURACY.
// ===========================================================================

// Forward 1D lift on a 4-vector with stride s. Lossy at the LSB due to >>1
// truncation; the loss is well below the noise floor after fixed-rate
// truncation, and is by design for the lossy ZFP modes.
__device__ __forceinline__ void zfp_fwd_lift(int32_t* p, int s)
{
  int32_t x = p[0 * s], y = p[1 * s], z = p[2 * s], w = p[3 * s];
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += (y >> 1); y -= (w >> 1);
  p[0 * s] = x; p[1 * s] = y; p[2 * s] = z; p[3 * s] = w;
}

__device__ __forceinline__ void zfp_inv_lift(int32_t* p, int s)
{
  int32_t x = p[0 * s], y = p[1 * s], z = p[2 * s], w = p[3 * s];
  y += (w >> 1); w -= (y >> 1);
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;
  p[0 * s] = x; p[1 * s] = y; p[2 * s] = z; p[3 * s] = w;
}

// 3D forward transform: 16 lifts per axis, in row-major (i = x + 4y + 16z).
__device__ static void zfp_fwd_xform_3d(int32_t b[kZfpValuesPerBlock])
{
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int y = 0; y < 4; y++)
      zfp_fwd_lift(b + 16 * z + 4 * y, 1);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      zfp_fwd_lift(b + 16 * z + x, 4);
  #pragma unroll
  for (int y = 0; y < 4; y++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      zfp_fwd_lift(b + 4 * y + x, 16);
}

__device__ static void zfp_inv_xform_3d(int32_t b[kZfpValuesPerBlock])
{
  #pragma unroll
  for (int y = 0; y < 4; y++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      zfp_inv_lift(b + 4 * y + x, 16);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      zfp_inv_lift(b + 16 * z + x, 4);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int y = 0; y < 4; y++)
      zfp_inv_lift(b + 16 * z + 4 * y, 1);
}

// Block-floating-point alignment: returns the largest binary exponent
// across finite, non-zero values in the block (frexpf convention: f = m*2^e
// with 0.5 <= |m| < 1). Returns INT_MIN for an all-zero / all-non-finite
// block, which the cast routines treat as a zero block.
__device__ static int zfp_compute_emax(const float v[kZfpValuesPerBlock])
{
  int emax = INT_MIN;
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    const float a = fabsf(v[i]);
    if (isfinite(a) && a > 0.0f) {
      int e;
      frexpf(a, &e);
      if (e > emax) emax = e;
    }
  }
  return emax;
}

// Forward cast: scale floats so |value| <= 2^kZfpEbias maps to int32 of
// magnitude ~2^kZfpEbias, then convert to signed int32. Clamps to int32
// range to avoid UB on extreme inputs.
__device__ static void zfp_fwd_cast(
    const float src[kZfpValuesPerBlock],
    int32_t     dst[kZfpValuesPerBlock],
    int         emax)
{
  if (emax == INT_MIN) {
    #pragma unroll
    for (int i = 0; i < kZfpValuesPerBlock; i++) dst[i] = 0;
    return;
  }
  const float scale = ldexpf(1.0f, kZfpEbias - emax);
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    const float f = src[i];
    if (!isfinite(f)) { dst[i] = 0; continue; }
    float s = f * scale;
    if (s >=  2147483520.0f) s =  2147483520.0f; // largest representable int32 in fp32
    if (s <= -2147483520.0f) s = -2147483520.0f;
    dst[i] = static_cast<int32_t>(s);
  }
}

__device__ static void zfp_inv_cast(
    const int32_t src[kZfpValuesPerBlock],
    float         dst[kZfpValuesPerBlock],
    int           emax)
{
  if (emax == INT_MIN) {
    #pragma unroll
    for (int i = 0; i < kZfpValuesPerBlock; i++) dst[i] = 0.0f;
    return;
  }
  const float scale = ldexpf(1.0f, emax - kZfpEbias);
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
}

// Two's complement <-> negabinary. Small-magnitude integers (positive or
// negative) become uint32s with long runs of leading zeros, which is what
// makes top-K-bit truncation a good lossy quantizer.
__device__ __forceinline__ uint32_t zfp_int_to_negabinary(int32_t x)
{
  return (static_cast<uint32_t>(x) + 0xAAAAAAAAu) ^ 0xAAAAAAAAu;
}

__device__ __forceinline__ int32_t zfp_negabinary_to_int(uint32_t x)
{
  return static_cast<int32_t>((x ^ 0xAAAAAAAAu) - 0xAAAAAAAAu);
}

// ===========================================================================
// FIXED_RATE per-block encode / decode
// Block layout: [int8 emax_byte][ceil(K*64/8) bytes of packed bits].
// ===========================================================================

__device__ static void zfp_fixed_rate_encode_block(
    const uint8_t* in,
    size_t         in_byte_offset,
    size_t         in_size,
    uint8_t*       out_block,
    int            K)
{
  // Load 64 floats, zero-padding past the end of the chunk.
  float v[kZfpValuesPerBlock];
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    const size_t off = in_byte_offset + i * sizeof(float);
    uint32_t bits = load_u32_padded(in, off, in_size);
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    v[i] = f;
  }

  const int emax = zfp_compute_emax(v);

  int32_t aligned[kZfpValuesPerBlock];
  zfp_fwd_cast(v, aligned, emax);

  zfp_fwd_xform_3d(aligned);

  uint32_t neg[kZfpValuesPerBlock];
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    neg[i] = zfp_int_to_negabinary(aligned[i]);
  }

  // emax to signed int8 ([-128, 127]). INT_MIN (all-zero block) → -128
  // sentinel; the decoder converts -128 back to INT_MIN and produces zeros.
  int8_t emax_byte;
  if (emax == INT_MIN)        emax_byte = -128;
  else if (emax >  127)       emax_byte =  127;
  else if (emax < -127)       emax_byte = -127;
  else                        emax_byte = static_cast<int8_t>(emax);
  out_block[0] = static_cast<uint8_t>(emax_byte);

  // Pack the top K bits of each negabinary coefficient into the body.
  uint8_t* bits_out = out_block + 1;
  uint64_t acc      = 0;
  int      acc_bits = 0;
  int      byte_idx = 0;
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    const uint32_t top = (K == 32) ? neg[i] : (neg[i] >> (32 - K));
    acc |= (static_cast<uint64_t>(top) << acc_bits);
    acc_bits += K;
    while (acc_bits >= 8) {
      bits_out[byte_idx++] = static_cast<uint8_t>(acc & 0xffu);
      acc >>= 8;
      acc_bits -= 8;
    }
  }
  if (acc_bits > 0) {
    bits_out[byte_idx++] = static_cast<uint8_t>(acc & 0xffu);
  }
}

__device__ static void zfp_fixed_rate_decode_block(
    const uint8_t* in_block,
    int            K,
    uint8_t*       out,
    size_t         out_byte_offset,
    size_t         out_size)
{
  const int8_t emax_byte = static_cast<int8_t>(in_block[0]);
  const int    emax      = (emax_byte == -128) ? INT_MIN : static_cast<int>(emax_byte);

  // Unpack K bits per coefficient into the top of a uint32.
  uint32_t        neg[kZfpValuesPerBlock];
  const uint8_t*  bits_in  = in_block + 1;
  uint64_t        acc      = 0;
  int             acc_bits = 0;
  int             byte_idx = 0;
  const uint64_t  mask     = (K == 32) ? 0xFFFFFFFFull : ((1ull << K) - 1ull);
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    while (acc_bits < K) {
      acc |= (static_cast<uint64_t>(bits_in[byte_idx++]) << acc_bits);
      acc_bits += 8;
    }
    const uint32_t top = static_cast<uint32_t>(acc & mask);
    acc      >>= K;
    acc_bits  -= K;
    neg[i] = (K == 32) ? top : (top << (32 - K));
  }

  int32_t aligned[kZfpValuesPerBlock];
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    aligned[i] = zfp_negabinary_to_int(neg[i]);
  }

  zfp_inv_xform_3d(aligned);

  float v[kZfpValuesPerBlock];
  zfp_inv_cast(aligned, v, emax);

  // Write back, truncating at out_size for partial trailing blocks.
  #pragma unroll
  for (int i = 0; i < kZfpValuesPerBlock; i++) {
    const size_t off = out_byte_offset + i * sizeof(float);
    if (off + sizeof(float) <= out_size) {
      std::memcpy(out + off, &v[i], sizeof(float));
    } else if (off < out_size) {
      uint32_t bits;
      std::memcpy(&bits, &v[i], sizeof(float));
      for (size_t b = off; b < out_size; b++) {
        out[b] = static_cast<uint8_t>(bits >> (8 * (b - off)));
      }
    }
  }
}

// ===========================================================================
// REVERSIBLE kernels (Phase 1a, retained)
// ===========================================================================

__global__ void zfp_reversible_compress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const int chunk_idx = blockIdx.x;
  if (static_cast<size_t>(chunk_idx) >= batch_size) return;

  const uint8_t* in      = in_ptrs[chunk_idx];
  const size_t   in_size = in_sizes[chunk_idx];
  uint8_t*       out     = out_ptrs[chunk_idx];
  const int      tid     = threadIdx.x;

  const size_t N = (in_size + 3) / 4;

  uint32_t local_max = 0;
  for (size_t i = tid; i < N; i += blockDim.x) {
    const uint32_t cur  = load_u32_padded(in, 4 * i, in_size);
    const uint32_t prev = (i == 0) ? 0u
                                   : load_u32_padded(in, 4 * (i - 1), in_size);
    const uint32_t delta = cur ^ prev;
    if (delta > local_max) local_max = delta;
  }

  __shared__ uint32_t reduce_buf[kThreadsPerChunkCTA];
  reduce_buf[tid] = local_max;
  __syncthreads();

  for (int s = kThreadsPerChunkCTA / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const uint32_t a = reduce_buf[tid];
      const uint32_t b = reduce_buf[tid + s];
      reduce_buf[tid] = (a > b) ? a : b;
    }
    __syncthreads();
  }

  __shared__ int K_bytes_s;
  if (tid == 0) {
    const uint32_t global_max = reduce_buf[0];
    const int K = bytes_for_max(global_max);
    K_bytes_s = K;

    store_u32_le(out + 0, static_cast<uint32_t>(in_size));
    out[4] = static_cast<uint8_t>(K);
    out[5] = static_cast<uint8_t>(kZfpFormatVersion);
    out[6] = kZfpModeReversible;
    out[7] = 0;

    const uint32_t first = (N == 0) ? 0u : load_u32_padded(in, 0, in_size);
    store_u32_le(out + kZfpHeaderBytes, first);

    const size_t body_bytes = (N == 0) ? 0
                                       : (N - 1) * static_cast<size_t>(K);
    out_sizes[chunk_idx] = kZfpHeaderBytes
                         + kZfpReversiblePreambleBytes
                         + body_bytes;
  }
  __syncthreads();

  const int      K    = K_bytes_s;
  uint8_t* const body = out + kZfpHeaderBytes + kZfpReversiblePreambleBytes;

  if (K == 0) return;

  for (size_t i = tid; i < N; i += blockDim.x) {
    if (i == 0) continue;
    const uint32_t cur   = load_u32_padded(in, 4 * i, in_size);
    const uint32_t prev  = load_u32_padded(in, 4 * (i - 1), in_size);
    const uint32_t delta = cur ^ prev;

    uint8_t* dst = body + (i - 1) * static_cast<size_t>(K);
    #pragma unroll
    for (int b = 0; b < 4; b++) {
      if (b < K) dst[b] = static_cast<uint8_t>(delta >> (8 * b));
    }
  }
}

__device__ static void zfp_reversible_decompress_serial(
    int chunk_idx,
    const uint8_t* const* in_ptrs,
    const size_t* in_sizes,
    const size_t* out_capacity,
    uint8_t* const* out_ptrs,
    size_t* actual_out_sizes,
    arctoStatus_t* statuses)
{
  if (threadIdx.x != 0) return;

  const uint8_t* in       = in_ptrs[chunk_idx];
  const size_t   in_size  = in_sizes[chunk_idx];
  uint8_t*       out      = out_ptrs[chunk_idx];
  const size_t   capacity = out_capacity[chunk_idx];

  auto fail = [&]() {
    if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
    if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
  };

  const uint32_t actual_size = load_u32_le(in + 0);
  const int      K           = static_cast<int>(in[4]);
  if (K < 0 || K > 4)         { fail(); return; }
  if (actual_size > capacity) { fail(); return; }

  const size_t N = (actual_size + 3) / 4;
  if (N == 0) {
    if (statuses)         statuses[chunk_idx]         = arctoSuccess;
    if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    return;
  }

  const size_t expected = kZfpHeaderBytes
                        + kZfpReversiblePreambleBytes
                        + (N - 1) * static_cast<size_t>(K);
  if (in_size < expected) { fail(); return; }

  uint32_t prev = load_u32_le(in + kZfpHeaderBytes);

  auto write_value = [&](size_t i, uint32_t v) {
    const size_t off = 4 * i;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
      const size_t pos = off + b;
      if (pos < actual_size) {
        out[pos] = static_cast<uint8_t>(v >> (8 * b));
      }
    }
  };

  write_value(0, prev);

  const uint8_t* body = in + kZfpHeaderBytes + kZfpReversiblePreambleBytes;
  for (size_t i = 1; i < N; i++) {
    uint32_t delta = 0;
    const uint8_t* src = body + (i - 1) * static_cast<size_t>(K);
    #pragma unroll
    for (int b = 0; b < 4; b++) {
      if (b < K) delta |= static_cast<uint32_t>(src[b]) << (8 * b);
    }
    const uint32_t cur = prev ^ delta;
    write_value(i, cur);
    prev = cur;
  }

  if (statuses)         statuses[chunk_idx]         = arctoSuccess;
  if (actual_out_sizes) actual_out_sizes[chunk_idx] = actual_size;
}

// ===========================================================================
// FIXED_RATE kernels
// ===========================================================================

__global__ void zfp_fixed_rate_compress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ out_sizes,
    int K,
    size_t batch_size)
{
  const int chunk_idx = blockIdx.x;
  if (static_cast<size_t>(chunk_idx) >= batch_size) return;

  const uint8_t* in      = in_ptrs[chunk_idx];
  const size_t   in_size = in_sizes[chunk_idx];
  uint8_t*       out     = out_ptrs[chunk_idx];
  const int      tid     = threadIdx.x;

  const size_t per_block_out = 1 + static_cast<size_t>((K * kZfpValuesPerBlock + 7) / 8);
  const size_t num_blocks    = (in_size + kZfpBytesPerBlock - 1) / kZfpBytesPerBlock;

  if (tid == 0) {
    store_u32_le(out + 0, static_cast<uint32_t>(in_size));
    out[4] = static_cast<uint8_t>(K);
    out[5] = static_cast<uint8_t>(kZfpFormatVersion);
    out[6] = kZfpModeFixedRate;
    out[7] = 0;
    out_sizes[chunk_idx] = kZfpHeaderBytes + num_blocks * per_block_out;
  }
  // No __syncthreads needed: the per-block writes below do not depend on
  // the header bytes.

  uint8_t* const body = out + kZfpHeaderBytes;
  for (size_t b = tid; b < num_blocks; b += blockDim.x) {
    zfp_fixed_rate_encode_block(
        in,
        b * kZfpBytesPerBlock,
        in_size,
        body + b * per_block_out,
        K);
  }
}

__device__ static void zfp_fixed_rate_decompress_chunk(
    int chunk_idx,
    const uint8_t* const* in_ptrs,
    const size_t* in_sizes,
    const size_t* out_capacity,
    uint8_t* const* out_ptrs,
    size_t* actual_out_sizes,
    arctoStatus_t* statuses)
{
  const uint8_t* in       = in_ptrs[chunk_idx];
  const size_t   in_size  = in_sizes[chunk_idx];
  uint8_t*       out      = out_ptrs[chunk_idx];
  const size_t   capacity = out_capacity[chunk_idx];
  const int      tid      = threadIdx.x;

  const uint32_t actual_size = load_u32_le(in + 0);
  const int      K           = static_cast<int>(in[4]);

  auto fail = [&]() {
    if (tid == 0) {
      if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    }
  };

  if (actual_size > capacity || K < 1 || K > 32) { fail(); return; }

  const size_t per_block_in = 1 + static_cast<size_t>((K * kZfpValuesPerBlock + 7) / 8);
  const size_t num_blocks   = (actual_size + kZfpBytesPerBlock - 1) / kZfpBytesPerBlock;
  const size_t expected     = kZfpHeaderBytes + num_blocks * per_block_in;
  if (in_size < expected) { fail(); return; }

  const uint8_t* body = in + kZfpHeaderBytes;
  for (size_t b = tid; b < num_blocks; b += blockDim.x) {
    zfp_fixed_rate_decode_block(
        body + b * per_block_in,
        K,
        out,
        b * kZfpBytesPerBlock,
        actual_size);
  }

  if (tid == 0) {
    if (statuses)         statuses[chunk_idx]         = arctoSuccess;
    if (actual_out_sizes) actual_out_sizes[chunk_idx] = actual_size;
  }
}

// ===========================================================================
// Unified decompress kernel: reads the mode byte from each chunk header and
// dispatches. All threads of a CTA process the same chunk, so the dispatch
// branch is uniform across the warp.
// ===========================================================================

__global__ void zfp_decompress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    const size_t* __restrict__ out_capacity,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ actual_out_sizes,
    arctoStatus_t* __restrict__ statuses,
    size_t batch_size)
{
  const int chunk_idx = blockIdx.x;
  if (static_cast<size_t>(chunk_idx) >= batch_size) return;

  const uint8_t* in      = in_ptrs[chunk_idx];
  const size_t   in_size = in_sizes[chunk_idx];
  const int      tid     = threadIdx.x;

  if (in_size < kZfpHeaderBytes || in[5] != kZfpFormatVersion) {
    if (tid == 0) {
      if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    }
    return;
  }

  const uint8_t mode = in[6];
  if (mode == kZfpModeReversible) {
    zfp_reversible_decompress_serial(
        chunk_idx, in_ptrs, in_sizes, out_capacity,
        out_ptrs, actual_out_sizes, statuses);
  } else if (mode == kZfpModeFixedRate) {
    zfp_fixed_rate_decompress_chunk(
        chunk_idx, in_ptrs, in_sizes, out_capacity,
        out_ptrs, actual_out_sizes, statuses);
  } else {
    if (tid == 0) {
      if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    }
  }
}

__global__ void zfp_get_decompress_sizes_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
  if (chunk_idx >= batch_size) return;

  const uint8_t* in      = in_ptrs[chunk_idx];
  const size_t   in_size = in_sizes[chunk_idx];

  if (in_size < kZfpHeaderBytes || in[5] != kZfpFormatVersion) {
    out_sizes[chunk_idx] = 0;
    return;
  }
  out_sizes[chunk_idx] = static_cast<size_t>(load_u32_le(in + 0));
}

// ===========================================================================
// Helpers used by host launchers
// ===========================================================================

int clamp_rate(double raw)
{
  if (raw < 1.0)   return 1;
  if (raw > 32.0)  return 32;
  return static_cast<int>(raw + 0.5);
}

} // namespace

// ===========================================================================
// Host launchers
// ===========================================================================

void zfpBatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* decomp_sizes_device,
    size_t /* max_chunk_size */,
    size_t batch_size,
    void* /* temp_data */,
    size_t /* temp_bytes */,
    uint8_t* const* comp_data_device,
    size_t* comp_sizes_device,
    arctoBatchedZFPOpts_t opts,
    hipStream_t stream)
{
  if (batch_size == 0) return;

  const dim3 grid(static_cast<unsigned>(batch_size), 1u, 1u);
  const dim3 block(kThreadsPerChunkCTA, 1u, 1u);

  if (opts.mode == ARCTO_ZFP_MODE_REVERSIBLE) {
    zfp_reversible_compress_kernel<<<grid, block, 0, stream>>>(
        decomp_data_device,
        decomp_sizes_device,
        comp_data_device,
        comp_sizes_device,
        batch_size);
    HipUtils::check_last_error("zfp_reversible_compress_kernel launch");
    return;
  }

  if (opts.mode == ARCTO_ZFP_MODE_FIXED_RATE) {
    const int K = clamp_rate(opts.param);
    zfp_fixed_rate_compress_kernel<<<grid, block, 0, stream>>>(
        decomp_data_device,
        decomp_sizes_device,
        comp_data_device,
        comp_sizes_device,
        K,
        batch_size);
    HipUtils::check_last_error("zfp_fixed_rate_compress_kernel launch");
    return;
  }

  throw std::runtime_error(
      "arcto ZFP: only ARCTO_ZFP_MODE_REVERSIBLE and ARCTO_ZFP_MODE_FIXED_RATE "
      "are implemented; FIXED_PRECISION / FIXED_ACCURACY land in a later phase");
}

void zfpBatchDecompress(
    const uint8_t* const* device_in_ptrs,
    const size_t* device_in_bytes,
    const size_t* device_out_bytes,
    size_t batch_size,
    void* /* temp_ptr */,
    size_t /* temp_bytes */,
    uint8_t* const* device_out_ptrs,
    size_t* device_actual_uncompressed_bytes,
    arctoStatus_t* device_status_ptrs,
    hipStream_t stream)
{
  if (batch_size == 0) return;
  const dim3 grid(static_cast<unsigned>(batch_size), 1u, 1u);
  const dim3 block(kThreadsPerChunkCTA, 1u, 1u);
  zfp_decompress_kernel<<<grid, block, 0, stream>>>(
      device_in_ptrs,
      device_in_bytes,
      device_out_bytes,
      device_out_ptrs,
      device_actual_uncompressed_bytes,
      device_status_ptrs,
      batch_size);
  HipUtils::check_last_error("zfp_decompress_kernel launch");
}

void zfpBatchGetDecompressSizes(
    const uint8_t* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  if (batch_size == 0) return;
  const size_t blocks =
      (batch_size + kThreadsPerChunkCTA - 1) / kThreadsPerChunkCTA;
  zfp_get_decompress_sizes_kernel<<<blocks, kThreadsPerChunkCTA, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_bytes,
      batch_size);
  HipUtils::check_last_error("zfp_get_decompress_sizes_kernel launch");
}

size_t zfpComputeMaxSize(size_t chunk_size, arctoBatchedZFPOpts_t opts)
{
  if (opts.mode == ARCTO_ZFP_MODE_FIXED_RATE) {
    const int K = clamp_rate(opts.param);
    const size_t num_blocks   = (chunk_size + kZfpBytesPerBlock - 1)
                              / kZfpBytesPerBlock;
    const size_t per_block_out = 1 + static_cast<size_t>(
                                         (K * kZfpValuesPerBlock + 7) / 8);
    return kZfpHeaderBytes + num_blocks * per_block_out;
  }
  // REVERSIBLE (and conservative upper bound for unimplemented modes):
  // header + K=4 worst-case payload = 8 + ceil(chunk_size/4) * 4.
  const size_t N = (chunk_size + 3) / 4;
  return kZfpHeaderBytes + N * 4u;
}

size_t zfpBatchCompressComputeTempSize(
    size_t /* max_chunk_size */,
    size_t /* batch_size */,
    arctoBatchedZFPOpts_t /* opts */)
{
  return 0;
}

size_t zfpBatchDecompressComputeTempSize(
    size_t /* max_chunks_in_batch */,
    size_t /* max_uncompressed_chunk_size */)
{
  return 0;
}

} // namespace lowlevel
} // namespace arcto
