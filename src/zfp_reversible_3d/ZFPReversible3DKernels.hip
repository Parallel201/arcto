/*
 * ARCTO ZFP-Reversible3D kernels.
 *
 * Algorithm (all four stages copied from canonical ZFP at the math layer,
 * implemented from scratch in our own code so we stay independent of
 * third_party/zfp/ at the source level):
 *
 *   1. Decompose the cube into 4x4x4 float32 blocks (zero-padded at edges).
 *   2. Block-floating-point cast: pick the largest binary exponent in the
 *      block, scale every value by 2^(ebias - emax) and round to int64.
 *      With ebias=30 and float32 input, the cast is strictly invertible.
 *   3. Apply the canonical ZFP reversible integer-lift transform on int64,
 *      sequentially in x, y, z (same coefficient ordering as canonical).
 *   4. Map signed int64 to unsigned uint64 via negabinary so the bits we
 *      pack are bit-position-significance ordered.
 *
 * On top of those primitives, the encoder runs the Phase 1e hybrid per-block
 * coder: every block tries BOTH a bit-plane coder (16-bit group-test +
 * 4 bits per active group, sparse-friendly) AND a flat K-bit-per-coefficient
 * coder (dense-friendly) and keeps whichever produced fewer bytes. The
 * choice is stored as the MSB of K[b]; the actual K value (0..64) lives in
 * the low 7 bits.
 *
 * Output layout (32-byte header + 6-byte/block metadata + variable body):
 *   bytes 0-3:   actual uncompressed size (= nx*ny*nz*sizeof(float))
 *   byte  4:     sub_format = kZfpRev3DSubFormatBitPlane (= 2)
 *   byte  5:     version
 *   byte  6:     mode = kZfpModeReversible3D (= 2)
 *   byte  7:     dim = kZfpDim3D (= 3)
 *   bytes 8-11:  nx
 *   bytes 12-15: ny
 *   bytes 16-19: nz
 *   bytes 20-23: last block's encoded byte count (the only value that the
 *                pack kernel can't recover from the scanned offsets[] alone)
 *   bytes 24-27: K_max across the cube (informational; atomicMax target)
 *   bytes 28-31: num_blocks
 *   bytes 32 + 0 .. 4*N:     offsets[] uint32_le, body-relative
 *   bytes 32 + 4*N .. 5*N:   K[] uint8 (bit 7 = scheme, bits 0-6 = K)
 *   bytes 32 + 5*N .. 6*N:   emax[] int8 (-128 sentinel for INT_MIN)
 *   bytes 32 + 6*N .. EOF:   body (variable per-block stream, byte-aligned)
 *
 * Three kernel passes on the encode side:
 *   1. compute_K: BFP + lift + negabinary; emit K[b], emax[b], race both
 *      coders to pick the cheaper one; write the chosen byte count into
 *      offsets[b].
 *   2. hipcub::DeviceScan::ExclusiveSum on offsets[] in place: byte counts
 *      become body-relative byte offsets.
 *   3. pack: re-run BFP+lift+negabinary; dispatch on K[b]'s MSB and write
 *      the encoded block at body+offsets[b]; thread (0,0) writes *out_size.
 *
 * The decoder is single-pass: read K[b]/offsets[b]/emax[b], dispatch on
 * the scheme bit, invert lift, invert BFP, store back to the output cube.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp_reversible_3d.h"
#include "../HipUtils.h"
#include "../arcto_hipcub.hiph"

#include <hip/hip_runtime.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace arcto {
namespace zfp_reversible_3d {

namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int      kThreadsPerCTA       = 256;
constexpr uint32_t kHeaderBytes         = 32;        // 24 fixed + 8 extra
constexpr uint8_t  kFormatVersion       = 1;
constexpr uint8_t  kModeReversible3D    = 2;
constexpr uint8_t  kDim3D               = 3;
constexpr uint8_t  kSubFormatBitPlane   = 2;         // Phase 1e hybrid
constexpr int      kValuesPerBlock      = 64;        // 4x4x4
constexpr int      kEbiasReversible     = 30;        // BFP scale exponent
// K[b] byte layout: high bit = coder choice, low 7 bits = K (0..64).
constexpr uint8_t  kSchemeMask          = 0x80u;
constexpr uint8_t  kKMask               = 0x7Fu;
constexpr uint8_t  kSchemeBP            = 0x00u;
constexpr uint8_t  kSchemeFlat          = 0x80u;

// ---------------------------------------------------------------------------
// Byte-stream helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void store_u32_le(uint8_t* p, uint32_t v)
{
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
  p[2] = uint8_t(v >> 16);
  p[3] = uint8_t(v >> 24);
}

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p)
{
  return  uint32_t(p[0])
       | (uint32_t(p[1]) << 8)
       | (uint32_t(p[2]) << 16)
       | (uint32_t(p[3]) << 24);
}

__host__ __device__ __forceinline__ uint32_t div_ceil_4(uint32_t n)
{
  return (n + 3u) >> 2;
}

__device__ __forceinline__ int bits64_for_max(uint64_t m)
{
  if (m == 0) return 0;
  return 64 - __clzll(static_cast<long long>(m));
}

// ---------------------------------------------------------------------------
// Negabinary (int64 <-> uint64), same as canonical
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t int_to_nega64(int64_t x)
{
  return (static_cast<uint64_t>(x) + 0xAAAAAAAAAAAAAAAAull)
       ^ 0xAAAAAAAAAAAAAAAAull;
}

__device__ __forceinline__ int64_t nega_to_int64(uint64_t x)
{
  return static_cast<int64_t>((x ^ 0xAAAAAAAAAAAAAAAAull)
                            -  0xAAAAAAAAAAAAAAAAull);
}

// ---------------------------------------------------------------------------
// emax + BFP cast (float32 <-> int64). ebias=30 leaves headroom for the
// lift transform's worst-case bit growth (~5 bits) inside int64.
// ---------------------------------------------------------------------------
__device__ static int compute_emax(const float v[kValuesPerBlock])
{
  int emax = INT_MIN;
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    const float a = fabsf(v[i]);
    if (isfinite(a) && a > 0.0f) {
      int e;
      frexpf(a, &e);
      if (e > emax) emax = e;
    }
  }
  return emax;
}

__device__ static void fwd_cast64(
    const float src[kValuesPerBlock],
    int64_t dst[kValuesPerBlock],
    int emax)
{
  if (emax == INT_MIN) {
    #pragma unroll
    for (int i = 0; i < kValuesPerBlock; i++) dst[i] = 0;
    return;
  }
  const double scale = ldexp(1.0, kEbiasReversible - emax);
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    const float f = src[i];
    if (!isfinite(f)) { dst[i] = 0; continue; }
    dst[i] = static_cast<int64_t>(llround(static_cast<double>(f) * scale));
  }
}

__device__ static void inv_cast64(
    const int64_t src[kValuesPerBlock],
    float dst[kValuesPerBlock],
    int emax)
{
  if (emax == INT_MIN) {
    #pragma unroll
    for (int i = 0; i < kValuesPerBlock; i++) dst[i] = 0.0f;
    return;
  }
  const double scale = ldexp(1.0, emax - kEbiasReversible);
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    dst[i] = static_cast<float>(static_cast<double>(src[i]) * scale);
  }
}

// ---------------------------------------------------------------------------
// ZFP reversible integer lift (int64). Verified bit-exact by fwd/inv pair.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fwd_lift64(int64_t* p, int s)
{
  int64_t x = p[0*s], y = p[1*s], z = p[2*s], w = p[3*s];
  w -= x; z -= y; y -= (w >> 1); x -= (z >> 1);
  y += x; z += w;
  p[0*s] = x; p[1*s] = y; p[2*s] = z; p[3*s] = w;
}

__device__ __forceinline__ void inv_lift64(int64_t* p, int s)
{
  int64_t x = p[0*s], y = p[1*s], z = p[2*s], w = p[3*s];
  z -= w; y -= x;
  x += (z >> 1); y += (w >> 1);
  z += y; w += x;
  p[0*s] = x; p[1*s] = y; p[2*s] = z; p[3*s] = w;
}

__device__ static void fwd_xform_3d64(int64_t b[kValuesPerBlock])
{
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int y = 0; y < 4; y++)
      fwd_lift64(b + 16*z + 4*y, 1);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      fwd_lift64(b + 16*z + x, 4);
  #pragma unroll
  for (int y = 0; y < 4; y++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      fwd_lift64(b + 4*y + x, 16);
}

__device__ static void inv_xform_3d64(int64_t b[kValuesPerBlock])
{
  // Inverse axis order.
  #pragma unroll
  for (int y = 0; y < 4; y++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      inv_lift64(b + 4*y + x, 16);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int x = 0; x < 4; x++)
      inv_lift64(b + 16*z + x, 4);
  #pragma unroll
  for (int z = 0; z < 4; z++)
    #pragma unroll
    for (int y = 0; y < 4; y++)
      inv_lift64(b + 16*z + 4*y, 1);
}

// ---------------------------------------------------------------------------
// Block load -> BFP -> lift -> negabinary; mirror inverse for store.
// ---------------------------------------------------------------------------
__device__ static void load_transform(
    const uint8_t* in_cube,
    uint32_t nx, uint32_t ny, uint32_t nz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint64_t neg[kValuesPerBlock],
    int& emax)
{
  float v[kValuesPerBlock];
  #pragma unroll
  for (int dz = 0; dz < 4; dz++)
    #pragma unroll
    for (int dy = 0; dy < 4; dy++)
      #pragma unroll
      for (int dx = 0; dx < 4; dx++) {
        const uint32_t x = bx * 4u + dx;
        const uint32_t y = by * 4u + dy;
        const uint32_t z = bz * 4u + dz;
        const int idx = 16 * dz + 4 * dy + dx;
        if (x < nx && y < ny && z < nz) {
          const size_t lin = (size_t(z) * ny + y) * nx + x;
          float f;
          std::memcpy(&f, in_cube + lin * sizeof(float), sizeof(float));
          v[idx] = f;
        } else {
          v[idx] = 0.0f;
        }
      }

  emax = compute_emax(v);

  int64_t aligned[kValuesPerBlock];
  fwd_cast64(v, aligned, emax);
  fwd_xform_3d64(aligned);

  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    neg[i] = int_to_nega64(aligned[i]);
  }
}

__device__ static void inv_transform_store(
    const uint64_t neg[kValuesPerBlock],
    int emax,
    uint8_t* out_cube,
    uint32_t nx, uint32_t ny, uint32_t nz,
    uint32_t bx, uint32_t by, uint32_t bz)
{
  int64_t aligned[kValuesPerBlock];
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    aligned[i] = nega_to_int64(neg[i]);
  }
  inv_xform_3d64(aligned);

  float v[kValuesPerBlock];
  inv_cast64(aligned, v, emax);

  #pragma unroll
  for (int dz = 0; dz < 4; dz++)
    #pragma unroll
    for (int dy = 0; dy < 4; dy++)
      #pragma unroll
      for (int dx = 0; dx < 4; dx++) {
        const uint32_t x = bx * 4u + dx;
        const uint32_t y = by * 4u + dy;
        const uint32_t z = bz * 4u + dz;
        if (x < nx && y < ny && z < nz) {
          const int idx = 16 * dz + 4 * dy + dx;
          const size_t lin = (size_t(z) * ny + y) * nx + x;
          std::memcpy(out_cube + lin * sizeof(float), &v[idx], sizeof(float));
        }
      }
}

// ---------------------------------------------------------------------------
// Per-block hybrid coder: flat (K low bits/coef) + bit-plane (group-test).
// Encoder races both at compute_K time and keeps the cheaper one; choice
// signalled in MSB of K[b].
// ---------------------------------------------------------------------------

// Flat: K low-order bits per coef, packed contiguously. Total K*64 bits.
__device__ static void flat_pack(
    const uint64_t neg[kValuesPerBlock], int K, uint8_t* out_bits)
{
  uint64_t acc = 0; int acc_bits = 0; int byte_idx = 0;
  const uint64_t mask = (K >= 64) ? ~uint64_t(0) : ((uint64_t(1) << K) - 1);
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    const uint64_t lo = neg[i] & mask;
    acc |= (lo << acc_bits);
    if (acc_bits + K < 64) {
      acc_bits += K;
    } else {
      const int produced = 64 - acc_bits;
      for (int b = 0; b < 8; b++) {
        out_bits[byte_idx++] = uint8_t(acc >> (8 * b));
      }
      acc = (K > produced) ? (lo >> produced) : uint64_t(0);
      acc_bits = K - produced;
    }
  }
  while (acc_bits > 0) {
    out_bits[byte_idx++] = uint8_t(acc & 0xFFu);
    acc >>= 8;
    acc_bits -= 8;
  }
}

__device__ static void flat_unpack(
    const uint8_t* in_bits, int K, uint64_t neg[kValuesPerBlock])
{
  uint64_t acc = 0; int acc_bits = 0; int byte_idx = 0;
  const uint64_t mask = (K >= 64) ? ~uint64_t(0) : ((uint64_t(1) << K) - 1);
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) {
    while (acc_bits < K) {
      acc |= uint64_t(in_bits[byte_idx++]) << acc_bits;
      acc_bits += 8;
      if (acc_bits >= 64) {
        neg[i] = acc & mask;
        acc = 0; acc_bits = 0;
        goto next_i;
      }
    }
    neg[i] = acc & mask;
    acc >>= K;
    acc_bits -= K;
    next_i:;
  }
}

// Bit-plane: walk planes MSB->LSB; per plane emit 16-bit group-test bitmap
// + 4 bits per active group.
__device__ __forceinline__ uint32_t bp_test_bits(
    const uint64_t neg[kValuesPerBlock], uint64_t mask)
{
  uint32_t t = 0;
  #pragma unroll
  for (int g = 0; g < 16; g++) {
    uint64_t any = (neg[4*g] | neg[4*g+1] | neg[4*g+2] | neg[4*g+3]) & mask;
    if (any) t |= (1u << g);
  }
  return t;
}

__device__ static uint32_t bp_count_bits(
    const uint64_t neg[kValuesPerBlock], int K)
{
  uint32_t total = 0;
  for (int p = K - 1; p >= 0; p--) {
    const uint64_t mask = uint64_t(1) << p;
    total += 16u + 4u * uint32_t(__popc(bp_test_bits(neg, mask)));
  }
  return total;
}

__device__ static void bp_pack(
    const uint64_t neg[kValuesPerBlock], int K, uint8_t* out_bits)
{
  uint64_t acc = 0; int acc_bits = 0; int byte_idx = 0;

  for (int p = K - 1; p >= 0; p--) {
    const uint64_t mask = uint64_t(1) << p;
    const uint32_t test = bp_test_bits(neg, mask);

    acc |= uint64_t(test) << acc_bits;
    acc_bits += 16;
    while (acc_bits >= 8) {
      out_bits[byte_idx++] = uint8_t(acc & 0xFFu);
      acc >>= 8; acc_bits -= 8;
    }

    for (int g = 0; g < 16; g++) {
      if (test & (1u << g)) {
        uint32_t coeff_bits = 0;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          coeff_bits |= uint32_t((neg[4*g+c] >> p) & 1u) << c;
        }
        acc |= uint64_t(coeff_bits) << acc_bits;
        acc_bits += 4;
        while (acc_bits >= 8) {
          out_bits[byte_idx++] = uint8_t(acc & 0xFFu);
          acc >>= 8; acc_bits -= 8;
        }
      }
    }
  }

  if (acc_bits > 0) {
    out_bits[byte_idx++] = uint8_t(acc & 0xFFu);
  }
}

__device__ static void bp_unpack(
    const uint8_t* in_bits, int K, uint64_t neg[kValuesPerBlock])
{
  #pragma unroll
  for (int i = 0; i < kValuesPerBlock; i++) neg[i] = 0;
  if (K == 0) return;

  uint64_t acc = 0; int acc_bits = 0; int byte_idx = 0;

  for (int p = K - 1; p >= 0; p--) {
    while (acc_bits < 16) {
      acc |= uint64_t(in_bits[byte_idx++]) << acc_bits;
      acc_bits += 8;
    }
    const uint32_t test = uint32_t(acc & 0xFFFFu);
    acc >>= 16; acc_bits -= 16;

    for (int g = 0; g < 16; g++) {
      if (test & (1u << g)) {
        while (acc_bits < 4) {
          acc |= uint64_t(in_bits[byte_idx++]) << acc_bits;
          acc_bits += 8;
        }
        const uint32_t coeff_bits = uint32_t(acc & 0xFu);
        acc >>= 4; acc_bits -= 4;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          neg[4*g+c] |= uint64_t((coeff_bits >> c) & 1u) << p;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Format-layout helpers: pointers into header + metadata arrays.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t* Kmax_slot(uint8_t* out) {
  return reinterpret_cast<uint32_t*>(out + 24);
}
__device__ __forceinline__ uint32_t* offsets_arr(uint8_t* out) {
  return reinterpret_cast<uint32_t*>(out + kHeaderBytes);
}
__device__ __forceinline__ const uint32_t* offsets_arr_c(const uint8_t* in) {
  return reinterpret_cast<const uint32_t*>(in + kHeaderBytes);
}
__device__ __forceinline__ uint8_t* K_arr_p(uint8_t* out, uint64_t nblocks) {
  return out + kHeaderBytes + nblocks * sizeof(uint32_t);
}
__device__ __forceinline__ const uint8_t* K_arr_c(const uint8_t* in, uint64_t nblocks) {
  return in + kHeaderBytes + nblocks * sizeof(uint32_t);
}
__device__ __forceinline__ uint8_t* emax_arr_p(uint8_t* out, uint64_t nblocks) {
  return out + kHeaderBytes + nblocks * (sizeof(uint32_t) + 1);
}
__device__ __forceinline__ const uint8_t* emax_arr_c(const uint8_t* in, uint64_t nblocks) {
  return in + kHeaderBytes + nblocks * (sizeof(uint32_t) + 1);
}
__host__ __device__ __forceinline__ size_t metadata_bytes(uint64_t nblocks) {
  return kHeaderBytes + nblocks * 6u;
}

// ===========================================================================
// Encode kernels
// ===========================================================================

__global__ void compute_K_kernel(
    const uint8_t* __restrict__ in_cube,
    uint32_t nx, uint32_t ny, uint32_t nz,
    uint8_t* __restrict__ out,
    uint64_t num_blocks)
{
  const uint64_t gid    = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  const uint32_t bx_count = div_ceil_4(nx);
  const uint32_t by_count = div_ceil_4(ny);

  // Thread 0 writes the fixed header. Bytes 20-27 (last-block byte count
  // + K_max atomicMax target) were memset to 0 by the host launcher.
  if (gid == 0) {
    store_u32_le(out + 0, nx * ny * nz * uint32_t(sizeof(float)));
    out[4] = kSubFormatBitPlane;
    out[5] = kFormatVersion;
    out[6] = kModeReversible3D;
    out[7] = kDim3D;
    store_u32_le(out +  8, nx);
    store_u32_le(out + 12, ny);
    store_u32_le(out + 16, nz);
    store_u32_le(out + 28, uint32_t(num_blocks));
  }

  uint32_t* offsets   = offsets_arr(out);
  uint8_t*  K_array   = K_arr_p(out, num_blocks);
  uint8_t*  emax_arr  = emax_arr_p(out, num_blocks);

  for (uint64_t b = gid; b < num_blocks; b += stride) {
    const uint32_t bx = uint32_t(b % bx_count);
    const uint32_t by = uint32_t((b / bx_count) % by_count);
    const uint32_t bz = uint32_t(b / (uint64_t(bx_count) * by_count));

    uint64_t neg[kValuesPerBlock];
    int emax;
    load_transform(in_cube, nx, ny, nz, bx, by, bz, neg, emax);

    uint64_t block_max = 0;
    #pragma unroll
    for (int i = 0; i < kValuesPerBlock; i++) {
      if (neg[i] > block_max) block_max = neg[i];
    }
    const int K_block = bits64_for_max(block_max);
    atomicMax(Kmax_slot(out), uint32_t(K_block));

    // Race the two coders: count bytes each would emit.
    const uint32_t bp_bits    = bp_count_bits(neg, K_block);
    const uint32_t bp_bytes   = (bp_bits + 7u) >> 3;
    const uint32_t flat_bytes = uint32_t(K_block) * 8u;
    const uint32_t bytes      = (bp_bytes <= flat_bytes) ? bp_bytes : flat_bytes;
    const uint8_t  scheme     = (bp_bytes <= flat_bytes) ? kSchemeBP : kSchemeFlat;

    K_array[b]  = uint8_t(K_block) | scheme;
    offsets[b]  = bytes;

    // Stash the LAST block's byte count for the pack kernel (it cannot
    // recover it from the exclusive-scan offsets[] alone).
    if (b == num_blocks - 1) {
      store_u32_le(out + 20, bytes);
    }

    int8_t emax_byte;
    if (emax == INT_MIN)   emax_byte = -128;
    else if (emax > 127)   emax_byte =  127;
    else if (emax < -127)  emax_byte = -127;
    else                   emax_byte = int8_t(emax);
    emax_arr[b] = uint8_t(emax_byte);
  }
}

__global__ void pack_kernel(
    const uint8_t* __restrict__ in_cube,
    uint8_t* __restrict__ out,
    size_t out_capacity,
    size_t* __restrict__ out_size,
    uint64_t num_blocks)
{
  const uint32_t nx = load_u32_le(out +  8);
  const uint32_t ny = load_u32_le(out + 12);
  const uint32_t nz = load_u32_le(out + 16);
  const uint32_t bx_count = div_ceil_4(nx);
  const uint32_t by_count = div_ceil_4(ny);

  const size_t   meta       = metadata_bytes(num_blocks);
  const uint32_t* offsets   = offsets_arr_c(out);
  const uint8_t*  K_array   = K_arr_c(out, num_blocks);

  const uint64_t gid    = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;

  const uint32_t bytes_last  = (num_blocks > 0) ? load_u32_le(out + 20) : 0u;
  const uint32_t offset_last = (num_blocks > 0) ? offsets[num_blocks - 1] : 0u;
  const size_t   total       = meta + offset_last + bytes_last;

  if (gid == 0) {
    *out_size = (total <= out_capacity) ? total : 0;
  }
  if (total > out_capacity) return;

  uint8_t* const body = out + meta;
  for (uint64_t b = gid; b < num_blocks; b += stride) {
    const uint8_t  K_byte  = K_array[b];
    const uint32_t K_block = uint32_t(K_byte & kKMask);
    if (K_block == 0u) continue;
    const bool use_flat = (K_byte & kSchemeMask) != 0;

    const uint32_t bx = uint32_t(b % bx_count);
    const uint32_t by = uint32_t((b / bx_count) % by_count);
    const uint32_t bz = uint32_t(b / (uint64_t(bx_count) * by_count));

    uint64_t neg[kValuesPerBlock];
    int unused;
    load_transform(in_cube, nx, ny, nz, bx, by, bz, neg, unused);

    if (use_flat) flat_pack(neg, int(K_block), body + offsets[b]);
    else          bp_pack  (neg, int(K_block), body + offsets[b]);
  }
}

// ===========================================================================
// Decode kernels
// ===========================================================================

__global__ void decompress_kernel(
    const uint8_t* __restrict__ in,
    size_t in_size,
    uint8_t* __restrict__ out_cube,
    size_t out_capacity,
    size_t* __restrict__ actual_out_size,
    arctoStatus_t* __restrict__ status)
{
  const uint64_t gid    = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;

  if (in_size < kHeaderBytes
   || in[4] != kSubFormatBitPlane
   || in[5] != kFormatVersion
   || in[6] != kModeReversible3D
   || in[7] != kDim3D) {
    if (gid == 0) {
      if (status)          *status          = arctoErrorCannotDecompress;
      if (actual_out_size) *actual_out_size = 0;
    }
    return;
  }

  const uint32_t actual_size = load_u32_le(in + 0);
  const uint32_t nx          = load_u32_le(in +  8);
  const uint32_t ny          = load_u32_le(in + 12);
  const uint32_t nz          = load_u32_le(in + 16);
  const uint32_t num_blocks  = load_u32_le(in + 28);

  if (actual_size > out_capacity) {
    if (gid == 0) {
      if (status)          *status          = arctoErrorCannotDecompress;
      if (actual_out_size) *actual_out_size = 0;
    }
    return;
  }

  const size_t meta = metadata_bytes(num_blocks);
  if (in_size < meta) {
    if (gid == 0) {
      if (status)          *status          = arctoErrorCannotDecompress;
      if (actual_out_size) *actual_out_size = 0;
    }
    return;
  }

  const uint32_t* offsets  = offsets_arr_c(in);
  const uint8_t*  K_array  = K_arr_c(in, num_blocks);
  const uint8_t*  emax_arr = emax_arr_c(in, num_blocks);
  const uint8_t*  body     = in + meta;

  const uint32_t bx_count = div_ceil_4(nx);
  const uint32_t by_count = div_ceil_4(ny);

  for (uint64_t b = gid; b < num_blocks; b += stride) {
    const uint32_t bx = uint32_t(b % bx_count);
    const uint32_t by = uint32_t((b / bx_count) % by_count);
    const uint32_t bz = uint32_t(b / (uint64_t(bx_count) * by_count));

    const uint8_t  K_byte  = K_array[b];
    const uint32_t K_block = uint32_t(K_byte & kKMask);
    const bool use_flat    = (K_byte & kSchemeMask) != 0;

    uint64_t neg[kValuesPerBlock];
    if (K_block == 0u) {
      #pragma unroll
      for (int i = 0; i < kValuesPerBlock; i++) neg[i] = 0;
    } else if (use_flat) {
      flat_unpack(body + offsets[b], int(K_block), neg);
    } else {
      bp_unpack(body + offsets[b], int(K_block), neg);
    }

    const int8_t emax_byte = int8_t(emax_arr[b]);
    const int    emax = (emax_byte == -128) ? INT_MIN : int(emax_byte);

    inv_transform_store(neg, emax, out_cube, nx, ny, nz, bx, by, bz);
  }

  if (gid == 0) {
    if (status)          *status          = arctoSuccess;
    if (actual_out_size) *actual_out_size = actual_size;
  }
}

__global__ void get_decompress_size_kernel(
    const uint8_t* __restrict__ in,
    size_t in_size,
    size_t* __restrict__ out_size)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (in_size < kHeaderBytes
   || in[4] != kSubFormatBitPlane
   || in[5] != kFormatVersion
   || in[6] != kModeReversible3D
   || in[7] != kDim3D) {
    *out_size = 0;
    return;
  }
  *out_size = size_t(load_u32_le(in + 0));
}

} // anonymous namespace

// ===========================================================================
// Host launchers
// ===========================================================================
static void launch_compress(
    const uint8_t* d_in,
    uint32_t nx, uint32_t ny, uint32_t nz,
    uint8_t* d_out, size_t d_out_capacity, size_t* d_out_size,
    hipStream_t stream)
{
  const uint64_t num_blocks =
      uint64_t(div_ceil_4(nx)) * div_ceil_4(ny) * div_ceil_4(nz);

  const unsigned tpb = kThreadsPerCTA;
  const uint64_t ctas = (num_blocks + tpb - 1) / tpb;
  const unsigned grid = unsigned(ctas > 65535u ? 65535u
                                              : (ctas == 0u ? 1u : ctas));

  // Clear bytes 20-27: bytes_last + K_max atomicMax target.
  HipUtils::check(hipMemsetAsync(d_out + 20, 0, 8, stream),
                  "rev3d compress: clear bytes_last + K_max");

  compute_K_kernel<<<grid, tpb, 0, stream>>>(d_in, nx, ny, nz, d_out, num_blocks);
  HipUtils::check_last_error("rev3d compute_K_kernel");

  // In-place ExclusiveSum on offsets[]: byte-counts -> body-relative offsets.
  uint32_t* offsets = reinterpret_cast<uint32_t*>(d_out + kHeaderBytes);
  void*  scan_ws = nullptr;
  size_t scan_ws_bytes = 0;
  HipUtils::check(hipcub::DeviceScan::ExclusiveSum(
      nullptr, scan_ws_bytes, offsets, offsets, int(num_blocks), stream),
      "rev3d compress: ExclusiveSum sizing");
  if (scan_ws_bytes > 0) {
    HipUtils::check(hipMallocAsync(&scan_ws, scan_ws_bytes, stream),
                    "rev3d compress: scan workspace alloc");
  }
  HipUtils::check(hipcub::DeviceScan::ExclusiveSum(
      scan_ws, scan_ws_bytes, offsets, offsets, int(num_blocks), stream),
      "rev3d compress: hipcub ExclusiveSum");
  if (scan_ws != nullptr) {
    HipUtils::check(hipFreeAsync(scan_ws, stream),
                    "rev3d compress: scan workspace free");
  }

  pack_kernel<<<grid, tpb, 0, stream>>>(
      d_in, d_out, d_out_capacity, d_out_size, num_blocks);
  HipUtils::check_last_error("rev3d pack_kernel");
}

static void launch_decompress(
    const uint8_t* d_in, size_t d_in_size,
    uint8_t* d_out, size_t d_out_capacity,
    size_t* d_actual_size, arctoStatus_t* d_status,
    hipStream_t stream)
{
  const unsigned grid = 65535u;
  decompress_kernel<<<grid, kThreadsPerCTA, 0, stream>>>(
      d_in, d_in_size, d_out, d_out_capacity, d_actual_size, d_status);
  HipUtils::check_last_error("rev3d decompress_kernel");
}

} // namespace zfp_reversible_3d
} // namespace arcto

// ===========================================================================
// C public API
// ===========================================================================
extern "C" {

arctoStatus_t arctoZFPReversible3DCompressGetMaxOutputSize(
    arctoZFPReversible3DOpts_t opts, size_t* max_bytes)
{
  if (!max_bytes) return arctoErrorInvalidValue;
  if (opts.type != ARCTO_ZFP_REV3D_TYPE_FLOAT) return arctoErrorNotSupported;
  if (opts.nx == 0 || opts.ny == 0 || opts.nz == 0) return arctoErrorInvalidValue;

  using namespace arcto::zfp_reversible_3d;
  const uint64_t nblocks = uint64_t((opts.nx + 3u) / 4u)
                         * ((opts.ny + 3u) / 4u)
                         * ((opts.nz + 3u) / 4u);
  // Worst case: bit-plane coder with K=64 and every group active on every
  // plane -> 16 + 4*16 = 80 bits/plane * 64 planes = 640 bytes/block.
  // Metadata = 4 offset + 1 K + 1 emax = 6 bytes/block. Plus 32-byte header.
  *max_bytes = kHeaderBytes + nblocks * (6u + 64u * 10u);
  return arctoSuccess;
}

arctoStatus_t arctoZFPReversible3DCompressAsync(
    const void* d_input,
    arctoZFPReversible3DOpts_t opts,
    void* d_output, size_t d_output_capacity,
    size_t* d_output_size,
    hipStream_t stream)
{
  if (!d_input || !d_output || !d_output_size) return arctoErrorInvalidValue;
  if (opts.type != ARCTO_ZFP_REV3D_TYPE_FLOAT) return arctoErrorNotSupported;
  if (opts.nx == 0 || opts.ny == 0 || opts.nz == 0) return arctoErrorInvalidValue;

  arcto::zfp_reversible_3d::launch_compress(
      static_cast<const uint8_t*>(d_input),
      opts.nx, opts.ny, opts.nz,
      static_cast<uint8_t*>(d_output), d_output_capacity, d_output_size,
      stream);
  return arctoSuccess;
}

arctoStatus_t arctoZFPReversible3DDecompressAsync(
    const void* d_input, size_t d_input_size,
    void* d_output, size_t d_output_capacity,
    size_t* d_actual_size,
    arctoStatus_t* d_status,
    hipStream_t stream)
{
  if (!d_input || !d_output) return arctoErrorInvalidValue;

  arcto::zfp_reversible_3d::launch_decompress(
      static_cast<const uint8_t*>(d_input), d_input_size,
      static_cast<uint8_t*>(d_output), d_output_capacity,
      d_actual_size, d_status, stream);
  return arctoSuccess;
}

arctoStatus_t arctoZFPReversible3DGetDecompressSizeAsync(
    const void* d_input, size_t d_input_size,
    size_t* d_uncomp_bytes, hipStream_t stream)
{
  if (!d_input || !d_uncomp_bytes) return arctoErrorInvalidValue;

  arcto::zfp_reversible_3d::get_decompress_size_kernel<<<1, 1, 0, stream>>>(
      static_cast<const uint8_t*>(d_input), d_input_size, d_uncomp_bytes);
  arcto::HipUtils::check_last_error("rev3d get_decompress_size_kernel");
  return arctoSuccess;
}

} // extern "C"
