/*
 * ARCTO ZFP — low-level kernel implementation.
 *
 * Phase 1a (current): REVERSIBLE mode = lossless XOR-delta + per-chunk
 * byte-aligned bit-width packing. Real compression on correlated data,
 * lossless round-trip. ZFP-inspired (block-based, bit-plane-oriented)
 * but not yet the full ZFP reversible algorithm (that lands in Phase 1c,
 * along with the integer lifting transform).
 *
 * Other modes (FIXED_RATE / FIXED_PRECISION / FIXED_ACCURACY) currently
 * return arctoErrorNotSupported; FIXED_RATE lands in Phase 1b.
 *
 * Compressed chunk layout (REVERSIBLE):
 *   [ u32 actual_uncomp_size ]   header
 *   [ u8  bytes_per_delta (K)]
 *   [ u8  format_version (=1)]
 *   [ u16 reserved          ]
 *   [ u32 first_value       ]   preamble = raw value[0]
 *   [ u8  delta_1[K], ..., delta_{N-1}[K] ]
 * where delta_i = value[i] XOR value[i-1], packed as the K lowest bytes
 * (little-endian). N = ceil(actual_uncomp_size / 4); trailing bytes when
 * the size is not a multiple of 4 are zero-padded on encode and only the
 * first actual_uncomp_size bytes are emitted on decode.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "ZFPCompressionKernels.h"
#include "../HipUtils.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

namespace arcto
{
namespace lowlevel
{

namespace
{

constexpr int      kThreadsPerChunkCTA = 256;
constexpr uint32_t kZfpHeaderBytes     = 8;
constexpr uint32_t kZfpPreambleBytes   = 4;
constexpr uint32_t kZfpFormatVersion   = 1;

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* p, size_t i)
{
  // Read value[i] from a byte buffer, zero-padding any bytes past the end.
  // The encoder rounds the chunk up to a multiple of 4 so padding only
  // matters for the final value in chunks with non-4-aligned sizes.
  uint32_t v = 0;
  v |= static_cast<uint32_t>(p[4 * i + 0]) << 0;
  v |= static_cast<uint32_t>(p[4 * i + 1]) << 8;
  v |= static_cast<uint32_t>(p[4 * i + 2]) << 16;
  v |= static_cast<uint32_t>(p[4 * i + 3]) << 24;
  return v;
}

__device__ __forceinline__ uint32_t load_u32_padded(
    const uint8_t* p, size_t byte_offset, size_t in_size)
{
  uint32_t v = 0;
  #pragma unroll
  for (int b = 0; b < 4; b++) {
    const size_t off = byte_offset + b;
    if (off < in_size) {
      v |= static_cast<uint32_t>(p[off]) << (8 * b);
    }
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
  // Number of significant bits = 32 - clz(m). Round up to bytes.
  const int bits = 32 - __clz(static_cast<int>(m));
  return (bits + 7) / 8;
}

// ---------------------------------------------------------------------------
// REVERSIBLE mode kernels (XOR-delta + byte-aligned bit-width).
// One CTA per chunk; threads cooperate to scan, reduce, and pack.
// ---------------------------------------------------------------------------

__global__ void zfp_reversible_compress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const int chunk_idx = blockIdx.x;
  if (static_cast<size_t>(chunk_idx) >= batch_size) return;

  const uint8_t* in       = in_ptrs[chunk_idx];
  const size_t   in_size  = in_sizes[chunk_idx];
  uint8_t*       out      = out_ptrs[chunk_idx];
  const int      tid      = threadIdx.x;

  // Number of 4-byte values (rounding up; the trailing partial value is
  // zero-padded by load_u32_padded).
  const size_t N = (in_size + 3) / 4;

  // Pass 1: each thread scans its strided slice and tracks the maximum
  // XOR-delta magnitude.
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

    // Header
    store_u32_le(out + 0, static_cast<uint32_t>(in_size));
    out[4] = static_cast<uint8_t>(K);
    out[5] = static_cast<uint8_t>(kZfpFormatVersion);
    out[6] = 0;
    out[7] = 0;

    // Preamble (raw value[0], or zero for an empty chunk)
    const uint32_t first = (N == 0) ? 0u : load_u32_padded(in, 0, in_size);
    store_u32_le(out + kZfpHeaderBytes, first);

    const size_t body_bytes = (N == 0) ? 0 : (N - 1) * static_cast<size_t>(K);
    out_sizes[chunk_idx] = kZfpHeaderBytes + kZfpPreambleBytes + body_bytes;
  }
  __syncthreads();

  // Pass 2: pack deltas (value[i] XOR value[i-1]) into K bytes each, at
  // deterministic offset (i - 1) * K from the start of the body.
  const int      K    = K_bytes_s;
  uint8_t* const body = out + kZfpHeaderBytes + kZfpPreambleBytes;

  if (K == 0) return; // all values identical to value[0] → no body to write

  for (size_t i = tid; i < N; i += blockDim.x) {
    if (i == 0) continue; // covered by the preamble
    const uint32_t cur  = load_u32_padded(in, 4 * i, in_size);
    const uint32_t prev = load_u32_padded(in, 4 * (i - 1), in_size);
    const uint32_t delta = cur ^ prev;

    uint8_t* dst = body + (i - 1) * static_cast<size_t>(K);
    #pragma unroll
    for (int b = 0; b < 4; b++) {
      if (b < K) dst[b] = static_cast<uint8_t>(delta >> (8 * b));
    }
  }
}

// Sequential decode (Phase 1a): one thread reconstructs the chain
// value[i] = value[i-1] XOR delta[i]. Parallel prefix-XOR scan deferred
// to Phase 1b along with the kernel rework.
__global__ void zfp_reversible_decompress_kernel(
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
  if (threadIdx.x != 0) return;

  const uint8_t* in       = in_ptrs[chunk_idx];
  const size_t   in_size  = in_sizes[chunk_idx];
  uint8_t*       out      = out_ptrs[chunk_idx];
  const size_t   capacity = out_capacity[chunk_idx];

  auto set_status = [&](arctoStatus_t s, size_t sz) {
    if (statuses)         statuses[chunk_idx]         = s;
    if (actual_out_sizes) actual_out_sizes[chunk_idx] = sz;
  };

  if (in_size < kZfpHeaderBytes) { set_status(arctoErrorCannotDecompress, 0); return; }
  if (in[5] != kZfpFormatVersion) { set_status(arctoErrorCannotDecompress, 0); return; }

  const uint32_t actual_size = load_u32_le(in + 0);
  const int      K           = static_cast<int>(in[4]);
  if (K < 0 || K > 4)             { set_status(arctoErrorCannotDecompress, 0); return; }
  if (actual_size > capacity)     { set_status(arctoErrorCannotDecompress, 0); return; }

  const size_t N = (actual_size + 3) / 4;
  if (N == 0) { set_status(arctoSuccess, 0); return; }

  // Expected compressed body size (preamble + (N-1)*K)
  const size_t expected = kZfpHeaderBytes + kZfpPreambleBytes
                        + (N - 1) * static_cast<size_t>(K);
  if (in_size < expected) { set_status(arctoErrorCannotDecompress, 0); return; }

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

  const uint8_t* body = in + kZfpHeaderBytes + kZfpPreambleBytes;
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

  set_status(arctoSuccess, actual_size);
}

__global__ void zfp_reversible_get_sizes_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
  if (chunk_idx >= batch_size) return;

  const uint8_t* in       = in_ptrs[chunk_idx];
  const size_t   in_size  = in_sizes[chunk_idx];

  if (in_size < kZfpHeaderBytes || in[5] != kZfpFormatVersion) {
    out_sizes[chunk_idx] = 0;
    return;
  }
  out_sizes[chunk_idx] = static_cast<size_t>(load_u32_le(in + 0));
}

} // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

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

  if (opts.mode != ARCTO_ZFP_MODE_REVERSIBLE) {
    // Phase 1a ships REVERSIBLE only; surface a clear error so callers
    // know fixed-rate/precision/accuracy are not yet implemented.
    throw std::runtime_error(
        "arcto ZFP: only ARCTO_ZFP_MODE_REVERSIBLE is implemented in Phase 1a");
  }

  const dim3 grid(static_cast<unsigned>(batch_size), 1u, 1u);
  const dim3 block(kThreadsPerChunkCTA, 1u, 1u);
  zfp_reversible_compress_kernel<<<grid, block, 0, stream>>>(
      decomp_data_device,
      decomp_sizes_device,
      comp_data_device,
      comp_sizes_device,
      batch_size);
  HipUtils::check_last_error("zfp_reversible_compress_kernel launch");
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
  zfp_reversible_decompress_kernel<<<grid, block, 0, stream>>>(
      device_in_ptrs,
      device_in_bytes,
      device_out_bytes,
      device_out_ptrs,
      device_actual_uncompressed_bytes,
      device_status_ptrs,
      batch_size);
  HipUtils::check_last_error("zfp_reversible_decompress_kernel launch");
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
  zfp_reversible_get_sizes_kernel<<<blocks, kThreadsPerChunkCTA, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_bytes,
      batch_size);
  HipUtils::check_last_error("zfp_reversible_get_sizes_kernel launch");
}

size_t zfpComputeMaxSize(size_t chunk_size, arctoBatchedZFPOpts_t /* opts */)
{
  // REVERSIBLE worst case: K = 4 (no compression), 8 byte header,
  // 4 byte preamble, (N-1) * 4 byte body. With N = ceil(chunk_size/4),
  // total = 8 + 4 + (N-1)*4 = 8 + N*4 = 8 + ceil(chunk_size/4)*4.
  const size_t N = (chunk_size + 3) / 4;
  return kZfpHeaderBytes + N * 4u; // = 8 + rounded-up chunk_size
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
