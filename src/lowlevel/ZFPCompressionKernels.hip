/*
 * ARCTO ZFP — low-level kernel implementation (Phase 0 passthrough stub).
 *
 * The compressed layout for Phase 0 is:
 *   [uint64_t original_size_le][original_size bytes verbatim]
 *
 * This stub exists so the rest of the toolkit plumbing (API, factory wiring
 * planned for Phase 1, benchmarks, tests) can be built and round-trip
 * validated before the real ZFP transform/encode kernel lands in Phase 1.
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

constexpr size_t kHeaderBytes = sizeof(uint64_t);
constexpr int    kThreadsPerBlock = 256;

__global__ void zfp_passthrough_compress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const size_t chunk_idx = blockIdx.y;
  if (chunk_idx >= batch_size) return;

  const size_t   in_size = in_sizes[chunk_idx];
  const uint8_t* src     = in_ptrs[chunk_idx];
  uint8_t*       dst     = out_ptrs[chunk_idx];

  // Thread 0 of block 0 writes the header and the output size.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *reinterpret_cast<uint64_t*>(dst) = static_cast<uint64_t>(in_size);
    out_sizes[chunk_idx] = in_size + kHeaderBytes;
  }

  // All threads copy the body byte-wise.
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint8_t* body = dst + kHeaderBytes;
  for (size_t i = tid; i < in_size; i += stride) {
    body[i] = src[i];
  }
}

__global__ void zfp_passthrough_decompress_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    const size_t* __restrict__ out_capacity,
    uint8_t* const* __restrict__ out_ptrs,
    size_t* __restrict__ actual_out_sizes,
    arctoStatus_t* __restrict__ statuses,
    size_t batch_size)
{
  const size_t chunk_idx = blockIdx.y;
  if (chunk_idx >= batch_size) return;

  const uint8_t* src     = in_ptrs[chunk_idx];
  const size_t   in_size = in_sizes[chunk_idx];
  uint8_t*       dst     = out_ptrs[chunk_idx];

  if (in_size < kHeaderBytes) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    }
    return;
  }

  const uint64_t original = *reinterpret_cast<const uint64_t*>(src);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const size_t cap = out_capacity[chunk_idx];
    if (original > cap) {
      if (statuses)         statuses[chunk_idx]         = arctoErrorCannotDecompress;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = 0;
    } else {
      if (statuses)         statuses[chunk_idx]         = arctoSuccess;
      if (actual_out_sizes) actual_out_sizes[chunk_idx] = static_cast<size_t>(original);
    }
  }

  const size_t cap = out_capacity[chunk_idx];
  if (original > cap) return;

  const uint8_t* body = src + kHeaderBytes;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < original; i += stride) {
    dst[i] = body[i];
  }
}

__global__ void zfp_get_decomp_sizes_kernel(
    const uint8_t* const* __restrict__ in_ptrs,
    const size_t* __restrict__ in_sizes,
    size_t* __restrict__ out_sizes,
    size_t batch_size)
{
  const size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk_idx >= batch_size) return;

  const size_t in_size = in_sizes[chunk_idx];
  if (in_size < kHeaderBytes) {
    out_sizes[chunk_idx] = 0;
    return;
  }
  out_sizes[chunk_idx] = static_cast<size_t>(*reinterpret_cast<const uint64_t*>(in_ptrs[chunk_idx]));
}

dim3 grid_for(size_t max_chunk_size, size_t batch_size)
{
  const size_t bytes_per_block = static_cast<size_t>(kThreadsPerBlock);
  const size_t blocks_x =
      max_chunk_size == 0 ? 1u : (max_chunk_size + bytes_per_block - 1) / bytes_per_block;
  return dim3(static_cast<unsigned>(blocks_x),
              static_cast<unsigned>(batch_size),
              1u);
}

} // namespace

void zfpBatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* decomp_sizes_device,
    size_t max_chunk_size,
    size_t batch_size,
    void* /* temp_data */,
    size_t /* temp_bytes */,
    uint8_t* const* comp_data_device,
    size_t* comp_sizes_device,
    arctoBatchedZFPOpts_t /* opts */,
    hipStream_t stream)
{
  if (batch_size == 0) return;
  dim3 grid = grid_for(max_chunk_size, batch_size);
  zfp_passthrough_compress_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      decomp_data_device,
      decomp_sizes_device,
      comp_data_device,
      comp_sizes_device,
      batch_size);
  HipUtils::check_last_error("zfp_passthrough_compress_kernel launch");
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
  // Use a generous grid since we don't know the decompressed size up front;
  // each block strides over the chunk body, so over-launching is benign.
  dim3 grid(64u, static_cast<unsigned>(batch_size), 1u);
  zfp_passthrough_decompress_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
      device_in_ptrs,
      device_in_bytes,
      device_out_bytes,
      device_out_ptrs,
      device_actual_uncompressed_bytes,
      device_status_ptrs,
      batch_size);
  HipUtils::check_last_error("zfp_passthrough_decompress_kernel launch");
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
      (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  zfp_get_decomp_sizes_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_bytes,
      batch_size);
  HipUtils::check_last_error("zfp_get_decomp_sizes_kernel launch");
}

size_t zfpComputeMaxSize(size_t chunk_size, arctoBatchedZFPOpts_t /* opts */)
{
  // Phase 0: passthrough + 8-byte header.
  // Phase 1 (fixed-rate): num_blocks * rate * values_per_block / 8 (+ header).
  return chunk_size + kHeaderBytes;
}

size_t zfpBatchCompressComputeTempSize(
    size_t /* max_chunk_size */,
    size_t /* batch_size */,
    arctoBatchedZFPOpts_t /* opts */)
{
  return 0; // Phase 0 stub needs no scratch.
}

size_t zfpBatchDecompressComputeTempSize(
    size_t /* max_chunks_in_batch */,
    size_t /* max_uncompressed_chunk_size */)
{
  return 0;
}

} // namespace lowlevel
} // namespace arcto
