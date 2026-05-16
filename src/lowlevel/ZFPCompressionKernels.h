/*
 * ARCTO ZFP — low-level kernel declarations.
 *
 * Phase 0: stub passthrough kernels. Real ZFP transform/encode coming in
 * Phase 1 (fixed-rate 3D).
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#pragma once

#include "../common.h"
#include "arcto/zfp.h"

namespace arcto
{
namespace lowlevel
{

/**
 * @brief Compress a batch of memory locations using ZFP.
 *
 * Phase 0: each output chunk is laid out as
 *   [8-byte little-endian original_size][original bytes verbatim].
 *
 * @param decomp_data_device   Device pointers to uncompressed chunks.
 * @param decomp_sizes_device  Device array of per-chunk uncompressed sizes.
 * @param max_chunk_size       Upper bound on any single chunk size in the batch.
 * @param batch_size           Number of chunks in the batch.
 * @param temp_data            Temporary workspace (unused in Phase 0).
 * @param temp_bytes           Size of temp workspace.
 * @param comp_data_device     Device pointers to where each compressed chunk
 *                             will be written.
 * @param comp_sizes_device    Device array where per-chunk compressed sizes
 *                             will be written.
 * @param opts                 ZFP format options (mode, dim, shape, param).
 * @param stream               HIP stream to launch on.
 */
void zfpBatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* decomp_sizes_device,
    size_t max_chunk_size,
    size_t batch_size,
    void* temp_data,
    size_t temp_bytes,
    uint8_t* const* comp_data_device,
    size_t* comp_sizes_device,
    arctoBatchedZFPOpts_t opts,
    hipStream_t stream);

/**
 * @brief Decompress a batch of ZFP-compressed chunks.
 *
 * Phase 0: reads the 8-byte original size header and copies the trailing
 * bytes back out verbatim.
 */
void zfpBatchDecompress(
    const uint8_t* const* device_in_ptrs,
    const size_t* device_in_bytes,
    const size_t* device_out_bytes,
    size_t batch_size,
    void* temp_ptr,
    size_t temp_bytes,
    uint8_t* const* device_out_ptrs,
    size_t* device_actual_uncompressed_bytes,
    arctoStatus_t* device_status_ptrs,
    hipStream_t stream);

/**
 * @brief Recover per-chunk decompressed sizes from the compressed buffers.
 *
 * Phase 0: reads the leading 8-byte header from each compressed chunk.
 */
void zfpBatchGetDecompressSizes(
    const uint8_t* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream);

/**
 * @brief Maximum output size for any chunk of size <= chunk_size.
 *
 * Phase 0: chunk_size + sizeof(header).
 * Phase 1 (fixed-rate): ceil(num_blocks * rate * values_per_block / 8) + header.
 */
size_t zfpComputeMaxSize(size_t chunk_size, arctoBatchedZFPOpts_t opts);

/**
 * @brief Compute temporary workspace size for a batched compression call.
 */
size_t zfpBatchCompressComputeTempSize(
    size_t max_chunk_size, size_t batch_size, arctoBatchedZFPOpts_t opts);

/**
 * @brief Compute temporary workspace size for a batched decompression call.
 */
size_t zfpBatchDecompressComputeTempSize(
    size_t max_chunks_in_batch, size_t max_uncompressed_chunk_size);

} // namespace lowlevel
} // namespace arcto
