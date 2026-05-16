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

// ---------------------------------------------------------------------------
// Phase 1c — single-cube 3D entry points (high-level API).
//
// One contiguous nx*ny*nz float32 cube in, one contiguous compressed buffer
// out. Output is fully self-describing: header carries mode, rate, and shape.
// ---------------------------------------------------------------------------

/**
 * @brief Worst-case output size in bytes for one 3D cube at the given rate.
 *        opts.shape[0..2] must hold nx, ny, nz.
 */
size_t zfp3DComputeMaxOutputSize(arctoBatchedZFPOpts_t opts);

/**
 * @brief Launch a single-cube 3D compress on the given stream.
 *        Caller must pass FIXED_RATE in opts.mode and the cube shape in
 *        opts.shape[0..2]; param holds the bit rate.
 */
void zfp3DCompress(
    const uint8_t* device_input,
    arctoBatchedZFPOpts_t opts,
    uint8_t* device_output,
    size_t device_output_capacity,
    size_t* device_output_size,
    hipStream_t stream);

/**
 * @brief Launch a single-cube 3D decompress. Shape, mode and rate are read
 *        from the header in device_input.
 */
void zfp3DDecompress(
    const uint8_t* device_input,
    size_t input_size,
    uint8_t* device_output,
    size_t device_output_capacity,
    size_t* device_actual_output_size,
    arctoStatus_t* device_status,
    hipStream_t stream);

/**
 * @brief Recover the uncompressed byte count from a 3D chunk header.
 */
void zfp3DGetDecompressSize(
    const uint8_t* device_input,
    size_t input_size,
    size_t* device_uncompressed_bytes,
    hipStream_t stream);

} // namespace lowlevel
} // namespace arcto
