/*
 * ARCTO ZFP — HIP-native floating-point compression.
 *
 * Implements the ZFP block-based fixed-rate/precision/accuracy/reversible
 * compression scheme as a GPU kernel for AMD via HIP, written from scratch
 * (no dependency on the reference ZFP library).
 *
 * Phase 0: scaffolding + passthrough stub kernel (no actual compression yet).
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#ifndef ARCTO_ZFP_H
#define ARCTO_ZFP_H

#include "arcto.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ZFP compression mode.
 *
 * - FIXED_RATE: each block compresses to exactly N bits/value (param = rate).
 *   Predictable output size, primary mode supported on GPU.
 * - FIXED_PRECISION: each block keeps the top N bit-planes (param = precision).
 * - FIXED_ACCURACY: absolute error bounded by 2^param (param = log2 tolerance).
 * - REVERSIBLE: lossless mode, encodes all bit-planes (param ignored).
 */
typedef enum
{
  ARCTO_ZFP_MODE_FIXED_RATE      = 0,
  ARCTO_ZFP_MODE_FIXED_PRECISION = 1,
  ARCTO_ZFP_MODE_FIXED_ACCURACY  = 2,
  ARCTO_ZFP_MODE_REVERSIBLE      = 3
} arctoZFPMode_t;

/**
 * @brief Logical dimensionality of the input field.
 *
 * ZFP partitions the field into 4^d blocks of values, where d is the
 * dimensionality. 4D is handled as a stack of 3D blocks across n4
 * (typically time), reusing the 3D kernel per timestep.
 */
typedef enum
{
  ARCTO_ZFP_DIM_1D = 1,
  ARCTO_ZFP_DIM_2D = 2,
  ARCTO_ZFP_DIM_3D = 3,
  ARCTO_ZFP_DIM_4D = 4
} arctoZFPDim_t;

/**
 * @brief ZFP compression options for the low-level API.
 *
 * shape[i] gives the extent along dimension i+1 (n1, n2, n3, n4 in RSF
 * convention). Unused dimensions must be set to 1.
 *
 * For ARCTO_ZFP_MODE_FIXED_RATE, param is the rate in bits per value
 * (typical range 1.0..64.0 for float32).
 */
typedef struct
{
  arctoZFPMode_t mode;
  arctoZFPDim_t  dim;
  arctoType_t    data_type; /* float32 supported in Phase 0+ */
  double         param;
  uint32_t       shape[4];
} arctoBatchedZFPOpts_t;

static const arctoBatchedZFPOpts_t arctoBatchedZFPDefaultOpts = {
    ARCTO_ZFP_MODE_FIXED_RATE,
    ARCTO_ZFP_DIM_3D,
    ARCTO_TYPE_INT, /* float32 reinterpreted; ZFP uses bit-pattern access */
    16.0,           /* default rate: 16 bits/value (2x compression) */
    {0u, 0u, 0u, 1u}};

/******************************************************************************
 * Batched compression/decompression interface
 *****************************************************************************/

/**
 * @brief Get temporary GPU workspace required for compression.
 */
arctoStatus_t arctoBatchedZFPCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    arctoBatchedZFPOpts_t format_opts,
    size_t* temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch.
 *
 * For passthrough (Phase 0) this is max_chunk_size + a small header.
 * For real ZFP fixed-rate this will be ceil(num_blocks * rate * values_per_block / 8) + header.
 */
arctoStatus_t arctoBatchedZFPCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    arctoBatchedZFPOpts_t format_opts,
    size_t* max_compressed_bytes);

/**
 * @brief Perform compression asynchronously. All pointers must be GPU accessible.
 */
arctoStatus_t arctoBatchedZFPCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    arctoBatchedZFPOpts_t format_opts,
    hipStream_t stream);

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 */
arctoStatus_t arctoBatchedZFPDecompressGetTempSize(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    size_t* temp_bytes);

/**
 * @brief Perform decompression asynchronously. All pointers must be GPU accessible.
 */
arctoStatus_t arctoBatchedZFPDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    arctoStatus_t* device_statuses,
    hipStream_t stream);

/**
 * @brief Compute the decompressed sizes of each chunk asynchronously.
 */
arctoStatus_t arctoBatchedZFPGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
