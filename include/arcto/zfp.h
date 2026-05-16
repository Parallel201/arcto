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
 * - REVERSIBLE: lossless for arbitrary byte streams via XOR-delta + adaptive
 *   byte-aligned bit-width. Use this for general data (the batched API
 *   defaults to it; the existing bit-exact test suite exercises it).
 * - REVERSIBLE_3D: lossless for float32 cubes via block-floating-point
 *   alignment + ZFP reversible integer lift transform + global-K bit-plane
 *   packing. Only available through the single-cube 3D API. Strictly
 *   lossless for float32 inputs to within one ULP of the original IEEE 754
 *   representation; the kernel reduces a global K across the cube so block
 *   sizes stay deterministic (per-block adaptive K is a future optimization).
 */
typedef enum
{
  ARCTO_ZFP_MODE_FIXED_RATE      = 0,
  ARCTO_ZFP_MODE_FIXED_PRECISION = 1,
  ARCTO_ZFP_MODE_FIXED_ACCURACY  = 2,
  ARCTO_ZFP_MODE_REVERSIBLE      = 3,
  ARCTO_ZFP_MODE_REVERSIBLE_3D   = 4
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

/*
 * Phase 1a ships REVERSIBLE only. FIXED_RATE / FIXED_PRECISION /
 * FIXED_ACCURACY return arctoErrorNotSupported until their kernels land
 * in subsequent phases. Default mode is therefore REVERSIBLE so the
 * existing bit-exact bench/test infrastructure continues to validate
 * round-trip correctness.
 */
static const arctoBatchedZFPOpts_t arctoBatchedZFPDefaultOpts = {
    ARCTO_ZFP_MODE_REVERSIBLE,
    ARCTO_ZFP_DIM_3D,
    ARCTO_TYPE_INT,
    0.0, /* unused in REVERSIBLE */
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

/******************************************************************************
 * Single-cube 3D interface (Phase 1c)
 *
 * A high-level API that mirrors the shape of hipcompZfpCompressAsync: one
 * GPU buffer in, one GPU buffer out, the field's 3D shape passed in opts.
 * Internally the kernel tiles the cube into real 4x4x4 sub-cubes (not the
 * linear 64-float groups used by the batched API), so spatial correlation
 * along y and z is exploited the way ZFP intends.
 *
 * opts must have:
 *   - mode = ARCTO_ZFP_MODE_FIXED_RATE   (other modes return NotSupported)
 *   - dim  = ARCTO_ZFP_DIM_3D
 *   - shape[0..2] = (nx, ny, nz); shape[3] is ignored (use the batched API
 *                   or call this once per timestep for 4D)
 *   - param = rate (bits per value, clamped to [1, 32])
 *****************************************************************************/

/**
 * @brief Worst-case compressed-output size for a single 3D cube.
 */
arctoStatus_t arctoZFP3DCompressGetMaxOutputSize(
    arctoBatchedZFPOpts_t opts,
    size_t* max_compressed_bytes);

/**
 * @brief Compress one nx*ny*nz float32 cube into a contiguous output buffer.
 *
 * The compressed payload is fully self-describing (header carries the shape
 * and rate), so the caller does not need to remember opts at decompress
 * time.
 *
 * device_output_size is a GPU-side size_t* that receives the actual number
 * of bytes written, suitable for D2H async copy.
 */
arctoStatus_t arctoZFP3DCompressAsync(
    const void* device_input,
    arctoBatchedZFPOpts_t opts,
    void* device_output,
    size_t device_output_capacity,
    size_t* device_output_size,
    hipStream_t stream);

/**
 * @brief Decompress a 3D cube previously written by arctoZFP3DCompressAsync.
 *
 * Shape, mode and rate are recovered from the compressed header; the
 * decompressed buffer must be at least nx*ny*nz*sizeof(float) bytes.
 */
arctoStatus_t arctoZFP3DDecompressAsync(
    const void* device_input,
    size_t input_size,
    void* device_output,
    size_t device_output_capacity,
    size_t* device_actual_output_size,
    arctoStatus_t* device_status,
    hipStream_t stream);

/**
 * @brief Recover the decompressed cube size (in bytes) from a compressed
 * buffer asynchronously. Reads only the chunk header.
 */
arctoStatus_t arctoZFP3DGetDecompressSizeAsync(
    const void* device_input,
    size_t input_size,
    size_t* device_uncompressed_bytes,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
