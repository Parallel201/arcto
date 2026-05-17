/*
 * ARCTO ZFP — thin C wrapper around the canonical LLNL/zfp HIP backend.
 *
 * The implementation in src/lowlevel/ZFPBatch.cpp delegates entirely to the
 * vendored zfp:: namespace (third_party/zfp). Output bitstreams are
 * byte-for-byte compatible with the upstream `zfp` CLI: a file written by
 * arctoZFPCompress() can be read by `zfp -h -z file` (and vice versa).
 *
 * Memory protocol (mirrors the canonical CLI):
 *   - Uncompressed field: HIP DEVICE pointer (from hipMalloc). The
 *     canonical's HIP backend autodetects it via is_gpu_ptr() and runs
 *     the kernel zero-copy.
 *   - Compressed stream:  HOST pointer. The canonical bitstream layer
 *     (stream_open / zfp_read_header / zfp_write_header) is CPU-only;
 *     storing the stream on the host avoids implicit host-staging by
 *     the HIP backend and matches how RTM checkpoints are written to /
 *     read from disk in practice.
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
 * @brief Compression mode mirroring canonical ZFP's modes.
 *
 * - FIXED_RATE:      param is bits/value; output size is deterministic.
 * - FIXED_PRECISION: param is target precision (bit-planes kept per block).
 * - FIXED_ACCURACY:  param is absolute error tolerance (power of two).
 * - REVERSIBLE:      lossless; param is ignored.
 *
 * Phase 3 limitation: only FIXED_RATE is currently supported on the HIP
 * path. The variable-rate modes produce blocks of differing sizes and the
 * canonical HIP decompressor requires a separately-serialized block-offset
 * index that this wrapper does not yet emit. Phase 4 will close that gap
 * by embedding the index in the output stream.
 */
typedef enum
{
  ARCTO_ZFP_MODE_FIXED_RATE      = 0,
  ARCTO_ZFP_MODE_FIXED_PRECISION = 1,
  ARCTO_ZFP_MODE_FIXED_ACCURACY  = 2,
  ARCTO_ZFP_MODE_REVERSIBLE      = 3
} arctoZFPMode_t;

/**
 * @brief Scalar element type of the input field.
 *
 * The canonical's HIP backend currently supports float / double / int32 /
 * int64; arctoZFP* forwards the choice unchanged.
 */
typedef enum
{
  ARCTO_ZFP_TYPE_FLOAT  = 0,
  ARCTO_ZFP_TYPE_DOUBLE = 1,
  ARCTO_ZFP_TYPE_INT32  = 2,
  ARCTO_ZFP_TYPE_INT64  = 3
} arctoZFPType_t;

/**
 * @brief Full compression specification.
 *
 * shape[i] for i >= ndims is ignored. ndims must be in [1, 4].
 */
typedef struct
{
  arctoZFPMode_t mode;
  arctoZFPType_t type;
  double         param;       /**< rate / precision / tolerance; ignored for REVERSIBLE */
  uint32_t       ndims;       /**< 1, 2, 3, or 4 */
  uint32_t       shape[4];    /**< unused trailing entries should be set to 1 */
} arctoZFPOpts_t;

static const arctoZFPOpts_t arctoZFPDefaultOpts = {
    ARCTO_ZFP_MODE_FIXED_RATE,
    ARCTO_ZFP_TYPE_FLOAT,
    16.0,
    3,
    {1u, 1u, 1u, 1u}
};

/**
 * @brief Conservative upper bound for the compressed size, including the
 * canonical bitstream header. Use to size the device output buffer for
 * arctoZFPCompress().
 */
arctoStatus_t arctoZFPCompressGetMaxOutputSize(
    arctoZFPOpts_t opts, size_t* max_bytes);

/**
 * @brief Compress a HIP device field into a host-resident canonical-format
 * zfp bitstream (full header included).
 *
 * @param d_input             DEVICE pointer to the source field.
 * @param opts                Mode + type + shape + parameter.
 * @param h_output            HOST pointer for compressed output.
 * @param h_output_capacity   Allocation size of h_output in bytes; should be
 *                            at least arctoZFPCompressGetMaxOutputSize().
 * @param actual_size_out     Receives the number of bytes actually written.
 *
 * @note Synchronous: the canonical HIP backend uses its own internal stream
 *       and calls hipDeviceSynchronize before returning. There is no
 *       per-call HIP stream parameter.
 *
 * @return arctoSuccess on success, an error code on failure.
 */
arctoStatus_t arctoZFPCompress(
    const void* d_input,
    arctoZFPOpts_t opts,
    void* h_output, size_t h_output_capacity,
    size_t* actual_size_out);

/**
 * @brief Decompress a host-resident canonical zfp bitstream into a HIP
 * device buffer.
 *
 * @param h_input              HOST pointer to the compressed stream.
 * @param h_input_size         Size of the compressed stream in bytes.
 * @param d_output             DEVICE pointer for the decompressed field.
 * @param d_output_capacity    Allocation size of d_output in bytes.
 * @param actual_size_out      Receives the number of bytes written to
 *                             d_output (the uncompressed field size).
 *
 * @return arctoSuccess on success, an error code on failure.
 */
arctoStatus_t arctoZFPDecompress(
    const void* h_input, size_t h_input_size,
    void* d_output, size_t d_output_capacity,
    size_t* actual_size_out);

/**
 * @brief Read the canonical header to learn the uncompressed size, without
 * actually decompressing. Useful for buffer pre-allocation.
 *
 * @param h_input    HOST pointer to the compressed stream (header bytes
 *                   suffice; the payload is not read).
 */
arctoStatus_t arctoZFPGetDecompressSize(
    const void* h_input, size_t h_input_size,
    size_t* uncomp_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARCTO_ZFP_H */
