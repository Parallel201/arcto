/*
 * ARCTO ZFP-Reversible3D -- GPU-native lossless compression of float32 3D
 * cubes using the ZFP reversible integer-lift transform.
 *
 * This is ARCTO's own addition on top of the canonical ZFP modes delegated
 * to third_party/zfp. The canonical's HIP backend implements FIXED_RATE /
 * FIXED_PRECISION / FIXED_ACCURACY on GPU but has no REVERSIBLE path --
 * its zfp_internal_hip_compress/_decompress switch on
 * zfp_mode_reversible falls through to "compression mode not supported on
 * GPU". This module fills that gap with kernels that share canonical's
 * mathematical primitives (block-floating-point cast, ZFP reversible
 * integer lift, negabinary encoding) but use a hybrid bit-plane / flat
 * per-block coder tailored to seismic wave-propagation checkpoints.
 *
 * Naming: the "ZFP" prefix is deliberate -- the algorithm IS the ZFP
 * reversible transform pipeline, just with our own embedded coder and an
 * ARCTO-specific bitstream format. It is NOT a generic reversible 3D
 * codec; the choice of BFP bias, lift coefficients, and 4x4x4 block
 * geometry are all inherited from ZFP and must match anyone reading our
 * coefficient layer.
 *
 * The module is fully self-contained in src/zfp_reversible_3d/ -- it
 * does NOT link any extra symbol from third_party/zfp/ and it does NOT
 * modify the vendored canonical source. The shared algorithmic choices
 * are re-implemented from scratch and tested for bit-exact round-trip;
 * cross-tool comparability is preserved at the coefficient layer, not
 * at the bitstream layer.
 *
 * Memory protocol: input AND output are HIP DEVICE pointers. Compressed
 * payloads are typically passed to a stream-ordered D2H copy by the
 * caller; we do not consume host buffers here.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#ifndef ARCTO_ZFP_REVERSIBLE_3D_H
#define ARCTO_ZFP_REVERSIBLE_3D_H

#include "arcto.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Element type. Only float32 is supported in this iteration --
 * the lift transform runs on int64 (REV lift) which preserves float32
 * mantissa losslessly with the ebias=30 BFP scaling.
 */
typedef enum
{
  ARCTO_ZFP_REV3D_TYPE_FLOAT = 0
} arctoZFPReversible3DType_t;

/**
 * @brief Single-cube compression specification.
 */
typedef struct
{
  arctoZFPReversible3DType_t type;
  uint32_t nx;    /**< x extent (inner) */
  uint32_t ny;    /**< y extent */
  uint32_t nz;    /**< z extent (outer) */
} arctoZFPReversible3DOpts_t;

static const arctoZFPReversible3DOpts_t arctoZFPReversible3DDefaultOpts = {
    ARCTO_ZFP_REV3D_TYPE_FLOAT, 1u, 1u, 1u
};

/**
 * @brief Worst-case compressed-output size, in bytes.
 *
 * The hybrid coder's worst case per block is the bit-plane path with K=64
 * and every group active on every plane: 16 (group test) + 4*16 (coeff
 * bits) per plane * 64 planes = 5120 bits = 640 bytes. Metadata per block
 * is 4 (offset) + 1 (K + scheme) + 1 (emax) = 6 bytes. A 32-byte header
 * sits in front.
 */
arctoStatus_t arctoZFPReversible3DCompressGetMaxOutputSize(
    arctoZFPReversible3DOpts_t opts, size_t* max_bytes);

/**
 * @brief Compress one float32 cube into a HIP device buffer.
 *
 * @param d_input             DEVICE pointer to nx*ny*nz contiguous float32
 *                            values in (z, y, x) C order (x innermost).
 * @param opts                Type + shape.
 * @param d_output            DEVICE pointer for compressed output.
 * @param d_output_capacity   Allocation size of d_output; should be at least
 *                            arctoZFPReversible3DCompressGetMaxOutputSize().
 * @param d_output_size       DEVICE pointer (size_t) that receives the
 *                            actual compressed byte count. The kernel
 *                            writes it stream-ordered so the caller can
 *                            chain a D2H copy without an explicit sync.
 * @param stream              HIP stream.
 */
arctoStatus_t arctoZFPReversible3DCompressAsync(
    const void* d_input,
    arctoZFPReversible3DOpts_t opts,
    void* d_output, size_t d_output_capacity,
    size_t* d_output_size,
    hipStream_t stream);

/**
 * @brief Decompress a previously-written cube.
 */
arctoStatus_t arctoZFPReversible3DDecompressAsync(
    const void* d_input, size_t d_input_size,
    void* d_output, size_t d_output_capacity,
    size_t* d_actual_size,
    arctoStatus_t* d_status,
    hipStream_t stream);

/**
 * @brief Recover the uncompressed cube size from the header alone.
 */
arctoStatus_t arctoZFPReversible3DGetDecompressSizeAsync(
    const void* d_input, size_t d_input_size,
    size_t* d_uncomp_bytes,
    hipStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARCTO_ZFP_REVERSIBLE_3D_H */
