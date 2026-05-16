/*
 * ARCTO ZFP — C API wrappers around the low-level ZFP batch kernels.
 *
 * Phase 0: thin pass-through that forwards to the stub kernels.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"

#include "Check.h"
#include "HipUtils.h"
#include "ZFPCompressionKernels.h"
#include "common.h"

using namespace arcto;

arctoStatus_t arctoBatchedZFPCompressGetTempSize(
    size_t batch_size,
    size_t max_uncompressed_chunk_bytes,
    arctoBatchedZFPOpts_t format_opts,
    size_t* temp_bytes)
{
  try {
    CHECK_NOT_NULL(temp_bytes);
    *temp_bytes = lowlevel::zfpBatchCompressComputeTempSize(
        max_uncompressed_chunk_bytes, batch_size, format_opts);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "arctoBatchedZFPCompressGetTempSize()");
  }
  return arctoSuccess;
}

arctoStatus_t arctoBatchedZFPCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    arctoBatchedZFPOpts_t format_opts,
    size_t* max_compressed_bytes)
{
  try {
    CHECK_NOT_NULL(max_compressed_bytes);
    *max_compressed_bytes =
        lowlevel::zfpComputeMaxSize(max_uncompressed_chunk_bytes, format_opts);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "arctoBatchedZFPCompressGetMaxOutputChunkSize()");
  }
  return arctoSuccess;
}

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
    hipStream_t stream)
{
  try {
    CHECK_NOT_NULL(device_uncompressed_ptrs);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);

    lowlevel::zfpBatchCompress(
        reinterpret_cast<const uint8_t* const*>(device_uncompressed_ptrs),
        device_uncompressed_bytes,
        max_uncompressed_chunk_bytes,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        reinterpret_cast<uint8_t* const*>(device_compressed_ptrs),
        device_compressed_bytes,
        format_opts,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "arctoBatchedZFPCompressAsync()");
  }
  return arctoSuccess;
}

arctoStatus_t arctoBatchedZFPDecompressGetTempSize(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    size_t* temp_bytes)
{
  try {
    CHECK_NOT_NULL(temp_bytes);
    *temp_bytes = lowlevel::zfpBatchDecompressComputeTempSize(
        num_chunks, max_uncompressed_chunk_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "arctoBatchedZFPDecompressGetTempSize()");
  }
  return arctoSuccess;
}

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
    hipStream_t stream)
{
  try {
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_ptrs);

    lowlevel::zfpBatchDecompress(
        reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs),
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        reinterpret_cast<uint8_t* const*>(device_uncompressed_ptrs),
        device_actual_uncompressed_bytes,
        device_statuses,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "arctoBatchedZFPDecompressAsync()");
  }
  return arctoSuccess;
}

arctoStatus_t arctoBatchedZFPGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  try {
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);

    lowlevel::zfpBatchGetDecompressSizes(
        reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs),
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "arctoBatchedZFPGetDecompressSizeAsync()");
  }
  return arctoSuccess;
}
