/*
 * ARCTO host batch -- implementation. See include/arcto/host_batch.h
 * for the design rationale.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/host_batch.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <new>
#include <vector>

struct arctoHostBatch
{
  void*               base = nullptr;     /* pinned host allocation       */
  size_t              total_bytes = 0;
  std::vector<size_t> offsets;            /* size = batch_size + 1        */
};

arctoStatus_t arctoHostBatchCreate(
    size_t batch_size,
    const size_t* chunk_sizes,
    arctoHostBatch_t** out_batch)
{
  if (out_batch == nullptr) return arctoErrorInvalidValue;
  if (batch_size > 0 && chunk_sizes == nullptr) return arctoErrorInvalidValue;

  arctoHostBatch_t* b = nullptr;
  try {
    b = new arctoHostBatch_t();
    b->offsets.assign(batch_size + 1, 0);
    for (size_t i = 0; i < batch_size; ++i) {
      b->offsets[i + 1] = b->offsets[i] + chunk_sizes[i];
    }
    b->total_bytes = b->offsets[batch_size];
  } catch (const std::bad_alloc&) {
    delete b;
    return arctoErrorInternal;
  }

  if (b->total_bytes > 0) {
    hipError_t e = hipHostMalloc(&b->base, b->total_bytes, hipHostMallocDefault);
    if (e != hipSuccess) {
      delete b;
      return arctoErrorInternal;
    }
  }

  *out_batch = b;
  return arctoSuccess;
}

void arctoHostBatchDestroy(arctoHostBatch_t* batch)
{
  if (batch == nullptr) return;
  if (batch->base != nullptr) {
    (void) hipHostFree(batch->base);
  }
  delete batch;
}

void* arctoHostBatchChunkPtr(arctoHostBatch_t* batch, size_t i)
{
  if (batch == nullptr) return nullptr;
  if (i + 1 >= batch->offsets.size()) return nullptr;
  return static_cast<unsigned char*>(batch->base) + batch->offsets[i];
}

size_t arctoHostBatchChunkSize(const arctoHostBatch_t* batch, size_t i)
{
  if (batch == nullptr) return 0;
  if (i + 1 >= batch->offsets.size()) return 0;
  return batch->offsets[i + 1] - batch->offsets[i];
}

size_t arctoHostBatchSize(const arctoHostBatch_t* batch)
{
  if (batch == nullptr || batch->offsets.empty()) return 0;
  return batch->offsets.size() - 1;
}

size_t arctoHostBatchTotalSize(const arctoHostBatch_t* batch)
{
  return (batch == nullptr) ? 0 : batch->total_bytes;
}

arctoStatus_t arctoHostBatchUploadAsync(
    const arctoHostBatch_t* batch,
    void* d_dst,
    void** h_out_chunk_dev_ptrs,
    hipStream_t stream)
{
  if (batch == nullptr) return arctoErrorInvalidValue;
  if (batch->total_bytes > 0 && d_dst == nullptr) return arctoErrorInvalidValue;

  if (batch->total_bytes > 0) {
    hipError_t e = hipMemcpyAsync(
        d_dst, batch->base, batch->total_bytes,
        hipMemcpyHostToDevice, stream);
    if (e != hipSuccess) return arctoErrorInternal;
  }

  if (h_out_chunk_dev_ptrs != nullptr) {
    const size_t n = arctoHostBatchSize(batch);
    unsigned char* dev_base = static_cast<unsigned char*>(d_dst);
    for (size_t i = 0; i < n; ++i) {
      h_out_chunk_dev_ptrs[i] = dev_base + batch->offsets[i];
    }
  }

  return arctoSuccess;
}
