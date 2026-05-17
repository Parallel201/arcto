/*
 * ARCTO host batch -- coalesced pinned-host staging for batched compress
 * / decompress APIs.
 *
 * Many ARCTO callers upload a batch of small chunks (seismic checkpoint
 * blocks, file segments, RAM tiles) and then invoke a batched compress
 * routine. The naive pattern -- one hipMemcpyAsync per chunk from a
 * scattered std::vector<std::vector<char>> -- caps the H2D bandwidth at
 * ~6 GB/s on PCIe Gen4 x16 because per-call overhead dominates the
 * actual transfer time. Two fixes are needed and they are multiplicative:
 *
 *   (1) coalesce the chunks into a single contiguous host buffer
 *   (2) allocate that buffer with hipHostMalloc so the H2D runs at
 *       PCIe peak (~27 GB/s on RX 7900 XT)
 *
 * arctoHostBatch_t packages both. The caller provides per-chunk byte
 * sizes once, gets back per-chunk host pointers that are slices into a
 * single pinned allocation, copies its data into those slices, and
 * finally uploads the whole batch to GPU in ONE hipMemcpyAsync.
 *
 * Measured speedup vs the scattered-pageable baseline:
 *   ~4.3x on H2D throughput
 *   (RX 7900 XT, see compression-experiments@458d5ad:
 *    paper/ARCTO-optim/results/RX7900XT_PCIe_20260517_195247/FINDINGS.md)
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#ifndef ARCTO_HOST_BATCH_H
#define ARCTO_HOST_BATCH_H

#include "arcto.h"

#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arctoHostBatch arctoHostBatch_t;

/**
 * @brief Create a coalesced pinned-host batch sized for the given chunks.
 *
 * Allocates one contiguous pinned host buffer of `sum(chunk_sizes)` bytes
 * via hipHostMalloc and computes per-chunk pointers as slices into it.
 * The buffer contents are not initialized; the caller is expected to
 * memcpy per-chunk data into the returned per-chunk pointers (host-to-
 * host, very fast) and then upload the whole batch with
 * arctoHostBatchUploadAsync().
 *
 * @param batch_size   Number of chunks. May be 0 (creates an empty batch).
 * @param chunk_sizes  Per-chunk byte counts. Must point to `batch_size`
 *                     size_t entries; may be NULL only if batch_size == 0.
 * @param out_batch    On success, receives the new batch handle.
 *
 * @return arctoSuccess on success; arctoErrorInternal if any HIP call
 *         fails (the partially-constructed batch is freed before return).
 */
arctoStatus_t arctoHostBatchCreate(
    size_t batch_size,
    const size_t* chunk_sizes,
    arctoHostBatch_t** out_batch);

/**
 * @brief Destroy a batch and release its pinned host storage. Passing
 * NULL is a no-op.
 */
void arctoHostBatchDestroy(arctoHostBatch_t* batch);

/**
 * @brief Get the host (pinned) pointer for chunk `i`. Use to memcpy data
 * into the batch. Returns NULL if `batch` is NULL or `i` is out of range.
 */
void* arctoHostBatchChunkPtr(arctoHostBatch_t* batch, size_t i);

/**
 * @brief Get the size of chunk `i`. Returns 0 if `batch` is NULL or `i`
 * is out of range.
 */
size_t arctoHostBatchChunkSize(const arctoHostBatch_t* batch, size_t i);

/**
 * @brief Get the number of chunks in the batch.
 */
size_t arctoHostBatchSize(const arctoHostBatch_t* batch);

/**
 * @brief Get the total byte count of all chunks (== size of the underlying
 * pinned allocation).
 */
size_t arctoHostBatchTotalSize(const arctoHostBatch_t* batch);

/**
 * @brief Upload the entire pinned host batch to a device buffer in ONE
 * hipMemcpyAsync. Equivalent in semantics to issuing `batch_size`
 * individual hipMemcpyAsync calls (one per chunk) but at PCIe peak instead
 * of launch-overhead-bound.
 *
 * Optionally also writes per-chunk DEVICE pointers (slices into `d_dst`)
 * into a host array, suitable for staging to a batched compress API's
 * `void**` argument via a subsequent hipMemcpyAsync.
 *
 * @param batch                The host batch.
 * @param d_dst                Device destination buffer. Must be at least
 *                             arctoHostBatchTotalSize(batch) bytes.
 * @param h_out_chunk_dev_ptrs (optional) On return, writes per-chunk
 *                             device pointers in `d_dst`. Must be either
 *                             NULL (skip) or point to
 *                             arctoHostBatchSize(batch) void* entries on
 *                             HOST memory.
 * @param stream               Stream to enqueue the transfer on. The
 *                             function returns as soon as the async copy
 *                             is enqueued; the caller is responsible for
 *                             synchronization.
 *
 * @return arctoSuccess on success.
 */
arctoStatus_t arctoHostBatchUploadAsync(
    const arctoHostBatch_t* batch,
    void* d_dst,
    void** h_out_chunk_dev_ptrs,
    hipStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARCTO_HOST_BATCH_H */
