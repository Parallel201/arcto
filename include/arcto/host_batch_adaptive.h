/*
 * ARCTO adaptive host batch -- profile-driven tiled aggregation for the
 * batched compress / decompress APIs at scale.
 *
 * The single-shot arctoHostBatch (host_batch.h) coalesces an entire
 * input into one pinned host buffer and uploads it in one
 * hipMemcpyAsync. For small to medium inputs (10s of MB to ~1 GB) this
 * is the right thing to do. For large inputs (multi-GB to tens of GB),
 * the up-front cost of hipHostMalloc + the pageable->pinned memcpy
 * dominates, and the supposed PCIe-peak gain on the bulk H2D is
 * overwhelmed by the host-side prep cost. See
 * compression-experiments/paper/ARCTO-optim/results/RX7900XT_SCALING_*
 * for the empirical characterization.
 *
 * arctoHostBatchAdaptive solves this by tiling the input into windows
 * of a profile-driven size W_opt, allocating ONE pinned host buffer of
 * size W_opt that is reused across all windows, and processing
 * windows sequentially. The cost model from which W_opt is computed
 * is refined online from per-window measurements, so the choice
 * adapts to the actual host, GPU, and algorithm in use.
 *
 * Lifecycle:
 *
 *   1. arctoHostBatchAdaptiveCreate -- declare total input bytes,
 *      optional kernel-throughput hint. No allocations yet.
 *
 *   2. arctoHostBatchAdaptiveChooseWindow -- compute W_opt using the
 *      cost model (with priors on first call, refined values on
 *      subsequent calls). Allocates the pinned window buffer on first
 *      call. Returns W_opt in bytes.
 *
 *   3. Per window N from 0 to ceil(input / W_opt) - 1:
 *
 *      a. arctoHostBatchAdaptiveNextWindow -- returns a descriptor
 *         with the pinned slice the caller fills via memcpy from its
 *         pageable source.
 *
 *      b. arctoHostBatchAdaptiveUpload -- enqueue one bulk H2D of the
 *         current window's bytes to d_dst (caller advances d_dst by
 *         window_bytes between calls).
 *
 *      c. caller runs the compress kernel on the device-resident
 *         window, measures its time, calls
 *         arctoHostBatchAdaptiveRecordKernelTime to feed the cost
 *         model.
 *
 *   4. arctoHostBatchAdaptiveDestroy releases the pinned buffer.
 *
 * Cost model (single-buffered, no overlap):
 *
 *   t_total(W) = t_alloc(W)                                 (paid once)
 *              + ceil(input / W) * (t_h2h(W) + t_h2d(W) + t_kernel(W))
 *
 * Each per-window term is approximately linear in W: t_h2h ~ W/R_dram,
 * t_h2d ~ W/R_pcie, t_kernel ~ W/R_kernel. For W much smaller than
 * input, t_total is monotonically increasing in W (because t_alloc
 * grows with W), so W_opt is the SMALLEST W that satisfies three
 * practical floors:
 *
 *   W_kernel_sat   = wave_slots * chunk_size_opt    (per-arch)
 *   W_pcie_amort   = const lower bound (~64 MB) for PCIe DMA amort
 *   W_launch_amort = const lower bound (~16 MB) for kernel-launch amort
 *
 *   W_opt = max(W_kernel_sat, W_pcie_amort, W_launch_amort)
 *
 * The online adaptation refines R_dram, R_pcie, R_kernel from the
 * first window's measurements; subsequent windows reuse the same W_opt
 * unless the constraints change significantly (logged as adaptation
 * events).
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#ifndef ARCTO_HOST_BATCH_ADAPTIVE_H
#define ARCTO_HOST_BATCH_ADAPTIVE_H

#include "arcto.h"

#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arctoHostBatchAdaptive arctoHostBatchAdaptive_t;

/**
 * @brief Descriptor for one window of the adaptive batch.
 */
typedef struct {
  void*  pinned_ptr;       /**< base of pinned window buffer slice */
  size_t window_bytes;     /**< bytes in this window (may be < W_opt
                                for the last window) */
  size_t input_offset;     /**< byte offset of this window into the
                                global input */
  size_t window_index;     /**< 0-based index of this window */
} arctoAdaptiveWindow_t;

/**
 * @brief Snapshot of the cost-model state for logging / CSV.
 */
typedef struct {
  size_t  window_bytes;           /**< current W_opt */
  size_t  total_input_bytes;
  size_t  windows_consumed;
  double  R_alloc_gbps;           /**< measured or default */
  double  R_dram_gbps;            /**< host-to-host bandwidth */
  double  R_pcie_gbps;            /**< pinned H2D bandwidth */
  double  R_kernel_gbps;          /**< kernel throughput */
  int     adaptation_count;       /**< times W_opt was recomputed */
} arctoAdaptiveStats_t;

/**
 * @brief Create an adaptive tiled batch handle.
 *
 * No allocations are performed yet. The pinned window buffer is
 * allocated on the first call to arctoHostBatchAdaptiveChooseWindow.
 *
 * @param total_input_bytes   total bytes the caller intends to upload
 *                            across all windows (used by the cost
 *                            model)
 * @param hint_kernel_gbps    initial guess of kernel throughput in
 *                            GB/s for the target compressor; 0 = use
 *                            a conservative built-in default. Refined
 *                            online from per-window measurements.
 * @param out_batch           on success, receives the handle
 */
arctoStatus_t arctoHostBatchAdaptiveCreate(
    size_t total_input_bytes,
    double hint_kernel_gbps,
    arctoHostBatchAdaptive_t** out_batch);

/**
 * @brief Compute W_opt via the cost model and allocate the pinned
 * window buffer if not yet allocated. Idempotent after the first call
 * unless the cost-model state changed significantly (handled
 * internally; the buffer is currently never reallocated to avoid the
 * cold-start hipHostMalloc cost).
 *
 * @return the chosen W_opt in bytes; 0 on error
 */
size_t arctoHostBatchAdaptiveChooseWindow(arctoHostBatchAdaptive_t* batch);

/**
 * @brief Begin the next window. Returns arctoSuccess and fills out_w
 * for each window from 0 to ceil(input / W_opt) - 1; after the last
 * window, returns arctoSuccess with out_w->window_bytes == 0 (caller
 * should stop iterating).
 */
arctoStatus_t arctoHostBatchAdaptiveNextWindow(
    arctoHostBatchAdaptive_t* batch,
    arctoAdaptiveWindow_t* out_w);

/**
 * @brief Enqueue one bulk hipMemcpyAsync of window_bytes from the
 * pinned buffer to d_dst on the given stream. The caller is
 * responsible for ensuring d_dst has at least window_bytes available
 * and for advancing d_dst between successive calls if it writes
 * windows contiguously.
 */
arctoStatus_t arctoHostBatchAdaptiveUpload(
    const arctoHostBatchAdaptive_t* batch,
    void* d_dst,
    hipStream_t stream);

/**
 * @brief Record the host-to-host memcpy time for the current window.
 * Used to refine R_dram in the cost model.
 */
void arctoHostBatchAdaptiveRecordH2HTime(
    arctoHostBatchAdaptive_t* batch,
    double h2h_ms);

/**
 * @brief Record the H2D time for the current window. Used to refine
 * R_pcie in the cost model.
 */
void arctoHostBatchAdaptiveRecordH2DTime(
    arctoHostBatchAdaptive_t* batch,
    double h2d_ms);

/**
 * @brief Record the kernel time for the current window. Used to refine
 * R_kernel in the cost model.
 */
void arctoHostBatchAdaptiveRecordKernelTime(
    arctoHostBatchAdaptive_t* batch,
    double kernel_ms);

/**
 * @brief Snapshot the cost-model state.
 */
arctoStatus_t arctoHostBatchAdaptiveStats(
    const arctoHostBatchAdaptive_t* batch,
    arctoAdaptiveStats_t* out_stats);

/**
 * @brief Release the pinned window buffer (if allocated) and destroy
 * the handle. NULL is a no-op.
 */
void arctoHostBatchAdaptiveDestroy(arctoHostBatchAdaptive_t* batch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARCTO_HOST_BATCH_ADAPTIVE_H */
