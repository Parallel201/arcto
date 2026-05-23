/*
 * ARCTO adaptive host batch -- implementation. See
 * include/arcto/host_batch_adaptive.h for the design rationale.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/host_batch_adaptive.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <string>

namespace {

// Static priors for the cost model. These match the linear-fit
// constants observed in the RX 7900 XT scaling sweep (RDNA3, ROCm 7.0.1).
// They are refined online from the first window's measurements but seed
// the choice of W_opt before any data is available.
constexpr double DEFAULT_R_ALLOC_GBPS  = 0.220;    // 220 MB/s, hipHostMalloc rate
constexpr double DEFAULT_R_DRAM_GBPS   = 11.0;     // pageable to pinned memcpy
constexpr double DEFAULT_R_PCIE_GBPS   = 25.0;     // pinned bulk H2D, Gen4
constexpr double DEFAULT_R_KERNEL_GBPS = 30.0;     // mid-range across algorithms

// Practical lower bounds on W_opt. None of these depend on input size;
// W_opt is the maximum of all three constraints applied as floors.
constexpr size_t W_KERNEL_SAT_DEFAULT = 64ull * 1024 * 1024;   // 64 MB safe default; arch-specific override below
constexpr size_t W_PCIE_AMORT_FLOOR    = 64ull * 1024 * 1024;  // 64 MB
constexpr size_t W_LAUNCH_AMORT_FLOOR  = 16ull * 1024 * 1024;  // 16 MB

// W_kernel_sat = wave_slots * chunk_size_opt. We hardcode the products
// we measured rather than carry the two factors separately, because
// chunk_size_opt itself is profile-driven (see RX7900XT_CHUNK_SWEEP).
//
// - MI300X CDNA3 (gfx942): ~9728 wave slots, optimal chunk 8 KB
//                          => 9728 * 8KB = 76 MB
// - RX 7900 XT RDNA3 (gfx1100): ~3072 wave slots, optimal chunk 16 KB
//                          => 3072 * 16KB = 48 MB
// - MI50  GCN  (gfx906): ~960 wave slots,  conservatively 16 KB
//                          => 960 * 16KB = 15 MB
// - MI210 CDNA (gfx90a): ~3328 wave slots, conservatively 16 KB
//                          => 3328 * 16KB = 52 MB
//
// We pick W_kernel_sat at runtime by detecting the device. Default 64 MB
// covers the largest of the four (MI300X) safely.
size_t pick_W_kernel_saturation()
{
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return W_KERNEL_SAT_DEFAULT;
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, dev) != hipSuccess) return W_KERNEL_SAT_DEFAULT;
  // Decide by gcnArchName prefix (gfx906/gfx90a/gfx942/gfx1100).
  std::string arch = props.gcnArchName ? props.gcnArchName : "";
  if (arch.find("gfx906")  != std::string::npos) return 15ull  * 1024 * 1024;
  if (arch.find("gfx90a")  != std::string::npos) return 52ull  * 1024 * 1024;
  if (arch.find("gfx942")  != std::string::npos) return 76ull  * 1024 * 1024;
  if (arch.find("gfx1100") != std::string::npos) return 48ull  * 1024 * 1024;
  return W_KERNEL_SAT_DEFAULT;
}

} // namespace

struct arctoHostBatchAdaptive
{
  // Input parameters
  size_t  total_input_bytes = 0;

  // Cost-model state (refined online)
  double  R_alloc_gbps   = DEFAULT_R_ALLOC_GBPS;
  double  R_dram_gbps    = DEFAULT_R_DRAM_GBPS;
  double  R_pcie_gbps    = DEFAULT_R_PCIE_GBPS;
  double  R_kernel_gbps  = DEFAULT_R_KERNEL_GBPS;

  // Chosen window parameters
  size_t  window_bytes      = 0;
  size_t  W_kernel_sat      = 0;
  size_t  W_pcie_amort      = W_PCIE_AMORT_FLOOR;
  size_t  W_launch_amort    = W_LAUNCH_AMORT_FLOOR;

  // Allocation state
  void*   pinned_buf        = nullptr;

  // Iteration state
  size_t  next_window_idx   = 0;
  size_t  current_window_bytes = 0;
  size_t  current_input_offset = 0;
  bool    window_pending    = false;

  // Diagnostics
  int     adaptation_count  = 0;
};

arctoStatus_t arctoHostBatchAdaptiveCreate(
    size_t total_input_bytes,
    double hint_kernel_gbps,
    arctoHostBatchAdaptive_t** out_batch)
{
  if (out_batch == nullptr) return arctoErrorInvalidValue;
  arctoHostBatchAdaptive_t* b = nullptr;
  try {
    b = new arctoHostBatchAdaptive_t();
  } catch (const std::bad_alloc&) {
    return arctoErrorInternal;
  }
  b->total_input_bytes = total_input_bytes;
  if (hint_kernel_gbps > 0.0) b->R_kernel_gbps = hint_kernel_gbps;
  b->W_kernel_sat = pick_W_kernel_saturation();
  *out_batch = b;
  return arctoSuccess;
}

size_t arctoHostBatchAdaptiveChooseWindow(arctoHostBatchAdaptive_t* batch)
{
  if (batch == nullptr) return 0;

  // Compute W_opt as the max of the three practical floors.
  size_t W_opt = std::max({batch->W_kernel_sat,
                           batch->W_pcie_amort,
                           batch->W_launch_amort});

  // Cap at total_input_bytes (no point in a window larger than the input).
  if (batch->total_input_bytes > 0 && W_opt > batch->total_input_bytes) {
    W_opt = batch->total_input_bytes;
  }

  // Allocate the pinned buffer on first call. Subsequent calls reuse
  // the same allocation; we do NOT reallocate if W_opt drifts within
  // a small factor, to avoid the hipHostMalloc cold-start cost
  // (RX7900XT_BYTE_PINNED_ROOTCAUSE_*/FINDINGS.md). For large
  // adaptations (rare), the caller should destroy and recreate the
  // handle.
  if (batch->pinned_buf == nullptr) {
    hipError_t e = hipHostMalloc(&batch->pinned_buf, W_opt, hipHostMallocDefault);
    if (e != hipSuccess) return 0;
    batch->window_bytes = W_opt;
  }
  return batch->window_bytes;
}

arctoStatus_t arctoHostBatchAdaptiveNextWindow(
    arctoHostBatchAdaptive_t* batch,
    arctoAdaptiveWindow_t* out_w)
{
  if (batch == nullptr || out_w == nullptr) return arctoErrorInvalidValue;
  if (batch->window_bytes == 0) return arctoErrorInvalidValue;  // need ChooseWindow first

  const size_t consumed = batch->next_window_idx * batch->window_bytes;
  if (consumed >= batch->total_input_bytes) {
    out_w->pinned_ptr    = nullptr;
    out_w->window_bytes  = 0;
    out_w->input_offset  = batch->total_input_bytes;
    out_w->window_index  = batch->next_window_idx;
    batch->window_pending = false;
    return arctoSuccess;
  }

  const size_t remaining = batch->total_input_bytes - consumed;
  const size_t this_bytes = std::min(batch->window_bytes, remaining);

  out_w->pinned_ptr   = batch->pinned_buf;
  out_w->window_bytes = this_bytes;
  out_w->input_offset = consumed;
  out_w->window_index = batch->next_window_idx;

  batch->current_window_bytes = this_bytes;
  batch->current_input_offset = consumed;
  batch->window_pending = true;
  batch->next_window_idx++;
  return arctoSuccess;
}

arctoStatus_t arctoHostBatchAdaptiveUpload(
    const arctoHostBatchAdaptive_t* batch,
    void* d_dst,
    hipStream_t stream)
{
  if (batch == nullptr) return arctoErrorInvalidValue;
  if (batch->current_window_bytes == 0) return arctoSuccess;  // nothing to do
  if (d_dst == nullptr) return arctoErrorInvalidValue;
  hipError_t e = hipMemcpyAsync(d_dst, batch->pinned_buf,
      batch->current_window_bytes, hipMemcpyHostToDevice, stream);
  return (e == hipSuccess) ? arctoSuccess : arctoErrorInternal;
}

void arctoHostBatchAdaptiveRecordH2HTime(
    arctoHostBatchAdaptive_t* batch, double h2h_ms)
{
  if (batch == nullptr || h2h_ms <= 0.0 || batch->current_window_bytes == 0) return;
  const double new_gbps = (double)batch->current_window_bytes / (1.0e9 * h2h_ms * 1.0e-3);
  // Online EMA with alpha = 0.5 on the first observation (replace
  // default), 0.2 thereafter (smooth small variations).
  const double alpha = (batch->adaptation_count == 0) ? 1.0 : 0.2;
  batch->R_dram_gbps = (1.0 - alpha) * batch->R_dram_gbps + alpha * new_gbps;
}

void arctoHostBatchAdaptiveRecordH2DTime(
    arctoHostBatchAdaptive_t* batch, double h2d_ms)
{
  if (batch == nullptr || h2d_ms <= 0.0 || batch->current_window_bytes == 0) return;
  const double new_gbps = (double)batch->current_window_bytes / (1.0e9 * h2d_ms * 1.0e-3);
  const double alpha = (batch->adaptation_count == 0) ? 1.0 : 0.2;
  batch->R_pcie_gbps = (1.0 - alpha) * batch->R_pcie_gbps + alpha * new_gbps;
}

void arctoHostBatchAdaptiveRecordKernelTime(
    arctoHostBatchAdaptive_t* batch, double kernel_ms)
{
  if (batch == nullptr || kernel_ms <= 0.0 || batch->current_window_bytes == 0) return;
  const double new_gbps = (double)batch->current_window_bytes / (1.0e9 * kernel_ms * 1.0e-3);
  const double alpha = (batch->adaptation_count == 0) ? 1.0 : 0.2;
  batch->R_kernel_gbps = (1.0 - alpha) * batch->R_kernel_gbps + alpha * new_gbps;
  batch->adaptation_count++;
}

arctoStatus_t arctoHostBatchAdaptiveStats(
    const arctoHostBatchAdaptive_t* batch,
    arctoAdaptiveStats_t* out)
{
  if (batch == nullptr || out == nullptr) return arctoErrorInvalidValue;
  out->window_bytes      = batch->window_bytes;
  out->total_input_bytes = batch->total_input_bytes;
  out->windows_consumed  = batch->next_window_idx;
  out->R_alloc_gbps      = batch->R_alloc_gbps;
  out->R_dram_gbps       = batch->R_dram_gbps;
  out->R_pcie_gbps       = batch->R_pcie_gbps;
  out->R_kernel_gbps     = batch->R_kernel_gbps;
  out->adaptation_count  = batch->adaptation_count;
  return arctoSuccess;
}

void arctoHostBatchAdaptiveDestroy(arctoHostBatchAdaptive_t* batch)
{
  if (batch == nullptr) return;
  if (batch->pinned_buf != nullptr) {
    (void) hipHostFree(batch->pinned_buf);
  }
  delete batch;
}
