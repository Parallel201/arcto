/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file amd_optimizations.h
 * @brief AMD GPU specific optimizations for wavefront-64 architectures
 *
 * This header provides optimizations specifically tailored for AMD GPUs,
 * particularly targeting:
 * - Wavefront size of 64 (vs NVIDIA warp size of 32)
 * - CDNA/RDNA architectures (MI300X, MI250X, MI210, MI50, RX 7900 XT, etc.)
 * - ROCm runtime optimizations
 */

#ifndef AMD_OPTIMIZATIONS_H
#define AMD_OPTIMIZATIONS_H

#include "device_types.h"

namespace arcto {
namespace amd {

// ============================================================================
// WAVEFRONT/WARP SIZE CONFIGURATION
// ============================================================================

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  #define AMD_GPU_TARGET 1

  // AMD GPUs use wavefront of 64 threads (except gfx1100+ which can use 32)
  #if defined(USE_WARPSIZE_32)
    constexpr int WAVEFRONT_SIZE = 32;
    constexpr int WAVEFRONT_SHIFT = 5;  // log2(32)
  #else
    constexpr int WAVEFRONT_SIZE = 64;
    constexpr int WAVEFRONT_SHIFT = 6;  // log2(64)
  #endif

  #define WAVEFRONT_MASK ((1ULL << WAVEFRONT_SIZE) - 1)

#else
  #define AMD_GPU_TARGET 0
  constexpr int WAVEFRONT_SIZE = 32;  // NVIDIA warp size
  constexpr int WAVEFRONT_SHIFT = 5;
  #define WAVEFRONT_MASK 0xFFFFFFFFU
#endif

// ============================================================================
// OPTIMIZED BLOCK SIZES FOR AMD GPUS
// ============================================================================

// AMD GPUs benefit from block sizes that are multiples of wavefront size (64)
// and aligned to LDS banks (32 banks, 4 bytes each)

namespace block_size {
  // Conservative: 4 wavefronts per block
  constexpr int SMALL = 4 * WAVEFRONT_SIZE;   // 256 threads

  // Balanced: 8 wavefronts per block (good for most kernels)
  constexpr int MEDIUM = 8 * WAVEFRONT_SIZE;  // 512 threads

  // Aggressive: 16 wavefronts per block (for memory-bound kernels)
  constexpr int LARGE = 16 * WAVEFRONT_SIZE;  // 1024 threads

  // Maximum supported by AMD GPUs
  constexpr int MAX = 1024;
}

// ============================================================================
// LAUNCH BOUNDS OPTIMIZATIONS
// ============================================================================

/**
 * @brief Optimal launch bounds for AMD GPUs
 *
 * AMD GPUs benefit from higher occupancy. These macros help the compiler
 * optimize register usage and LDS allocation.
 *
 * Format: __launch_bounds__(max_threads_per_block, min_blocks_per_cu)
 *
 * MI300X has 304 CUs, so we can have many blocks in flight.
 */

// For compute-intensive kernels (prefer more registers)
#define AMD_LAUNCH_BOUNDS_COMPUTE __launch_bounds__(512, 2)

// For memory-intensive kernels (prefer more blocks for hiding latency)
#define AMD_LAUNCH_BOUNDS_MEMORY __launch_bounds__(1024, 4)

// For balanced kernels
#define AMD_LAUNCH_BOUNDS_BALANCED __launch_bounds__(512, 3)

// For small kernels that need high occupancy
#define AMD_LAUNCH_BOUNDS_HIGH_OCC __launch_bounds__(256, 8)

// ============================================================================
// WAVE-LEVEL SYNCHRONIZATION
// ============================================================================

#if AMD_GPU_TARGET

/**
 * @brief Wave-level barrier for AMD GPUs
 *
 * AMD provides intrinsic for wave-level synchronization which is faster
 * than full block synchronization for wave-local operations.
 */
__device__ __forceinline__ void wave_barrier()
{
#if defined(__HIP_PLATFORM_AMD__)
  __builtin_amdgcn_wave_barrier();
#else
  __syncwarp(WAVEFRONT_MASK);
#endif
}

/**
 * @brief Full fence ensuring all memory operations are visible
 */
__device__ __forceinline__ void wave_fence()
{
#if defined(__HIP_PLATFORM_AMD__)
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
#else
  __threadfence();
#endif
}

#else

// Fallback for NVIDIA GPUs
__device__ __forceinline__ void wave_barrier()
{
  __syncwarp(WAVEFRONT_MASK);
}

__device__ __forceinline__ void wave_fence()
{
  __threadfence();
}

#endif

// ============================================================================
// WAVEFRONT COLLECTIVE OPERATIONS
// ============================================================================

/**
 * @brief Ballot operation across wavefront
 * Returns bitmask of threads where predicate is true
 */
__device__ __forceinline__ uint64_t wave_ballot(int predicate)
{
#if AMD_GPU_TARGET && (WAVEFRONT_SIZE == 64)
  return __ballot(predicate);  // Returns uint64_t on AMD with wave64
#else
  return __ballot(predicate);  // Returns uint32_t on NVIDIA
#endif
}

/**
 * @brief Count number of active threads in wavefront
 */
__device__ __forceinline__ int wave_active_count()
{
#if AMD_GPU_TARGET && (WAVEFRONT_SIZE == 64)
  return __popcll(__ballot(1));  // 64-bit popc for wave64
#else
  return __popc(__ballot(1));    // 32-bit popc for warp32
#endif
}

/**
 * @brief Get lane ID within wavefront (0 to WAVEFRONT_SIZE-1)
 */
__device__ __forceinline__ int wave_lane_id()
{
  return threadIdx.x & (WAVEFRONT_SIZE - 1);
}

/**
 * @brief Get wavefront ID within block
 */
__device__ __forceinline__ int wave_id()
{
  return threadIdx.x >> WAVEFRONT_SHIFT;
}

/**
 * @brief Check if this is the first lane in the wavefront
 */
__device__ __forceinline__ bool is_wave_first_lane()
{
  return wave_lane_id() == 0;
}

// ============================================================================
// WAVEFRONT REDUCTIONS
// ============================================================================

/**
 * @brief Sum reduction across wavefront
 * Optimized for wave64 on AMD GPUs
 */
template<typename T>
__device__ __forceinline__ T wave_reduce_sum(T val)
{
#if AMD_GPU_TARGET && (WAVEFRONT_SIZE == 64)
  // Optimized tree reduction for wave64
  #pragma unroll
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, WAVEFRONT_SIZE);
  }
#else
  // Standard reduction for warp32
  #pragma unroll
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, WAVEFRONT_SIZE);
  }
#endif
  return val;
}

/**
 * @brief Max reduction across wavefront
 */
template<typename T>
__device__ __forceinline__ T wave_reduce_max(T val)
{
#pragma unroll
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down(val, offset, WAVEFRONT_SIZE);
    val = (val > other) ? val : other;
  }
  return val;
}

/**
 * @brief Min reduction across wavefront
 */
template<typename T>
__device__ __forceinline__ T wave_reduce_min(T val)
{
#pragma unroll
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down(val, offset, WAVEFRONT_SIZE);
    val = (val < other) ? val : other;
  }
  return val;
}

// ============================================================================
// WAVEFRONT SCAN OPERATIONS
// ============================================================================

/**
 * @brief Inclusive prefix sum (scan) across wavefront
 * Each thread gets sum of all values from lane 0 to current lane (inclusive)
 */
template<typename T>
__device__ __forceinline__ T wave_scan_inclusive_sum(T val)
{
  // Hillis-Steele scan optimized for wave64
#pragma unroll
  for (int offset = 1; offset < WAVEFRONT_SIZE; offset <<= 1) {
    T other = __shfl_up(val, offset, WAVEFRONT_SIZE);
    if (wave_lane_id() >= offset) {
      val += other;
    }
  }
  return val;
}

/**
 * @brief Exclusive prefix sum (scan) across wavefront
 * Each thread gets sum of all values from lane 0 to current lane (exclusive)
 */
template<typename T>
__device__ __forceinline__ T wave_scan_exclusive_sum(T val)
{
  T inclusive = wave_scan_inclusive_sum(val);
  T exclusive = __shfl_up(inclusive, 1, WAVEFRONT_SIZE);
  return (wave_lane_id() == 0) ? 0 : exclusive;
}

// ============================================================================
// MEMORY ALIGNMENT HELPERS
// ============================================================================

/**
 * @brief Check if pointer is aligned to AMD LDS bank size (128 bytes)
 */
template<typename T>
__device__ __forceinline__ bool is_lds_aligned(const T* ptr)
{
  return (reinterpret_cast<uintptr_t>(ptr) & 127) == 0;
}

/**
 * @brief Align value to wavefront size
 */
__device__ __forceinline__ int align_to_wavefront(int value)
{
  return ((value + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE) * WAVEFRONT_SIZE;
}

// ============================================================================
// DIVERGENCE HINTS
// ============================================================================

/**
 * @brief Hint to compiler that branches are likely to be uniform across wavefront
 * This can help reduce divergence overhead
 */
#if defined(__HIP_PLATFORM_AMD__)
  #define AMD_LIKELY_UNIFORM __attribute__((likely_uniform))
#else
  #define AMD_LIKELY_UNIFORM
#endif

/**
 * @brief Hint that condition is likely true
 */
#define AMD_LIKELY(x) __builtin_expect(!!(x), 1)

/**
 * @brief Hint that condition is likely false
 */
#define AMD_UNLIKELY(x) __builtin_expect(!!(x), 0)

} // namespace amd
} // namespace arcto

#endif // AMD_OPTIMIZATIONS_H
