/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// MIT License
//
// Modifications Copyright (C) 2025-2026 Cristiano Künas.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "arcto.h"
#include "arcto.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <time.h>
using ssize_t = ptrdiff_t;
#endif

// Restrict qualification on kernel pointer parameters pays only on AMD
// wave64 together with the 64-VGPR pin (E17: +18.5% MI210, +12.7% MI50);
// on wave32 it costs -1.8% (gfx1100) and on the CUDA backend it was never
// measured, so both keep the inherited unqualified form.
#if defined(__HIP_PLATFORM_AMD__) && !defined(USE_WARPSIZE_32)
#define ARCTO_KRESTRICT __restrict__
#else
#define ARCTO_KRESTRICT
#endif

/**
 * @brief Launch bounds with explicit, per-platform semantics.
 *
 * The two platforms give the SECOND argument of __launch_bounds__ different
 * meanings:
 *   CUDA: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
 *   HIP:  __launch_bounds__(maxThreadsPerBlock, minWavesPerExecutionUnit)
 *         i.e. the minimum number of wavefronts the compiler must keep
 *         resident per SIMD (amdgpu_waves_per_eu) -- it caps the VGPR budget
 *         (CDNA: <=64 VGPRs for 8 waves/SIMD, <=72 for 7, <=80 for 6, <=96 for
 *         5, <=128 for 4) and is NOT "blocks per CU".
 *
 * ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu, min_blocks_per_sm) lets a
 * kernel state both intents in one place and expands to whichever form the
 * active platform understands. ARCTO_LAUNCH_BOUNDS_1(threads) is the
 * single-argument form (identical on both platforms).
 */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  #define ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu, min_blocks_per_sm) \
    __launch_bounds__(threads, min_waves_per_eu)
#else
  #define ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu, min_blocks_per_sm) \
    __launch_bounds__(threads, min_blocks_per_sm)
#endif
#define ARCTO_LAUNCH_BOUNDS_1(threads) __launch_bounds__(threads)

namespace arcto {

namespace {

template <typename T>
T* align(T* const ptr, const size_t alignment)
{
  const size_t bits = reinterpret_cast<size_t>(ptr);
  const size_t mask = alignment - 1;

  return reinterpret_cast<T*>(((bits - 1) | mask) + 1);
}

template <typename T>
size_t
relativeEndOffset(const void* start, const T* subsection, const size_t length)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<size_t>(diff) + length * sizeof(T);
}

template <typename T = size_t>
T relativeEndOffset(const void* start, const void* subsection)
{
  std::ptrdiff_t diff = reinterpret_cast<const char*>(subsection)
                        - static_cast<const char*>(start);
  return static_cast<T>(diff);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const chunk)
{
  return (num / chunk) + (num % chunk > 0);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundDownTo(U const num, T const chunk)
{
  return (num / chunk) * chunk;
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

/**
 * @brief Calculate the first aligned location after `ptr`.
 *
 * @tparam T Type such that the alignment requirement is satisfied.
 * @param ptr Input pointer.
 * @return The first pointer after `ptr` that satisfy the alignment requirement.
 */
template <typename T>
constexpr __host__ __device__ T* roundUpToAlignment(void* ptr)
{
  return reinterpret_cast<T*>(
      roundUpTo(reinterpret_cast<uintptr_t>(ptr), sizeof(T)));
}

template <typename T>
constexpr __host__ __device__ const T* roundUpToAlignment(const void* ptr)
{
  return reinterpret_cast<const T*>(
      roundUpTo(reinterpret_cast<uintptr_t>(ptr), sizeof(T)));
}

/**
 * @brief Provide a type that is the larger of `U` and `T` in terms of size.
 */
template <typename U, typename T>
struct make_larger
{
  typedef std::conditional_t<(sizeof(U) >= sizeof(T)), U, T> type;
};

template <typename U, typename T>
using larger_t = typename make_larger<U, T>::type;

} // namespace 

__inline__ size_t sizeOfarctoType(arctoType_t type)
{
  switch (type) {
  case ARCTO_TYPE_BITS:
    return 1;
  case ARCTO_TYPE_CHAR:
    return sizeof(int8_t);
  case ARCTO_TYPE_UCHAR:
    return sizeof(uint8_t);
  case ARCTO_TYPE_SHORT:
    return sizeof(int16_t);
  case ARCTO_TYPE_USHORT:
    return sizeof(uint16_t);
  case ARCTO_TYPE_INT:
    return sizeof(int32_t);
  case ARCTO_TYPE_UINT:
    return sizeof(uint32_t);
  case ARCTO_TYPE_LONGLONG:
    return sizeof(int64_t);
  case ARCTO_TYPE_ULONGLONG:
    return sizeof(uint64_t);
  default:
    throw std::runtime_error("Unsupported type " + std::to_string(type));
  }
}

} // namespace arcto