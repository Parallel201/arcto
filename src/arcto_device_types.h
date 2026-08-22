// Copyright (C) 2025-2026 Cristiano Künas.
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

#ifndef ARCTO_DEVICE_TYPES_HIPH
#define ARCTO_DEVICE_TYPES_HIPH

#include <stdint.h>

namespace arcto {

#if !defined(USE_WARPSIZE_32)
# if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
#   define USE_WARPSIZE_64
#  endif
#endif

// Snappy GPU types
#if defined(USE_WARPSIZE_64)
typedef uint64_t warp_mask_t;
typedef int64_t signed_warp_mask_t;
#else
typedef uint32_t warp_mask_t;
typedef int32_t signed_warp_mask_t;
#endif

constexpr uint32_t uwarpsize = sizeof(warp_mask_t)*8;
constexpr int32_t warpsize = uwarpsize;

// The compile-time wave size above must be the wavefront size the device code
// is actually compiled for (the wave-level primitives, ballots and masks
// assume it). hip-clang exposes the latter in device passes; a mismatch is a
// build error, not a runtime surprise (e.g. a wave32 build on gfx9, or a
// wave64 build of an RDNA target without -mwavefrontsize64 — see CMake).
#if defined(__HIP_DEVICE_COMPILE__)
# if defined(__AMDGCN_WAVEFRONT_SIZE__)
static_assert(warpsize == __AMDGCN_WAVEFRONT_SIZE__,
    "ARCTO wave size (USE_WARPSIZE_32 / default 64) differs from the wavefront size of this device compilation");
# elif defined(__AMDGCN_WAVEFRONT_SIZE)
static_assert(warpsize == __AMDGCN_WAVEFRONT_SIZE,
    "ARCTO wave size (USE_WARPSIZE_32 / default 64) differs from the wavefront size of this device compilation");
# endif
#endif

} // namespace arcto

#endif // ARCTO_DEVICE_TYPES_HIPH
