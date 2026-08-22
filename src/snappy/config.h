/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#ifndef SNAPPY_CONFIG_HIPH
#define SNAPPY_CONFIG_HIPH

#include "arcto_device_types.h"

// Decompression default settings that can be influenced via compiler flags

#ifndef LOG2_BATCH_SIZE
#  if defined(USE_WARPSIZE_64)
     // results in BATCH_SIZE 64 (LZ77 symbols)
#    define LOG2_BATCH_SIZE 6
#  else
     // results in BATCH_SIZE 32 (LZ77 symbols)
#    define LOG2_BATCH_SIZE 5
#  endif
#endif

#ifndef LOG2_BATCH_COUNT
#  define LOG2_BATCH_COUNT 2
#endif

#ifndef LOG2_PREFETCH_SIZE
   // results in PREFETCH_SIZE 4096 (bytes)
#  define LOG2_PREFETCH_SIZE 12
#endif

#ifndef PREFETCH_SECTORS
  // How many loads in flight when prefetching
#  define PREFETCH_SECTORS 8
#endif

#ifndef LITERAL_SECTORS
   // How many loads in flight when processing the literal (bytes per lane per
   // literal step). SNP-D11: with the dword literal path (SNP-D8) two dwords per
   // lane (8) measured +4 % / +10 % on literal-heavy data on the wave32 gfx1100
   // and neutral-to-slightly-negative on the wave64 gfx942, so 8 is the default
   // for wave32 AMD builds only; wave64 and CUDA keep the inherited 4.
#  if defined(USE_WARPSIZE_32)
#    define LITERAL_SECTORS 8
#  else
#    define LITERAL_SECTORS 4
#  endif
#endif

namespace arcto
{
  namespace snappy
  {

    //////////////
    // COMPRESSION
    //////////////

    constexpr unsigned HASH_BITS = 12;
    //: results in 4096 hash map entries a 2 Byte

    // TBD: Tentatively limits to 2-byte codes to prevent long copy search followed by long literal
    // encoding
    constexpr unsigned MAX_LITERAL_LENGTH = 256;

    constexpr unsigned MAX_COPY_LENGTH = 64;      // Syntax limit
    constexpr unsigned MAX_COPY_DISTANCE = 32768; // Matches encoder limit as described in snappy format description

    constexpr unsigned COMP_THREADS_PER_BLOCK = 2 * warpsize; // 2 warps per stream, 1 stream per block

    ////////////////
    // DECOMPRESSION
    ////////////////

    constexpr unsigned DECOMP_THREADS_PER_BLOCK = 3 * warpsize; // 3 warps per stream, 1 stream per block

    // Back-off amounts for the three spin-wait loops of the decoder (see
    // SPIN_SLEEP in device_functions.hiph). The unit is platform dependent.
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // AMD: s_sleep units, each roughly 64 clocks, immediate 0..127. The
    // purpose is to yield the SIMD to productive waves, not precise timing;
    // the values are tunables (override with -DARCTO_SNAPPY_*_SLEEP=n).
#  ifndef ARCTO_SNAPPY_PREFETCH_SLEEP
#    define ARCTO_SNAPPY_PREFETCH_SLEEP 2
#  endif
#  ifndef ARCTO_SNAPPY_DECODE_SLEEP
#    define ARCTO_SNAPPY_DECODE_SLEEP 1
#  endif
#  ifndef ARCTO_SNAPPY_PROCESS_SLEEP
#    define ARCTO_SNAPPY_PROCESS_SLEEP 1
#  endif
    constexpr unsigned PREFETCH_SLEEP = ARCTO_SNAPPY_PREFETCH_SLEEP;
    constexpr unsigned DECODE_SLEEP = ARCTO_SNAPPY_DECODE_SLEEP;
    constexpr unsigned PROCESS_SLEEP = ARCTO_SNAPPY_PROCESS_SLEEP;
#else
    // CUDA: nanoseconds for __nanosleep on sm_70+ (no-op below); the
    // inherited nvCOMP values.
    constexpr unsigned PREFETCH_SLEEP = 1600;
    constexpr unsigned DECODE_SLEEP = 50;
    constexpr unsigned PROCESS_SLEEP = 100;
#endif

    // Not supporting streams longer than this (not what snappy is intended for)
    constexpr unsigned SNAPPY_MAX_STREAM_SIZE = 0x7fffffff;

    constexpr unsigned BATCH_SIZE = (1 << LOG2_BATCH_SIZE);
    constexpr unsigned BATCH_COUNT = (1 << LOG2_BATCH_COUNT);
    constexpr unsigned PREFETCH_SIZE = (1 << LOG2_PREFETCH_SIZE); // 4KB, in 32B chunks
                                                                  //: TODO: amd: does it make sense to tune this for AMD to have the same amount of chunks?

    constexpr unsigned LOG_CYCLECOUNT = 0;
  } // namespace snappy
} // namespace arcto

#endif // SNAPPY_CONFIG_HIPH