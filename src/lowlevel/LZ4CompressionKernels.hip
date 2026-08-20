/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "HipUtils.h"
#include "LZ4CompressionKernels.h"
#include "LZ4Kernels.hiph"
#include "TempSpaceBroker.h"
#include "common.h"

#include "hip/hip_runtime.h"
#include "arcto_hipcub.hiph"

#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using double_word_type = uint64_t;
using item_type = uint32_t;

#define OOB_CHECKING 1 // Prevent's crashing of corrupt lz4 sequences

namespace arcto {

namespace lowlevel {

// CDNA (wave64) sits exactly on the 64-VGPR occupancy step (<=64 VGPRs ->
// 8 waves/SIMD; 65-96 -> 5). Without bounds the compiler allocates ~90
// VGPRs for the compress kernel and occupancy drops 8->5 waves (measured
// on gfx942). Pin the budget back with a waves-per-EU hint. On wave32
// (RDNA3, 3x larger register file) the hint would only restrict the
// scheduler, so cap the block size alone.
// E16 knob: on wave32 the compress kernel compiles to 136 arch VGPRs
// (read from the rocprofv3 kernel-symbol table), which caps residency at
// about 10 of 16 waves per SIMD on RDNA3. A nonzero min-waves value asks
// the compiler to fit more waves (bounding VGPRs), trading possible spills
// for occupancy. Default 0 keeps the shipped behavior exactly.
#ifndef ARCTO_LZ4_W32_MIN_WAVES
#define ARCTO_LZ4_W32_MIN_WAVES 0
#endif

#if defined(USE_WARPSIZE_32)
#if ARCTO_LZ4_W32_MIN_WAVES > 0
#define LZ4_COMP_LAUNCH_BOUNDS __launch_bounds__(LZ4_COMP_THREADS_PER_CHUNK, ARCTO_LZ4_W32_MIN_WAVES)
#else
#define LZ4_COMP_LAUNCH_BOUNDS __launch_bounds__(LZ4_COMP_THREADS_PER_CHUNK)
#endif
#else
#define LZ4_COMP_LAUNCH_BOUNDS __launch_bounds__(LZ4_COMP_THREADS_PER_CHUNK, 8)
#endif

#ifndef ARCTO_LZ4_TABLE_PAD_ENTRIES
#define ARCTO_LZ4_TABLE_PAD_ENTRIES 0u
#endif

template<typename T>
__global__ LZ4_COMP_LAUNCH_BOUNDS void lz4CompressBatchKernel(
    const uint8_t* const* __restrict__ device_in_ptr,
    const size_t* const __restrict__ device_in_bytes,
    uint8_t* const* const __restrict__ device_out_ptr,
    size_t* const __restrict__ device_out_bytes,
    offset_type* const __restrict__ temp_space,
    const position_type hash_table_size)
{
  const int bidx = blockIdx.x * blockDim.y + threadIdx.y;

  auto decomp_ptr = device_in_ptr[bidx];
  assert(reinterpret_cast<uintptr_t>(decomp_ptr) % sizeof(T) == 0 && "Input buffer not aligned");
  const size_t decomp_length = device_in_bytes[bidx];

  uint8_t* const comp_ptr = device_out_ptr[bidx];
  size_t* const comp_length = device_out_bytes + bidx;

#ifdef ARCTO_LZ4_LDS_TABLE
  // E11: the match-finding table lives in the Local Data Share instead of
  // global memory. At the measured optimum sizes (64-128 entries, 128-256 B)
  // the old occupancy objection is gone, and the probes leave the contended
  // per-CU vector cache entirely -- the mechanism the counter matrices showed
  // governing throughput. One table per block (one wave per chunk here).
  __shared__ offset_type lds_hash_table[MAX_HASH_TABLE_SIZE];
  offset_type* const hash_table = lds_hash_table;
  (void)temp_space;
#else
  // A6 experiment knob: per-chunk tables are laid out at a stride of
  // hash_table_size + ARCTO_LZ4_TABLE_PAD_ENTRIES entries. With the default
  // pad of 0 this is exactly the shipped layout. A nonzero pad breaks the
  // power-of-two alignment of consecutive tables while leaving the table
  // itself untouched, which is what separates a channel/bank-conflict account
  // of the table cost from a footprint account.
  const position_type table_stride = hash_table_size + ARCTO_LZ4_TABLE_PAD_ENTRIES;
  offset_type* const hash_table = temp_space + bidx * table_stride;
#endif

  compressStream(comp_ptr, reinterpret_cast<const T*>(decomp_ptr), hash_table, hash_table_size, decomp_length, comp_length);
}

__global__ void lz4DecompressBatchKernel(
    const uint8_t* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t* const device_out_bytes,
    const size_t batch_size,
    uint8_t* const* const device_out_ptrs,
    size_t* device_uncompressed_bytes,
    arctoStatus_t* device_status_ptrs,
    bool output_decompressed)
{
  const int bid = blockIdx.x * LZ4_DECOMP_CHUNKS_PER_BLOCK + threadIdx.y;

  __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * LZ4_DECOMP_CHUNKS_PER_BLOCK];

  assert(!output_decompressed || device_out_ptrs != nullptr);
  // device_uncompressed_bytes needs to be valid if we are precomputing
  // output size
  assert(output_decompressed || device_uncompressed_bytes != nullptr);

  if (bid < batch_size) {
    uint8_t* const decomp_ptr
        = device_out_ptrs == nullptr ? nullptr : device_out_ptrs[bid];
    const uint8_t* const comp_ptr = device_in_ptrs[bid];
    const position_type chunk_length
        = static_cast<position_type>(device_in_bytes[bid]);
    const position_type output_buf_length
        = output_decompressed
              ? static_cast<position_type>(device_out_bytes[bid])
              : UINT_MAX;

    decompressStream(
        buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE,
        decomp_ptr,
        comp_ptr,
        chunk_length,
        output_buf_length,
        device_uncompressed_bytes ? device_uncompressed_bytes + bid : nullptr,
        device_status_ptrs? device_status_ptrs + bid : nullptr,
        output_decompressed);
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS ***********************************************************
 *****************************************************************************/

#ifdef ARCTO_LZ4_RUNTIME_TABLE
// E15: the table cap derived at run time from the two device properties the
// model names -- per-CU vector cache size and the number of concurrently
// resident waves -- instead of a fixed constant. The knee measured on three
// architectures sits at L1 / concurrent_waves, and moved with the denominator
// when the input size changed; this encodes exactly that.
static size_t lz4RuntimeTableCap(size_t batch_size)
{
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev);
  size_t l1_bytes = 16 * 1024;                       // GCN/CDNA per-CU vector cache
  if (strstr(prop.gcnArchName, "gfx10") || strstr(prop.gcnArchName, "gfx11")
      || strstr(prop.gcnArchName, "gfx12")) {
    l1_bytes = 32 * 1024;                            // RDNA L0
  }
  const size_t cus = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
  size_t max_waves_per_cu = 32;
  if (prop.warpSize > 0 && prop.maxThreadsPerMultiProcessor > 0) {
    max_waves_per_cu = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  }
  size_t waves_per_cu = batch_size / cus;
  if (waves_per_cu < 1) waves_per_cu = 1;
  if (waves_per_cu > max_waves_per_cu) waves_per_cu = max_waves_per_cu;
  size_t entries = (l1_bytes / waves_per_cu) / sizeof(offset_type);
  size_t pow2 = 1;
  while (pow2 * 2 <= entries) pow2 *= 2;             // round DOWN to a power of two
  if (pow2 < 64) pow2 = 64;
  if (pow2 > (size_t)MAX_HASH_TABLE_SIZE) pow2 = MAX_HASH_TABLE_SIZE;
  return pow2;
}
#endif

static size_t lz4GetHashTableSizeForBatch(size_t max_chunk_size, size_t batch_size)
{
#ifdef ARCTO_LZ4_RUNTIME_TABLE
  const size_t base = lz4GetHashTableSize(max_chunk_size);
  const size_t cap = lz4RuntimeTableCap(batch_size);
  return base < cap ? base : cap;
#else
  (void)batch_size;
  return lz4GetHashTableSize(max_chunk_size);
#endif
}

size_t lz4GetHashTableSize(size_t max_chunk_size)
{
  auto roundUpPow2 = [](size_t x) {
    size_t ans = 1;
    while(ans < x)
      ans *= 2;
    return ans;
  };
  // when chunk size is smaller than the max hashtable size round the
  // hashtable size up to the nearest power of 2 of the chunk size.
  // The lower load factor from a significantly larger hashtable size compared
  // to the chunk size doesn't increase performance, however having a smaller
  // hashtable which yields much high cache utilization does.
  return min(roundUpPow2(max_chunk_size), (size_t)MAX_HASH_TABLE_SIZE);
}

void lz4BatchCompress(
    const uint8_t* const* decomp_data_device,
    const size_t* const decomp_sizes_device,
    const size_t max_chunk_size,
    const size_t batch_size,
    void* const temp_data,
    const size_t temp_bytes,
    uint8_t* const* const comp_data_device,
    size_t* const comp_sizes_device,
    arctoType_t data_type,
    hipStream_t stream)
{

  position_type HT_size = lz4GetHashTableSizeForBatch(max_chunk_size, batch_size);

  const size_t total_required_temp
      = batch_size * (HT_size + ARCTO_LZ4_TABLE_PAD_ENTRIES) * sizeof(offset_type);
  if (temp_bytes < total_required_temp) {
    throw std::runtime_error(
        "Insufficient temp space: got " + std::to_string(temp_bytes)
        + " bytes, but need " + std::to_string(total_required_temp)
        + " bytes.");
  }

  const dim3 grid(batch_size);
  const dim3 block(LZ4_COMP_THREADS_PER_CHUNK); //: 32

  switch (data_type) {
    case ARCTO_TYPE_BITS:
    case ARCTO_TYPE_CHAR:
    case ARCTO_TYPE_UCHAR:
      lz4CompressBatchKernel<uint8_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    case ARCTO_TYPE_SHORT:
    case ARCTO_TYPE_USHORT:
      lz4CompressBatchKernel<uint16_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    case ARCTO_TYPE_INT:
    case ARCTO_TYPE_UINT:
      lz4CompressBatchKernel<uint32_t><<<grid, block, 0, stream>>>(
          decomp_data_device,
          decomp_sizes_device,
          comp_data_device,
          comp_sizes_device,
          static_cast<offset_type*>(temp_data),
          HT_size);
      break;
    default:
      throw std::invalid_argument("Unsupported input data type");
  }

  HipUtils::check_last_error();
}

void lz4BatchDecompress(
    const uint8_t* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t* const device_out_bytes,
    const size_t batch_size,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    uint8_t* const* const device_out_ptrs,
    size_t* device_actual_uncompressed_bytes,
    arctoStatus_t* device_status_ptrs,
    hipStream_t stream)
{
  const dim3 grid(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK)); //: chunks per block: 2
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK); //: threads per chunk, chunks per block: 32,2

  lz4DecompressBatchKernel<<<grid, block, 0, stream>>>(
      device_in_ptrs,
      device_in_bytes,
      device_out_bytes,
      batch_size,
      device_out_ptrs,
      device_actual_uncompressed_bytes,
      device_status_ptrs,
      true);
  HipUtils::check_last_error("lz4DecompressBatchKernel()");
}

void lz4BatchGetDecompressSizes(
    const uint8_t* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  const dim3 grid(roundUpDiv(batch_size, LZ4_DECOMP_CHUNKS_PER_BLOCK)); //: chunks per block: 2
  const dim3 block(LZ4_DECOMP_THREADS_PER_CHUNK, LZ4_DECOMP_CHUNKS_PER_BLOCK); //: threads per chunk, chunks per block: 32,2

  lz4DecompressBatchKernel<<<grid, block, 0, stream>>>(
      device_compressed_ptrs,
      device_compressed_bytes,
      nullptr,
      batch_size,
      nullptr,
      device_uncompressed_bytes,
      nullptr,
      false);
  HipUtils::check_last_error("lz4DecompressBatchKernel()");
}

size_t lz4ComputeChunksInBatch(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size)
{
  size_t num_chunks = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    num_chunks += roundUpDiv(decomp_data_size[i], chunk_size);
  }

  return num_chunks;
}

size_t lz4BatchCompressComputeTempSize(
    const size_t max_chunk_size, const size_t batch_size)
{
  if (max_chunk_size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "Maximum chunk size for LZ4 is " + std::to_string(lz4MaxChunkSize()));
  }

  return (lz4GetHashTableSizeForBatch(max_chunk_size, batch_size) + ARCTO_LZ4_TABLE_PAD_ENTRIES) * sizeof(offset_type) * batch_size;
}

size_t lz4DecompressComputeTempSize(
    const size_t maxChunksInBatch, const size_t /* chunkSize */)
{
  const size_t header_size = sizeof(chunk_header) * maxChunksInBatch;

  return roundUpTo(header_size, sizeof(size_t));
}

size_t lz4ComputeMaxSize(const size_t size)
{
  if (size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "Maximum chunk size for LZ4 is " + std::to_string(lz4MaxChunkSize()));
  }
  return maxSizeOfStream(size);
}

size_t lz4MaxChunkSize()
{
  return MAX_CHUNK_SIZE;
}

} // namespace lowlevel
} // namespace arcto