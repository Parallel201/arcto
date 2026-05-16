/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef ARCTO_BENCHMARKS_BENCHMARK_TEMPLATE_CHUNKED_CUH
#define ARCTO_BENCHMARKS_BENCHMARK_TEMPLATE_CHUNKED_CUH

#include "benchmark_common.h"

#ifdef __HIP_PLATFORM_AMD__
#include "arcto.h"
#include "arcto/lz4.h"
#include "arcto/snappy.h"
#include "arcto/cascaded.h"
#include "arcto/gdeflate.h"
#include "arcto/bitcomp.h"
#include "arcto/ans.h"
#else
#include "nvcomp.h"
#include "nvcomp/lz4.h"
#include "nvcomp/snappy.h"
#include "nvcomp/cascaded.h"
#include "nvcomp/gdeflate.h"
#include "nvcomp/bitcomp.h"
#include "nvcomp/ans.h"
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cassert>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/system/hip/execution_policy.h>
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuSuccess hipSuccess
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuSetDevice hipSetDevice
// Note: arcto uses C API without namespaces
#else
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuSuccess cudaSuccess
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuSetDevice cudaSetDevice
// Note: nvcomp uses C API without namespaces
#endif

#define GENERATE_CHUNKED_BENCHMARK( \
    comp_get_temp, \
    comp_get_output, \
    comp_async, \
    decomp_get_temp, \
    decomp_async, \
    is_input_valid, \
    format_opts) \
void run_benchmark( \
    const std::vector<std::vector<char>>& data, \
    const bool warmup, \
    const size_t count, \
    const bool csv_output, \
    const bool tab_separator, \
    const size_t duplicate_count, \
    const size_t num_files) \
{ \
  run_benchmark_template( \
      comp_get_temp, \
      comp_get_output, \
      comp_async, \
      decomp_get_temp, \
      decomp_async, \
      is_input_valid, \
      format_opts, \
      data, \
      warmup, \
      count, \
      csv_output, \
      tab_separator, \
      duplicate_count, \
      num_files); \
}

// A helper function for if the input data requires no validation.
static bool inputAlwaysValid(const std::vector<std::vector<char>>& data)
{
  return true;
}

namespace
{

constexpr const char * const REQUIRED_PARAMTER = "_REQUIRED_";

static size_t compute_batch_size(
    const std::vector<std::vector<char>>& data, const size_t chunk_size)
{
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

std::vector<size_t> compute_chunk_sizes(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<size_t> sizes(batch_size, chunk_size);

  size_t offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset] = data[i].size() % chunk_size;
    }
    offset += num_chunks;
  }
  return sizes;
}

class BatchData
{
public:
  BatchData(
      const std::vector<std::vector<char>>& host_data) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(0)
  {
    m_size = host_data.size();

    // find max chunk size and build prefixsum
    std::vector<size_t> prefixsum(m_size+1,0);
    size_t chunk_size = 0;
    for (size_t i = 0; i < m_size; ++i) {
      if (chunk_size < host_data[i].size()) {
        chunk_size = host_data[i].size();
      }
      prefixsum[i+1] = prefixsum[i] + host_data[i].size();
    }

    m_data = thrust::device_vector<uint8_t>(prefixsum.back());

    std::vector<void*> uncompressed_ptrs(size());
    for (size_t i = 0; i < size(); ++i) {
      uncompressed_ptrs[i] = static_cast<void*>(data() + prefixsum[i]);
    }

    m_ptrs = thrust::device_vector<void*>(uncompressed_ptrs);
    std::vector<size_t> sizes(m_size);
    for (size_t i = 0; i < sizes.size(); ++i) {
      sizes[i] = host_data[i].size();
    }
    m_sizes = thrust::device_vector<size_t>(sizes);

    // copy data to GPU
    for (size_t i = 0; i < host_data.size(); ++i) {
      GPU_CHECK(gpuMemcpy(
          uncompressed_ptrs[i],
          host_data[i].data(),
          host_data[i].size(),
          gpuMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size) :
      m_ptrs(),
      m_sizes(),
      m_data(),
      m_size(batch_size)
  {
    m_data = thrust::device_vector<uint8_t>(max_output_size * size());

    std::vector<size_t> sizes(size(), max_output_size);
    m_sizes = thrust::device_vector<size_t>(sizes);

    std::vector<void*> ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      ptrs[i] = data() + max_output_size * i;
    }
    m_ptrs = thrust::device_vector<void*>(ptrs);
  }

  BatchData(BatchData&& other) = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  void** ptrs()
  {
    return m_ptrs.data().get();
  }

  size_t* sizes()
  {
    return m_sizes.data().get();
  }

  uint8_t* data()
  {
    return m_data.data().get();
  }

  size_t total_size() const
  {
    return m_data.size();
  }

  size_t size() const
  {
    return m_size;
  }

private:
  thrust::device_vector<void*> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

std::vector<char> readFile(const std::string& filename)
{
  std::ifstream fin(filename, std::ifstream::binary);
  if (!fin) {
    std::cerr << "ERROR: Unable to open \"" << filename << "\" for reading."
              << std::endl;
    throw std::runtime_error("Error opening file for reading.");
  }

  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  fin.seekg(0, std::ios_base::end);
  auto fileSize = static_cast<std::streamoff>(fin.tellg());
  fin.seekg(0, std::ios_base::beg);

  std::vector<char> host_data(fileSize);
  fin.read(host_data.data(), fileSize);

  if (!fin) {
    std::cerr << "ERROR: Unable to read all of file \"" << filename << "\"."
              << std::endl;
    throw std::runtime_error("Error reading file.");
  }

  return host_data;
}

std::vector<std::vector<char>> readFileWithPageSizes(const std::string& filename)
{
  std::vector<std::vector<char>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof()) {
    uint64_t chunk_size;
    fin.read((char *)(&chunk_size), sizeof(uint64_t));
    if (fin.eof())
      break;
    res.emplace_back(chunk_size);
    fin.read((char *)(res.back().data()), chunk_size);
  }

  return res;
}

std::vector<std::vector<char>>
multi_file(const std::vector<std::string>& filenames, const size_t chunk_size,
    const bool has_page_sizes, const size_t num_duplicates)
{
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    if (!has_page_sizes) {
      std::vector<char> filedata = readFile(filename);

      const size_t num_chunks
          = (filedata.size() + chunk_size - 1) / chunk_size;
      size_t offset = 0;
      for (size_t c = 0; c < num_chunks; ++c) {
        const size_t size_of_this_chunk = std::min(chunk_size, filedata.size()-offset);
        split_data.emplace_back(
            std::vector<char>(filedata.data() + offset,
                              filedata.data()+ offset + size_of_this_chunk));
        offset += size_of_this_chunk;
        assert(offset <= filedata.size());
      }
    } else {
      std::vector<std::vector<char>> filedata = readFileWithPageSizes(filename);
      split_data.insert(split_data.end(), filedata.begin(), filedata.end());
    }
  }

  const size_t num_chunks = split_data.size();
  for (size_t d = 0; d < num_duplicates; ++d) {
    split_data.insert(split_data.end(), split_data.begin(),
        split_data.begin()+num_chunks);
  }

  return split_data;
}
}

#ifdef __HIP_PLATFORM_AMD__
template<
    typename CompGetTempT,
    typename CompGetSizeT,
    typename CompAsyncT,
    typename DecompGetTempT,
    typename DecompAsyncT,
    typename IsInputValidT,
    typename FormatOptsT>
void
run_benchmark_template(
    CompGetTempT BatchedCompressGetTempSize,
    CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
    CompAsyncT BatchedCompressAsync,
    DecompGetTempT BatchedDecompressGetTempSize,
    DecompAsyncT BatchedDecompressAsync,
    IsInputValidT IsInputValid,
    const FormatOptsT format_opts,
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool use_tabs,
    const size_t duplicates,
    const size_t num_files)
{
  benchmark::benchmark_assert(IsInputValid(data), "Invalid input data");

  const std::string separator = use_tabs ? "\t" : ",";

  size_t total_bytes = 0;
  size_t chunk_size = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
    if (part.size() > chunk_size) {
      chunk_size = part.size();
    }
  }

  // build up metadata
  BatchData input_data(data);

  hipStream_t stream;
  GPU_CHECK(hipStreamCreate(&stream));

  const size_t batch_size = input_data.size();

  std::vector<size_t> h_input_sizes(batch_size);
  GPU_CHECK(hipMemcpy(h_input_sizes.data(), input_data.sizes(),
      sizeof(size_t)*batch_size, hipMemcpyDeviceToHost));

  size_t compressed_size = 0;
  double comp_time = 0.0;
  double decomp_time = 0.0;
  std::vector<double> comp_throughputs;
  std::vector<double> decomp_throughputs;
  std::vector<double> comp_times_ms;
  std::vector<double> decomp_times_ms;
  comp_throughputs.reserve(count);
  decomp_throughputs.reserve(count);
  comp_times_ms.reserve(count);
  decomp_times_ms.reserve(count);

  // Measure H2D transfer overhead (representative of input upload cost)
  float h2d_ms = 0.0f;
  if (!warmup && total_bytes > 0) {
    void* d_transfer_buf;
    GPU_CHECK(hipMalloc(&d_transfer_buf, total_bytes));
    uint8_t* dst = static_cast<uint8_t*>(d_transfer_buf);
    hipEvent_t ts, te;
    GPU_CHECK(hipEventCreate(&ts));
    GPU_CHECK(hipEventCreate(&te));
    GPU_CHECK(hipEventRecord(ts, stream));
    for (const auto& chunk : data) {
      GPU_CHECK(hipMemcpyAsync(dst, chunk.data(), chunk.size(),
          hipMemcpyHostToDevice, stream));
      dst += chunk.size();
    }
    GPU_CHECK(hipEventRecord(te, stream));
    GPU_CHECK(hipStreamSynchronize(stream));
    GPU_CHECK(hipEventElapsedTime(&h2d_ms, ts, te));
    GPU_CHECK(hipFree(d_transfer_buf));
    GPU_CHECK(hipEventDestroy(ts));
    GPU_CHECK(hipEventDestroy(te));
  }

  for (size_t iter = 0; iter < count; ++iter) {
    // compression
    arctoStatus_t status;

    // Compress on the GPU using batched API
    size_t comp_temp_bytes;
    status = BatchedCompressGetTempSize(
        batch_size, chunk_size, format_opts, &comp_temp_bytes);
    benchmark::benchmark_assert(status == arctoSuccess,
        "BatchedCompressGetTempSize() failed.");

    void* d_comp_temp;
    GPU_CHECK(hipMalloc(&d_comp_temp, comp_temp_bytes));

    size_t max_out_bytes;
    status = BatchedCompressGetMaxOutputChunkSize(
        chunk_size, format_opts, &max_out_bytes);
    benchmark::benchmark_assert(status == arctoSuccess,
        "BatchedGetMaxOutputChunkSize() failed.");

    BatchData compress_data(max_out_bytes, batch_size);

    hipEvent_t start, end;
    GPU_CHECK(hipEventCreate(&start));
    GPU_CHECK(hipEventCreate(&end));
    GPU_CHECK(hipEventRecord(start, stream));

    status = BatchedCompressAsync(
        input_data.ptrs(),
        input_data.sizes(),
        chunk_size,
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        compress_data.ptrs(),
        compress_data.sizes(),
        format_opts,
        stream);
    benchmark::benchmark_assert(status == arctoSuccess,
        "BatchedCompressAsync() failed.");

    GPU_CHECK(hipEventRecord(end, stream));
    GPU_CHECK(hipStreamSynchronize(stream));

    // free compression memory
    GPU_CHECK(hipFree(d_comp_temp));

    float compress_ms;
    GPU_CHECK(hipEventElapsedTime(&compress_ms, start, end));
    if (!warmup) {
      comp_throughputs.push_back((double)total_bytes / (1.0e9 * compress_ms * 1.0e-3));
      comp_times_ms.push_back(compress_ms);
    }

    // compute compression ratio
    std::vector<size_t> compressed_sizes_host(compress_data.size());
    GPU_CHECK(hipMemcpy(
        compressed_sizes_host.data(),
        compress_data.sizes(),
        compress_data.size() * sizeof(*compress_data.sizes()),
        hipMemcpyDeviceToHost));

    size_t comp_bytes = 0;
    for (const size_t s : compressed_sizes_host) {
      comp_bytes += s;
    }

    // Decompression
    size_t decomp_temp_bytes;
    status = BatchedDecompressGetTempSize(
        compress_data.size(), chunk_size, &decomp_temp_bytes);
    benchmark::benchmark_assert(status == arctoSuccess,
        "BatchedDecompressGetTempSize() failed.");

    void* d_decomp_temp;
    GPU_CHECK(hipMalloc(&d_decomp_temp, decomp_temp_bytes));

    size_t* d_decomp_sizes;
    GPU_CHECK(hipMalloc(
        (void**)&d_decomp_sizes, batch_size*sizeof(*d_decomp_sizes)));

    arctoStatus_t* d_decomp_statuses;
    GPU_CHECK(hipMalloc(
        (void**)&d_decomp_statuses, batch_size*sizeof(*d_decomp_statuses)));

    std::vector<void*> h_output_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_CHECK(hipMalloc((void**)&h_output_ptrs[i], h_input_sizes[i]));
    }
    void ** d_output_ptrs;
    GPU_CHECK(hipMalloc((void**)&d_output_ptrs,
        sizeof(*d_output_ptrs)*batch_size));
    GPU_CHECK(hipMemcpy(d_output_ptrs, h_output_ptrs.data(),
        sizeof(*d_output_ptrs)*batch_size, hipMemcpyHostToDevice));

    GPU_CHECK(hipEventRecord(start, stream));
    status = BatchedDecompressAsync(
        compress_data.ptrs(),
        compress_data.sizes(),
        input_data.sizes(),
        d_decomp_sizes,
        batch_size,
        d_decomp_temp,
        decomp_temp_bytes,
        d_output_ptrs,
        d_decomp_statuses,
        stream);
    benchmark::benchmark_assert(
        status == arctoSuccess,
        "BatchedDecompressAsync() not successful");

    GPU_CHECK(hipEventRecord(end, stream));
    GPU_CHECK(hipStreamSynchronize(stream));

    float decompress_ms;
    GPU_CHECK(hipEventElapsedTime(&decompress_ms, start, end));
    GPU_CHECK(hipEventDestroy(start));
    GPU_CHECK(hipEventDestroy(end));
    if (!warmup) {
      decomp_throughputs.push_back((double)total_bytes / (1.0e9 * decompress_ms * 1.0e-3));
      decomp_times_ms.push_back(decompress_ms);
    }

    GPU_CHECK(hipFree(d_output_ptrs));

    // verify success each time
    std::vector<size_t> h_decomp_sizes(batch_size);
    GPU_CHECK(hipMemcpy(h_decomp_sizes.data(), d_decomp_sizes,
      sizeof(*d_decomp_sizes)*batch_size, hipMemcpyDeviceToHost));

    std::vector<arctoStatus_t> h_decomp_statuses(batch_size);
    GPU_CHECK(hipMemcpy(h_decomp_statuses.data(), d_decomp_statuses,
      sizeof(*d_decomp_statuses)*batch_size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < batch_size; ++i) {
      benchmark::benchmark_assert(h_decomp_statuses[i] == arctoSuccess, "Batch item not successfuly decompressed: i=" + std::to_string(i) + ": status=" +
      std::to_string(h_decomp_statuses[i]));
      benchmark::benchmark_assert(h_decomp_sizes[i] == h_input_sizes[i], "Batch item of wrong size: i=" + std::to_string(i) + ": act_size=" +
      std::to_string(h_decomp_sizes[i]) + " exp_size=" +
      std::to_string(h_input_sizes[i]));
    }

    GPU_CHECK(hipFree(d_decomp_temp));
    GPU_CHECK(hipFree(d_decomp_sizes));
    GPU_CHECK(hipFree(d_decomp_statuses));

    // only verify last iteration
    if (iter + 1 == count) {
      std::vector<void*> h_input_ptrs(batch_size);
      GPU_CHECK(hipMemcpy(h_input_ptrs.data(), input_data.ptrs(),
          sizeof(void*)*batch_size, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < batch_size; ++i) {
        std::vector<uint8_t> exp_data(h_input_sizes[i]);
        GPU_CHECK(hipMemcpy(exp_data.data(), h_input_ptrs[i],
            h_input_sizes[i], hipMemcpyDeviceToHost));
        std::vector<uint8_t> act_data(h_decomp_sizes[i]);
        GPU_CHECK(hipMemcpy(act_data.data(), h_output_ptrs[i],
            h_decomp_sizes[i], hipMemcpyDeviceToHost));
        for (size_t j = 0; j < h_input_sizes[i]; ++j) {
          if (act_data[j] != exp_data[j]) {
            benchmark::benchmark_assert(false, "Batch item decompressed output did not match input: i="+std::to_string(i) + ": j=" + std::to_string(j) + " act=" + std::to_string(act_data[j]) + " exp=" +
            std::to_string(exp_data[j]));
          }
        }
      }
    }

    for (size_t i = 0; i < batch_size; ++i) {
      GPU_CHECK(hipFree(h_output_ptrs[i]));
    }

    // count everything from our iteration
    compressed_size += comp_bytes;
    comp_time += compress_ms * 1.0e-3;
    decomp_time += decompress_ms * 1.0e-3;
  }
  GPU_CHECK(hipStreamDestroy(stream));

  // Save accumulated values before averaging
  const size_t total_compressed_accumulated = compressed_size;
  const double total_comp_time_s = comp_time;

  // average iterations
  compressed_size /= count;
  comp_time /= count;
  decomp_time /= count;

  if (!warmup) {
    const double comp_ratio = (double)total_bytes / compressed_size;
    const double compression_throughput_gbs = (double)total_bytes / (1.0e9 * comp_time);
    const double decompression_throughput_gbs = (double)total_bytes / (1.0e9 * decomp_time);
    const double effective_bw_gbs = compression_throughput_gbs / comp_ratio;
    const double space_saved_bytes = (double)(total_bytes - compressed_size);
    const double space_saved_pct = (total_bytes > 0) ? (100.0 * space_saved_bytes / total_bytes) : 0.0;
    const float d2h_ms = (total_bytes > 0)
        ? static_cast<float>(h2d_ms * (double)compressed_size / (double)total_bytes)
        : 0.0f;
    const double transfer_ms = h2d_ms + d2h_ms;
    const double comp_time_ms = comp_time * 1.0e3;
    const double decomp_time_ms = decomp_time * 1.0e3;
    const double comp_total_ms = comp_time_ms + transfer_ms;
    const double decomp_total_ms = decomp_time_ms + transfer_ms;
    const double comp_pct = (comp_total_ms > 0) ? (100.0 * comp_time_ms / comp_total_ms) : 100.0;
    const double decomp_pct = (decomp_total_ms > 0) ? (100.0 * decomp_time_ms / decomp_total_ms) : 100.0;
    const double avg_chunk_time_ms = (batch_size > 0) ? (comp_time_ms / batch_size) : 0.0;
    const double chunks_per_second = (comp_time > 0) ? ((double)batch_size / comp_time) : 0.0;

    if (!csv_output) {
      const int LW = 26;
      auto lbl = [&](const std::string& s) -> std::string {
        return "  " + s + std::string(std::max(0, LW - (int)s.size()), ' ');
      };
      std::cout << std::fixed;

      std::cout << "Compression:" << std::endl;
      std::cout << lbl("Throughput:")
                << std::setprecision(2) << compression_throughput_gbs << " GB/s" << std::endl;
      std::cout << lbl("Effective Bandwidth:")
                << std::setprecision(2) << effective_bw_gbs << " GB/s" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(1) << comp_total_ms << " ms" << std::endl;
      std::cout << lbl("Compression Time:")
                << std::setprecision(1) << comp_time_ms
                << " ms (" << std::setprecision(1) << comp_pct << "%)" << std::endl;
      std::cout << lbl("Transfer Time:")
                << std::setprecision(1) << transfer_ms
                << " ms (" << std::setprecision(1) << (100.0 - comp_pct) << "%)" << std::endl;
      std::cout << lbl("Compression Ratio:")
                << std::setprecision(2) << comp_ratio << "x" << std::endl;
      std::cout << lbl("Space Saved:")
                << std::setprecision(2) << (space_saved_bytes * 1.0e-6)
                << " MB (" << std::setprecision(1) << space_saved_pct << "%)" << std::endl;
      std::cout << std::endl;

      std::cout << "Decompression:" << std::endl;
      std::cout << lbl("Throughput:")
                << std::setprecision(2) << decompression_throughput_gbs << " GB/s" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(1) << decomp_total_ms << " ms" << std::endl;
      std::cout << lbl("Decompression Time:")
                << std::setprecision(1) << decomp_time_ms
                << " ms (" << std::setprecision(1) << decomp_pct << "%)" << std::endl;
      std::cout << lbl("Transfer Time:")
                << std::setprecision(1) << transfer_ms
                << " ms (" << std::setprecision(1) << (100.0 - decomp_pct) << "%)" << std::endl;
      std::cout << std::endl;

      std::cout << "Chunk Statistics:" << std::endl;
      std::cout << lbl("Number of Chunks:") << batch_size << std::endl;
      std::cout << lbl("Chunk Size:")
                << std::setprecision(0) << (chunk_size / 1024.0) << " KB" << std::endl;
      std::cout << lbl("Avg Time per Chunk:")
                << std::setprecision(3) << avg_chunk_time_ms << " ms" << std::endl;
      std::cout << lbl("Chunks per Second:")
                << std::setprecision(0) << chunks_per_second << std::endl;
      std::cout << std::endl;

      const double total_input_gb = (double)total_bytes * count * 1.0e-9;
      const double total_output_gb = (double)total_compressed_accumulated * 1.0e-9;
      std::cout << "Accumulated Statistics (" << count << " iteration"
                << (count != 1 ? "s" : "") << "):" << std::endl;
      std::cout << lbl("Total Data Processed:")
                << std::setprecision(2) << total_input_gb
                << " GB -> " << total_output_gb << " GB" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(2) << total_comp_time_s << " s" << std::endl;
      std::cout << lbl("Avg Compression Ratio:")
                << std::setprecision(2) << comp_ratio << "x" << std::endl;
      auto stddev = [](const std::vector<double>& v) -> double {
        if (v.size() < 2) return 0.0;
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq = 0.0;
        for (double x : v) sq += (x - mean) * (x - mean);
        return std::sqrt(sq / v.size());
      };
      if (count > 1 && !comp_throughputs.empty()) {
        const double tmin = *std::min_element(comp_throughputs.begin(), comp_throughputs.end());
        const double tmax = *std::max_element(comp_throughputs.begin(), comp_throughputs.end());
        const double comp_std  = stddev(comp_throughputs);
        const double decomp_std = stddev(decomp_throughputs);
        const double comp_time_std = stddev(comp_times_ms);
        const double decomp_time_std = stddev(decomp_times_ms);
        std::cout << lbl("Throughput Range:")
                  << std::setprecision(2) << tmin << " - " << tmax << " GB/s" << std::endl;
        std::cout << lbl("Comp Throughput StdDev:")
                  << std::setprecision(2) << comp_std << " GB/s" << std::endl;
        std::cout << lbl("Decomp Throughput StdDev:")
                  << std::setprecision(2) << decomp_std << " GB/s" << std::endl;
        std::cout << lbl("Comp Time StdDev:")
                  << std::setprecision(3) << comp_time_std << " ms" << std::endl;
        std::cout << lbl("Decomp Time StdDev:")
                  << std::setprecision(3) << decomp_time_std << " ms" << std::endl;
      }
      std::cout << lbl("Avg Throughput:")
                << std::setprecision(2) << compression_throughput_gbs << " GB/s" << std::endl;
      std::cout << std::endl;
    } else {
      // CSV header
      std::cout << "Files";
      std::cout << separator << "Duplicate data";
      std::cout << separator << "Size in MB";
      std::cout << separator << "Pages";
      std::cout << separator << "Avg page size in KB";
      std::cout << separator << "Max page size in KB";
      std::cout << separator << "Ucompressed size in bytes";
      std::cout << separator << "Compressed size in bytes";
      std::cout << separator << "Compression ratio";
      std::cout << separator << "Compression throughput (uncompressed) in GB/s";
      std::cout << separator << "Decompression throughput (uncompressed) in GB/s";
      std::cout << separator << "Compression time (ms)";
      std::cout << separator << "Decompression time (ms)";
      std::cout << separator << "Transfer H2D (ms)";
      std::cout << separator << "Transfer D2H (ms)";
      std::cout << separator << "Total time (ms)";
      std::cout << separator << "Avg chunk time (ms)";
      std::cout << separator << "Comp throughput stddev (GB/s)";
      std::cout << separator << "Decomp throughput stddev (GB/s)";
      std::cout << separator << "Comp time stddev (ms)";
      std::cout << separator << "Decomp time stddev (ms)";
      std::cout << std::endl;

      // CSV values
      std::cout << num_files;
      std::cout << separator << duplicates;
      std::cout << separator << std::fixed << std::setprecision(6) << (total_bytes * 1e-6);
      std::cout << separator << data.size();
      std::cout << separator << ((1e-3 * total_bytes) / data.size());
      std::cout << separator << (1e-3 * chunk_size);
      std::cout << separator << total_bytes;
      std::cout << separator << compressed_size;
      std::cout << separator << std::setprecision(2) << comp_ratio;
      std::cout << separator << compression_throughput_gbs;
      std::cout << separator << decompression_throughput_gbs;
      std::cout << separator << std::setprecision(3) << comp_time_ms;
      std::cout << separator << decomp_time_ms;
      std::cout << separator << h2d_ms;
      std::cout << separator << d2h_ms;
      std::cout << separator << comp_total_ms;
      std::cout << separator << std::setprecision(6) << avg_chunk_time_ms;
      {
        auto stddev = [](const std::vector<double>& v) -> double {
          if (v.size() < 2) return 0.0;
          const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
          double sq = 0.0;
          for (double x : v) sq += (x - mean) * (x - mean);
          return std::sqrt(sq / v.size());
        };
        std::cout << separator << std::setprecision(4) << stddev(comp_throughputs);
        std::cout << separator << stddev(decomp_throughputs);
        std::cout << separator << stddev(comp_times_ms);
        std::cout << separator << stddev(decomp_times_ms);
      }
      std::cout << std::endl;
    }
  }
}
#else
// CUDA version
template<
    typename CompGetTempT,
    typename CompGetSizeT,
    typename CompAsyncT,
    typename DecompGetTempT,
    typename DecompAsyncT,
    typename IsInputValidT,
    typename FormatOptsT>
void
run_benchmark_template(
    CompGetTempT BatchedCompressGetTempSize,
    CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
    CompAsyncT BatchedCompressAsync,
    DecompGetTempT BatchedDecompressGetTempSize,
    DecompAsyncT BatchedDecompressAsync,
    IsInputValidT IsInputValid,
    const FormatOptsT format_opts,
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool use_tabs,
    const size_t duplicates,
    const size_t num_files)
{
  benchmark::benchmark_assert(IsInputValid(data), "Invalid input data");

  const std::string separator = use_tabs ? "\t" : ",";

  size_t total_bytes = 0;
  size_t chunk_size = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
    if (part.size() > chunk_size) {
      chunk_size = part.size();
    }
  }

  // build up metadata
  BatchData input_data(data);

  cudaStream_t stream;
  GPU_CHECK(cudaStreamCreate(&stream));

  const size_t batch_size = input_data.size();

  std::vector<size_t> h_input_sizes(batch_size);
  GPU_CHECK(cudaMemcpy(h_input_sizes.data(), input_data.sizes(),
      sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost));

  size_t compressed_size = 0;
  double comp_time = 0.0;
  double decomp_time = 0.0;
  std::vector<double> comp_throughputs;
  std::vector<double> decomp_throughputs;
  std::vector<double> comp_times_ms;
  std::vector<double> decomp_times_ms;
  comp_throughputs.reserve(count);
  decomp_throughputs.reserve(count);
  comp_times_ms.reserve(count);
  decomp_times_ms.reserve(count);

  // Measure H2D transfer overhead (representative of input upload cost)
  float h2d_ms = 0.0f;
  if (!warmup && total_bytes > 0) {
    void* d_transfer_buf;
    GPU_CHECK(cudaMalloc(&d_transfer_buf, total_bytes));
    uint8_t* dst = static_cast<uint8_t*>(d_transfer_buf);
    cudaEvent_t ts, te;
    GPU_CHECK(cudaEventCreate(&ts));
    GPU_CHECK(cudaEventCreate(&te));
    GPU_CHECK(cudaEventRecord(ts, stream));
    for (const auto& chunk : data) {
      GPU_CHECK(cudaMemcpyAsync(dst, chunk.data(), chunk.size(),
          cudaMemcpyHostToDevice, stream));
      dst += chunk.size();
    }
    GPU_CHECK(cudaEventRecord(te, stream));
    GPU_CHECK(cudaStreamSynchronize(stream));
    GPU_CHECK(cudaEventElapsedTime(&h2d_ms, ts, te));
    GPU_CHECK(cudaFree(d_transfer_buf));
    GPU_CHECK(cudaEventDestroy(ts));
    GPU_CHECK(cudaEventDestroy(te));
  }

  for (size_t iter = 0; iter < count; ++iter) {
    // compression
    nvcompStatus_t status;

    // Compress on the GPU using batched API
    size_t comp_temp_bytes;
    status = BatchedCompressGetTempSize(
        batch_size, chunk_size, format_opts, &comp_temp_bytes);
    benchmark::benchmark_assert(status == nvcompSuccess,
        "BatchedCompressGetTempSize() failed.");

    void* d_comp_temp;
    GPU_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    size_t max_out_bytes;
    status = BatchedCompressGetMaxOutputChunkSize(
        chunk_size, format_opts, &max_out_bytes);
    benchmark::benchmark_assert(status == nvcompSuccess,
        "BatchedGetMaxOutputChunkSize() failed.");

    BatchData compress_data(max_out_bytes, batch_size);

    cudaEvent_t start, end;
    GPU_CHECK(cudaEventCreate(&start));
    GPU_CHECK(cudaEventCreate(&end));
    GPU_CHECK(cudaEventRecord(start, stream));

    status = BatchedCompressAsync(
        input_data.ptrs(),
        input_data.sizes(),
        chunk_size,
        batch_size,
        d_comp_temp,
        comp_temp_bytes,
        compress_data.ptrs(),
        compress_data.sizes(),
        format_opts,
        stream);
    benchmark::benchmark_assert(status == nvcompSuccess,
        "BatchedCompressAsync() failed.");

    GPU_CHECK(cudaEventRecord(end, stream));
    GPU_CHECK(cudaStreamSynchronize(stream));

    // free compression memory
    GPU_CHECK(cudaFree(d_comp_temp));

    float compress_ms;
    GPU_CHECK(cudaEventElapsedTime(&compress_ms, start, end));
    if (!warmup) {
      comp_throughputs.push_back((double)total_bytes / (1.0e9 * compress_ms * 1.0e-3));
      comp_times_ms.push_back(compress_ms);
    }

    // compute compression ratio
    std::vector<size_t> compressed_sizes_host(compress_data.size());
    GPU_CHECK(cudaMemcpy(
        compressed_sizes_host.data(),
        compress_data.sizes(),
        compress_data.size() * sizeof(*compress_data.sizes()),
        cudaMemcpyDeviceToHost));

    size_t comp_bytes = 0;
    for (const size_t s : compressed_sizes_host) {
      comp_bytes += s;
    }

    // Decompression
    size_t decomp_temp_bytes;
    status = BatchedDecompressGetTempSize(
        compress_data.size(), chunk_size, &decomp_temp_bytes);
    benchmark::benchmark_assert(status == nvcompSuccess,
        "BatchedDecompressGetTempSize() failed.");

    void* d_decomp_temp;
    GPU_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

    size_t* d_decomp_sizes;
    GPU_CHECK(cudaMalloc(
        (void**)&d_decomp_sizes, batch_size*sizeof(*d_decomp_sizes)));

    nvcompStatus_t* d_decomp_statuses;
    GPU_CHECK(cudaMalloc(
        (void**)&d_decomp_statuses, batch_size*sizeof(*d_decomp_statuses)));

    std::vector<void*> h_output_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      GPU_CHECK(cudaMalloc((void**)&h_output_ptrs[i], h_input_sizes[i]));
    }
    void ** d_output_ptrs;
    GPU_CHECK(cudaMalloc((void**)&d_output_ptrs,
        sizeof(*d_output_ptrs)*batch_size));
    GPU_CHECK(cudaMemcpy(d_output_ptrs, h_output_ptrs.data(),
        sizeof(*d_output_ptrs)*batch_size, cudaMemcpyHostToDevice));

    GPU_CHECK(cudaEventRecord(start, stream));
    status = BatchedDecompressAsync(
        compress_data.ptrs(),
        compress_data.sizes(),
        input_data.sizes(),
        d_decomp_sizes,
        batch_size,
        d_decomp_temp,
        decomp_temp_bytes,
        d_output_ptrs,
        d_decomp_statuses,
        stream);
    benchmark::benchmark_assert(
        status == nvcompSuccess,
        "BatchedDecompressAsync() not successful");

    GPU_CHECK(cudaEventRecord(end, stream));
    GPU_CHECK(cudaStreamSynchronize(stream));

    float decompress_ms;
    GPU_CHECK(cudaEventElapsedTime(&decompress_ms, start, end));
    GPU_CHECK(cudaEventDestroy(start));
    GPU_CHECK(cudaEventDestroy(end));
    if (!warmup) {
      decomp_throughputs.push_back((double)total_bytes / (1.0e9 * decompress_ms * 1.0e-3));
      decomp_times_ms.push_back(decompress_ms);
    }

    GPU_CHECK(cudaFree(d_output_ptrs));

    // verify success each time
    std::vector<size_t> h_decomp_sizes(batch_size);
    GPU_CHECK(cudaMemcpy(h_decomp_sizes.data(), d_decomp_sizes,
      sizeof(*d_decomp_sizes)*batch_size, cudaMemcpyDeviceToHost));

    std::vector<nvcompStatus_t> h_decomp_statuses(batch_size);
    GPU_CHECK(cudaMemcpy(h_decomp_statuses.data(), d_decomp_statuses,
      sizeof(*d_decomp_statuses)*batch_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < batch_size; ++i) {
      benchmark::benchmark_assert(h_decomp_statuses[i] == nvcompSuccess, "Batch item not successfuly decompressed: i=" + std::to_string(i) + ": status=" +
      std::to_string(h_decomp_statuses[i]));
      benchmark::benchmark_assert(h_decomp_sizes[i] == h_input_sizes[i], "Batch item of wrong size: i=" + std::to_string(i) + ": act_size=" +
      std::to_string(h_decomp_sizes[i]) + " exp_size=" +
      std::to_string(h_input_sizes[i]));
    }

    GPU_CHECK(cudaFree(d_decomp_temp));
    GPU_CHECK(cudaFree(d_decomp_sizes));
    GPU_CHECK(cudaFree(d_decomp_statuses));

    // only verify last iteration
    if (iter + 1 == count) {
      std::vector<void*> h_input_ptrs(batch_size);
      GPU_CHECK(cudaMemcpy(h_input_ptrs.data(), input_data.ptrs(),
          sizeof(void*)*batch_size, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < batch_size; ++i) {
        std::vector<uint8_t> exp_data(h_input_sizes[i]);
        GPU_CHECK(cudaMemcpy(exp_data.data(), h_input_ptrs[i],
            h_input_sizes[i], cudaMemcpyDeviceToHost));
        std::vector<uint8_t> act_data(h_decomp_sizes[i]);
        GPU_CHECK(cudaMemcpy(act_data.data(), h_output_ptrs[i],
            h_decomp_sizes[i], cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < h_input_sizes[i]; ++j) {
          if (act_data[j] != exp_data[j]) {
            benchmark::benchmark_assert(false, "Batch item decompressed output did not match input: i="+std::to_string(i) + ": j=" + std::to_string(j) + " act=" + std::to_string(act_data[j]) + " exp=" +
            std::to_string(exp_data[j]));
          }
        }
      }
    }

    for (size_t i = 0; i < batch_size; ++i) {
      GPU_CHECK(cudaFree(h_output_ptrs[i]));
    }

    // count everything from our iteration
    compressed_size += comp_bytes;
    comp_time += compress_ms * 1.0e-3;
    decomp_time += decompress_ms * 1.0e-3;
  }
  GPU_CHECK(cudaStreamDestroy(stream));

  // Save accumulated values before averaging
  const size_t total_compressed_accumulated = compressed_size;
  const double total_comp_time_s = comp_time;

  // average iterations
  compressed_size /= count;
  comp_time /= count;
  decomp_time /= count;

  if (!warmup) {
    const double comp_ratio = (double)total_bytes / compressed_size;
    const double compression_throughput_gbs = (double)total_bytes / (1.0e9 * comp_time);
    const double decompression_throughput_gbs = (double)total_bytes / (1.0e9 * decomp_time);
    const double effective_bw_gbs = compression_throughput_gbs / comp_ratio;
    const double space_saved_bytes = (double)(total_bytes - compressed_size);
    const double space_saved_pct = (total_bytes > 0) ? (100.0 * space_saved_bytes / total_bytes) : 0.0;
    const float d2h_ms = (total_bytes > 0)
        ? static_cast<float>(h2d_ms * (double)compressed_size / (double)total_bytes)
        : 0.0f;
    const double transfer_ms = h2d_ms + d2h_ms;
    const double comp_time_ms = comp_time * 1.0e3;
    const double decomp_time_ms = decomp_time * 1.0e3;
    const double comp_total_ms = comp_time_ms + transfer_ms;
    const double decomp_total_ms = decomp_time_ms + transfer_ms;
    const double comp_pct = (comp_total_ms > 0) ? (100.0 * comp_time_ms / comp_total_ms) : 100.0;
    const double decomp_pct = (decomp_total_ms > 0) ? (100.0 * decomp_time_ms / decomp_total_ms) : 100.0;
    const double avg_chunk_time_ms = (batch_size > 0) ? (comp_time_ms / batch_size) : 0.0;
    const double chunks_per_second = (comp_time > 0) ? ((double)batch_size / comp_time) : 0.0;

    if (!csv_output) {
      const int LW = 26;
      auto lbl = [&](const std::string& s) -> std::string {
        return "  " + s + std::string(std::max(0, LW - (int)s.size()), ' ');
      };
      std::cout << std::fixed;

      std::cout << "Compression:" << std::endl;
      std::cout << lbl("Throughput:")
                << std::setprecision(2) << compression_throughput_gbs << " GB/s" << std::endl;
      std::cout << lbl("Effective Bandwidth:")
                << std::setprecision(2) << effective_bw_gbs << " GB/s" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(1) << comp_total_ms << " ms" << std::endl;
      std::cout << lbl("Compression Time:")
                << std::setprecision(1) << comp_time_ms
                << " ms (" << std::setprecision(1) << comp_pct << "%)" << std::endl;
      std::cout << lbl("Transfer Time:")
                << std::setprecision(1) << transfer_ms
                << " ms (" << std::setprecision(1) << (100.0 - comp_pct) << "%)" << std::endl;
      std::cout << lbl("Compression Ratio:")
                << std::setprecision(2) << comp_ratio << "x" << std::endl;
      std::cout << lbl("Space Saved:")
                << std::setprecision(2) << (space_saved_bytes * 1.0e-6)
                << " MB (" << std::setprecision(1) << space_saved_pct << "%)" << std::endl;
      std::cout << std::endl;

      std::cout << "Decompression:" << std::endl;
      std::cout << lbl("Throughput:")
                << std::setprecision(2) << decompression_throughput_gbs << " GB/s" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(1) << decomp_total_ms << " ms" << std::endl;
      std::cout << lbl("Decompression Time:")
                << std::setprecision(1) << decomp_time_ms
                << " ms (" << std::setprecision(1) << decomp_pct << "%)" << std::endl;
      std::cout << lbl("Transfer Time:")
                << std::setprecision(1) << transfer_ms
                << " ms (" << std::setprecision(1) << (100.0 - decomp_pct) << "%)" << std::endl;
      std::cout << std::endl;

      std::cout << "Chunk Statistics:" << std::endl;
      std::cout << lbl("Number of Chunks:") << batch_size << std::endl;
      std::cout << lbl("Chunk Size:")
                << std::setprecision(0) << (chunk_size / 1024.0) << " KB" << std::endl;
      std::cout << lbl("Avg Time per Chunk:")
                << std::setprecision(3) << avg_chunk_time_ms << " ms" << std::endl;
      std::cout << lbl("Chunks per Second:")
                << std::setprecision(0) << chunks_per_second << std::endl;
      std::cout << std::endl;

      const double total_input_gb = (double)total_bytes * count * 1.0e-9;
      const double total_output_gb = (double)total_compressed_accumulated * 1.0e-9;
      std::cout << "Accumulated Statistics (" << count << " iteration"
                << (count != 1 ? "s" : "") << "):" << std::endl;
      std::cout << lbl("Total Data Processed:")
                << std::setprecision(2) << total_input_gb
                << " GB -> " << total_output_gb << " GB" << std::endl;
      std::cout << lbl("Total Time:")
                << std::setprecision(2) << total_comp_time_s << " s" << std::endl;
      std::cout << lbl("Avg Compression Ratio:")
                << std::setprecision(2) << comp_ratio << "x" << std::endl;
      auto stddev = [](const std::vector<double>& v) -> double {
        if (v.size() < 2) return 0.0;
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq = 0.0;
        for (double x : v) sq += (x - mean) * (x - mean);
        return std::sqrt(sq / v.size());
      };
      if (count > 1 && !comp_throughputs.empty()) {
        const double tmin = *std::min_element(comp_throughputs.begin(), comp_throughputs.end());
        const double tmax = *std::max_element(comp_throughputs.begin(), comp_throughputs.end());
        const double comp_std  = stddev(comp_throughputs);
        const double decomp_std = stddev(decomp_throughputs);
        const double comp_time_std = stddev(comp_times_ms);
        const double decomp_time_std = stddev(decomp_times_ms);
        std::cout << lbl("Throughput Range:")
                  << std::setprecision(2) << tmin << " - " << tmax << " GB/s" << std::endl;
        std::cout << lbl("Comp Throughput StdDev:")
                  << std::setprecision(2) << comp_std << " GB/s" << std::endl;
        std::cout << lbl("Decomp Throughput StdDev:")
                  << std::setprecision(2) << decomp_std << " GB/s" << std::endl;
        std::cout << lbl("Comp Time StdDev:")
                  << std::setprecision(3) << comp_time_std << " ms" << std::endl;
        std::cout << lbl("Decomp Time StdDev:")
                  << std::setprecision(3) << decomp_time_std << " ms" << std::endl;
      }
      std::cout << lbl("Avg Throughput:")
                << std::setprecision(2) << compression_throughput_gbs << " GB/s" << std::endl;
      std::cout << std::endl;
    } else {
      // CSV header
      std::cout << "Files";
      std::cout << separator << "Duplicate data";
      std::cout << separator << "Size in MB";
      std::cout << separator << "Pages";
      std::cout << separator << "Avg page size in KB";
      std::cout << separator << "Max page size in KB";
      std::cout << separator << "Ucompressed size in bytes";
      std::cout << separator << "Compressed size in bytes";
      std::cout << separator << "Compression ratio";
      std::cout << separator << "Compression throughput (uncompressed) in GB/s";
      std::cout << separator << "Decompression throughput (uncompressed) in GB/s";
      std::cout << separator << "Compression time (ms)";
      std::cout << separator << "Decompression time (ms)";
      std::cout << separator << "Transfer H2D (ms)";
      std::cout << separator << "Transfer D2H (ms)";
      std::cout << separator << "Total time (ms)";
      std::cout << separator << "Avg chunk time (ms)";
      std::cout << separator << "Comp throughput stddev (GB/s)";
      std::cout << separator << "Decomp throughput stddev (GB/s)";
      std::cout << separator << "Comp time stddev (ms)";
      std::cout << separator << "Decomp time stddev (ms)";
      std::cout << std::endl;

      // CSV values
      std::cout << num_files;
      std::cout << separator << duplicates;
      std::cout << separator << std::fixed << std::setprecision(6) << (total_bytes * 1e-6);
      std::cout << separator << data.size();
      std::cout << separator << ((1e-3 * total_bytes) / data.size());
      std::cout << separator << (1e-3 * chunk_size);
      std::cout << separator << total_bytes;
      std::cout << separator << compressed_size;
      std::cout << separator << std::setprecision(2) << comp_ratio;
      std::cout << separator << compression_throughput_gbs;
      std::cout << separator << decompression_throughput_gbs;
      std::cout << separator << std::setprecision(3) << comp_time_ms;
      std::cout << separator << decomp_time_ms;
      std::cout << separator << h2d_ms;
      std::cout << separator << d2h_ms;
      std::cout << separator << comp_total_ms;
      std::cout << separator << std::setprecision(6) << avg_chunk_time_ms;
      {
        auto stddev = [](const std::vector<double>& v) -> double {
          if (v.size() < 2) return 0.0;
          const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
          double sq = 0.0;
          for (double x : v) sq += (x - mean) * (x - mean);
          return std::sqrt(sq / v.size());
        };
        std::cout << separator << std::setprecision(4) << stddev(comp_throughputs);
        std::cout << separator << stddev(decomp_throughputs);
        std::cout << separator << stddev(comp_times_ms);
        std::cout << separator << stddev(decomp_times_ms);
      }
      std::cout << std::endl;
    }
  }
}
#endif

void run_benchmark(
    const std::vector<std::vector<char>>& data,
    const bool warmup,
    const size_t count,
    const bool csv_output,
    const bool tab_separator,
    const size_t duplicate_count,
    const size_t num_files);

struct args_type {
  int gpu;
  std::vector<std::string> filenames;
  size_t warmup_count;
  size_t iteration_count;
  size_t duplicate_count;
  bool csv_output;
  bool use_tabs;
  bool has_page_sizes;
  size_t chunk_size;
};

struct parameter_type {
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::string default_value;
};

bool parse_bool(const std::string& val)
{
  std::istringstream ss(val);
  std::boolalpha(ss);
  bool x;
  if (!(ss >> x)) {
    std::cerr << "ERROR: Invalid boolean: '" << val << "', only 'true' and 'false' are accepted." << std::endl;
    std::exit(1);
  }
  return x;
}

void usage(const std::string& name, const std::vector<parameter_type>& parameters)
{
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;
  for (const parameter_type& parameter : parameters) {
    std::cout << "  -" << parameter.short_flag << ",--" << parameter.long_flag;
    std::cout << "  : " << parameter.description << std::endl;
    if (parameter.default_value.empty()) {
      // no default value
    } else if (parameter.default_value == REQUIRED_PARAMTER) {
      std::cout << "    required" << std::endl;
    } else {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  }
}

std::string bool_to_string(const bool b) {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

args_type parse_args(int argc, char ** argv) {
  args_type args;
  args.gpu = 0;
  args.warmup_count = 1;
  args.iteration_count = 1;
  args.duplicate_count = 0;
  args.csv_output = false;
  args.use_tabs = false;
  args.has_page_sizes = false;
  args.chunk_size = 65536;

  const std::vector<parameter_type> params{
    {"h", "help", "Show options.", ""},
    {"g", "gpu", "GPU device number", std::to_string(args.gpu)},
    {"f", "input_file", "The list of inputs files. All files must start "
        "with a character other than '-'", "_required_"},
    {"w", "warmup_count", "The number of warmup iterations to perform.",
        std::to_string(args.warmup_count)},
    {"i", "iteration_count", "The number of runs to average.",
        std::to_string(args.iteration_count)},
    {"x", "duplicate_data", "CLone uncompressed chunks multiple times.",
        std::to_string(args.duplicate_count)},
    {"c", "csv_output", "Output in column/csv format.",
        bool_to_string(args.csv_output)},
    {"t", "tab_separator", "Use tabs instead of commas when "
        "'--csv_output' is specificed.",
        bool_to_string(args.use_tabs)},
    {"w", "file_with_page_sizes", "File(s) contain pages, each prefix "
        "with int64 size.", bool_to_string(args.has_page_sizes)},
    {"p", "chunk_size", "Chunk size when splitting uncompressed data.",
        std::to_string(args.chunk_size)},
  };

  char** argv_end = argv + argc;
  const std::string name(argv[0]);
  argv += 1;

  while (argv != argv_end) {
    std::string arg(*(argv++));
    bool found = false;
    for (const parameter_type& param : params) {
      if (arg == "-" + param.short_flag || arg == "--" + param.long_flag) {
        found = true;

        // found the parameter
        if (param.long_flag == "help") {
          usage(name, params);
          std::exit(0);
        }

        // everything from here on out requires an extra parameter
        if (argv >= argv_end) {
          std::cerr << "ERROR: Missing argument" << std::endl;
          usage(name, params);
          std::exit(1);
        }

        if (param.long_flag == "gpu") {
          args.gpu = std::stol(*(argv++));
          break;
        } else if (param.long_flag == "input_file") {
          // read all following arguments until a new flag is found
          char ** next_argv_ptr = argv;
          while (next_argv_ptr < argv_end && (*next_argv_ptr)[0] != '-') {
            args.filenames.emplace_back(*next_argv_ptr);
            next_argv_ptr = ++argv;
          }
          break;
        } else if (param.long_flag == "warmup_count") {
          args.warmup_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "iteration_count") {
          args.iteration_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "duplicate_data") {
          args.duplicate_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "csv_output") {
          std::string on(*(argv++));
          args.csv_output = parse_bool(on);
          break;
        } else if (param.long_flag == "tab_separator") {
          std::string on(*(argv++));
          args.use_tabs = parse_bool(on);
          break;
        } else if (param.long_flag == "file_with_page_sizes") {
          std::string on(*(argv++));
          args.has_page_sizes = parse_bool(on);
          break;
        } else if (param.long_flag == "chunk_size") {
          args.chunk_size = size_t(std::stoull(*(argv++)));
          break;
        } else {
          std::cerr << "INTERNAL ERROR: Unhandled paramter '" << arg << "'." << std::endl;
          usage(name, params);
          std::exit(1);
        }
      }
    }
    if (!found) {
      std::cerr << "ERROR: Unknown argument '" << arg << "'." << std::endl;
      usage(name, params);
      std::exit(1);
    }
  }

  if (args.filenames.empty()) {
    std::cerr << "ERROR: Must specify at least one input file." << std::endl;
    std::exit(1);
  }

  return args;
}

int main(int argc, char** argv)
{
  args_type args = parse_args(argc, argv);

  GPU_CHECK(gpuSetDevice(args.gpu));

  if (!args.csv_output) {
#ifdef __HIP_PLATFORM_AMD__
    hipDeviceProp_t props;
    GPU_CHECK(hipGetDeviceProperties(&props, args.gpu));
#else
    cudaDeviceProp props;
    GPU_CHECK(cudaGetDeviceProperties(&props, args.gpu));
#endif
    std::cout << "GPU Information:" << std::endl;
    std::cout << "  Device:          " << props.name << std::endl;
    std::cout << "  Compute Units:   " << props.multiProcessorCount << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total Memory:    "
              << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << std::endl;
  }

  auto data = multi_file(args.filenames, args.chunk_size, args.has_page_sizes,
      args.duplicate_count);

  // one warmup to allow cuda to initialize
  run_benchmark(data, true, args.warmup_count, false, false,
      args.duplicate_count, args.filenames.size());

  // second run to report times
  run_benchmark(data, false, args.iteration_count, args.csv_output,
      args.use_tabs, args.duplicate_count, args.filenames.size());

  return 0;
}

#endif
