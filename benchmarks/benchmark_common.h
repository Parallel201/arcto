/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BENCHMARK_COMMON_H
#define BENCHMARK_COMMON_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define GPU_PREFIX "hip"
#define GPU_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(error) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#else
#include <cuda_runtime.h>
#define GPU_PREFIX "cuda"
#define GPU_CHECK(condition)                                                   \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

// Legacy compatibility
#define CUDA_CHECK GPU_CHECK

#pragma GCC diagnostic ignored "-Wunused-function"

namespace benchmark
{

static double get_time(const struct timespec start, const struct timespec end)
{
  double const BILLION = 1000000000.0;
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / BILLION;
}

#ifdef __HIP_PLATFORM_AMD__
static double gibs(const size_t bytes, hipEvent_t start, hipEvent_t end)
{
  float ms;
  GPU_CHECK(hipEventElapsedTime(&ms, start, end));
  return ((double)bytes / (1.0e6 * ms));
}
#else
static double gibs(const size_t bytes, cudaEvent_t start, cudaEvent_t end)
{
  float ms;
  GPU_CHECK(cudaEventElapsedTime(&ms, start, end));
  return ((double)bytes / (1.0e6 * ms));
}
#endif

static double gbs(const size_t bytes, const size_t time_ns)
{
  return ((double)bytes / time_ns);
}

static double average_gbs(
    const std::vector<size_t>& bytes, const std::vector<size_t>& durations_ns)
{
  if (bytes.size() != durations_ns.size()) {
    throw std::runtime_error(
        "Mismatched length of bytes and durations_ns: "
        + std::to_string(bytes.size()) + " vs "
        + std::to_string(durations_ns.size()) + ".");
  }

  double total_gbs = 0.0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    total_gbs += gbs(bytes[i], durations_ns[i]);
  }

  return total_gbs / bytes.size();
}

static std::vector<uint8_t>
gen_data(const size_t size, const int min_byte = 0, const int max_byte = 255)
{
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(min_byte, max_byte);

  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>(dist(rng));
  }

  return data;
}

template <typename T>
static std::vector<T> load_dataset_from_binary(
    const std::string& filename, size_t* const num = nullptr)
{
  std::vector<T> buffer;
  std::ifstream fin(filename, std::ifstream::binary);

  fin.seekg(0, std::ifstream::end);
  size_t file_size = static_cast<size_t>(fin.tellg());
  fin.seekg(0, std::ifstream::beg);

  size_t num_elements;
  if (num) {
    num_elements = *num;
  } else {
    num_elements = file_size / sizeof(T);
  }

  buffer.resize(num_elements);
  fin.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(T));
  fin.close();

  return buffer;
}

template <typename T>
static std::vector<T> load_dataset_from_txt(const std::string& filename)
{
  std::vector<T> buffer;
  std::ifstream fin(filename);

  std::string line;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    T val;
    if (iss >> val) {
      buffer.push_back(val);
    }
  }
  fin.close();

  return buffer;
}

static bool startsWith(const std::string& str, const std::string& prefix)
{
  if (str.length() < prefix.length()) {
    return false;
  }

  return str.substr(0, prefix.length()) == prefix;
}

static void benchmark_assert(
    const bool pass, const std::string& msg = "ERROR")
{
  if (!pass) {
    throw std::runtime_error("ERROR: " + msg);
  }
}

} // namespace benchmark

#endif
