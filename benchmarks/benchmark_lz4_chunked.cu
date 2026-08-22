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

#include "benchmark_template_chunked.cuh"

#include "arcto/lz4.h"

// Test for the asynchronous C++ interface
static arctoBatchedLZ4Opts_t arctoBatchedLZ4TestOpts
    = {ARCTO_TYPE_CHAR};

static bool isLZ4InputValid(const std::vector<std::vector<char>>& data)
{
  arctoType_t data_type = arctoBatchedLZ4TestOpts.data_type;

  size_t typeSize = 0;
  if (data_type == ARCTO_TYPE_CHAR || data_type == ARCTO_TYPE_UCHAR
      || data_type == ARCTO_TYPE_BITS) {
    typeSize = 1;
  } else if (
      data_type == ARCTO_TYPE_SHORT || data_type == ARCTO_TYPE_USHORT) {
    typeSize = 2;
  } else if (
      data_type == ARCTO_TYPE_INT || data_type == ARCTO_TYPE_UINT) {
    typeSize = 4;
  }

  bool valid = true;
  for (const auto& chunk : data) {
    if (chunk.size() % typeSize != 0) {
      std::cerr << "ERROR: Input data must have a length and chunk size "
                << "that are a multiple of " << typeSize << ", but found a "
                << "chunk of size " << chunk.size() << "." << std::endl;
      valid = false;
    }
  }

  return valid;
}

GENERATE_CHUNKED_BENCHMARK(
    arctoBatchedLZ4CompressGetTempSize,
    arctoBatchedLZ4CompressGetMaxOutputChunkSize,
    arctoBatchedLZ4CompressAsync,
    arctoBatchedLZ4DecompressGetTempSize,
    arctoBatchedLZ4DecompressAsync,
    isLZ4InputValid,
    arctoBatchedLZ4TestOpts);
