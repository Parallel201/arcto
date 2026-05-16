# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

# Collect all benchmark source files
file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_*.hip")

# Remove template and common files from sources
list(FILTER BENCHMARK_SOURCES EXCLUDE REGEX ".*benchmark_template_.*")
list(FILTER BENCHMARK_SOURCES EXCLUDE REGEX ".*benchmark_common.*")

# Build each benchmark as a separate executable
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
  get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)

  if(CUDA_BACKEND)
    # CUDA backend
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
    target_include_directories(${BENCHMARK_NAME}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${BENCHMARK_NAME}
      PRIVATE
        arcto
    )
    set_target_properties(${BENCHMARK_NAME} PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  else()
    # HIP/AMD backend
    # Mark .cu files as HIP language for AMD backend
    if(BENCHMARK_SOURCE MATCHES "\\.cu$")
      set_source_files_properties(${BENCHMARK_SOURCE} PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
    target_include_directories(${BENCHMARK_NAME}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${BENCHMARK_NAME}
      PRIVATE
        arcto
        hip::host
    )
    set_target_properties(${BENCHMARK_NAME} PROPERTIES
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
    )
  endif()

  # Set C++14 standard
  set_target_properties(${BENCHMARK_NAME} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
  )

  # Install benchmark executable
  install(TARGETS ${BENCHMARK_NAME}
    RUNTIME DESTINATION bin/benchmarks
  )
endforeach()



message(STATUS "Benchmarks will be built: ${BENCHMARK_SOURCES}")
