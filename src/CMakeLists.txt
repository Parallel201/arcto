# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
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
#

# MIT License
#
# Modifications Copyright (C) 2025-2026 Cristiano Künas.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

macro (setup_cuda_devices)
  if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    # an explicit -D CMAKE_CUDA_ARCHITECTURES=... always wins, exactly as
    # CMAKE_HIP_ARCHITECTURES does on the AMD side
    set(GPU_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
  else()
    # Default fan-out, bounded by what the toolkit still accepts: CUDA 12
    # dropped Kepler/Maxwell, CUDA 13 dropped Pascal (sm_60) and Volta
    # (sm_70), so an inherited fixed list stops compiling with new toolkits.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
      set(GPU_ARCHS "75-real" "80-real" "86-real" "89-real" "90")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12")
      set(GPU_ARCHS "70-real" "75-real" "80-real" "86-real" "90")
    else()
      set(GPU_ARCHS "60")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8")
        set(GPU_ARCHS ${GPU_ARCHS} "70-real")
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "9")
        set(GPU_ARCHS ${GPU_ARCHS} "75-real")
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
        set(GPU_ARCHS ${GPU_ARCHS} "80")
      endif()
    endif()
  endif()
  message(STATUS "CUDA architectures: ${GPU_ARCHS}")
endmacro(setup_cuda_devices)

macro (set_cuda_compilation_options)
  set_property(TARGET arcto PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET arcto PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  # The device sources are compiled as CUDA on this backend (they are marked
  # LANGUAGE CUDA above), so the generator expression must name CUDA -- with
  # COMPILE_LANGUAGE:HIP these flags were silently never applied.
  #  --expt-relaxed-constexpr: hip-clang lets device code call constexpr host
  #    functions (std::numeric_limits<T>::max() and friends); nvcc needs this
  #    flag for the same code.
  target_compile_options(arcto PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(DEFINED CUB_DIR)
    target_include_directories(arcto PRIVATE "${CUB_DIR}")
  endif()
  target_include_directories(arcto PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
endmacro (set_cuda_compilation_options)

file(GLOB HIP_SOURCES
  ./*.hip
  ./lowlevel/*.hip
  ./highlevel/*.hip)
file(GLOB CPP_SOURCES
  ./*.cpp
  ./lowlevel/*.cpp
  ./highlevel/*.cpp)

if (CUDA_BACKEND)
  setup_cuda_devices()
  set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE CUDA)
endif (CUDA_BACKEND)

if (BUILD_STATIC)
  add_library(arcto STATIC ${HIP_SOURCES} ${CPP_SOURCES})
else()
  if (MSVC_VERSION)
    # we have to export all symbols on windows for a dll
    add_library(arcto STATIC ${HIP_SOURCES} ${CPP_SOURCES})
  else()
    add_library(arcto SHARED ${HIP_SOURCES} ${CPP_SOURCES})
  endif()
endif()

target_include_directories(arcto PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>")
target_include_directories(arcto PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>")

if (CUDA_BACKEND)
  set_cuda_compilation_options()
endif (CUDA_BACKEND)

if (ARCTO_DEVICE_REPORTS AND NOT CUDA_BACKEND)
  # -Rpass-analysis=kernel-resource-usage makes the AMDGPU backend print, per
  # kernel and per offload arch, the SGPR/VGPR/AGPR counts, LDS and scratch
  # bytes and the resulting occupancy (waves/SIMD) -- the numbers every
  # launch-bounds / register-pressure decision in docs/AMD_OPTIMIZATION_MAP.md
  # depends on. --save-temps keeps the device assembly (.s per arch) in the
  # build tree so ISA-level checks (global_load_dwordx4, ds_bpermute, s_barrier,
  # nt/slc bits) can be grepped without a profiler.
  target_compile_options(arcto PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-Rpass-analysis=kernel-resource-usage>
    $<$<COMPILE_LANGUAGE:HIP>:--save-temps>)
endif()

if(UNIX)
target_include_directories(arcto PUBLIC
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
endif()

include_directories("${arcto_SOURCE_DIR}/src")

# hipCUB/rocPRIM block and warp primitives are used by the Cascaded and RLE
# kernels (arcto_hipcub.hiph). Take the include directories from the imported
# target so the dependency is explicit instead of resolving by accident through
# HIP_INCLUDE_DIRS. Deliberately NOT target_link_libraries(): hip::hipcub pulls
# hip::device, whose interface adds --offload-arch=<gfx> to every source of the
# target -- including the host-only .cpp files compiled by the C++ compiler,
# which then fail with "unrecognized command-line option". The CUDA backend
# uses CUB from the toolkit.
if (NOT CUDA_BACKEND)
  target_include_directories(arcto PRIVATE
    $<TARGET_PROPERTY:hip::hipcub,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# Vendored canonical LLNL/zfp (added at the top-level CMakeLists via
# add_subdirectory(third_party/zfp)). The zfp::zfp target is a static
# library; linking is PRIVATE because we expose only the arcto/zfp.h C
# wrapper, never canonical's headers.
if (TARGET zfp::zfp)
  target_link_libraries(arcto PRIVATE zfp::zfp)
endif()

if (BUILD_TESTS)
  add_subdirectory("test")
  add_subdirectory("highlevel/test")
  add_subdirectory("lowlevel/test")
endif()
