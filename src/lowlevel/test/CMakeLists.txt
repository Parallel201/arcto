# Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
#
# Unit tests for the low-level (batched) infrastructure: one executable and
# one ctest entry per *_test.cpp in this directory (none yet). Binaries stay
# next to this directory's build output (build/src/lowlevel/test/<name>).

file(GLOB ARCTO_UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

foreach(test_source IN LISTS ARCTO_UNIT_TEST_SOURCES)
  get_filename_component(test_name "${test_source}" NAME_WE)
  add_executable(${test_name} "${test_source}")
  target_include_directories(${test_name} PRIVATE
    "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/lowlevel")
  if (CUDA_BACKEND)
    target_link_libraries(${test_name} PRIVATE arcto CUDA::cudart)
  else()
    target_link_libraries(${test_name} PRIVATE arcto hip::host)
  endif()
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
