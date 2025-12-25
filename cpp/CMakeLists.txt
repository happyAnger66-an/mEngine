#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake;/usr/share/glog/cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

INCLUDE_DIRECTORIES(/usr/include/x86_64-linux-gnu/)
#find_package(CUDAToolkit REQUIRED)
include(/usr/share/glog/cmake/FindUnwind.cmake)
find_package(glog REQUIRED)

include(resolve_dirs)
include(parse_make_options)
include(cuda_configuration)
include(sanitizers)

project(m_engine LANGUAGES CXX)

# Build options
option(BUILD_TESTS "Build Google tests" ON)
option(NVRTC_DYNAMIC_LINKING "Link against the dynamic NVRTC libraries" OFF)
option(CUBLAS_DYNAMIC_LINKING
       "Link against the dynamic cublas/cublasLt libraries" ON)
option(CURAND_DYNAMIC_LINKING "Link against the dynamic curand library" ON)

option(ENABLE_NVSHMEM "Enable building with NVSHMEM support" OFF)
if(NOT ${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "12.8")
  set(ENABLE_CUBLASLT_FP4_GEMM
      OFF
      CACHE BOOL "" FORCE)
  message(
    STATUS
      "CUDA ${CUDAToolkit_VERSION} < 12.8: disabling ENABLE_CUBLASLT_FP4_GEMM")
endif()

message(STATUS "ENABLE_NVSHMEM is ${ENABLE_NVSHMEM}")

if(NVTX_DISABLE)
  add_compile_definitions("NVTX_DISABLE")
  message(STATUS "NVTX is disabled")
else()
  message(STATUS "NVTX is enabled")
endif()

# Add TensorRT LLM Gen export interface and CUDA support
add_compile_definitions("MENGINE_GEN_EXPORT_INTERFACE")
add_compile_definitions("MENGINE_ENABLE_CUDA")

if(BUILD_TESTS)
  message(STATUS "Building Google tests")
else()
  message(STATUS "Not building Google tests")
endif()

# Read the project version
set(MENGINE_VERSION_DIR ${PROJECT_SOURCE_DIR}/../m_engine)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS
                                    ${MENGINE_VERSION_DIR}/version.py)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import version; print(version.__version__)"
  WORKING_DIRECTORY ${MENGINE_VERSION_DIR}
  OUTPUT_VARIABLE MENGINE_VERSION
  RESULT_VARIABLE MENGINE_VERSION_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(MENGINE_VERSION_RESULT "0.0.1")

if(MENGINE_VERSION_RESULT EQUAL 0)
  message(STATUS "MENGINE version: ${MENGINE_VERSION}")
else()
  message(FATAL_ERROR "Failed to determine MENGINE version")
endif()

configure_file(
  cmake/templates/version.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorrt_llm/executor/version.h)


setup_cuda_compiler()

enable_language(C CXX CUDA)

# Configure CUDA Architectures after enabling CUDA.

# Old CMake rejects family conditional architectures during enable_language, But
# after that CMake handles it just fine.
#setup_cuda_architectures()

set(CUDA_DRV_LIB CUDA::cuda_driver)
set(CUDA_NVML_LIB CUDA::nvml)
#set(CUDA_RT_LIB CUDA::cudart_static)
set(NVPTX_LIB CUDA::nvptxcompiler_static)

set(CUDA_TOOLKIT_COMPONENTS cudart_static cuda_driver nvml nvptxcompiler_static)

if(NVRTC_DYNAMIC_LINKING)
  set(NVRTC_LIB CUDA::nvrtc)
  set(NVRTC_BUILTINS_LIB CUDA::nvrtc_builtins)
  list(APPEND CUDA_TOOLKIT_COMPONENTS nvrtc nvrtc_builtins)
else()
  set(NVRTC_LIB CUDA::nvrtc_static)
  set(NVRTC_BUILTINS_LIB CUDA::nvrtc_builtins_static)
  list(APPEND CUDA_TOOLKIT_COMPONENTS nvrtc_static nvrtc_builtins_static)
endif()

find_package(CUDAToolkit 11.2 REQUIRED COMPONENTS ${CUDA_TOOLKIT_COMPONENTS})

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

resolve_dirs(CUDAToolkit_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")

message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA_NVML_LIB: ${CUDA_NVML_LIB}")

# Prevent CMake from creating a response file for CUDA compiler, so clangd can
# pick up on the includes
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

find_library(RT_LIB rt)

# TRT dependencies
find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser)
set(TRT_LIB TensorRT::NvInfer)

get_filename_component(TRT_LLM_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)

#FetchContent_MakeAvailable(cutlass cxxopts flashmla json xgrammar)

#if(NOT NVTX_DISABLE)
#  FetchContent_MakeAvailable(nvtx)
#endif()

#if(NOT NVTX_DISABLE)
#  set(maybe_nvtx_includedir ${CMAKE_BINARY_DIR}/_deps/nvtx-src/include)
#endif()

# include as system to suppress warnings
include_directories(
  SYSTEM
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}/cccl
  ${CUDNN_ROOT_DIR}/include
  $<TARGET_PROPERTY:TensorRT::NvInfer,INTERFACE_INCLUDE_DIRECTORIES>
  ${maybe_nvtx_includedir}
  ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include
  ${CMAKE_BINARY_DIR}/_deps/cutlass-src/tools/util/include
  ${CMAKE_BINARY_DIR}/_deps/json-src/include)

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "12.8")
  add_definitions("-DENABLE_FP4")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 12.8, enable -DENABLE_FP4 flag"
  )
endif()

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fno-inline")
endif()

# Note: The following are desirable settings that should be enabled if we
# decrease shared library size. See e.g.
# https://github.com/rapidsai/cudf/pull/6134 for a similar issue in another
# project.

# set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}
# --generate-line-info")

# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss ")

if(ENABLE_NVSHMEM)
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_NVSHMEM=1>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_NVSHMEM=1>)
else()
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:ENABLE_NVSHMEM=0>
                          $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_NVSHMEM=0>)
endif()

# Fix linking issue with TRT 10, the detailed description about `--mcmodel` can
# be found in
# https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mcmodel_003dmedium-1
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-relax")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-relax")
endif()

# Disable deprecated declarations warnings
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
  # /wd4996 is the Windows equivalent to turn off warnings for deprecated
  # declarations

  # /wd4505
  # https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements-2019?view=msvc-170#warning-for-unused-internal-linkage-functions
  # "warning C4505: <>: unreferenced function with internal linkage has been
  # removed"

  # /wd4100
  # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-4-c4100?view=msvc-170
  # warning C4100: 'c': unreferenced formal parameter

setup_sanitizers()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(FAST_MATH)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()
if(COMPRESS_FATBIN)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fatbin-options -compress-all")
endif()
if(NVCC_TIMING)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} --time ${CMAKE_CURRENT_BINARY_DIR}/nvcc-timing.csv")
endif()
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIR})
message(STATUS "COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

if(NOT WIN32 AND NOT DEFINED USE_CXX11_ABI)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c"
            "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE USE_CXX11_ABI)
  # Convert the bool variable to integer.
  if(USE_CXX11_ABI)
    set(USE_CXX11_ABI 1)
  else()
    set(USE_CXX11_ABI 0)
  endif()
  message(STATUS "USE_CXX11_ABI is set by python Torch to ${USE_CXX11_ABI}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_UCX=0")

list(APPEND COMMON_HEADER_DIRS)
include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_subdirectory(m_engine)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Measure the compile time
option(MEASURE_BUILD_TIME "Measure the build time of each module" OFF)
if(MEASURE_BUILD_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_CUSTOM "${CMAKE_COMMAND} -E time")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
endif()