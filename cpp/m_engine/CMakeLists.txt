# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
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
set(TARGET_NAME m_engine)
set(SHARED_TARGET ${TARGET_NAME})
set(SHARED_TARGET
    ${SHARED_TARGET}
    PARENT_SCOPE)
set(API_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${API_INCLUDE_DIR})

set(TARGET_ARCH "unknown")

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if(NOT WIN32) # Linux
  execute_process(
    COMMAND grep -oP "(?<=^ID=).+" /etc/os-release
    COMMAND tr -d "\""
    COMMAND tr -d "\n"
    RESULT_VARIABLE _OS_ID_SUCCESS
    OUTPUT_VARIABLE OS_ID)
  execute_process(
    COMMAND grep -oP "(?<=^VERSION_ID=).+" /etc/os-release
    COMMAND tr -d "\""
    COMMAND tr -d "\n"
    RESULT_VARIABLE _OS_VERSION_ID_SUCCESS
    OUTPUT_VARIABLE OS_VERSION_ID)
  message(STATUS "Operating System: ${OS_ID}, ${OS_VERSION_ID}")

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(TARGET_ARCH "x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TARGET_ARCH "aarch64-linux-gnu")
    if(NOT ${OS_ID} MATCHES "ubuntu" OR ${OS_VERSION_ID} VERSION_LESS 22.04)
      message(
        FATAL_ERROR
          "The minimum system requirement for aarch64 is Ubuntu 22.04.")
    endif()
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
else() # Windows
  # AMD64, IA64, ARM64, EM64T, X86
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    set(TARGET_ARCH "x86_64-windows-msvc")
  else()
    message(
      FATAL_ERROR
        "The system processor type is unsupported: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

add_subdirectory(common)
add_subdirectory(runtime)
add_subdirectory(executor)

find_package(Threads REQUIRED)

set(TRTLLM_LINK_LIBS
    ${CUDA_DRV_LIB}
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CURAND_LIB}
    ${CMAKE_DL_LIBS}
    ${TRT_LIB}
    common_src)

if(NOT WIN32) # Unix-like compilers
  set(UNDEFINED_FLAG "-Wl,--no-undefined")
  set(AS_NEEDED_FLAG "-Wl,--as-needed")
  set(NO_AS_NEEDED_FLAG "-Wl,--no-as-needed")
else() # Windows
  set(UNDEFINED_FLAG "")
  set(AS_NEEDED_FLAG "")
  set(NO_AS_NEEDED_FLAG "")
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(${SHARED_TARGET} SHARED)

set_target_properties(
  ${SHARED_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             LINK_FLAGS "${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}")

target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})

target_link_libraries(
  ${SHARED_TARGET}
  PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${BATCH_MANAGER_TARGET}>
          $<LINK_LIBRARY:WHOLE_ARCHIVE,${EXECUTOR_TARGET}>)