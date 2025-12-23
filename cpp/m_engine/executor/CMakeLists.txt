# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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
set(EXECUTOR_TARGET_NAME m_engine_executor)
set(EXECUTOR_STATIC_TARGET ${EXECUTOR_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    tensor.cpp
    types.cpp
    serialization.cpp)

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

add_library(${EXECUTOR_STATIC_TARGET} STATIC ${SRCS})
set_target_properties(
  ${EXECUTOR_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${EXECUTOR_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${EXECUTOR_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")