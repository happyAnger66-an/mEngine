/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <cmath>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "m_engine/executor/executor.h"
#include "m_engine/executor/types.h"
#include "m_engine/runtime/bufferManager.h"
#include "m_engine/runtime/common.h"
#include "m_engine/runtime/iBuffer.h"
#include "m_engine/runtime/iTensor.h"

namespace m_engine::testing {
namespace fs = std::filesystem;
namespace mr = m_engine::runtime;

auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";

auto const ENGINE_X86_PATH = TEST_RESOURCE_PATH / "models/engine/x86";
auto const ENGINE_THOR_PATH = TEST_RESOURCE_PATH / "models/engine/thor";

auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
auto const YOLO_DATA_PATH = DATA_PATH / "yolo";
auto const YOLO_INPUT_DATA_PATH = YOLO_DATA_PATH / "yolo_input.npy";
auto const YOLO_OUTPUT0_DATA_PATH = YOLO_DATA_PATH / "output0.npy";
auto const YOLO_OUTPUT1_DATA_PATH = YOLO_DATA_PATH / "output1.npy";

auto const YOLO_X86_ENGINE_PATH =
    ENGINE_X86_PATH / "yolo_5_class_full_dataset.251209_simplifier.engine";
auto const YOLO_THOR_ENGINE_PATH =
    ENGINE_THOR_PATH / "yolo_5_class_full_dataset.251209_simplifier.engine";

class PathUtil {
 public:
  // model paths
  // results
};

inline bool almostEqual(float a, float b, float atol = 1e-2,
                        float rtol = 1e-3) {
  // Params: a = value to compare and b = reference
  // This function follows implementation of numpy.isclose(), which checks
  //   abs(a - b) <= (atol + rtol * abs(b)).
  // Note that the inequality above is asymmetric where b is considered as
  // a reference value. To account into both absolute/relative errors, it
  // uses absolute tolerance and relative tolerance at the same time. The
  // default values of atol and rtol borrowed from numpy.isclose(). For the
  // case of nan value, the result will be true.
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id;
  cudaEvent_t _start;
  cudaEvent_t _stop;

  /// Construct`or
  GpuTimer() : _stream_id(0) {
    MENGINE_CUDA_CHECK(cudaEventCreate(&_start));
    MENGINE_CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    MENGINE_CUDA_CHECK(cudaEventDestroy(_start));
    MENGINE_CUDA_CHECK(cudaEventDestroy(_stop));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void start(cudaStream_t stream_id = 0) {
    _stream_id = stream_id;
    MENGINE_CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void stop() { MENGINE_CUDA_CHECK(cudaEventRecord(_stop, _stream_id)); }

  /// Return the elapsed time (in milliseconds)
  float elapsed_millis() {
    float elapsed = 0.0;
    MENGINE_CUDA_CHECK(cudaEventSynchronize(_stop));
    MENGINE_CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
};

}  // namespace m_engine::testing
