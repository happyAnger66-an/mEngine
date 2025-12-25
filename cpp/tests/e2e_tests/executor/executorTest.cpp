/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "executorTest.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "m_engine/common/assert.h"
#include "m_engine/common/logger.h"
#include "m_engine/common/memoryUtils.h"
#include "m_engine/executor/executor.h"
#include "m_engine/executor/types.h"
#include "m_engine/runtime/bufferManager.h"
#include "m_engine/runtime/cudaStream.h"
#include "m_engine/runtime/iBuffer.h"
#include "m_engine/runtime/iTensor.h"
#include "m_engine/runtime/memoryCounters.h"

namespace mr = m_engine::runtime;
namespace mc = m_engine::common;

using namespace m_engine::executor;
using namespace m_engine::testing;

using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {}  // namespace

void testInvalidCtor(std::filesystem::path const& enginePath,
                     ModelType modelType, std::string expectedErrMsg = "") {
  try {
    auto executor = TRTExecutor(enginePath);

    FAIL() << "Expected mEngineException";
  } catch (std::exception const& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrMsg));
  }
}

TEST_F(YoloExecutorTest, validInputBuffer) {
  auto trtEnginePath = "/tmp/quantize/1215/model.engine";
  auto mBufferManager =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());

  //  nvinfer1::Dims inputDims(1,3,224,224);
  // nvinfer1::DataType dataType = nvinfer1::DataType::KFLOAT;

  auto executor = TRTExecutor(trtEnginePath);
  auto input_host_buffer = executor.GetHostBuffer("gpu_0/data_0");
  EXPECT_EQ(input_host_buffer->getSize(), 1 * 3 * 224 * 224);
  EXPECT_EQ(input_host_buffer->getSizeInBytes(),
            1 * 3 * 224 * 224 * sizeof(float));

  auto input_device_buffer = executor.GetDeviceBuffer("gpu_0/data_0");
  EXPECT_EQ(input_device_buffer->getSize(), 1 * 3 * 224 * 224);
  EXPECT_EQ(input_device_buffer->getSizeInBytes(),
            1 * 3 * 224 * 224 * sizeof(float));

  EXPECT_EQ(input_host_buffer->getSize(), input_device_buffer->getSize());
  EXPECT_EQ(input_host_buffer->getSizeInBytes(),
            input_device_buffer->getSizeInBytes());
}

TEST_F(YoloExecutorTest, validOutputBuffer) {
  auto trtEnginePath = "/tmp/quantize/1215/model.engine";
  auto output_tensor_name = "gpu_0/softmax_1";
  auto mBufferManager =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());

  //  nvinfer1::Dims inputDims(1,3,224,224);
  // nvinfer1::DataType dataType = nvinfer1::DataType::KFLOAT;

  auto executor = TRTExecutor(trtEnginePath);
  auto output_host_buffer = executor.GetHostBuffer(output_tensor_name);
  EXPECT_EQ(output_host_buffer->getSize(), 1 * 1000);
  EXPECT_EQ(output_host_buffer->getSizeInBytes(), 1 * 1000 * sizeof(float));

  auto output_device_buffer = executor.GetDeviceBuffer(output_tensor_name);
  EXPECT_EQ(output_device_buffer->getSize(), 1 * 1000);
  EXPECT_EQ(output_device_buffer->getSizeInBytes(), 1 * 1000 * sizeof(float));

  EXPECT_EQ(output_host_buffer->getSize(), output_device_buffer->getSize());
  EXPECT_EQ(output_host_buffer->getSizeInBytes(),
            output_device_buffer->getSizeInBytes());

  int32_t numBytes = output_host_buffer->getSizeInBytes();
  const char* host_data =
      reinterpret_cast<const char*>(output_host_buffer->data());
  auto device2host_buff =
      mBufferManager->copyFrom(*output_device_buffer, mr::MemoryType::kCPU);
  const char* device_data =
      reinterpret_cast<const char*>(device2host_buff->data());
  for (int i = 0; i < numBytes; i++) {
    EXPECT_EQ(host_data[i], device_data[i]);
    EXPECT_EQ(host_data[i], 0);
  }
}

TEST_F(YoloExecutorTest, validInferOutputBuffer) {
  auto trtEnginePath = "/tmp/quantize/1215/model.engine";
  auto output_tensor_name = "gpu_0/softmax_1";
  auto mBufferManager =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());

  auto executor = TRTExecutor(trtEnginePath);
  auto output_host_buffer = executor.GetHostBuffer(output_tensor_name);
  auto output_device_buffer = executor.GetDeviceBuffer(output_tensor_name);

  executor.Infer();

  int32_t numBytes = output_host_buffer->getSizeInBytes();
  const char* host_data =
      reinterpret_cast<const char*>(output_host_buffer->data());
  auto device2host_buff =
      mBufferManager->copyFrom(*output_device_buffer, mr::MemoryType::kCPU);
  const char* device_data =
      reinterpret_cast<const char*>(device2host_buff->data());
  for (int i = 0; i < numBytes; i++) {
    EXPECT_EQ(host_data[i], device_data[i]);
  }
}

TEST_F(YoloExecutorTest, validMemUsed) {
  auto trtEnginePath = "/tmp/quantize/1215/model.engine";
  auto output_tensor_name = "gpu_0/softmax_1";
  auto mBufferManager =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());

  auto executor = TRTExecutor(trtEnginePath);

  auto & mem_counters = mr::MemoryCounters::getInstance();
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112);
 
  /* read model output device to host. */
  auto output_device_buffer = executor.GetDeviceBuffer(output_tensor_name);
  auto device2host_buff =
      mBufferManager->copyFrom(*output_device_buffer, mr::MemoryType::kCPU);
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112+1000*sizeof(float));

  device2host_buff.reset(nullptr);
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112);
}