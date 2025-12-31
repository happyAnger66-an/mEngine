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
#include "m_engine/runtime/utils/numpyUtils.h"
#include "tests/utils/common.h"

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
  auto mBufferManager =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());

  auto executor = TRTExecutor(trtEnginePath);
  auto input_host_buffer = executor.GetHostBuffer("images");
  EXPECT_EQ(input_host_buffer->getSize(), 1 * 3 * 640 * 640);
  EXPECT_EQ(input_host_buffer->getSizeInBytes(),
            1 * 3 * 640 * 640 * sizeof(float));

  auto input_device_buffer = executor.GetDeviceBuffer("images");
  EXPECT_EQ(input_device_buffer->getSize(), 1 * 3 * 640 * 640);
  EXPECT_EQ(input_device_buffer->getSizeInBytes(),
            1 * 3 * 640 * 640 * sizeof(float));

  EXPECT_EQ(input_host_buffer->getSize(), input_device_buffer->getSize());
  EXPECT_EQ(input_host_buffer->getSizeInBytes(),
            input_device_buffer->getSizeInBytes());

  auto output0_device_buffer = executor.GetDeviceBuffer("output0");
  EXPECT_EQ(output0_device_buffer->getSize(), 1 * 41 * 8400);
  EXPECT_EQ(output0_device_buffer->getSizeInBytes(),
            1 * 41 * 8400 * sizeof(float));

  auto output1_device_buffer = executor.GetDeviceBuffer("output1");
  EXPECT_EQ(output1_device_buffer->getSize(), 1 * 32 * 160 * 160);
  EXPECT_EQ(output1_device_buffer->getSizeInBytes(),
            1 * 32 * 160 * 160 * sizeof(float));
}

/*
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

  executor.PrepareData();
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

  auto& mem_counters = mr::MemoryCounters::getInstance();
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112);
*/
/* read model output device to host. */
/*  auto output_device_buffer = executor.GetDeviceBuffer(output_tensor_name);
  auto device2host_buff =
      mBufferManager->copyFrom(*output_device_buffer, mr::MemoryType::kCPU);
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112 + 1000 * sizeof(float));

  device2host_buff.reset(nullptr);
  EXPECT_EQ(mem_counters.getGpu(), 606112);
  EXPECT_EQ(mem_counters.getCpu(), 606112);
}*/

TEST_F(YoloExecutorTest, validOutput) {
  auto mBufferMgr =
      std::make_unique<mr::BufferManager>(std::make_unique<mr::CudaStream>());
  auto input_npy = YOLO_INPUT_DATA_PATH.c_str();
  auto output0_npy = YOLO_OUTPUT0_DATA_PATH.c_str();
  auto output1_npy = YOLO_OUTPUT1_DATA_PATH.c_str();
  auto loadedTensor =
      mr::utils::loadNpy(*mBufferMgr, input_npy, mr::MemoryType::kCPU);
  auto output0Tensor =
      mr::utils::loadNpy(*mBufferMgr, output0_npy, mr::MemoryType::kCPU);
  auto output1Tensor =
      mr::utils::loadNpy(*mBufferMgr, output1_npy, mr::MemoryType::kCPU);
  EXPECT_EQ(loadedTensor->getShape().nbDims, 4);
  EXPECT_EQ(loadedTensor->getShape().d[0], 1);
  EXPECT_EQ(loadedTensor->getShape().d[1], 3);
  EXPECT_EQ(loadedTensor->getShape().d[2], 640);
  EXPECT_EQ(loadedTensor->getShape().d[3], 640);

  EXPECT_EQ(output0Tensor->getShape().nbDims, 3);
  EXPECT_EQ(output0Tensor->getShape().d[0], 1);
  EXPECT_EQ(output0Tensor->getShape().d[1], 41);
  EXPECT_EQ(output0Tensor->getShape().d[2], 8400);

  EXPECT_EQ(output1Tensor->getShape().nbDims, 4);
  EXPECT_EQ(output1Tensor->getShape().d[0], 1);
  EXPECT_EQ(output1Tensor->getShape().d[1], 32);
  EXPECT_EQ(output1Tensor->getShape().d[2], 160);
  EXPECT_EQ(output1Tensor->getShape().d[2], 160);

  auto executor = TRTExecutor(trtEnginePath);
  auto input_tensor_name = "images";
  auto output0_tensor_name = "output0";
  auto output1_tensor_name = "output1";

  auto input_host_buffer = executor.GetHostBuffer(input_tensor_name);
  mBufferMgr->copy(loadedTensor->data(), *input_host_buffer);
  auto output0_device_buffer = executor.GetDeviceBuffer(output0_tensor_name);
  auto output1_device_buffer = executor.GetDeviceBuffer(output1_tensor_name);
  auto output0_host_buffer = executor.GetHostBuffer(output0_tensor_name);
  auto output1_host_buffer = executor.GetHostBuffer(output1_tensor_name);

  executor.PrepareData();
  int loops[] = {1, 2, 3, 4, 5};
  for (int i : loops) {
    auto start = std::chrono::system_clock::now();

    executor.Infer();

    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cerr << "duration: " << duration.count() << " ms" << std::endl;
  }

  EXPECT_EQ(memcmp(output0_host_buffer->data(), output0Tensor->data(),
                   output0Tensor->getSizeInBytes()),
            0);

  EXPECT_EQ(memcmp(output1_host_buffer->data(), output1Tensor->data(),
                   output1Tensor->getSizeInBytes()),
            0);
}