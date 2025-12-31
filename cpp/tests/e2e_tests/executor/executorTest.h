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

#pragma once
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "m_engine/common/cudaUtils.h"

#include "tests/utils/common.h"

namespace m_engine::testing {

class YoloExecutorTest
    : public ::testing::Test  // NOLINT(cppcoreguidelines-pro-type-member-init)
{
 public:
 protected:
  void SetUp() override {
    mDeviceCount = m_engine::common::getDeviceCount();
    if (mDeviceCount == 0) {
      GTEST_SKIP() << "No GPUs found";
    }
  }

  void TearDown() override {}

#ifdef X86_64
  const std::string trtEnginePath = YOLO_X86_ENGINE_PATH.c_str();
#else
  const std::string trtEnginePath = YOLO_THOR_ENGINE_PATH.c_str();
#endif

  int mDeviceCount{};
};

}  // namespace m_engine::testing
