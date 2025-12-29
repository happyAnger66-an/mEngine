/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "common.h"

#include "m_engine/common/assert.h"
#include "m_engine/common/memoryUtils.h"
#include "m_engine/executor/executor.h"
#include "m_engine/executor/types.h"
#include "m_engine/runtime/iBuffer.h"
#include "m_engine/runtime/iTensor.h"
#include "m_engine/runtime/utils/numpyUtils.h"
#include "tests/utils/common.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace m_engine::testing
{
namespace fs = std::filesystem;
namespace tr = m_engine::runtime;
namespace tc = m_engine::common;


} // namespace m_engine::testing
