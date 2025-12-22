/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "m_engine/common/mEngineException.h"

namespace m_engine {

class DebugConfig {
 public:
  static bool isCheckDebugEnabled();
};

}  // namespace m_engine

#define MENGINE_LIKELY(x) __builtin_expect((x), 1)
#define MENGINE_UNLIKELY(x) __builtin_expect((x), 0)

#define MENGINE_CHECK(val)                                           \
  do {                                                               \
    MENGINE_LIKELY(static_cast<bool>(val))                           \
    ? ((void)0)                                                      \
    : m_engine::common::throwRuntimeError(__FILE__, __LINE__, #val); \
  } while (0)

#define MENGINE_CHECK_WITH_INFO(val, info, ...)                   \
  do {                                                            \
    MENGINE_LIKELY(static_cast<bool>(val))                        \
    ? ((void)0)                                                   \
    : m_engine::common::throwRuntimeError(                        \
          __FILE__, __LINE__,                                     \
          m_engine::common::fmtstr(info, ##__VA_ARGS__).c_str()); \
  } while (0)

#define MENGINE_CHECK_DEBUG(val)                                          \
  do {                                                                    \
    if (MENGINE_UNLIKELY(m_engine::DebugConfig::isCheckDebugEnabled())) { \
      MENGINE_LIKELY(static_cast<bool>(val))                              \
      ? ((void)0)                                                         \
      : m_engine::common::throwRuntimeError(__FILE__, __LINE__, #val);    \
    }                                                                     \
  } while (0)

#define MENGINE_CHECK_DEBUG_WITH_INFO(val, info, ...)                     \
  do {                                                                    \
    if (MENGINE_UNLIKELY(m_engine::DebugConfig::isCheckDebugEnabled())) { \
      MENGINE_LIKELY(static_cast<bool>(val))                              \
      ? ((void)0)                                                         \
      : m_engine::common::throwRuntimeError(                              \
            __FILE__, __LINE__,                                           \
            m_engine::common::fmtstr(info, ##__VA_ARGS__).c_str());       \
    }                                                                     \
  } while (0)
