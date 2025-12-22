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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "m_engine/common/stringUtils.h"

#define MENGINE_THROW(...)                    \
  do {                                        \
    throw NEW_MENGINE_EXCEPTION(__VA_ARGS__); \
  } while (0)

#define MENGINE_WRAP(ex)                                                       \
  NEW_MENGINE_EXCEPTION(                                                       \
      "%s: %s",                                                                \
      m_engine::common::mEngineException::demangle(typeid(ex).name()).c_str(), \
      ex.what())

#define NEW_MENGINE_EXCEPTION(...)    \
  m_engine::common::mEngineException( \
      __FILE__, __LINE__, m_engine::common::fmtstr(__VA_ARGS__).c_str())

namespace m_engine {

namespace common {

/// @brief Enumeration of different error codes for request-specific exceptions
enum class RequestErrorCode : uint32_t {
  // General errors (0-999)
  kUNKNOWN_ERROR = 0,

  // Network and communication errors (1000-1999)
  kNETWORK_ERROR = 1000,
};

/// @brief Constant for unknown request ID
static constexpr uint64_t kUNKNOWN_REQUEST_ID =
    std::numeric_limits<uint64_t>::max();

class mEngineException : public std::runtime_error {
 public:
  static auto constexpr MAX_FRAMES = 128;

  explicit mEngineException(char const* file, std::size_t line,
                            char const* msg);

  ~mEngineException() noexcept override;

  [[nodiscard]] std::string getTrace() const;

  static std::string demangle(char const* name);

 private:
  std::array<void*, MAX_FRAMES> mCallstack{};
  int mNbFrames;
};

[[noreturn]] inline void throwRuntimeError(char const* const file,
                                           int const line, char const* info) {
  throw mEngineException(
      file, line,
      m_engine::common::fmtstr("[mEngine][ERROR] Assertion failed: %s", info)
          .c_str());
}

[[noreturn]] inline void throwRuntimeError(char const* const file,
                                           int const line,
                                           std::string const& info = "") {
  throw mEngineException(
      file, line,
      m_engine::common::fmtstr("[mEngine][ERROR] Assertion failed: %s",
                               info.c_str())
          .c_str());
}

}  // namespace common

}  // namespace m_engine