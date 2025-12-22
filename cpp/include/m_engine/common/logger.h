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

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "m_engine/common/assert.h"
#include "m_engine/common/mEngineException.h"
#include "m_engine/common/stringUtils.h"

namespace m_engine {

namespace common {

class Logger {
  // On Windows, the file wingdi.h is included which has
  // #define ERROR 0
  // This breaks everywhere ERROR is used in the Level enum
 public:
  enum Level { TRACE = 0, DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40 };

  static Logger* getLogger();

  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;

  template <typename... Args>
  void log(Level const level, char const* format, Args const&... args)
      __attribute__((format(printf, 3, 0)));

  template <typename... Args>
  void log(Level const level, int const rank, char const* format,
           Args const&... args) __attribute__((format(printf, 4, 0)));

  template <typename... Args>
  void log(Level const level, std::string const& format, Args const&... args) {
    return log(level, format.c_str(), args...);
  }

  template <typename... Args>
  void log(Level const level, int const rank, std::string const& format,
           Args const&... args) {
    return log(level, rank, format.c_str(), args...);
  }

  void log(std::exception const& ex, Level level = Level::ERROR);

  Level getLevel() const { return level_; }

  void setLevel(Level const level) {
    level_ = level;
    log(INFO, "Set logger level to %s", getLevelName(level));
  }

  bool isEnabled(Level const level) const { return level_ <= level; }

 private:
  static auto constexpr kPREFIX = "[M-Engine]";

#ifndef NDEBUG
  Level const DEFAULT_LOG_LEVEL = DEBUG;
#else
  Level const DEFAULT_LOG_LEVEL = INFO;
#endif
  Level level_ = DEFAULT_LOG_LEVEL;

  Logger();  // NOLINT(modernize-use-equals-delete)

  static inline char const* getLevelName(Level const level) {
    switch (level) {
      case TRACE:
        return "TRACE";
      case DEBUG:
        return "DEBUG";
      case INFO:
        return "INFO";
      case WARNING:
        return "WARNING";
      case ERROR:
        return "ERROR";
    }

    MENGINE_THROW("Unknown log level: %d", level);
  }

  static inline std::string getPrefix(Level const level) {
    return m_engine::common::fmtstr("%s[%s] ", kPREFIX, getLevelName(level));
  }

  static inline std::string getPrefix(Level const level, int const rank) {
    return m_engine::common::fmtstr("%s[%s][%d] ", kPREFIX, getLevelName(level),
                                    rank);
  }
};

template <typename... Args>
void Logger::log(Logger::Level const level, char const* format,
                 Args const&... args) {
  if (isEnabled(level)) {
    auto const fmt = getPrefix(level) + format;
    auto& out = level_ < WARNING ? std::cout : std::cerr;
    if constexpr (sizeof...(args) > 0) {
      out << fmtstr(fmt.c_str(), args...);
    } else {
      out << fmt;
    }
    out << std::endl;
  }
}

template <typename... Args>
void Logger::log(Logger::Level const level, int const rank, char const* format,
                 Args const&... args) {
  if (isEnabled(level)) {
    auto const fmt = getPrefix(level, rank) + format;
    auto& out = level_ < WARNING ? std::cout : std::cerr;
    if constexpr (sizeof...(args) > 0) {
      out << fmtstr(fmt.c_str(), args...);
    } else {
      out << fmt;
    }
    out << std::endl;
  }
}
}  // namespace common

}  // namespace m_engine

#define MENGINE_LOG(level, ...)                                 \
  do {                                                          \
    auto* const logger = m_engine::common::Logger::getLogger(); \
    if (logger->isEnabled(level)) {                             \
      logger->log(level, __VA_ARGS__);                          \
    }                                                           \
  } while (0)

#define MENGINE_LOG_TRACE(...) \
  MENGINE_LOG(m_engine::common::Logger::TRACE, __VA_ARGS__)
#define MENGINE_LOG_DEBUG(...) \
  MENGINE_LOG(m_engine::common::Logger::DEBUG, __VA_ARGS__)
#define MENGINE_LOG_INFO(...) \
  MENGINE_LOG(m_engine::common::Logger::INFO, __VA_ARGS__)
#define MENGINE_LOG_WARNING(...) \
  MENGINE_LOG(m_engine::common::Logger::WARNING, __VA_ARGS__)
#define MENGINE_LOG_ERROR(...) \
  MENGINE_LOG(m_engine::common::Logger::ERROR, __VA_ARGS__)
#define MENGINE_LOG_EXCEPTION(ex, ...) \
  m_engine::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)
