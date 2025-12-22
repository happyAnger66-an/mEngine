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

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <cinttypes>
#include <cstdlib>
#include <sstream>

#include "m_engine/common/mEngineException.h"
#include "m_engine/common/stringUtils.h"

namespace m_engine {

namespace common {

namespace {
int constexpr VOID_PTR_SZ = 2 + sizeof(void*) * 2;
}

mEngineException::mEngineException(char const* file, std::size_t line,
                             char const* msg)
    : std::runtime_error{""} {
  mNbFrames = backtrace(mCallstack.data(), MAX_FRAMES);
  auto const trace = getTrace();
  std::runtime_error::operator=(std::runtime_error{
      fmtstr("%s (%s:%zu)\n%s", msg, file, line, trace.c_str())});
}

mEngineException::~mEngineException() noexcept = default;

std::string mEngineException::getTrace() const {
  auto const trace = std::unique_ptr<char const*, void (*)(char const**)>(
      const_cast<char const**>(backtrace_symbols(mCallstack.data(), mNbFrames)),
      [](char const** p) { std::free(p); });
  if (trace == nullptr) {
    throw std::bad_alloc();
  }
  std::ostringstream buf;
  for (auto i = 1; i < mNbFrames; ++i) {
    Dl_info info;
    if (dladdr(mCallstack[i], &info) && info.dli_sname) {
      auto const clearName = demangle(info.dli_sname);
      buf << fmtstr("%-3d %*p %s + %zd", i, VOID_PTR_SZ, mCallstack[i],
                    clearName.c_str(),
                    static_cast<char*>(mCallstack[i]) -
                        static_cast<char*>(info.dli_saddr));
    } else {
      buf << fmtstr("%-3d %*p %s", i, VOID_PTR_SZ, mCallstack[i],
                    trace.get()[i]);
    }
    if (i < mNbFrames - 1) buf << std::endl;
  }

  if (mNbFrames == MAX_FRAMES) buf << std::endl << "[truncated]";

  return buf.str();
}

std::string mEngineException::demangle(char const* name) {
  std::string clearName{name};
  auto status = -1;
  auto const demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (status == 0) {
    clearName = demangled;
    std::free(demangled);
  }
  return clearName;
}

}  // namespace common

}  // namespace m_engine