#pragma once
#include <algorithm>

#include "m_engine/common/assert.h"
#include "m_engine/common/mEngineException.h"
#include "m_engine/common/cudaDriverWrapper.h"

namespace m_engine::common {
static char const* _cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

template <typename T>
void check(T ptr, char const* const func, char const* const file,
           int const line) {
  if (ptr) {
    throw mEngineException(
        file, line,
        fmtstr("[M-ENGINE][ERROR] CUDA runtime error in %s: %s", func,
               _cudaGetErrorEnum(ptr))
            .c_str());
  }
}

template <typename T>
void checkEx(T ptr, std::initializer_list<T> const& validReturns,
             char const* const func, char const* const file, int const line) {
  if (std::all_of(std::begin(validReturns), std::end(validReturns),
                  [&ptr](T const& t) { return t != ptr; })) {
    throw mEngineException(
        file, line,
        fmtstr("[M-ENGINE][ERROR] CUDA runtime error in %s: %s", func,
               _cudaGetErrorEnum(ptr))
            .c_str());
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline int getDevice() {
  int deviceID{0};
  check_cuda_error(cudaGetDevice(&deviceID));
  return deviceID;
}

inline int getDeviceCount() {
  int count{0};
  check_cuda_error(cudaGetDeviceCount(&count));
  return count;
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

}  // namespace m_engine::common

#define MENGINE_CUDA_CHECK(stat)                                \
  do {                                                          \
    m_engine::common::check((stat), #stat, __FILE__, __LINE__); \
  } while (0)

// We use singleton memory pool and the order of destructors depends on the
// compiler implementation. We find that the cudaFree/cudaFreeHost is called
// after cudaruntime destruction on Windows. There will be an
// cudaErrorCudartUnloading error.  However, it is safe to ignore this error
// because the cuda runtime is already exited, we are no more worried about the
// memory leaks.
#define MENGINE_CUDA_CHECK_FREE_RESOURCE(stat)                                 \
  do {                                                                         \
    m_engine::common::checkEx((stat), {cudaSuccess, cudaErrorCudartUnloading}, \
                              #stat, __FILE__, __LINE__);                      \
  } while (0)
