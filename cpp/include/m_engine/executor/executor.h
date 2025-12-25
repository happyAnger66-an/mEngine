#pragma once
#include <cassert>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

#include "m_engine/runtime/bufferManager.h"
#include "m_engine/runtime/common.h"
#include "m_engine/runtime/cudaStream.h"
#include "m_engine/runtime/iBuffer.h"

namespace m_engine::executor {
class BaseExecutor {
 public:
};

class TRTExecutor : BaseExecutor {
  using IBuffer = m_engine::runtime::IBuffer;
  using BufferManager = m_engine::runtime::BufferManager;
  using CudaStream = m_engine::runtime::CudaStream;

 public:
  explicit TRTExecutor(const std::string& engine_path,
                       const int32_t batchSize = 0);

  void Infer() {
    copyHost2Device();
    setIoTensors();

    mExecContext->enqueueV3(mStream->get());

    copyDevice2Host();
    mBufferMgr->getStream().synchronize();
  }

  IBuffer::SharedPtr GetHostBuffer(const std::string& name) {
    auto index = mName2Index.at(name);
    return mHostBuffers[index];
  }

  IBuffer::SharedPtr GetDeviceBuffer(const std::string& name) {
    auto index = mName2Index.at(name);
    return mDeviceBuffers[index];
  }

  int32_t GetBufferSize(std::string const& name) const {
    auto iter = mName2Index.find(name);
    if (iter == mName2Index.end()) return -1;
    return mHostBuffers[iter->second]->getSize();
  }

 private:
  bool checkIs_init() { return mIsInit; }
  bool tenosrIsInput(const std::string& tensorName) const {
    return mEngine->getTensorIOMode(tensorName.c_str()) ==
           nvinfer1::TensorIOMode::kINPUT;
  }

  int LoadEngine(const std::string& engine_file);

  std::vector<char> readEngineFile(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
      throw std::runtime_error("Failed to read engine file");
    }
    return buffer;
  }

  inline int64_t volume(nvinfer1::Dims const& d) {
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1},
                           std::multiplies<int64_t>{});
  }

  template <typename A, typename B>
  inline auto divUp(A m, B n) -> typename std::enable_if_t<
      std::is_integral<A>::value && std::is_integral<B>::value, A> {
    assert(n > 0);
    return (m + n - 1) / n;
  }

  void copyHost2Device() {
    for (auto const& n : mName2Index) {
      if (tenosrIsInput(n.first)) {
        auto index = n.second;
        mBufferMgr->copy(*mHostBuffers[index], mDeviceBuffers[index]->data());
      }
    }
  }

  void copyDevice2Host() {
    for (auto const& n : mName2Index) {
      if (!tenosrIsInput(n.first)) {
        auto index = n.second;
        mBufferMgr->copy(*mDeviceBuffers[index], mHostBuffers[index]->data());
      }
    }
  }

  void setIoTensors() {
    for (int32_t i = 0; i < mIoNums; i++) {
      auto const name = mEngine->getIOTensorName(i);
      mExecContext->setTensorAddress(name, mDeviceBindings[i]);
    }
  }

  void init();

  bool mIsInit = false;
  int32_t mIoNums;
  int32_t mBatchSize;
  std::unique_ptr<BufferManager> mBufferMgr;
  std::vector<IBuffer::SharedPtr> mDeviceBuffers;
  std::vector<IBuffer::SharedPtr> mHostBuffers;
  std::vector<void*> mDeviceBindings;

  std::unordered_map<std::string, int32_t> mName2Index;
  std::unordered_map<int32_t, std::string> mIndex2Name;

  std::shared_ptr<CudaStream> mStream;
  std::unique_ptr<nvinfer1::IRuntime> mRuntime;
  std::unique_ptr<nvinfer1::ICudaEngine>
      mEngine;  //!< The pointer to the engine
                //!< for engine execution
  std::unique_ptr<nvinfer1::IExecutionContext> mExecContext;
};

}  // namespace m_engine::executor