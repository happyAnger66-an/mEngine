#pragma once

#include "m_engine/runtime/common.h"

namespace m_engine::executor {
class BaseExecutor {
 public:
};

class TRTExecutor : BaseExecutor {
 public:
  size_t size(std::string const& tensorName) const {
    auto record = mNames.find(tensorName);
    if (record == mNames.end()) return kINVALID_SIZE_VALUE;
    return mManagedBuffers[record->second]->hostBuffer.nbBytes();
  }

  TRTExecutor(std::shared_ptr<nvinfer1::ICudaEngine> engine) {
  for (int32_t i = 0; i < mEngine->getNbIOTensors(); i++) {
    auto const name = engine->getIOTensorName(i);
    mNames[name] = i;

    nvinfer1::DataType type = mEngine->getTensorDataType(name);

    std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
    manBuf->deviceBuffer = DeviceBuffer(volumes[i], type);
    manBuf->hostBuffer = HostBuffer(volumes[i], type);
    void* deviceBuffer = manBuf->deviceBuffer.data();
    mDeviceBindings.emplace_back(deviceBuffer);
    mManagedBuffers.emplace_back(std::move(manBuf));
  }
}

 private:
  bool tenosrIsInput(const std::string& tensorName) const {
    return mEngine->getTensorIOMode(tensorName.c_str()) ==
           nvinfer1::TensorIOMode::kINPUT;
  }

  std::shared_ptr<nvinfer1::ICudaEngine>
      mEngine;     //!< The pointer to the engine
  int mBatchSize;  //!< The batch size for legacy networks, 0 otherwise.
  std::vector<std::unique_ptr<ManagedBuffer>>
      mManagedBuffers;  //!< The vector of pointers to managed buffers
  std::vector<void*> mDeviceBindings;  //!< The vector of device buffers needed
                                       //!< for engine execution
  std::unordered_map<std::string, int32_t>
      mNames;  //!< The map of tensor name and index pairs
};
}  // namespace m_engine::executor