#include "m_engine/executor/executor.h"

#include <glog/logging.h>

#include "m_engine/executor/trt_logger.h"

namespace m_engine::executor {

TRTExecutor::TRTExecutor(const std::string& engine_path,
                         const int32_t batchSize) {
  mStream = std::make_shared<CudaStream>();
  mBufferMgr = std::make_unique<BufferManager>(mStream);
  LoadEngine(engine_path);

  mExecContext = std::unique_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext());
  LOG(INFO) << "createExecutionContext ";

  init();
}

int TRTExecutor::LoadEngine(const std::string& engine_file) {
  mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(
      *m_engine::executor::TrtLogger::GetTrtLogger()));
  if (!mRuntime) {
    LOG(ERROR) << "Failed to create TensorRT runtime.";
    return -1;
  }
  LOG(INFO) << "Created TensorRT runtime.";

  auto plan = readEngineFile(engine_file);

  mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
      mRuntime->deserializeCudaEngine(plan.data(), plan.size()));
  return 0;
}

void TRTExecutor::copyHost2Device() {
  for (auto const& n : mName2Index) {
    if (tenosrIsInput(n.first)) {
      auto index = n.second;
      mBufferMgr->copy(*mHostBuffers[index], mDeviceBuffers[index]->data());
    }
  }
}

void TRTExecutor::copyDevice2Host() {
  for (auto const& n : mName2Index) {
    if (!tenosrIsInput(n.first)) {
      auto index = n.second;
      mBufferMgr->copy(*mDeviceBuffers[index], mHostBuffers[index]->data());
    }
  }
}

void TRTExecutor::setIoTensors() {
  for (int32_t i = 0; i < mIoNums; i++) {
    auto const name = mEngine->getIOTensorName(i);
    std::cerr << "setIoTensors " << name << std::endl;
    mExecContext->setTensorAddress(name, mDeviceBindings[i]);
  }
}

void TRTExecutor::PrepareData() { copyHost2Device(); }

void TRTExecutor::Infer() {
  setIoTensors();

  mExecContext->enqueueV3(mStream->get());

  copyDevice2Host();
  mBufferMgr->getStream().synchronize();
}

void TRTExecutor::init() {
  mIoNums = mEngine->getNbIOTensors();

  // Create host and device buffers
  for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++) {
    auto const name = mEngine->getIOTensorName(i);
    mName2Index[name] = i;
    mIndex2Name[i] = name;

    const auto num_opt_profiles = mEngine->getNbOptimizationProfiles();
    auto dims = mExecContext ? mExecContext->getTensorShape(name)
                             : mEngine->getTensorShape(name);
    const auto is_dynamic_tensor =
        std::any_of(dims.d, dims.d + dims.nbDims,
                    [](const int32_t dim) { return dim == -1; });
    if (!is_dynamic_tensor) {
      size_t vol =
          mExecContext || mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
      nvinfer1::DataType type = mEngine->getTensorDataType(name);
      int32_t vecDim = mEngine->getTensorVectorizedDim(name);
      std::cerr << "dims " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2]
                << std::endl;
      if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
      {
        int32_t scalarsPerVec = mEngine->getTensorComponentsPerElement(name);
        dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
        vol *= scalarsPerVec;
      }
      vol *= volume(dims);
      auto hostBuffer = BufferManager::cpu(vol, type);
      auto deviceBuffer = mBufferMgr->gpu(vol, type);
      mDeviceBindings.push_back(deviceBuffer->data());
      mDeviceBuffers.emplace_back(std::move(deviceBuffer));
      mHostBuffers.emplace_back(std::move(hostBuffer));
    } else {
      std::cerr << "dynamic shape: " << name << std::endl;
      mDeviceBindings.push_back(nullptr);
    }
  }
  mIsInit = true;
}

}  // namespace m_engine::executor