#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferVersion.h>
#include <driver_types.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
}

namespace m_engine::executor {

// TrtBindings provide memory management and upload/download functionalities for
// easy TensorRT deployment. There are two types of memory that TrtBindings
// manages: 1) binding buffers that TRT directly accesses; 2) auxiliary buffers
// that can be used for pre/post processing. The first type of memory is
// initialized with an TensorRT CUDA engine; the second auxiliary buffers need
// to be allocated by the user manually.
class TrtBindings {
 public:
  TrtBindings() = default;
  ~TrtBindings() = default;

  TrtBindings(const TrtBindings&) = delete;
  TrtBindings(const TrtBindings&&) = delete;
  TrtBindings& operator=(const TrtBindings&) = delete;
  TrtBindings& operator=(const TrtBindings&&) = delete;

 private:
  static void CheckBinding(const nvinfer1::ICudaEngine& engine, int binding_index);
  static int64_t GetBindingVolume(const nvinfer1::ICudaEngine& engine,
                                  int binding_index);
  static std::vector<int> GetBindingDims(const nvinfer1::ICudaEngine& engine,
                                         int binding_index);
};

}  // namespace cargo::onboard::inference
