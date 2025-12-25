#include "m_engine/executor/trt_bindings.h"

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace m_engine::executor {

int64_t TrtBindings::GetBindingVolume(const nvinfer1::ICudaEngine& engine,
                                      const int binding_index) {
  const auto& binding_dims = engine.getTensorShape(binding_index);
  const auto is_dynamic_tensor =
      std::any_of(binding_dims.d, binding_dims.d + binding_dims.nbDims,
                  [](const int32_t dim) { return dim == -1; });

  const auto num_opt_profiles = engine.getNbOptimizationProfiles();
  const bool is_shape_binding = engine.isShapeBinding(binding_index);
  // volume of a tensor is the total number of elements in the tensor
  int64_t volume = 0;
  if (is_dynamic_tensor) {
    // if the tensor has dynamic shapes, we use the max dimension in the
    // optimization profile to preallocate memory
    for (int opt_profile_index = 0; opt_profile_index < num_opt_profiles;
         ++opt_profile_index) {
      if (!is_shape_binding) {
        // nvinfer1::OptProfileSelector::kMAX
        for (const auto& selector : {nvinfer1::OptProfileSelector::kMAX,
                                     nvinfer1::OptProfileSelector::kMAX,
                                     nvinfer1::OptProfileSelector::kOPT}) {
          const auto& dims = engine.getProfileDimensions(
              binding_index, opt_profile_index, selector);
          LOG(INFO) << static_cast<int>(selector) << " : " << dims;
        }
        const auto& max_dims =
            engine.getProfileDimensions(binding_index, opt_profile_index,
                                        nvinfer1::OptProfileSelector::kMAX);
        volume = std::max(volume, Volume(max_dims));
      } else {
        LOG(FATAL) << "Shape value tensors are not currently supported yet.";
      }
    }
  } else {
    // For static shaped tensors, calculates the tensor volume by accumulating
    // each dimension via multiplication.
    volume = Volume(binding_dims);
  }
  return volume;
}

std::vector<int> TrtBindings::GetBindingDims(
    const nvinfer1::ICudaEngine& engine, const int binding_index) {
  const auto& binding_dims = engine.getBindingDimensions(binding_index);
  const auto is_dynamic_tensor =
      std::any_of(binding_dims.d, binding_dims.d + binding_dims.nbDims,
                  [](const int32_t dim) { return dim == -1; });

  const auto num_opt_profiles = engine.getNbOptimizationProfiles();
  const bool is_shape_binding = engine.isShapeBinding(binding_index);
  std::vector<int> dims;
  if (is_dynamic_tensor) {
    // if the tensor has dynamic shapes, we use the max dimension in the
    // optimization profile to preallocate memory
    for (int opt_profile_index = 0; opt_profile_index < num_opt_profiles;
         ++opt_profile_index) {
      if (!is_shape_binding) {
        // nvinfer1::OptProfileSelector::kMAX
        for (const auto& selector : {nvinfer1::OptProfileSelector::kMAX,
                                     nvinfer1::OptProfileSelector::kMAX,
                                     nvinfer1::OptProfileSelector::kOPT}) {
          const auto& dims = engine.getProfileDimensions(
              binding_index, opt_profile_index, selector);
          LOG(INFO) << static_cast<int>(selector) << " : " << dims;
        }
        const auto& max_dims =
            engine.getProfileDimensions(binding_index, opt_profile_index,
                                        nvinfer1::OptProfileSelector::kMAX);
        dims.resize(max_dims.nbDims);
        for (int i = 0; i < max_dims.nbDims; ++i) {
          dims[i] = max_dims.d[i];
        }
      } else {
        LOG(FATAL) << "Shape value tensors are not currently supported yet.";
      }
    }
  } else {
    // For static shaped tensors, calculates the tensor volume by accumulating
    // each dimension via multiplication.
    dims.resize(binding_dims.nbDims);
    for (int i = 0; i < binding_dims.nbDims; ++i) {
      dims[i] = binding_dims.d[i];
    }
  }

  return dims;
}

void TrtBindings::CheckBinding(const nvinfer1::ICudaEngine& engine,
                               const int binding_index) {
  const auto* binding_name = engine.getBindingName(binding_index);
  LOG(INFO) << "Binding " << binding_index << " name: " << binding_name;
  bool is_shape_binding = engine.isShapeBinding(binding_index);
  LOG(INFO) << "\tIs shape binding: " << is_shape_binding;
  bool is_binding_input = engine.bindingIsInput(binding_index);
  LOG(INFO) << "\tIs binding input: " << is_binding_input;
  if (is_binding_input) {
    input_binding_indices_.emplace_back(binding_index);
  } else {
    output_binding_indices_.emplace_back(binding_index);
  }
  const auto& binding_dims = engine.getBindingDimensions(binding_index);
  LOG(INFO) << "\tBinding dims: " << binding_dims;
  const auto is_dynamic_tensor =
      std::any_of(binding_dims.d, binding_dims.d + binding_dims.nbDims,
                  [](const int32_t dim) { return dim == -1; });
  LOG(INFO) << "\tIs dynamic: " << is_dynamic_tensor;
  const auto& binding_data_type = engine.getBindingDataType(binding_index);
  LOG(INFO) << "\tBinding data type: "
            << cuda::CudaDataType_Name(GetDataType(binding_data_type));
  const auto& binding_location = engine.getLocation(binding_index);
  LOG(INFO) << "\tBinding location: "
            << (binding_location == nvinfer1::TensorLocation::kDEVICE ? "Device"
                                                                      : "Host");
  // const auto& binding_bytes_per_component =
  //     engine.getBindingBytesPerComponent(binding_index);
  // LOG(INFO) << "\tBinding bytes per component: "
  //           << binding_bytes_per_component;
  // const auto& binding_components_per_element =
  //     engine.getBindingComponentsPerElement(binding_index);
  // LOG(INFO) << "\tBinding components per element: "
  //           << binding_components_per_element;
  const auto* binding_format_desc = engine.getBindingFormatDesc(binding_index);
  LOG(INFO) << "\tBinding format: " << binding_format_desc;
  const auto binding_vectorized_dim =
      engine.getBindingVectorizedDim(binding_index);
  LOG(INFO) << "\tBinding vectorized dim: " << binding_vectorized_dim;
  const auto is_execution_binding = engine.isExecutionBinding(binding_index);
  LOG(INFO) << "\tIs execution binding: " << is_execution_binding;
}

}  // namespace m_engine::executor
