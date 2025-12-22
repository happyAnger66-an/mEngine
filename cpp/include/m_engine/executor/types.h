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

#include <cuda_fp16.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace m_engine::runtime {
class CudaStream;
}  // namespace m_engine::runtime

namespace m_engine::executor {

class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using VecTokenExtraIds = std::vector<IdType>;
using IterationType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;
using StreamPtr = std::shared_ptr<m_engine::runtime::CudaStream>;
using MillisecondsType = std::chrono::milliseconds;
using CacheSaltIDType = std::uint64_t;
using LogitsPostProcessor =
    std::function<void(IdType, Tensor&, BeamTokens const&, StreamPtr const&,
                       std::optional<IdType>)>;
using LogitsPostProcessorMap =
    std::unordered_map<std::string, LogitsPostProcessor>;
using LogitsPostProcessorBatched = std::function<void(
    std::vector<IdType> const&, std::vector<Tensor>&,
    std::vector<std::reference_wrapper<BeamTokens const>> const&,
    StreamPtr const&, std::vector<std::optional<IdType>> const&)>;
using MedusaChoices = std::vector<std::vector<SizeType32>>;
using EagleChoices = std::vector<std::vector<SizeType32>>;
using PriorityType = float;
using BufferView = std::basic_string_view<uint8_t>;

enum class DataType {
  kBOOL,
  kUINT8,
  kINT8,
  kINT32,
  kINT64,
  kBF16,
  kFP8,
  kFP16,
  kFP32,
  kUNKNOWN
};

enum class RequestType {
  REQUEST_TYPE_CONTEXT_AND_GENERATION = 0,
  REQUEST_TYPE_CONTEXT_ONLY = 1,
  REQUEST_TYPE_GENERATION_ONLY = 2
};

//! \brief For converting a C++ data type to a `TrtLmmDataType`.
template <typename T, bool = false>
struct TypeTraits {};

template <>
struct TypeTraits<float> {
  static constexpr auto value = DataType::kFP32;
};

template <>
struct TypeTraits<half> {
  static constexpr auto value = DataType::kFP16;
};

template <>
struct TypeTraits<std::int8_t> {
  static constexpr auto value = DataType::kINT8;
};

template <>
struct TypeTraits<std::int32_t> {
  static constexpr auto value = DataType::kINT32;
};

template <>
struct TypeTraits<std::int64_t> {
  static constexpr auto value = DataType::kINT64;
};

template <>
struct TypeTraits<bool> {
  static constexpr auto value = DataType::kBOOL;
};

template <>
struct TypeTraits<std::uint8_t> {
  static constexpr auto value = DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TypeTraits<__nv_bfloat16> {
  static constexpr auto value = DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TypeTraits<__nv_fp8_e4m3> {
  static constexpr auto value = DataType::kFP8;
};
#endif

template <typename T>
struct TypeTraits<T*> {
  // Pointers are stored as int64_t.
  static constexpr auto value = DataType::kINT64;
};

enum class MemoryType {
  kCPU,
  kCPU_PINNED,
  kCPU_PINNEDPOOL,
  kGPU,
  kUVM,
  kUNKNOWN
};

enum class ModelType {
  kDECODER_ONLY = 0,
  kENCODER_ONLY = 1,
  kENCODER_DECODER = 2,
};

/// @brief Struct that holds the debug tensors in an iteration
struct DebugTensorsPerIteration {
  /// @brief The iteration id for these tensors
  IterationType iter;
  /// @brief The debug tensors for this iteration
  std::map<std::string, Tensor> debugTensors;
};

/// @brief The reason why the model stopped generating tokens for a request.
enum class FinishReason {
  /// @brief The request is not finished.
  kNOT_FINISHED = 0,

  /// @brief The request finished because the end id was generated.
  kEND_ID = 1,

  /// @brief The request finished because a stop word was generated.
  kSTOP_WORDS = 2,

  /// @brief The request finished because the maximum number of tokens was
  /// reached.
  kLENGTH = 3,

  /// @brief The request finished because it got timed out (via the mAllotedTime
  /// parameter)
  kTIMED_OUT = 4,

  /// @brief The request was cancelled by calling cancelRequest.
  kCANCELLED = 5
};

}  // namespace m_engine::executor
