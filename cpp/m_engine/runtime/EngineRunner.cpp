/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/EngineRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "common/version.h"
#include "executor/tensor.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "runtime/llmRuntimeUtils.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace m_engine::runtime;
using namespace nvinfer1;

namespace m_engine::runtime {

namespace {

using Tensor = m_engine::executor::Tensor;
size_t hashDecodingInput(Tensor const &inputIds, Tensor const &outputLogits) {
  // For vanilla decoding step, the shape can be distingusihed by active batch
  // size. Also capture the pointer address to ensure we are read/write correct
  // locations.
  int64_t const activeBatchSize = inputIds.getShape()[0];
  uintptr_t const inputIdsAddr =
      reinterpret_cast<uintptr_t>(inputIds.rawPointer());
  uintptr_t const outputLogitsAddr =
      reinterpret_cast<uintptr_t>(outputLogits.rawPointer());

  size_t hashValue = 0;
  hash_utils::hashCombine(hashValue, activeBatchSize);
  hash_utils::hashCombine(hashValue, inputIdsAddr);
  hash_utils::hashCombine(hashValue, outputLogitsAddr);
  return hashValue;
}

} // namespace

//! Current implementation limits to two optimization profiles per LLM engine.
static constexpr int32_t kPREFILL_PROFILE_INDEX{0};
static constexpr int32_t kGENERATION_PROFILE_INDEX{1};

EngineRunner::EngineRunner(std::filesystem::path const &enginePath,
                           std::filesystem::path const &configPath,
                           cudaStream_t stream) {

  if (!this->initializeConfigFromJson(configJson)) {
    LOG_ERROR("Failed to initialize LLMEngineRunner from config file: %s",
              configPath.string().c_str());
    throw std::runtime_error(
        "Failed to initialize LLMEngineRunner from config file: " +
        configPath.string());
  }

  // Load the engine after config loading succeeds
  LOG_INFO("Loading engine file: %s", enginePath.string().c_str());
  mRuntime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(gLogger));

  auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
  if (mmapReader->getData() == nullptr) {
    LOG_ERROR("LLMEngineRunner(): Failed to use MMap to read engine from file "
              "path: %s",
              enginePath.string());
    throw std::runtime_error(
        "Failed to use MMap to read engine from file path: " +
        enginePath.string());
  }
  mEngine =
      std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(
          mmapReader->getData(), mmapReader->getSize()));

  int64_t const execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
  // Allocate device memory for the execution contexts. UINT8 is used to
  // represent raw bytes.
  mExecContextMemory = rt::Tensor(
      {execContextMemoryInBytes}, rt::DeviceType::kGPU,
      nvinfer1::DataType::kUINT8, "LLMEngineRunner::mExecContextMemory");

  mPrefillExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext(
          ExecutionContextAllocationStrategy::kUSER_MANAGED));
  mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext(
          ExecutionContextAllocationStrategy::kUSER_MANAGED));

  // The prefill and generation contexts of the LLM engine execute serially, can
  // therefore share a single device memory block.
  mPrefillExecutionContext->setDeviceMemoryV2(mExecContextMemory.rawPointer(),
                                              execContextMemoryInBytes);
  mGenerationExecutionContext->setDeviceMemoryV2(
      mExecContextMemory.rawPointer(), execContextMemoryInBytes);
  LOG_INFO("Allocated a shared device memory of %zu bytes for the prefill and "
           "generation contexts.",
           execContextMemoryInBytes);

  bool setOptimizationProfileStatus{true};
  setOptimizationProfileStatus &=
      mPrefillExecutionContext->setOptimizationProfileAsync(
          kPREFILL_PROFILE_INDEX, stream);
  setOptimizationProfileStatus &=
      mGenerationExecutionContext->setOptimizationProfileAsync(
          kGENERATION_PROFILE_INDEX, stream);
  if (!setOptimizationProfileStatus) {
    LOG_ERROR("Failed to set optimization profile to the engine");
    throw std::runtime_error(
        "Failed to set optimization profile to the engine");
  }

  if (!this->validateConfigFromEngine()) {
    LOG_ERROR("Failed to match config file %s with engine file: %s",
              configPath.string().c_str(), enginePath.string().c_str());
    throw std::runtime_error("Failed to match config file " +
                             configPath.string() +
                             " with engine file: " + enginePath.string());
  }


  // Initialize the dummy tensor as TensorRT does not support nullptr for
  // binding Calculate maximum memory requirements across all use cases:
  // 1. Multimodal embeddings: {1, hiddenSize}
  // 2. Attention mask: {maxSupportedBatchSize, 1, 1}
  // 3. Attention position IDs: {maxSupportedBatchSize, 1}
  // 4. LoRA weights: max dimension across all adapters
  // 5. KV cache start index: {maxSupportedBatchSize}
  int64_t maxDummyElements = std::max({
      static_cast<int64_t>(mConfig.hiddenSize), // multimodal embeddings
      static_cast<int64_t>(
          mConfig.maxSupportedBatchSize), // attention mask/pos IDs/KV cache
                                          // start index
      static_cast<int64_t>(getMaxLoraWeightsDimension() *
                           kEMPTY_LORA_RANK), // LoRA weights
  });
  mDummyTensor =
      rt::Tensor({maxDummyElements}, rt::DeviceType::kGPU,
                 nvinfer1::DataType::kHALF, "LLMEngineRunner::mDummyTensor");
  // Initialize dummy tensor memory to zero
  CUDA_CHECK(cudaMemsetAsync(mDummyTensor.rawPointer(), 0,
                             mDummyTensor.getMemoryCapacity(), stream));

  // Set multimodal embeddings to dummy tensor for generation contexts if VLM is
  // enabled
  if (mConfig.isVlm) {
    bool setMultimodalStatus{true};
    setMultimodalStatus &= mGenerationExecutionContext->setTensorAddress(
        binding_names::kImageEmbeds, mDummyTensor.rawPointer());
    setMultimodalStatus &= mGenerationExecutionContext->setInputShape(
        binding_names::kImageEmbeds,
        rt::Coords{1, mConfig.hiddenSize}.getTRTDims());

    // Set deepstack features if exists.
    for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx) {
      std::string deepstackFeatureName =
          binding_names::formatDeepstackFeaturesName(idx);
      setMultimodalStatus &= mGenerationExecutionContext->setTensorAddress(
          deepstackFeatureName.c_str(), mDummyTensor.rawPointer());
      setMultimodalStatus &= mGenerationExecutionContext->setInputShape(
          deepstackFeatureName.c_str(),
          rt::Coords{1, mConfig.hiddenSize}.getTRTDims());
    }

    if (!setMultimodalStatus) {
      LOG_ERROR("Failed to set multimodal embeddings dummy tensor for "
                "generation context");
      throw std::runtime_error("Failed to set multimodal embeddings dummy "
                               "tensor for generation context");
    }
  }

  // Synchronize the stream to ensure all the operations have completed.
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool EngineRunner::initializeConfigFromJson(
    m_engine::inference::TrthEngineConfig const &configPb) {
  // Check model version
  std::string modelVersion =
      configJson.value(binding_names::kEdgellmVersion, "");
  version::checkVersion(modelVersion);

  // Define required fields for main config
  std::vector<std::string> const requiredConfigFields = {
      "num_hidden_layers", "num_key_value_heads", "head_dim", "vocab_size",
      "builder_config"};

  // Validate required fields exist in main config
  for (auto const &field : requiredConfigFields) {
    if (!configJson.contains(field)) {
      LOG_ERROR(
          "initializeConfigFromJson(): Missing required field '%s' in config",
          field.c_str());
      return false;
    }
  }

  LOG_INFO("initializeConfigFromJson(): Loaded LLMEngineRunner with config: %s",
           formatEngineConfig(mConfig).c_str());
  return true;
}

bool LLMEngineRunner::validateConfigFromEngine() {
  auto identifyKVCacheBinding = [](std::string const &bindingName,
                                   Dims const &tensorDim) {
    return tensorDim.nbDims == 5 &&
           bindingName.find(binding_names::kPastKeyValuesTemplate) !=
               std::string::npos;
  };

  // If the engine comes with multimodal embeddings binding, it means the engine
  // supports VLM.
  auto identifyMultimodalEmbeddingsBinding = [](std::string const &bindingName,
                                                Dims const &tensorDim) {
    return tensorDim.nbDims == 2 && bindingName == binding_names::kImageEmbeds;
  };

  // If the engine comes with deepstack features binding, it means the engine is
  // Qwen3-VL.
  auto identifyDeepstackFeaturesBinding = [](std::string const &bindingName,
                                             Dims const &tensorDim) {
    return tensorDim.nbDims == 2 &&
           bindingName.find(binding_names::kDeepstackFeaturesTemplate) !=
               std::string::npos;
  };

  int32_t nbKVCacheInputs{0};
  bool foundMultimodalEmbeddingsInput{false};
  int32_t numIOBindings = mEngine->getNbIOTensors();

  for (int32_t i = 0; i < numIOBindings; ++i) {
    std::string const bindingName = mEngine->getIOTensorName(i);
    Dims const tensorDim = mEngine->getTensorShape(bindingName.c_str());

  }
  // Validate hiddenSize from multimodal embeddings if VLM is enabled
  if (mConfig.isVlm && !foundMultimodalEmbeddingsInput) {
    LOG_ERROR("VLM is enabled but multimodal embeddings input (%s) not found "
              "in engine",
              binding_names::kImageEmbeds);
    return false;
  }
  Dims const minInputPrefillShape =
      mEngine->getProfileShape(binding_names::kInputIds, kPREFILL_PROFILE_INDEX,
                               OptProfileSelector::kMIN);
  Dims const maxInputPrefillShape =
      mEngine->getProfileShape(binding_names::kInputIds, kPREFILL_PROFILE_INDEX,
                               OptProfileSelector::kMAX);
  if (mConfig.minSupportedInputLength != minInputPrefillShape.d[1]) {
    LOG_ERROR("minSupportedInputLength is not consistent. From engine: %d, "
              "from config: %d",
              minInputPrefillShape.d[1], mConfig.minSupportedInputLength);
    return false;
  }
  if (mConfig.maxSupportedInputLength != maxInputPrefillShape.d[1]) {
    LOG_ERROR("maxSupportedInputLength is not consistent. From engine: %d, "
              "from config: %d",
              maxInputPrefillShape.d[1], mConfig.maxSupportedInputLength);
    return false;
  }

  // Validate and potentially override maxSupportedBatchSize from engine's
  // actual max profile
  int32_t const engineMaxBatchSize = maxInputPrefillShape.d[0];
  if (mConfig.maxSupportedBatchSize != engineMaxBatchSize) {
    LOG_ERROR("maxSupportedBatchSize mismatch! Config is %d, engine's max "
              "optimization profile is %d.",
              mConfig.maxSupportedBatchSize, engineMaxBatchSize);
    return false;
  }

  // Obtain vocab size from the engine.
  // Logits shape is [batch_size, num_tokens/num_selected_tokens, vocab_size]
  // for both EAGLE and vanilla models
  Dims const logitsDim = mEngine->getTensorShape(binding_names::kLogits);
  if (mConfig.outputVocabSize != logitsDim.d[2]) {
    LOG_ERROR("vocabSize is not consistent. From engine: %d, expected output "
              "vocab size: %d",
              logitsDim.d[2], mConfig.outputVocabSize);
    return false;
  }

  return true;
}

LLMEngineRunner::~LLMEngineRunner() {
  for (auto &[hashValue, graphPair] : mCudaGraphs) {
    CUDA_CHECK(cudaGraphDestroy(graphPair.first));
    CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
  }
  for (auto &[hashValue, graphPair] : mBaseTreeDecodingCudaGraphs) {
    CUDA_CHECK(cudaGraphDestroy(graphPair.first));
    CUDA_CHECK(cudaGraphExecDestroy(graphPair.second));
  }
}

bool EngineRunner::bindKVCacheToEngine(int32_t activeBatchSize) {
  // Prepare special input binding shape for prefill stage KVCache input.
  Dims const kvCacheDims = {5,
                            {activeBatchSize, 2, mConfig.numKVHeads,
                             mConfig.maxKVCacheCapacity, mConfig.headDim}};
  bool status{true};
  // Bind KV cache tensors to execution contexts
  for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i) {
    std::string const pastKeyValuesName =
        binding_names::formatKVCacheName(i, true);
    std::string const presentKeyValuesName =
        binding_names::formatKVCacheName(i, false);

    rt::Tensor kvCacheBlock = mKVCache.getKVCacheForDecoderLayer(i);
    status &= mPrefillExecutionContext->setTensorAddress(
        pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
    status &= mPrefillExecutionContext->setTensorAddress(
        presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());
    status &= mGenerationExecutionContext->setTensorAddress(
        pastKeyValuesName.c_str(), kvCacheBlock.rawPointer());
    status &= mGenerationExecutionContext->setTensorAddress(
        presentKeyValuesName.c_str(), kvCacheBlock.rawPointer());

    status &= mPrefillExecutionContext->setInputShape(pastKeyValuesName.c_str(),
                                                      kvCacheDims);
    status &= mGenerationExecutionContext->setInputShape(
        pastKeyValuesName.c_str(), kvCacheDims);
  }
  return status;
}

LLMEngineRunnerConfig LLMEngineRunner::getEngineConfig() const {
  return mConfig;
}

bool LLMEngineRunner::prefillStepInputValidation(
    rt::Tensor const &inputIds, rt::Tensor const &contextLengths,
    rt::Tensor const &outputLogits, OptionalOutputTensor outputHiddenStates,
    rt::OptionalInputTensor multimodalEmbeddings,
    rt::OptionalInputTensors extraInputTensors) {
  int32_t activeBatchSize = inputIds.getShape()[0];
  int32_t prefillSequenceLength = inputIds.getShape()[1];

  bool const checkInputsGPUTensor =
      inputIds.getDeviceType() == rt::DeviceType::kGPU &&
      contextLengths.getDeviceType() == rt::DeviceType::kCPU &&
      outputLogits.getDeviceType() == rt::DeviceType::kGPU;
  if (!checkInputsGPUTensor) {
    LOG_ERROR("Invalid device type of I/O tensors. ContextLengths input should "
              "reside on CPU and "
              "the rest should reside on GPU.");
    return false;
  }
  bool const isBatchValid = activeBatchSize <= mConfig.maxSupportedBatchSize &&
                            contextLengths.getShape()[0] == activeBatchSize &&
                            outputLogits.getShape()[0] == activeBatchSize;
  if (!isBatchValid) {
    LOG_ERROR("Invalid batchSize of the input tensors. Either batchSize is "
              "larger than "
              "maxSupportedBatchSize or batchSize is not consistent among the "
              "input tensors. "
              "Current inputIds shape: %s, contextLengths shape: %s, logits "
              "shape: %s",
              inputIds.getShape().formatString().c_str(),
              contextLengths.getShape().formatString().c_str(),
              outputLogits.getShape().formatString().c_str());
    return false;
  }
  if (prefillSequenceLength > mConfig.maxSupportedInputLength) {
    LOG_ERROR("Invalid sequence length of the input tensors. Input sequence "
              "length (%d) is larger "
              "than maxSupportedInputLength (%d). Current inputIds shape: %s.",
              prefillSequenceLength, mConfig.maxSupportedInputLength,
              inputIds.getShape().formatString().c_str());
    return false;
  }

  // Validate multimodal embeddings based on is_vlm flag
  bool const isMultimodalEmbeddingsValid =
      (mConfig.isVlm && multimodalEmbeddings.has_value() &&
       multimodalEmbeddings.value().get().getShape().getNumDims() == 2 &&
       multimodalEmbeddings.value().get().getShape()[1] ==
           mConfig.hiddenSize) ||
      (!mConfig.isVlm && !multimodalEmbeddings.has_value());
  if (!isMultimodalEmbeddingsValid) {
    LOG_ERROR("Invalid multimodal embeddings. VLM=%s, provided=%s, expected "
              "shape=[*, %d]. Current shape: %s",
              mConfig.isVlm ? "true" : "false",
              multimodalEmbeddings.has_value() ? "true" : "false",
              mConfig.hiddenSize,
              multimodalEmbeddings.has_value() ? multimodalEmbeddings.value()
                                                     .get()
                                                     .getShape()
                                                     .formatString()
                                                     .c_str()
                                               : "None");
    return false;
  }

  // Validate extra input tensors, e.g. deepstack features for Qwen3-VL
  int32_t deepstackFeaturesCount{0};
  for (auto const &tensorRef : extraInputTensors) {
    rt::Tensor const &tensor = tensorRef.get();
    std::string const tensorName = tensor.getName();

    // Deepstack features
    if (tensorName.find(binding_names::kDeepstackFeaturesTemplate) !=
        std::string::npos) {
      bool const isTensorValid =
          tensor.getDeviceType() == rt::DeviceType::kGPU &&
          tensor.getShape().getNumDims() == 2 &&
          tensor.getShape()[1] == mConfig.hiddenSize;
      if (!isTensorValid) {
        LOG_ERROR("Invalid deepstack feature '%s'. Expected device type: GPU, "
                  "shape: [*, %d]. Current shape: %s",
                  tensorName.c_str(), mConfig.hiddenSize,
                  tensor.getShape().formatString().c_str());
        return false;
      }
      ++deepstackFeaturesCount;
    }
  }
  if (deepstackFeaturesCount != mConfig.numDeepstackFeatures) {
    LOG_ERROR("Invalid deepstack features count. Expected %d, got %d",
              mConfig.numDeepstackFeatures, deepstackFeaturesCount);
    return false;
  }

  bool const isLogitsShapeValid =
      outputLogits.getShape().getNumDims() == 2 &&
      outputLogits.getShape()[1] == mConfig.outputVocabSize;
  if (!isLogitsShapeValid) {
    LOG_ERROR("Invalid shape of the output logits tensor. The output logits "
              "tensor should have shape "
              "[activeBatchSize, outputVocabSize]. Current logits shape is %s.",
              outputLogits.getShape().formatString().c_str());
    return false;
  }
  
  return true;
}

bool EngineRunner::executePrefillStep(
    rt::Tensor const &inputIds, rt::Tensor const &hostContextLengths,
    rt::OptionalInputTensor multimodalEmbeddings,
    rt::OptionalInputTensors extraInputTensors, rt::Tensor &outputLogits,
    rt::OptionalOutputTensor outputHiddenStates, cudaStream_t stream) {
  // Bind the KVCache IO to the engine.
  setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

  if (!setEngineIOStatus) {
    LOG_ERROR(
        "executePrefill(): Failed to bind engine input and output tensors.");
    return false;
  }

  // launch the engine execution.
  bool executeStatus{true};
  executeStatus &= mPrefillExecutionContext->enqueueV3(stream);
  if (!executeStatus) {
    LOG_ERROR(
        "executePrefill(): Failed on TensorRT prefill stage enqueueV3() call.");
    return false;
  }
  // Prefill operation has completed, commit the new contents with KVCache.
  mKVCache.commitSequenceLength(mSequenceContextLengths, stream);

  LOG_DEBUG("executePrefill(): Prefill stage execution completed for request "
            "with batch size %d.",
            activeBatchSize);
  return true;
}

bool EngineRunner::captureCudaGraph(rt::Tensor const &baseTreeDecodingInputIds,
                                    rt::Tensor const &baseTreeDecodingMask,
                                    rt::Tensor &outputLogits,
                                    rt::Tensor &outputHiddenStates,
                                    cudaStream_t stream) {
  size_t const hashValue = hashBaseTreeDecodingInput(
      baseTreeDecodingInputIds, outputLogits, outputHiddenStates);
  if (mBaseTreeDecodingCudaGraphs.find(hashValue) !=
      mBaseTreeDecodingCudaGraphs.end()) {
    LOG_INFO("captureEagleBaseTreeDecodingCudaGraph(): CUDA graph already "
             "captured for the input tensors.");
    return true;
  }

  // Here we will simulate the state of the EngineRunner after executing one
  // prefill request for a batched request.
  int32_t const activeBatchSize = baseTreeDecodingInputIds.getShape()[0];
  constexpr int32_t simulateCacheLength{128};
  std::vector<int32_t> reuseKVCacheLengths(activeBatchSize,
                                           simulateCacheLength);
  rt::Tensor const reuseKVCacheLengthsTensor(
      reuseKVCacheLengths.data(), {activeBatchSize}, rt::DeviceType::kCPU,
      DataType::kINT32);

  mKVCache.resetForNewSequences(reuseKVCacheLengthsTensor, stream);

  bool const validateInputStatus =
      this->eagleBaseTreeDecodingStepInputValidation(
          baseTreeDecodingInputIds, baseTreeDecodingMask, outputLogits,
          outputHiddenStates);
  if (!validateInputStatus) {
    LOG_ERROR("captureEagleBaseTreeDecodingCudaGraph(): Eagle base tree "
              "decoding request not performed due to invalid "
              "input "
              "tensors.");
    return false;
  }

  // Prepare extra input for engine execution. Assemble packed base tree
  // decoding mask, position indices, select token indices, sequence context
  // lengths.
  int32_t const baseTreeDecodingSize =
      static_cast<int32_t>(baseTreeDecodingInputIds.getShape()[1]);
  int32_t const packedBaseTreeDecodingMaskLen =
      static_cast<int32_t>(divUp(baseTreeDecodingSize, 32));
  mSelectTokenIndices.reshape(
      {activeBatchSize, baseTreeDecodingSize}); // 2D tensor [batch, num_tokens]
  mSequenceContextLengths.reshape({activeBatchSize});
  mEagleBasePositionIds.reshape({activeBatchSize, baseTreeDecodingSize});
  mEagleBasePackedMask.reshape(
      {activeBatchSize, baseTreeDecodingSize, packedBaseTreeDecodingMaskLen});

  rt::Tensor const &sequenceStartIndices = mKVCache.getKVCacheLengths();

  kernel::prepareEagleBaseTreeDecodingInputs(
      baseTreeDecodingMask, sequenceStartIndices, mEagleBasePackedMask,
      mEagleBasePositionIds, mSelectTokenIndices, mSequenceContextLengths,
      stream);

  // Bind the input and output tensor into the engine. RopeCosSinCache and
  // KVCache are pre-bind during runner initialization.
  bool setEngineIOStatus{true};

  // Update KV cache shapes to match activeBatchSize for CUDA graph capture
  setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kKVCacheStartIndex,
      mKVCache.getKVCacheLengths().getShape().getTRTDims());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kKVCacheStartIndex,
      mKVCache.getKVCacheLengths().rawPointer());

  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kInputIds,
      const_cast<void *>(baseTreeDecodingInputIds.rawPointer()));
  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kInputIds,
      baseTreeDecodingInputIds.getShape().getTRTDims());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kContextLengths, mSequenceContextLengths.rawPointer());
  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kContextLengths,
      mSequenceContextLengths.getShape().getTRTDims());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kLastTokenIds, mSelectTokenIndices.rawPointer());
  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kLastTokenIds,
      mSelectTokenIndices.getShape().getTRTDims());

  // For MRope (VLM), reshape the RopeCosSinCache to match the activeBatchSize
  if (mConfig.ropeConfig.type == RopeType::kMRope) {
    mPosEncCosSinCache.reshape(
        {activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim});
  }

  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kRopeCosSin, mPosEncCosSinCache.getShape().getTRTDims());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kAttentionMask, mEagleBasePackedMask.rawPointer());
  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kAttentionMask,
      mEagleBasePackedMask.getShape().getTRTDims());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kAttentionPosId, mEagleBasePositionIds.rawPointer());
  setEngineIOStatus &= mGenerationExecutionContext->setInputShape(
      binding_names::kAttentionPosId,
      mEagleBasePositionIds.getShape().getTRTDims());

  // Bind the output tensor into the engine.
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kLogits, outputLogits.rawPointer());
  setEngineIOStatus &= mGenerationExecutionContext->setTensorAddress(
      binding_names::kOutputHiddenStates, outputHiddenStates.rawPointer());

  setEngineIOStatus &= this->bindKVCacheToEngine(activeBatchSize);

  if (!setEngineIOStatus) {
    LOG_ERROR("captureEagleBaseTreeDecodingCudaGraph(): Failed to bind engine "
              "input and output tensors.");
    return false;
  }

  // launch the engine execution. This will trigger the shape machine of
  // TensorRT engine to avoid cudaGraph capture. error.
  bool executeStatus{true};
  executeStatus &= mGenerationExecutionContext->enqueueV3(stream);

  if (!executeStatus) {
    LOG_ERROR("captureEagleBaseTreeDecodingCudaGraph(): Failed on TensorRT "
              "eagle base tree decoding stage enqueueV3() "
              "call.");
    return false;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  executeStatus &= mGenerationExecutionContext->enqueueV3(stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
  mBaseTreeDecodingCudaGraphs[hashValue] = std::make_pair(graph, graphExec);

  if (!executeStatus) {
    LOG_WARNING("captureEagleBaseTreeDecodingCudaGraph(): Failed on TensorRT "
                "engine enqueueV3() call during CUDA graph "
                "capture.");
    return false;
  } else {
    LOG_DEBUG("captureEagleBaseTreeDecodingCudaGraph(): CUDA graph captured "
              "successfully for input shape %s.",
              baseTreeDecodingInputIds.getShape().formatString().c_str());
  }

  return true;
}

} // namespace m_engine::runtime
