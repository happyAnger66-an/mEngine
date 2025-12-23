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

#include "m_engine/executor/serialization.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

#include "m_engine/executor/executor.h"
#include "m_engine/executor/types.h"
#include "m_engine/runtime/cudaStream.h"

namespace m_engine::executor {

// Tensor
Tensor Serialization::deserializeTensor(std::istream& is) {
  // DataType
  DataType dataType{};
  is.read(reinterpret_cast<char*>(&dataType), sizeof(dataType));

  // Shape
  size_t shapeSize{0};
  is.read(reinterpret_cast<char*>(&shapeSize), sizeof(size_t));
  static constexpr int32_t MAX_DIMS{8};
  MENGINE_CHECK(shapeSize < MAX_DIMS);

  Shape::DimType64 dims[MAX_DIMS];
  is.read(reinterpret_cast<char*>(&dims[0]),
          shapeSize * sizeof(Shape::DimType64));
  Shape shape(&dims[0], shapeSize);

  // Memory Type
  MemoryType memoryType{};
  is.read(reinterpret_cast<char*>(&memoryType), sizeof(memoryType));

  // Size in bytes
  size_t sizeInBytes{0};
  is.read(reinterpret_cast<char*>(&sizeInBytes), sizeof(size_t));

  Tensor tensor;
  switch (memoryType) {
    case MemoryType::kCPU: {
      tensor = Tensor::cpu(dataType, shape);
      is.read(reinterpret_cast<char*>(tensor.getData()),
              static_cast<std::streamsize>(sizeInBytes));
      break;
    }
    case MemoryType::kCPU_PINNED: {
      tensor = Tensor::pinned(dataType, shape);
      is.read(reinterpret_cast<char*>(tensor.getData()),
              static_cast<std::streamsize>(sizeInBytes));
      break;
    }
    case MemoryType::kUVM: {
      tensor = Tensor::managed(dataType, shape);
      is.read(reinterpret_cast<char*>(tensor.getData()),
              static_cast<std::streamsize>(sizeInBytes));
      break;
    }
    case MemoryType::kGPU: {
      // TODO: Eventually we might want to support serialization/deserialization
      // in GPU memory
      //       Until then created Pinned tensor and move to GPU
      auto pinnedTensor = Tensor::pinned(dataType, shape);
      is.read(reinterpret_cast<char*>(pinnedTensor.getData()),
              static_cast<std::streamsize>(sizeInBytes));
      auto stream = std::make_shared<m_engine::runtime::CudaStream>();
      tensor = pinnedTensor.copyToGpu(stream);
      stream->synchronize();
      break;
    }
    case MemoryType::kUNKNOWN: {
      MENGINE_THROW("Cannot deserialize tensor with UNKNOWN type.");
      break;
    }
    default: {
      MENGINE_THROW("Memory type not supported in deserializeTensor.");
      break;
    }
  }

  return tensor;
}

void Serialization::serialize(Tensor const& tensor, std::ostream& os) {
  auto dataType = tensor.getDataType();
  os.write(reinterpret_cast<char const*>(&dataType), sizeof(dataType));
  auto shape = tensor.getShape();
  auto shapeSize = shape.size();
  os.write(reinterpret_cast<char const*>(&shapeSize), sizeof(shapeSize));
  os.write(reinterpret_cast<char const*>(&shape[0]),
           shapeSize * sizeof(Shape::DimType64));

  // Memory Type
  auto memoryType = tensor.getMemoryType();
  os.write(reinterpret_cast<char const*>(&memoryType), sizeof(memoryType));

  std::size_t sizeInBytes = tensor.getSizeInBytes();
  os.write(reinterpret_cast<char const*>(&sizeInBytes), sizeof(sizeInBytes));

  if (memoryType == MemoryType::kCPU || memoryType == MemoryType::kCPU_PINNED ||
      memoryType == MemoryType::kUVM) {
    void const* data = tensor.getData();
    os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
  }
  // Need special treatment for GPU type
  else if (memoryType == MemoryType::kGPU) {
    auto stream = std::make_shared<m_engine::runtime::CudaStream>();
    auto pinnedTensor = tensor.copyToPinned(stream);
    stream->synchronize();
    void const* data = pinnedTensor.getData();
    os.write(reinterpret_cast<char const*>(data), std::streamsize(sizeInBytes));
  } else if (memoryType == MemoryType::kUNKNOWN) {
    MENGINE_THROW("Memory type unknown when serializing tensor");
  }
}

size_t Serialization::serializedSize(Tensor const& tensor) {
  size_t totalSize = 0;
  totalSize += sizeof(tensor.getDataType());  // datatype
  auto const shape = tensor.getShape();
  auto const shapeSize = shape.size();
  totalSize += sizeof(decltype(shapeSize));  // number of dims
  MENGINE_CHECK(shapeSize > 0);
  totalSize += shapeSize * sizeof(decltype(shape[0]));

  auto memoryType = tensor.getMemoryType();
  totalSize += sizeof(memoryType);  // memory type

  totalSize += sizeof(size_t);  // Size in bytes
  totalSize += tensor.getSizeInBytes();
  return totalSize;
}

}  // namespace m_engine::executor