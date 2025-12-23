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

#include <istream>
#include <ostream>

#include "m_engine/executor/executor.h"
#include "m_engine/executor/tensor.h"
#include "m_engine/executor/types.h"

namespace m_engine::executor {

class Serialization {
 public:
  // Tensor
  [[nodiscard]] static Tensor deserializeTensor(std::istream& is);
  static void serialize(Tensor const& tensor, std::ostream& os);
  [[nodiscard]] static size_t serializedSize(Tensor const& tensor);

};

}  // namespace m_engine::executor
