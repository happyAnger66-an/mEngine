/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>

#include "m_engine/common/cudaUtils.h"

namespace m_engine {

namespace common {

// cudaMemcpyAsync with extra check via ASan for D2H copy
cudaError_t cudaMemcpyAsyncSanitized(
    void* dst, void const* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = nullptr);

// The following functions implement conversion of multi-dimensional indices to
// an index in a flat array. The shape of the Tensor dimensions is passed as one
// array (`dims`), the indices are given as individual arguments. For examples
// on how to use these functions, see their tests `test_memory_utils.cu`. All of
// these functions can be evaluated at compile time by recursive template
// expansion.

template <typename TDim, typename T, typename TIndex>
__inline__ __host__ __device__
    std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
        T const& acc, TDim dims, TIndex const& index) {
  assert(index < dims[0]);
  return acc * dims[0] + index;
}

template <typename TDim, typename T, typename TIndex, typename... TIndices>
__inline__ __host__ __device__
    std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
        T const& acc, TDim dims, TIndex const& index, TIndices... indices) {
  assert(index < dims[0]);
  return flat_index(acc * dims[0] + index, dims + 1, indices...);
}

template <typename TDim, typename T>
__inline__ __host__ __device__
    std::enable_if_t<std::is_pointer<TDim>::value, T> constexpr flat_index(
        [[maybe_unused]] TDim dims, T const& index) {
  assert(index < dims[0]);
  return index;
}

template <typename TDim, typename TIndex, typename... TIndices>
__inline__ __host__ __device__
    std::enable_if_t<std::is_pointer<TDim>::value,
                     typename std::remove_pointer<TDim>::
                         type> constexpr flat_index(TDim dims,
                                                    TIndex const& index,
                                                    TIndices... indices) {
  assert(index < dims[0]);
  return flat_index(
      static_cast<typename std::remove_pointer<TDim>::type>(index), dims + 1,
      indices...);
}

template <unsigned skip = 0, typename T, std::size_t N, typename TIndex,
          typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(
    std::array<T, N> const& dims, TIndex const& index, TIndices... indices) {
  static_assert(skip < N);
  static_assert(sizeof...(TIndices) < N - skip,
                "Number of indices exceeds number of dimensions");
  return flat_index(&dims[skip], index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N,
          typename... TIndices>
__inline__ __host__ __device__
    T constexpr flat_index(T const& acc, std::array<T, N> const& dims,
                           TIndex const& index, TIndices... indices) {
  static_assert(skip < N);
  static_assert(sizeof...(TIndices) < N - skip,
                "Number of indices exceeds number of dimensions");
  return flat_index(acc, &dims[skip], index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N,
          typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(T const (&dims)[N],
                                                      TIndex const& index,
                                                      TIndices... indices) {
  static_assert(skip < N);
  static_assert(sizeof...(TIndices) < N - skip,
                "Number of indices exceeds number of dimensions");
  return flat_index(static_cast<T const*>(dims) + skip, index, indices...);
}

template <unsigned skip = 0, typename T, typename TIndex, std::size_t N,
          typename... TIndices>
__inline__ __host__ __device__ T constexpr flat_index(T const& acc,
                                                      T const (&dims)[N],
                                                      TIndex const& index,
                                                      TIndices... indices) {
  static_assert(skip < N);
  static_assert(sizeof...(TIndices) < N - skip,
                "Number of indices exceeds number of dimensions");
  return flat_index(acc, static_cast<T const*>(dims) + skip, index, indices...);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// These are simpler functions for multi-dimensional index conversion. Indices
// and dimensions are passed as individual arguments. These functions are more
// suitable for usage inside kernels than the corresponding flat_index functions
// which require arrays as arguments. Usage examples can be found in
// `test_memory_utils.cu`. The functions can be evaluated at compile time.

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index2(TIndex const& index_0,
                                                       TIndex const& index_1,
                                                       T const& dim_1) {
  assert(index_1 < dim_1);
  return index_0 * dim_1 + index_1;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index3(TIndex const& index_0,
                                                       TIndex const& index_1,
                                                       TIndex const& index_2,
                                                       T const& dim_1,
                                                       T const& dim_2) {
  assert(index_2 < dim_2);
  return flat_index2(index_0, index_1, dim_1) * dim_2 + index_2;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index4(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2,
    TIndex const& index_3, T const& dim_1, T const& dim_2, T const& dim_3) {
  assert(index_3 < dim_3);
  return flat_index3(index_0, index_1, index_2, dim_1, dim_2) * dim_3 + index_3;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index5(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2,
    TIndex const& index_3, TIndex const& index_4, T const& dim_1,
    T const& dim_2, T const& dim_3, T const& dim_4) {
  assert(index_4 < dim_4);
  return flat_index4(index_0, index_1, index_2, index_3, dim_1, dim_2, dim_3) *
             dim_4 +
         index_4;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index_strided3(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2,
    T const& stride_1, T const& stride_2) {
  assert(index_1 < stride_1 / stride_2);
  assert(index_2 < stride_2);
  return index_0 * stride_1 + index_1 * stride_2 + index_2;
}

template <typename T, typename TIndex>
__inline__ __host__ __device__ T constexpr flat_index_strided4(
    TIndex const& index_0, TIndex const& index_1, TIndex const& index_2,
    TIndex const& index_3, T const& stride_1, T const& stride_2,
    T const& stride_3) {
  assert(index_1 < stride_1 / stride_2);
  assert(index_2 < stride_2 / stride_3);
  assert(index_3 < stride_3);
  return index_0 * stride_1 + index_1 * stride_2 + index_2 * stride_3 + index_3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace common

}  // namespace m_engine