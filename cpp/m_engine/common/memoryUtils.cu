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

#include "m_engine/common/assert.h"
//#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "m_engine/common/logger.h"
#include "m_engine/common/memoryUtils.h"

#include <curand_kernel.h>
#include <sys/stat.h>
#include <unordered_map>

#include <sanitizer/asan_interface.h>

namespace m_engine {

namespace common
{

#ifdef __has_feature
#if __has_feature(address_sanitizer)
#define MENGINE_HAS_ASAN
#endif
#elif defined(__SANITIZE_ADDRESS__)
#define MENGINE_HAS_ASAN
#endif

cudaError_t cudaMemcpyAsyncSanitized(
    void* dst, void const* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#if defined(MENGINE_HAS_ASAN)
    bool needASAN = false;
    if (kind == cudaMemcpyDeviceToHost)
    {
        needASAN = true;
    }
    else if (kind == cudaMemcpyDefault)
    {
        auto const srcType = getPtrCudaMemoryType(src);
        auto const dstType = getPtrCudaMemoryType(dst);
        needASAN = srcType == cudaMemoryTypeDevice && dstType != cudaMemoryTypeDevice;
    }

    // Poison the memory area during async copy
    if (needASAN)
    {
        ASAN_POISON_MEMORY_REGION(dst, count);
    }

    auto const result = cudaMemcpyAsync(dst, src, count, kind, stream);

    if (result == cudaSuccess && needASAN)
    {
        struct ctxType
        {
            void* ptr;
            size_t count;
        };

        auto const ctx = new ctxType{dst, count};
        auto cb = [](cudaStream_t, cudaError_t, void* data)
        {
            auto const ctx = static_cast<ctxType*>(data);
            ASAN_UNPOISON_MEMORY_REGION(ctx->ptr, ctx->count);
            delete ctx;
        };
        MENGINE_CUDA_CHECK(cudaStreamAddCallback(stream, cb, ctx, 0));
    }

    return result;
#else
    return cudaMemcpyAsync(dst, src, count, kind, stream);
#endif
}

}

}