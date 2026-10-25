/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OPS_SROTM_KERNEL_H__
#define __OPS_SROTM_KERNEL_H__

#include <cstdint>
#include "kernel_operator.h"
#include "srotm_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_SECTION_NUM = 6;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t GM_ALIGNMENT_BYTES = 512;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t SFLAG_MODE_GENERAL = 0;
constexpr uint32_t SFLAG_MODE_ZERO = 1;
constexpr uint32_t SFLAG_MODE_POSITIVE = 2;
constexpr uint32_t SFLAG_MODE_NO_OP = 3;

template <typename T>
class SrotmKernel {
public:
    __aicore__ inline SrotmKernel() = default;
    __aicore__ inline void Init(const SrotmTilingData &tiling, TPipe *pipeObj);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const SrotmTilingData &tiling);
    __aicore__ inline void CalculateBlockRange();
    __aicore__ inline uint32_t AlignToBlock(uint32_t dataCount) const;
    __aicore__ inline void CopyIn(uint32_t curOffset, uint32_t dataCount, uint32_t computeCount);
    __aicore__ inline void WaitScalarToVec();
    __aicore__ inline void WaitVecToScalar();
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline uint64_t GetXIndex(uint32_t logicalIndex) const;
    __aicore__ inline uint64_t GetYIndex(uint32_t logicalIndex) const;

    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue;
    TBuf<TPosition::VECCALC> xOutBuf;
    TBuf<TPosition::VECCALC> yOutBuf;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
    uint32_t useCoreNum = 0;
    uint32_t startOffset = 0;
    uint32_t calNum = 0;
    uint32_t maxDataCount = 0;
    uint32_t flagMode = SFLAG_MODE_NO_OP;

    uint64_t xStorageSize = 0;
    uint64_t yStorageSize = 0;
    int64_t incx = 1;
    int64_t incy = 1;
    int64_t xStartIndex = 0;
    int64_t yStartIndex = 0;

    T sflag = static_cast<T>(-2);
    T h11 = static_cast<T>(1);
    T h12 = static_cast<T>(0);
    T h21 = static_cast<T>(0);
    T h22 = static_cast<T>(1);

    uint32_t elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
    uint32_t elementsPerAlignment = GM_ALIGNMENT_BYTES / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void SrotmKernel<T>::ParseTilingData(const SrotmTilingData &tiling)
{
    n = tiling.n;
    useCoreNum = tiling.useCoreNum;
    xStorageSize = tiling.xStorageSize;
    yStorageSize = tiling.yStorageSize;
    incx = tiling.incx;
    incy = tiling.incy;
    sflag = static_cast<T>(tiling.sflag);
    h11 = static_cast<T>(tiling.h11);
    h12 = static_cast<T>(tiling.h12);
    h21 = static_cast<T>(tiling.h21);
    h22 = static_cast<T>(tiling.h22);
    xStartIndex = incx >= 0 ? 0 : (1 - static_cast<int64_t>(n)) * incx;
    yStartIndex = incy >= 0 ? 0 : (1 - static_cast<int64_t>(n)) * incy;

    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.x), xStorageSize);
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tiling.y), yStorageSize);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::CalculateBlockRange()
{
    // 多核切分以 512B 对齐粒度为主，只让最后一个有效块承接尾数。
    startOffset = 0;
    calNum = 0;
    if (useCoreNum == 0 || vecIdx >= useCoreNum || n == 0) {
        return;
    }

    const uint32_t alignedChunkCount = n / elementsPerAlignment;
    const uint32_t tailElementCount = n % elementsPerAlignment;
    if (alignedChunkCount == 0) {
        if (vecIdx == 0) {
            calNum = n;
        }
        return;
    }

    uint32_t alignedCoreNum = useCoreNum;
    if (alignedCoreNum > alignedChunkCount) {
        alignedCoreNum = alignedChunkCount;
    }
    if (vecIdx >= alignedCoreNum) {
        return;
    }

    const uint32_t chunksPerCore = alignedChunkCount / alignedCoreNum;
    const uint32_t remainChunks = alignedChunkCount % alignedCoreNum;
    const uint32_t priorExtraChunks = vecIdx < remainChunks ? vecIdx : remainChunks;
    const uint32_t curChunkCount = chunksPerCore + (vecIdx < remainChunks ? 1U : 0U);

    startOffset = (vecIdx * chunksPerCore + priorExtraChunks) * elementsPerAlignment;
    calNum = curChunkCount * elementsPerAlignment;
    if (tailElementCount > 0 && vecIdx == alignedCoreNum - 1) {
        calNum += tailElementCount;
    }
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::Init(const SrotmTilingData &tiling, TPipe *pipeObj)
{
    vecIdx = GetBlockIdx();
    ParseTilingData(tiling);
    CalculateBlockRange();

    uint32_t ubSizePerBuffer = UB_SIZE / BUFFER_SECTION_NUM;
    maxDataCount = (ubSizePerBuffer / sizeof(T) / elementsPerBlock) * elementsPerBlock;
    if (maxDataCount == 0) {
        maxDataCount = elementsPerBlock;
    }

    pipeObj->InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipeObj->InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipeObj->InitBuffer(xOutBuf, maxDataCount * sizeof(T));
    pipeObj->InitBuffer(yOutBuf, maxDataCount * sizeof(T));

    if (sflag == static_cast<T>(-2)) {
        flagMode = SFLAG_MODE_NO_OP;
    } else if (sflag == static_cast<T>(0)) {
        flagMode = SFLAG_MODE_ZERO;
    } else if (sflag > static_cast<T>(0)) {
        flagMode = SFLAG_MODE_POSITIVE;
    } else {
        flagMode = SFLAG_MODE_GENERAL;
    }
}

template <typename T>
__aicore__ inline uint64_t SrotmKernel<T>::GetXIndex(uint32_t logicalIndex) const
{
    return static_cast<uint64_t>(xStartIndex + static_cast<int64_t>(logicalIndex) * incx);
}

template <typename T>
__aicore__ inline uint64_t SrotmKernel<T>::GetYIndex(uint32_t logicalIndex) const
{
    return static_cast<uint64_t>(yStartIndex + static_cast<int64_t>(logicalIndex) * incy);
}

template <typename T>
__aicore__ inline uint32_t SrotmKernel<T>::AlignToBlock(uint32_t dataCount) const
{
    return ((dataCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::CopyIn(uint32_t curOffset, uint32_t dataCount, uint32_t computeCount)
{
    // 对齐扩展部分补零，避免向量计算读到未定义值。
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    for (uint32_t i = 0; i < dataCount; ++i) {
        uint32_t logicalIndex = curOffset + i;
        xLocal.SetValue(i, xGM.GetValue(GetXIndex(logicalIndex)));
        yLocal.SetValue(i, yGM.GetValue(GetYIndex(logicalIndex)));
    }
    for (uint32_t i = dataCount; i < computeCount; ++i) {
        xLocal.SetValue(i, static_cast<T>(0));
        yLocal.SetValue(i, static_cast<T>(0));
    }
    xQueue.EnQue<T>(xLocal);
    yQueue.EnQue<T>(yLocal);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::WaitScalarToVec()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::WaitVecToScalar()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::Compute(uint32_t dataCount)
{
    // 根据 sflag 映射后的模式执行对应的旋转公式。
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    LocalTensor<T> xOutLocal = xOutBuf.Get<T>();
    LocalTensor<T> yOutLocal = yOutBuf.Get<T>();

    if (flagMode == SFLAG_MODE_GENERAL) {
        Muls(xOutLocal, xLocal, h11, dataCount);
        Muls(yOutLocal, yLocal, h22, dataCount);
        Muls(xLocal, xLocal, h21, dataCount);
        Add(yOutLocal, yOutLocal, xLocal, dataCount);
        Muls(yLocal, yLocal, h12, dataCount);
        Add(xOutLocal, xOutLocal, yLocal, dataCount);
    } else if (flagMode == SFLAG_MODE_ZERO) {
        Muls(xOutLocal, yLocal, h12, dataCount);
        Add(xOutLocal, xOutLocal, xLocal, dataCount);
        Muls(yOutLocal, xLocal, h21, dataCount);
        Add(yOutLocal, yOutLocal, yLocal, dataCount);
    } else {
        Muls(xOutLocal, xLocal, h11, dataCount);
        Add(xOutLocal, xOutLocal, yLocal, dataCount);
        Muls(yOutLocal, yLocal, h22, dataCount);
        Muls(xLocal, xLocal, h21, dataCount);
        Add(yOutLocal, yOutLocal, xLocal, dataCount);
    }

    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::CopyOut(uint32_t curOffset, uint32_t dataCount)
{
    LocalTensor<T> xOutLocal = xOutBuf.Get<T>();
    LocalTensor<T> yOutLocal = yOutBuf.Get<T>();
    for (uint32_t i = 0; i < dataCount; ++i) {
        uint32_t logicalIndex = curOffset + i;
        xGM.SetValue(GetXIndex(logicalIndex), xOutLocal.GetValue(i));
        yGM.SetValue(GetYIndex(logicalIndex), yOutLocal.GetValue(i));
    }
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t computeCount = AlignToBlock(dataCount);
    CopyIn(curOffset, dataCount, computeCount);
    WaitScalarToVec();
    Compute(computeCount);
    WaitVecToScalar();
    CopyOut(curOffset, dataCount);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::Process()
{
    // 当前核没有分配到工作时直接退出。
    if (vecIdx >= useCoreNum || calNum == 0 || flagMode == SFLAG_MODE_NO_OP) {
        return;
    }

    uint32_t curOffset = startOffset;
    uint32_t remain = calNum;
    while (remain > 0) {
        uint32_t dataCount = remain > maxDataCount ? maxDataCount : remain;
        SingleIteration(curOffset, dataCount);
        curOffset += dataCount;
        remain -= dataCount;
    }
}

__global__ __aicore__ void srotm_kernel(const SrotmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SrotmKernel<float> op;
    op.Init(tiling, &pipe);
    op.Process();
}

void srotm_kernel_do(const SrotmTilingData &tiling, uint32_t numBlocks, void *stream)
{
    srotm_kernel<<<numBlocks, nullptr, stream>>>(tiling);
}

#endif // __OPS_SROTM_KERNEL_H__
