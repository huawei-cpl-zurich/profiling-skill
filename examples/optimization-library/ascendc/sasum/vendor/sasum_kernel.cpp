/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SASUM_KERNEL_H
#define SASUM_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;

template <typename T>
class SasumAIV {
public:
    __aicore__ inline SasumAIV(){};
    __aicore__ inline void Init(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SingleIteration(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut();

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TBuf<TPosition::VECCALC> workBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    int32_t blockNum;
    int32_t vecIdx;

    uint32_t totalVecCoreNum = 40;
    uint32_t computeNum = 0;
    uint32_t startOffset = 0;
    uint32_t maxDataCount = 0;
    uint32_t coreNum = 0;
    uint32_t n = 0;
    int32_t incx = 1;
};

template <typename T>
__aicore__ inline void SasumAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingAddr = reinterpret_cast<__gm__ uint8_t*>(tilingGm);

    this->n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr));
    this->incx = (*(__gm__ int32_t*)((__gm__ uint8_t*)tilingAddr + sizeof(uint32_t)));
    this->coreNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + 2 * sizeof(uint32_t)));
    this->startOffset =
        (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + sizeof(uint32_t) * this->vecIdx + 3 * sizeof(uint32_t)));
    this->computeNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + this->totalVecCoreNum * sizeof(uint32_t) +
                                            sizeof(uint32_t) * this->vecIdx + 3 * sizeof(uint32_t)));
}

template <typename T>
__aicore__ inline void SasumAIV<T>::Init(GM_ADDR inDevice, GM_ADDR outDevice, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    const uint64_t inputSize =
        (this->n > 0 && this->incx > 1) ? static_cast<uint64_t>(this->n - 1) * this->incx + 1 : this->n;
    inGM.SetGlobalBuffer((__gm__ T*)inDevice, inputSize);
    outGM.SetGlobalBuffer((__gm__ T*)outDevice, 1);

    maxDataCount = 80 * 1024 / BYTENUM_PER_FLOAT32; // 80kb

    // compute the minimum workspace for ReduceSum
    int typeSize = 4;
    int elementsPerBlock = 32 / typeSize;
    int elementsPerRepeat = 256 / typeSize;
    int firstMaxRepeat = maxDataCount / 64; // 376

    int iter1OutputCount = firstMaxRepeat;
    int iter1AlignEnd = (iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    int finalWorkLocalNeedSize = iter1AlignEnd;

    uint32_t byteLen = finalWorkLocalNeedSize * sizeof(T); // 1504
    pipe.InitBuffer(workBuf, byteLen + 32);

    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, 8 * sizeof(T));

    LocalTensor<T> workLocal = workBuf.Get<T>(8);
    Duplicate<float>(workLocal, 0.0, 8);
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, workLocal, copyParams);

    SyncAll();

    return;
}

template <typename T>
__aicore__ inline void SasumAIV<T>::Process()
{
    SetAtomicAdd<T>();

    uint32_t repeatTimes = computeNum / maxDataCount;
    uint32_t remainNum = computeNum % maxDataCount;
    uint32_t maxCopyPadNum = (UINT16_MAX + 1) / sizeof(T);

    uint32_t currOffset = startOffset;
    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, maxDataCount);
        currOffset += maxDataCount;
    }
    if (remainNum > 0) {
        if (remainNum >= maxCopyPadNum) {
            SingleIteration(currOffset, maxCopyPadNum);
            currOffset += maxCopyPadNum;
            remainNum -= maxCopyPadNum;
        }

        if (remainNum > 0) {
            SingleIterationAligned(currOffset, remainNum);
        }
    }
    SetAtomicNone();
    return;
}

template <typename T>
__aicore__ inline void SasumAIV<T>::SingleIteration(uint32_t offset, uint32_t dataCount)
{
    CopyIn(offset, dataCount);
    Compute(dataCount);
    CopyOut();
}

template <typename T>
__aicore__ inline void SasumAIV<T>::SingleIterationAligned(uint32_t offset, uint32_t dataCount)
{
    uint32_t dataCountAligned = (dataCount + 7) / 8 * 8;
    CopyInPad(offset, dataCount);
    Compute(dataCountAligned);
    CopyOut();
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    if (this->incx > 1) {
        for (uint32_t i = 0; i < dataCount; i++) {
            inLocal.SetValue(i, inGM.GetValue((static_cast<uint64_t>(offset) + i) * this->incx));
        }
    } else {
        DataCopy(inLocal, inGM[offset], dataCount);
    }
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyInPad(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    if (this->incx > 1) {
        uint32_t dataCountAligned = (dataCount + 7) / 8 * 8;
        for (uint32_t i = 0; i < dataCount; i++) {
            inLocal.SetValue(i, inGM.GetValue((static_cast<uint64_t>(offset) + i) * this->incx));
        }
        for (uint32_t i = dataCount; i < dataCountAligned; i++) {
            inLocal.SetValue(i, static_cast<T>(0));
        }
    } else {
        DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
        uint8_t paddingNum = 8 - dataCount % 8;
        DataCopyPadParams padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal, inGM[offset], copyParams, padParams);
    }
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::Compute(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Abs(inLocal, inLocal, dataCount);
    pipe_barrier(PIPE_V);
    ReduceSum(outLocal, inLocal, workLocal, dataCount);

    outQueue.EnQue<T>(outLocal);
    inQueue.FreeTensor(inLocal);
}

template <typename T>
__aicore__ inline void SasumAIV<T>::CopyOut()
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

__global__ __aicore__ void sasum_kernel(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SasumAIV<float> op;
    op.Init(inGM, outGM, nullptr, tilingGm);
    op.Process();
}

void sasum_kernel_do(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    sasum_kernel<<<numBlocks, nullptr, stream>>>(inGM, outGM, nullptr, tilingGm);
}

#endif // SASUM_KERNEL_H
