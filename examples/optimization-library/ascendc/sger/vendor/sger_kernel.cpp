/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "sger_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_NUM = 4;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t MAX_UB_ELEMENTS = UB_SIZE / BYTENUM_PER_FLOAT32;
constexpr uint32_t ELEM_PER_32B_BLOCK = 8;
constexpr uint32_t CHUNK_BASE = UB_SIZE / BUFFER_NUM / QUEUE_NUM / BYTENUM_PER_FLOAT32 / ELEM_PER_32B_BLOCK;

template <typename T>
class SgerKernel {
public:
    __aicore__ inline SgerKernel() {}
    __aicore__ inline void Init(const SgerTilingData& tiling, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t col, uint32_t rowOff, uint32_t chunk);
    __aicore__ inline void Compute(uint32_t col, uint32_t rowOff, uint32_t chunk);
    __aicore__ inline void CopyOut(uint32_t col, uint32_t rowOff, uint32_t chunk);

    TPipe* pipe;
    TQue<TPosition::VECIN, 2> xQueue;
    TQue<TPosition::VECIN, 2> aInQueue;
    TQue<TPosition::VECOUT, 2> aOutQueue;
    TQue<TPosition::VECCALC, 1> tmpQueue;

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    T alpha;

    int32_t blockIdx;
    int32_t blockNum;
    int64_t m;
    int64_t n;
    int64_t lda;
    int incx;
    int incy;
    uint32_t absIncx;
    uint32_t absIncy;

    uint32_t useCoreNum;
    uint32_t colsPerBlock;
    uint32_t colStart;
    uint32_t colCount;
    uint32_t rowCount;
    uint32_t chunkMax;
};

template <typename T>
__aicore__ inline void SgerKernel<T>::Init(const SgerTilingData& tiling, TPipe* pipeIn)
{
    this->blockNum = GetBlockNum();
    this->blockIdx = GetBlockIdx();
    this->pipe = pipeIn;
    this->m = tiling.m;
    this->n = tiling.n;
    this->lda = tiling.lda;
    this->useCoreNum = tiling.useCoreNum;
    this->colsPerBlock = tiling.colsPerBlock;
    this->incx = tiling.incx;
    this->incy = tiling.incy;
    this->absIncx = (this->incx < 0) ? static_cast<uint32_t>(-this->incx) : static_cast<uint32_t>(this->incx);
    this->absIncy = (this->incy < 0) ? static_cast<uint32_t>(-this->incy) : static_cast<uint32_t>(this->incy);
    this->alpha = tiling.alpha;
    this->colStart = this->blockIdx * this->colsPerBlock;
    this->chunkMax = (this->incx > 1) ? CHUNK_BASE : (CHUNK_BASE * ELEM_PER_32B_BLOCK);

    if (this->colStart >= this->n) {
        this->colCount = 0;
    } else {
        this->colCount = (this->blockIdx == this->useCoreNum - 1) ? (this->n - this->colStart) : this->colsPerBlock;
    }

    this->rowCount = this->m;

    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tiling.A), static_cast<uint64_t>(n * lda));
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tiling.x), static_cast<uint64_t>((m - 1) * absIncx + 1));
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tiling.y), static_cast<uint64_t>((n - 1) * absIncy + 1));

    uint32_t ubSizePerBuffer = UB_SIZE / BUFFER_NUM / QUEUE_NUM;
    pipe->InitBuffer(xQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe->InitBuffer(aInQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe->InitBuffer(aOutQueue, BUFFER_NUM, ubSizePerBuffer);
    pipe->InitBuffer(tmpQueue, BUFFER_NUM, ubSizePerBuffer);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyIn(uint32_t col, uint32_t rowOff, uint32_t chunk)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyPadExtParams<T> noPad{false, 0, 0, T(0)};
    if (incx == 1) {
        DataCopyExtParams xParams{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};
        DataCopyPad(xLocal, xGM[rowOff], xParams, noPad);
    } else if (incx > 1) {
        DataCopyExtParams xParams{
            static_cast<uint16_t>(chunk), static_cast<uint32_t>(sizeof(T)),
            static_cast<uint32_t>((absIncx - 1) * sizeof(T)), 0, 0};
        DataCopyPad(xLocal, xGM[rowOff * absIncx], xParams, noPad);
    } else {
        for (uint32_t i = 0; i < chunk; ++i) {
            uint64_t xIdx = (incx > 0) ? static_cast<uint64_t>(static_cast<int64_t>(rowOff + i) * incx) :
                                         static_cast<uint64_t>(static_cast<int64_t>(m - 1 - (rowOff + i)) * (-incx));
            xLocal.SetValue(i, xGM.GetValue(xIdx));
        }
    }
    xQueue.EnQue<T>(xLocal);

    LocalTensor<T> aLocal = aInQueue.AllocTensor<T>();
    uint32_t aCol = colStart + col;
    DataCopyExtParams aParams{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};
    uint64_t aOff = static_cast<uint64_t>(static_cast<int64_t>(aCol) * lda) + rowOff;
    DataCopyPad(aLocal, aGM[aOff], aParams, noPad);
    aInQueue.EnQue<T>(aLocal);

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::Compute(uint32_t col, uint32_t rowOff, uint32_t chunk)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> aInLocal = aInQueue.DeQue<T>();
    LocalTensor<T> aOutLocal = aOutQueue.AllocTensor<T>();
    LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();

    uint32_t aCol = colStart + col;
    uint64_t yIdx = (incy > 0) ? static_cast<uint64_t>(static_cast<int64_t>(aCol) * incy) :
                                 static_cast<uint64_t>(static_cast<int64_t>(n - 1 - aCol) * (-incy));
    T yVal = yGM.GetValue(yIdx);

    T alphaY = alpha * yVal;
    if (incx > 1) {
        for (uint32_t i = 0; i < chunk; ++i) {
            xLocal.SetValue(i, xLocal.GetValue(i * ELEM_PER_32B_BLOCK));
        }
    }
    Muls(tmpLocal, xLocal, alphaY, chunk);
    Add(aOutLocal, aInLocal, tmpLocal, chunk);

    tmpQueue.FreeTensor(tmpLocal);
    xQueue.FreeTensor(xLocal);
    aInQueue.FreeTensor(aInLocal);
    aOutQueue.EnQue<T>(aOutLocal);

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::CopyOut(uint32_t col, uint32_t rowOff, uint32_t chunk)
{
    LocalTensor<T> aLocal = aOutQueue.DeQue<T>();
    uint32_t aCol = colStart + col;
    DataCopyExtParams outParams{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};
    uint64_t aOff = static_cast<uint64_t>(static_cast<int64_t>(aCol) * lda) + rowOff;
    DataCopyPad(aGM[aOff], aLocal, outParams);
    aOutQueue.FreeTensor(aLocal);

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void SgerKernel<T>::Process()
{
    if (rowCount == 0 || colCount == 0) {
        return;
    }

    for (uint32_t col = 0; col < colCount; ++col) {
        for (uint32_t rowOff = 0; rowOff < rowCount; rowOff += chunkMax) {
            const uint32_t chunk = (rowOff + chunkMax <= rowCount) ? chunkMax : (rowCount - rowOff);
            CopyIn(col, rowOff, chunk);
            Compute(col, rowOff, chunk);
            CopyOut(col, rowOff, chunk);
        }
    }
}

__global__ __aicore__ void sger_kernel(const SgerTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgerKernel<float> op;
    op.Init(tiling, &pipe);
    op.Process();
}

void sger_kernel_do(const SgerTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sger_kernel<<<numBlocks, nullptr, stream>>>(tiling);
}