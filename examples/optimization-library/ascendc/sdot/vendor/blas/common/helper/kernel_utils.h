/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

/* ========== Tiling utilities ========== */

#define CONST_2 2

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif

#include <limits>
#include <type_traits>

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#else
#include "kernel_macros.h"
#endif

template <uint32_t ALIGN, typename T = uint32_t>
inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUpPow2(T v)
{
    static_assert(std::is_unsigned<T>::value, "RoundUpPow2 requires unsigned type");
    if (v == 0) {
        return 1;
    }
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    if (sizeof(T) >= 4) {
        v |= v >> 16;
    }
    if (sizeof(T) >= 8) {
        v |= v >> 32;
    }
    v++;
    return v;
}

template <uint32_t DIVISOR, typename T = uint32_t>
inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint32_t BlockSize()
{
    return 32 / sizeof(Dtype);
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint32_t MatrixSize()
{
    return 512 / sizeof(Dtype);
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t BlockSizeRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>() * BlockSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t NumBlocksRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t MatrixSizeRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>() * MatrixSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t NumMatrixsRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>();
}

template <typename Dtype>
__aicore__ __attribute__((always_inline)) inline uint64_t L0HalfSize()
{
    return 32 * 1024 / sizeof(Dtype);
}

/* ========== Pipe / sync utilities ========== */

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()

#ifndef KERNEL_UTILS_LITE

template <typename IN_DTYPE>
__aicore__ inline void CreateCaMatrix(
    const AscendC::LocalTensor<IN_DTYPE>& dst, const uint16_t repeats, const uint16_t blockNum, const uint16_t dstGap,
    const IN_DTYPE initValue)
{
    AscendC::InitConstValue<IN_DTYPE>(
        dst, AscendC::InitConstValueParams<IN_DTYPE>(repeats, blockNum, dstGap, initValue));
}

__aicore__ inline void SetFftsBaseAddr(uint64_t config) { AscendC::SetSyncBaseAddr(config); }

template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue);
}

__aicore__ inline void SetAtomicnone() { AscendC::SetAtomicNone(); }

__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
    AscendC::SetMaskNorm();
}

__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
}

template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
    AscendC::SetVectorMask<IN_DTYPE>(maskHigh, maskLow);
}

__aicore__ inline int64_t GetSubBlockidx() { return AscendC::GetSubBlockIdx(); }

__aicore__ inline void WaitFlagDev(uint16_t flagId) { AscendC::WaitEvent(flagId); }

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    AscendC::CrossCoreSetFlag<mode, pipe>(flagId);
}

template <typename IN_DTYPE, bool setRelu = false>
__aicore__ inline void SetFpc(const AscendC::LocalTensor<IN_DTYPE>& preTensor, bool isUnitFlag = false)
{
    AscendC::SetFixPipeConfig<IN_DTYPE, setRelu>(preTensor, isUnitFlag);
}

template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToFbuf(
    AscendC::LocalTensor<IN_DTYPE>& dst, AscendC::LocalTensor<IN_DTYPE>& src, uint16_t burstNum, uint16_t burstLen,
    uint16_t srcGapSize, uint16_t dstGapSize)
{
    dst.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2PIPE2GM);
    AscendC::DataCopy(
        dst, src,
        AscendC::DataCopyParams(
            burstNum,     // nBurst
            burstLen,     // lenBurst
            srcGapSize,   // srcGap
            dstGapSize)); // dstGap);
}

template <typename IN_DTYPE>
__aicore__ inline void CopyCbufToBt(
    uint64_t dst, const AscendC::LocalTensor<IN_DTYPE>& src, uint16_t convControl, uint16_t nBurst, uint16_t lenBurst,
    uint16_t sourceGap, uint16_t dstGap)
{
    AscendC::LocalTensor<IN_DTYPE> dstTensor;
    dstTensor.InitBuffer(dst, nBurst * lenBurst);
    dstTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::DataCopy(
        dstTensor, src,
        AscendC::DataCopyParams(
            nBurst,    // nBurst
            lenBurst,  // lenBurst
            sourceGap, // srcGap
            dstGap));  // dstGap);
}

/* ========== Ascend BLAS kernel utilities ========== */

#include "cann_ops_blas_common.h"

static constexpr int64_t BIT_4 = 4;
static constexpr int64_t BIT_8 = 8;

__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId) { return 1 | (mode << BIT_4) | (flagId << BIT_8); }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
#ifndef INCLUDE_ITERTOR_H
#include "../iterator/iterator.h"
#endif
#endif

namespace fp32 {
constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(float);
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);
constexpr int64_t NUM_ELE_PERBLOCK = 32 / sizeof(float);
constexpr int64_t CUBE_M0 = 16;
constexpr int64_t CUBE_N0 = 16;
constexpr int64_t CUBE_K0 = 32 / sizeof(float);
constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 256 * 1024 / sizeof(float);
constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

__aicore__ inline int64_t ROUND(int64_t num, int64_t paddingNum)
{
    return ((num + paddingNum - 1) / paddingNum * paddingNum);
}

__aicore__ __inline__ void matrix_gm2cbuf_ND2nZ(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, int64_t CBUF_M0, int64_t CBUF_N0,
    int64_t mActual, int64_t nActual, size_t stride)
{
    if (stride < UINT16_STRIDE_LIMIT) {
        AscendC::DataCopy(dst, src, AscendC::Nd2NzParams(1, nActual, mActual, 0, stride, CBUF_N0, 1, 0));
    } else {
        for (int i = 0; i < nActual; i++) {
            AscendC::DataCopy(
                dst[i * CUBE_K0], src[i * stride], AscendC::Nd2NzParams(1, 1, mActual, 0, 0, CBUF_N0, 0, 0));
        }
    }
}

__aicore__ __inline__ void matrix_gm2cbuf_ND2nN(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, int64_t CBUF_M0, int64_t CBUF_N0,
    int64_t mActual, int64_t nActual, size_t stride)
{
    int64_t srcNdStride = CUBE_N0 * stride;
    int64_t srcNStride = stride;
    if (srcNdStride < UINT16_STRIDE_LIMIT) {
        int ndNum = nActual / CUBE_N0;
        int remains = nActual % CUBE_N0;
        if (ndNum > 0) {
            AscendC::DataCopy(
                dst, src,
                AscendC::Nd2NzParams(ndNum, CUBE_N0, mActual, srcNdStride, srcNStride, CUBE_N0, 1, CUBE_N0 * CBUF_M0));
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * CUBE_N0 * CBUF_M0], src[ndNum * CUBE_N0 * stride],
                AscendC::Nd2NzParams(1, remains, mActual, 0, srcNStride, CUBE_N0, 1, 0));
        }
    } else if (srcNStride < UINT16_STRIDE_LIMIT) {
        int ndNum = nActual / CUBE_N0;
        int remains = nActual % CUBE_N0;
        for (int i = 0; i < ndNum; i++) {
            AscendC::DataCopy(
                dst[i * CUBE_N0 * CBUF_M0], src[i * CUBE_N0 * stride],
                AscendC::Nd2NzParams(1, CUBE_N0, mActual, 0, srcNStride, CUBE_N0, 1, 0));
        }
        if (remains > 0) {
            AscendC::DataCopy(
                dst[ndNum * CUBE_N0 * CBUF_M0], src[ndNum * CUBE_N0 * stride],
                AscendC::Nd2NzParams(1, remains, mActual, 0, srcNStride, CUBE_N0, 1, 0));
        }
    } else {
        for (int i = 0; i < nActual; i++) {
            int idxR0 = i / CUBE_N0;
            int idxInR0 = i % CUBE_N0;
            AscendC::DataCopy(
                dst[idxR0 * CUBE_N0 * CBUF_M0 + idxInR0 * CUBE_K0], src[i * stride],
                AscendC::Nd2NzParams(1, 1, mActual, 0, 0, CUBE_N0, 0, 0));
        }
    }
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 3510)
__aicore__ __inline__ void matrix_gm2ubuf(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, int64_t mActual, int64_t nActual,
    size_t srcStride, size_t dstStride)
{
    int64_t mRound = ROUND(mActual, NUM_ELE_PERBLOCK);
    for (int i = 0; i < nActual; i++) {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            dst[i * dstStride], src[i * srcStride], 0, 1, mActual * sizeof(float), 0, 0, 0, 0);
    }
}

__aicore__ __inline__ void matrix_ubuf2gm(
    AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, int64_t mActual, int64_t nActual,
    size_t srcStride, size_t dstStride)
{
    for (int i = 0; i < nActual; i++) {
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            dst[i * dstStride], src[i * srcStride], 0, 1, mActual * sizeof(float), 0, 0, 0, 0);
    }
}
#endif
} // namespace fp32

#endif // KERNEL_UTILS_LITE

// ========== Mirror kernel shared prologue ==========

struct MirrorKernelCtx {
    const __gm__ float* aGm;
    __gm__ float* workspaceGm;
    uint32_t dimA;
    uint32_t lda;
    uint32_t rowStart;
    uint32_t rowEnd;
};

__force_inline__ __aicore__ MirrorKernelCtx InitMirrorKernelCtx(
    const __gm__ uint8_t* gmA, __gm__ uint8_t* gmWorkspaceA,
    uint32_t dimA, uint32_t lda, uint32_t mirrorRowsPerCore)
{
    MirrorKernelCtx ctx;
    ctx.aGm = reinterpret_cast<const __gm__ float*>(gmA);
    ctx.workspaceGm = reinterpret_cast<__gm__ float*>(gmWorkspaceA);
    ctx.dimA = dimA;
    ctx.lda = lda;
    int32_t blkIdx = AscendC::GetBlockIdx();
    ctx.rowStart = static_cast<uint32_t>(blkIdx) * mirrorRowsPerCore;
    ctx.rowEnd = Min<uint32_t>(ctx.rowStart + mirrorRowsPerCore, dimA);
    return ctx;
}
