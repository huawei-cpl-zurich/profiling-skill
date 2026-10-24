/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */
// Independent standalone adaptation of SpmvConvertVals, ops-sparse
// bfc7c19d83ce246b03c65dc916ed4753e5467cf3, spmv_kernel.h:267-302.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
constexpr uint32_t tile = 4096;
#if AFTER
#define ENTRY wave006_int8_widening_after
#else
#define ENTRY wave006_int8_widening_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void ENTRY(GM_ADDR x, GM_ADDR y) {
    const uint32_t offset = GetBlockIdx() * tile;
    const uint32_t count = STUDY_N - offset < tile ? STUDY_N - offset : tile;
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuffer, halfBuffer, outBuffer;
    pipe.InitBuffer(rawBuffer, tile);
    pipe.InitBuffer(halfBuffer, tile * sizeof(half)); // Identical reservation both arms.
    pipe.InitBuffer(outBuffer, tile * sizeof(float));
    auto raw = rawBuffer.Get<int8_t>();
    auto temporary = halfBuffer.Get<half>();
    auto result = outBuffer.Get<float>();
    GlobalTensor<int8_t> input;
    GlobalTensor<float> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x), STUDY_N);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), STUDY_N);
    DataCopyExtParams inCopy{1, count, 0, 0, 0}; // GM byte count, UB aligned tile.
    DataCopyPadExtParams<int8_t> padding{false, 0, 0, 0};
    DataCopyPad(raw, input[offset], inCopy, padding);
    // Same conservative full-pipe ownership fence: MTE2 -> scalar/V.
    PipeBarrier<PIPE_ALL>();
#if AFTER
    Cast<half, int8_t>(temporary, raw, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(result, temporary, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
#else
    __ubuf__ int8_t* src = reinterpret_cast<__ubuf__ int8_t*>(raw.GetPhyAddr());
    __ubuf__ float* dst = reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr());
    for (uint32_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
#endif
    // Same scalar/V -> MTE3 ownership fence, including scalar UB completion.
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams outCopy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(output[offset], result, outCopy);
    PipeBarrier<PIPE_ALL>();
}
extern "C" uint32_t int8_widening_n() { return STUDY_N; }
extern "C" uint32_t int8_widening_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_int8_widening_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_int8_widening_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** pointers, uint64_t count, void* stream) {
    if (count != 2 || !pointers[0] || !pointers[1]) return 1;
    ENTRY<<<(STUDY_N + tile - 1) / tile, nullptr, stream>>>((uint8_t*)pointers[0], (uint8_t*)pointers[1]);
    return 0;
}
