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
#pragma once
#include "kernel_operator.h"
// Extracted from cann/ops-sparse bfc7c19d83ce246b03c65dc916ed4753e5467cf3
// sparse/spmv/arch22/kernels/spmv_kernel.h:L22-L153.
// 1. 拆分需先计算 4097*a / 4097*b，|a| 或 |b| > FLT_MAX/4097（≈8.3e34）时该
//    乘积溢出为 Inf，拆分将产出 NaN。此类极端输入回退为不补偿（e=0），该项退化为普通
//    乘积精度（与 fl(a*b) 一致），保证「大数×小数且积有限」的合法输入输出有限值。
// 2. 积量级守卫：即使 |a|,|b| 都在拆分安全域内，拆分后的高位积 ah*bh 仍可能溢出——
//    |ah*bh| <= |a*b|*(1+2^-11)^2，|a*b| 接近 FLT_MAX 时 ah*bh 可到 ~2^128 变 Inf
//    （反例：a=b=2^64，积 FLT_MAX 有限但 ah*bh=2^128 溢出）。|p| > FLT_MAX/2 时
//    先把 a 精确减半（乘 2^-1 不引入舍入）再拆分，最后把 p、e 精确翻倍还原，
//    保证参与拆分的积量级 <= FLT_MAX/2，ah*bh 必有限。
__aicore__ inline void SpmvTwoProd(float a, float b, float &p, float &e) {
    p = a * b;
    // 拆分安全上限：4097*|x| 不溢出当且仅当 |x| <= FLT_MAX/4097 ≈ 8.306e34，留裕量
    constexpr float SPLIT_SAFE_MAX = 8.0e34f;
    if (a >= -SPLIT_SAFE_MAX && a <= SPLIT_SAFE_MAX && b >= -SPLIT_SAFE_MAX && b <= SPLIT_SAFE_MAX) {
        // 积量级守卫：|ah*bh| <= |a*b|*(1+2^-11)^2，|a*b| > PROD_SAFE_MAX 时 ah*bh 可能溢出。
        // 将 a 精确减半后拆分（2 的幂缩放不引入舍入），p/e 翻倍还原，保证 ah*bh <= 0.5*FLT_MAX*(1+eps) 有限。
        constexpr float PROD_SAFE_MAX = 1.7e38f; // ≈ FLT_MAX/2 = 2^127*(1-2^-24)
        float sa = a;
        bool scaled = false;
        if (p > PROD_SAFE_MAX || p < -PROD_SAFE_MAX) {
            sa = a * 0.5f;
            p = sa * b;
            scaled = true;
        }
        constexpr float SPLITTER = 4097.0f; // 2^12 + 1
        float ca = SPLITTER * sa;
        float ah = ca - (ca - sa);
        float al = sa - ah;
        float cb = SPLITTER * b;
        float bh = cb - (cb - b);
        float bl = b - bh;
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        if (scaled) {
            p = p + p;  // 翻倍精确（2 的幂），p <= FLT_MAX 不溢出
            e = e + e;
        }
    }
    else {
        e = 0.0f;
    }
}

// Knuth TwoSum：s = fl(a+b)，e 为 a+b - s 的精确误差项（不要求 |a|>=|b|）
__aicore__ inline void SpmvTwoSum(float a, float b, float &s, float &e) {
    s = a + b;
    float bp = s - a;
    e = (a - (s - bp)) + (b - bp);
}

// TwoProd 守卫外提的行/窗口级安全阈值（与 SpmvTwoProd 内两道守卫同源）：
// 拆分安全要求 |a|,|b| <= FLT_MAX/4097 ≈ 8.3e34；积量级安全要求 |a*b| 使
// ah*bh <= |a*b|*(1+2^-11)^2 不溢出。谓词统一取 1.6e38（< FLT_MAX/2），为
// 上界乘法（maxVal*maxX）的舍入与 (1+2^-11)^2 留双倍裕量
constexpr float SPMV_TP_SPLIT_SAFE_MAX = 8.0e34f;
constexpr float SPMV_TP_PROD_SAFE_MAX = 1.6e38f;

// 标量绝对值（设备侧无 fabsf）
__aicore__ inline float SpmvAbsF(float v) { return v >= 0.0f ? v : -v; }

// 无守卫 Dekker TwoProd（快路径）：调用方已按行/窗口级谓词保证
// |a|,|b| <= SPMV_TP_SPLIT_SAFE_MAX 且 |a*b| <= SPMV_TP_PROD_SAFE_MAX，
// 此时拆分与高位积必有限，逐元素守卫恒不动作，逐位等价于守卫版
__aicore__ inline void SpmvTwoProdFast(float a, float b, float &p, float &e) {
    p = a * b;
    constexpr float SPLITTER = 4097.0f; // 2^12 + 1
    float ca = SPLITTER * a;
    float ah = ca - (ca - a);
    float al = a - ah;
    float cb = SPLITTER * b;
    float bh = cb - (cb - b);
    float bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// max|src[0..count)| 矢量归约（max 与 -min 取大）。dstTmp 为至少 256 floats 的
// 死区（dst 用前 8，tmp 用后 248，与 ReduceProdSegment 的 bufReduceWk 用法同口径）；
// count <= 4096（归约 tmp 的实测安全上界，与长行段 ReduceSum 一致）
__aicore__ inline float SpmvVecMaxAbs(const AscendC::LocalTensor<float> &src, int32_t count,
                                      const AscendC::LocalTensor<float> &dstTmp) {
    AscendC::ReduceMax<float>(dstTmp, src, dstTmp[8], count, false);
    // 矢量写 dst → 标量读
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float mx = dstTmp.GetValue(0);
    // 标量读 → 矢量写 dst（WAR）
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::ReduceMin<float>(dstTmp, src, dstTmp[8], count, false);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float mn = dstTmp.GetValue(0);
// 标量读 → 后续矢量覆写 dstTmp（WAR）
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    return mx > -mn ? mx : -mn;
}

// TwoProd 分派：GUARDED=true 走逐点守卫版，false 走无守卫快路径（调用方谓词
// 已保证安全）。让 fast/slow 共用同一份主体代码，避免双份实现的重复块
template <bool GUARDED>
__aicore__ inline void SpmvTwoProdSel(float a, float b, float &p, float &e) {
    if constexpr (GUARDED) {
        SpmvTwoProd(a, b, p, e);
    }
    else {
        SpmvTwoProdFast(a, b, p, e);
    }
}

// DS 双字窗口单槽位 TwoSum 合并：(acc[slot], accLo[slot]) += (h, l)
__aicore__ inline void SpmvDsMerge(__ubuf__ float *accPtr, __ubuf__ float *accLoPtr,
                                   uint32_t slot, float h, float l) {
    float s, e;
    SpmvTwoSum(accPtr[slot], h, s, e);
    accLoPtr[slot] += e + l;
    accPtr[slot] = s;
}

// N 路径 tile 级守卫谓词：返回 tileSafe（本 tile 全部非零元可否走无守卫快路径），
// rowSafeMax 输出行级阈值（max|x| 超拆分安全域时为 -1，禁止一切快路径（含零值元））；
// redWk 为归约死区工作区（须在 Gather/ReduceSum 之前使用）
__aicore__ inline bool SpmvTileSafeN(const AscendC::LocalTensor<float> &valsF, int32_t valid,
                                     float maxAbsX, const AscendC::LocalTensor<float> &redWk,
                                     float &rowSafeMax) {
    float tileMaxV = SpmvVecMaxAbs(valsF, valid, redWk);
    // max|x| 超拆分安全域时置 -1 哨兵：tileMaxV/rowMax 恒 >= 0，<= -1 恒假，
    // tile 级与行级快路径全禁——含零值元：快路径拆分需 4097*b 有限，b 越限时
    // 即使 a=0 也会 Inf-Inf 产出 NaN，必须走守卫路径（SPLIT 守卫回退 e=0、
    // p=fl(a*b)=0，结果正确）
    rowSafeMax = -1.0f;
    if (maxAbsX <= SPMV_TP_SPLIT_SAFE_MAX) {
        rowSafeMax = SPMV_TP_SPLIT_SAFE_MAX;
        if (maxAbsX > 0.0f) {
            float lim = SPMV_TP_PROD_SAFE_MAX / maxAbsX;
            rowSafeMax = lim < rowSafeMax ? lim : rowSafeMax;
        }
    }
    return tileMaxV <= rowSafeMax;
}
