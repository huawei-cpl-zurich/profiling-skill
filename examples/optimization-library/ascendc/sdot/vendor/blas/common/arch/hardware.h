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

enum class ArchType { ASCEND_V220, ASCEND_V200, ASCEND_M200, ASCEND_V350 };

template <ArchType ArchTag>
struct HardwareInfo {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 128 * 1024;
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 192 * 1024;
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 32;
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};

// Specialization for arch35 (DAV_3510 / Ascend 950)
template <>
struct HardwareInfo<ArchType::ASCEND_V350> {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 32 * 1024;      // arch35 L1 = 32KB
    static uint32_t const l0ASize = 64 * 1024;     // arch35 L0A = 64KB
    static uint32_t const l0BSize = 64 * 1024;     // arch35 L0B = 64KB
    static uint32_t const l0CSize = 256 * 1024;    // arch35 L0C = 256KB
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 248 * 1024;     // arch35 UB = 248KB
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 16;      // arch35 Cube block = 16
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};

