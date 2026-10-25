/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#ifndef SGER_TILING_H
#define SGER_TILING_H

#include <cstdint>

struct SgerTilingData {
    uint64_t A;
    uint64_t x;
    uint64_t y;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t colsPerBlock;
    int incx;
    int incy;
    float alpha;
};

#endif // SGER_TILING_H
