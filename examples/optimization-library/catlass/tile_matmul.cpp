/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Original license text is retained here as CANN_LICENSE.
// Source-backed adaptation of Catlass 40aced927fda examples/00_basic_matmul.
// The separately supplied pinned Catlass headers retain their own license.
#define K_MAX_SHAPE_DIM 0
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201
#error This pilot supports Atlas A2/A3 only
#endif
#ifndef AFTER
#define AFTER 0
#endif
using namespace Catlass;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;
#if AFTER
using L1TileShape = GemmShape<256,128,256>;
using L0TileShape = GemmShape<256,128,64>;
#define ENTRY wave1_tile_after
#else
using L1TileShape = GemmShape<128,256,256>;
using L0TileShape = GemmShape<128,256,64>;
#define ENTRY wave1_tile_before
#endif
using BlockMmadType = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>, L1TileShape,L0TileShape,AType,BType,CType>;
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmadType,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
extern "C" CATLASS_GLOBAL void ENTRY(MatmulKernel::Params params) { MatmulKernel op; op(params); }
static uint32_t m_, n_, k_, cores_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16) return 1;
    m_=m; n_=n; k_=k; cores_=cores; return 0;
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
extern "C" const char *catlass_bridge_kernel_names() {
#if AFTER
    return "[\"wave1_tile_after\"]";
#else
    return "[\"wave1_tile_before\"]";
#endif
}
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=3 || !cores_ || !p[0] || !p[1] || !p[2]) return 2;
    MatmulKernel::Arguments args{GemmCoord(m_,n_,k_), (uint8_t*)p[0], (uint8_t*)p[1], (uint8_t*)p[2]};
    auto params=MatmulKernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_, nullptr, stream>>>(params);
    return 0;
}
