/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source-backed AtlasA2 adaptation; pinned Catlass dependency retains its notices.
#define K_MAX_SHAPE_DIM 0
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#include "interval_config.h"
#if CATLASS_ARCH != 2201
#error Interval DMA example supports Atlas A2/A3 only
#endif
using namespace Catlass;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;
using DefaultCopy = Gemm::Tile::TileCopy<Arch::AtlasA2,AType,BType,CType>;
struct IntervalCopy : DefaultCopy {
#if AFTER
    using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<Arch::AtlasA2,AType>;
#else
    using CopyGmToL1A = typename DefaultCopy::CopyGmToL1A;
#endif
};
// All geometry, stages, B movement, arithmetic and scheduling are identical.
using L1TileShape = GemmShape<128,256,256>;
using L0TileShape = GemmShape<128,256,64>;
using BlockMmadType = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,
    L1TileShape,L0TileShape,AType,BType,CType,void,IntervalCopy>;
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmadType,void,
    Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
static_assert((BlockMmadType::L1A_SIZE+BlockMmadType::L1B_SIZE)*BlockMmadType::STAGES==393216);
#if AFTER
#define ENTRY wave2_interval_after
#else
#define ENTRY wave2_interval_before
#endif
extern "C" CATLASS_GLOBAL void ENTRY(MatmulKernel::Params params) { MatmulKernel op; op(params); }
extern "C" uint64_t catlass_interval_l1_bytes() { return 393216; }
extern "C" uint64_t catlass_interval_workspace() {
    MatmulKernel::Arguments args{GemmCoord(interval_m,interval_n,interval_k),nullptr,nullptr,nullptr};
    return MatmulKernel::GetWorkspaceSize(args);
}
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=3 || !interval_cores || !p[0] || !p[1] || !p[2]) return 2;
    MatmulKernel::Arguments args{GemmCoord(interval_m,interval_n,interval_k),
        (uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2]};
    auto params=MatmulKernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<interval_cores,nullptr,stream>>>(params);
    return 0;
}
