/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from Catlass 40aced927fda; see CANN_LICENSE and recipe citations.
#define K_MAX_SHAPE_DIM 0
#include "acl/acl.h"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/gemm/kernel/matmul_full_loadA.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201 || (FAMILY != 8 && FAMILY != 9 && FAMILY != 11)
#error This example requires Atlas A2/A3 and family 8, 9 or 11
#endif
using namespace Catlass;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;
using L1TileShape = GemmShape<128,256,256>;
using L0TileShape = GemmShape<128,256,64>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3,0>;
#if FAMILY == 8
#if AFTER
#define ENTRY wave1_preload_after
using Policy = Gemm::MmadAtlasA2Preload<true,false>;
#else
#define ENTRY wave1_preload_before
using Policy = Gemm::MmadAtlasA2Pingpong<true>;
#endif
#elif FAMILY == 9
#if AFTER
#define ENTRY wave1_resident_after
using Policy = Gemm::MmadAtlasA2FullLoadA<true>;
#else
#define ENTRY wave1_resident_before
using Policy = Gemm::MmadAtlasA2Pingpong<true>;
#endif
#else
#if AFTER
#define ENTRY wave1_shuffle_after
#else
#define ENTRY wave1_shuffle_before
#endif
using Policy = Gemm::MmadAtlasA2Preload<true,bool(AFTER)>;
#endif
using BlockMmadType = Gemm::Block::BlockMmad<Policy,L1TileShape,L0TileShape,AType,BType,CType>;
#if FAMILY == 9 && AFTER
using MatmulKernel = Gemm::Kernel::MatmulFullLoadA<BlockMmadType,void,Scheduler>;
#elif (FAMILY == 8 && AFTER) || FAMILY == 11
using MatmulKernel = Gemm::Kernel::OptimizedMatmul<void,void,BlockMmadType,void,Scheduler>;
#else
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmadType,void,Scheduler>;
#endif
extern "C" CATLASS_GLOBAL void ENTRY(MatmulKernel::Params params) { MatmulKernel op; op(params); }
static uint32_t m_,n_,k_,cores_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16) return 1;
#if FAMILY == 9
    // Match source's fixed L1 A partition and its M<=tile-M scheduler path.
    if (m>128 || uint64_t(128)*k*2 + uint64_t(256)*256*2*2 > 512*1024) return 2;
#endif
    m_=m; n_=n; k_=k; cores_=cores; return 0;
}
extern "C" uint32_t catlass_preload_family() { return FAMILY; }
extern "C" uint32_t catlass_preload_after() { return AFTER; }
extern "C" uint64_t catlass_preload_l1_bytes() {
#if FAMILY == 9 && AFTER
    return BlockMmadType::L1A_SIZE + BlockMmadType::L1B_SIZE*BlockMmadType::STAGES;
#else
    return (BlockMmadType::L1A_SIZE+BlockMmadType::L1B_SIZE)*BlockMmadType::STAGES;
#endif
}
extern "C" uint64_t catlass_preload_workspace() {
    MatmulKernel::Arguments args{GemmCoord(m_,n_,k_),nullptr,nullptr,nullptr};
    return MatmulKernel::GetWorkspaceSize(args);
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR_INNER(x) #x
#define STR(x) STR_INNER(x)
extern "C" const char *catlass_bridge_kernel_names() { return "[\"" STR(ENTRY) "\"]"; }
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=3 || !cores_ || !p[0] || !p[1] || !p[2]) return 3;
    MatmulKernel::Arguments args{GemmCoord(m_,n_,k_),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2]};
    auto params=MatmulKernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_, nullptr, stream>>>(params);
    return 0;
}
