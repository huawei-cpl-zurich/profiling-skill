/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from Catlass 40aced927fda examples/03_matmul_add and 06_optimized_matmul.
// Original license is retained in CANN_LICENSE; dependency headers remain pinned.
// Keep the subcontext ownership rule executable in a CPU ABI regression test.
#ifdef MIXED_SCHEDULE_UNIT_TEST
#include <stdint.h>
#define MIXED_HOST_DEVICE
#else
#define K_MAX_SHAPE_DIM 0
#include "acl/acl.h"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201 || (FAMILY != 2 && FAMILY != 10)
#error These examples require Atlas A2/A3 and family 2 or 10
#endif
#define MIXED_HOST_DEVICE CATLASS_HOST_DEVICE
#endif
MIXED_HOST_DEVICE inline uint32_t EpilogueTaskOwner(uint32_t block,uint32_t subblocks) {
    return block/subblocks;
}
#ifdef MIXED_SCHEDULE_UNIT_TEST
extern "C" uint32_t catlass_epilogue_task_owner(uint32_t block,uint32_t subblocks) {
    return EpilogueTaskOwner(block,subblocks);
}
#else
using namespace Catlass;
using InputLayout = layout::RowMajor;
using L1TileShape = GemmShape<128,256,256>;
using L0TileShape = GemmShape<128,256,64>;
using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3,0>;
#if FAMILY == 2
using AType = Gemm::GemmType<half,InputLayout>;
using BType = Gemm::GemmType<half,InputLayout>;
using CType = Gemm::GemmType<float,InputLayout>;
using BlockMmadType = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Pingpong<true>,
    L1TileShape,L0TileShape,AType,BType,CType>;
using TileAdd = Epilogue::Tile::TileElemWiseAdd<Arch::AtlasA2,CType,16384>;
using EpilogueCopy = Epilogue::Tile::TileCopy<Arch::AtlasA2,CType,CType,CType>;
using BlockAdd = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2ElemWiseOneSource,
    CType,CType,CType,TileAdd,EpilogueCopy>;
using AddKernel = Gemm::Kernel::MatmulEpilogue<BlockMmadType,BlockAdd,Scheduler>;
using PlainKernel = Gemm::Kernel::BasicMatmul<BlockMmadType,void,Scheduler>;
#if AFTER
#define ENTRY wave1_add_fused
extern "C" CATLASS_GLOBAL void ENTRY(AddKernel::Params params,uint64_t sync) {
    AscendC::SetSyncBaseAddr(sync); AddKernel op; op(params);
}
#else
#define ENTRY wave1_add_matmul
extern "C" CATLASS_GLOBAL void ENTRY(PlainKernel::Params params) { PlainKernel op; op(params); }
extern "C" CATLASS_GLOBAL void wave1_add_vector(AddKernel::Params params) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (g_coreType == AscendC::AIV) {
        Arch::Resource<Arch::AtlasA2> resource;
        BlockAdd add(resource,params.epilogueParams);
        GemmCoord tile(128,256,256);
        Scheduler scheduler(params.problemShape,MakeCoord(tile.m(),tile.n()));
        AscendC::GlobalTensor<float> intermediate;
        intermediate.SetGlobalBuffer((__gm__ float*)params.ptrWorkspace);
        layout::RowMajor layout(params.problemShape.m(),params.problemShape.n());
        for (uint32_t task=EpilogueTaskOwner(AscendC::GetBlockIdx(),AscendC::GetSubBlockNum());
             task<scheduler.GetCoreLoops();task+=AscendC::GetBlockNum()) {
            auto coord=scheduler.GetBlockCoord(task);
            auto actual=scheduler.GetActualBlockShape(coord);
            auto offset=coord.GetCoordMN()*tile.GetCoordMN();
            add(tile,coord,actual,intermediate[layout.GetOffset(offset)],layout.GetTileLayout(actual.GetCoordMN()));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
#endif
#else
using Policy = Gemm::MmadAtlasA2Preload<true,false>;
#if AFTER
using PadA = Gemm::Kernel::PaddingBuilder<Gemm::Kernel::PaddingTag::PADDING_NZ,
    Arch::AtlasA2,half,InputLayout,48*1024/sizeof(half)>;
using PadB = PadA;
using PrologueA = typename PadA::Padding;
using PrologueB = typename PadB::Padding;
using AType = Gemm::GemmType<half,typename PadA::LayoutAfterPadding>;
using BType = Gemm::GemmType<half,typename PadB::LayoutAfterPadding>;
#define ENTRY wave1_padding_after
#else
using PrologueA = void;
using PrologueB = void;
using AType = Gemm::GemmType<half,InputLayout>;
using BType = Gemm::GemmType<half,InputLayout>;
#define ENTRY wave1_padding_before
#endif
using CType = Gemm::GemmType<half,InputLayout>;
using BlockMmadType = Gemm::Block::BlockMmad<Policy,L1TileShape,L0TileShape,AType,BType,CType>;
using PaddingKernel = Gemm::Kernel::OptimizedMatmul<PrologueA,PrologueB,BlockMmadType,void,Scheduler>;
extern "C" CATLASS_GLOBAL void ENTRY(PaddingKernel::Params params,uint64_t sync) {
    AscendC::SetSyncBaseAddr(sync); PaddingKernel op; op(params);
}
#endif
static uint32_t m_,n_,k_,cores_;
static uint64_t sync_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || !cores || m>32768 || n>32768 || k>32768) return 1;
#if FAMILY == 2
    if (m%16 || n%16 || k%16) return 1;
#endif
    m_=m;n_=n;k_=k;cores_=cores;
    return aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync_));
}
extern "C" uint32_t catlass_mixed_family() { return FAMILY; }
extern "C" uint64_t catlass_mixed_workspace() {
#if FAMILY == 2
    return uint64_t(m_)*n_*sizeof(float);
#else
    PaddingKernel::Arguments args{GemmCoord(m_,n_,k_),nullptr,nullptr,nullptr};
    return PaddingKernel::GetWorkspaceSize(args);
#endif
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR_INNER(x) #x
#define STR(x) STR_INNER(x)
extern "C" const char* catlass_bridge_kernel_names() {
#if FAMILY == 2 && !AFTER
    return "[\"wave1_add_matmul\",\"wave1_add_vector\"]";
#else
    return "[\"" STR(ENTRY) "\"]";
#endif
}
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=5 || !cores_ || !sync_ || !p[0] || !p[1] || !p[4]) return 2;
    if (catlass_mixed_workspace() && !p[3]) return 3;
    GemmCoord shape(m_,n_,k_);
#if FAMILY == 2
    if (!p[2]) return 4;
    InputLayout la(m_,k_),lb(k_,n_),lc(m_,n_);
    BlockAdd::Params add{(uint8_t*)p[2],lc,(uint8_t*)p[4],lc};
    AddKernel::Params params(shape,(uint8_t*)p[0],la,(uint8_t*)p[1],lb,(uint8_t*)p[3],add);
#if AFTER
    ENTRY<<<cores_,nullptr,stream>>>(params,sync_);
#else
    PlainKernel::Arguments args{shape,(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[3]};
    auto plain=PlainKernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_,nullptr,stream>>>(plain);
    // A mixed logical block owns two AIV subcontexts, matching the fused epilogue.
    wave1_add_vector<<<cores_,nullptr,stream>>>(params);
#endif
#else
    PaddingKernel::Arguments args{shape,(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[4]};
    auto params=PaddingKernel::ToUnderlyingArguments(args,(uint8_t*)p[3]);
    ENTRY<<<cores_,nullptr,stream>>>(params,sync_);
#endif
    return 0;
}

#endif // MIXED_SCHEDULE_UNIT_TEST
