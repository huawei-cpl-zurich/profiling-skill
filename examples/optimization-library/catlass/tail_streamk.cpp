/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See CANN_LICENSE. Adapted from Catlass 40aced927fda example 37.
 */
#define K_MAX_SHAPE_DIM 0
#include "acl/acl.h"
#include "catlass/gemm/kernel/streamk_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201
#error Requires Atlas A2/A3
#endif
using namespace Catlass;
using TailScheduler = Gemm::Block::StreamkGemmIdentityBlockSwizzle<3,0>;
// The control uses the SAME BlockMmad, ShuffleK, buffers and mixed-core shell.
// Its scheduler assigns each output tile's entire K range to one Cube core.
struct OrdinaryScheduler : TailScheduler {
    CATLASS_DEVICE OrdinaryScheduler(GemmCoord p,GemmCoord t,uint32_t cores)
        : TailScheduler(p,t,cores) {
        normalBlocks=loopsMNK.m()*loopsMNK.n(); streamkBlocks=0;
    }
    CATLASS_DEVICE uint32_t GetCoreLoops() const { return normalBlocks; }
    CATLASS_DEVICE GemmCoord GetBlockCoord(uint32_t task) {
        Gemm::Block::GemmIdentityBlockSwizzle<3,0> ordinary(problemShape,MatrixCoord(tileMNK.m(),tileMNK.n()));
        return ordinary.GetBlockCoord(task);
    }
    CATLASS_DEVICE void GetStreamkBlockDec(uint32_t task,StreamkBlockDec &d) {
        d.isStreamkBlock=false; d.isCrossBlock=false;
        d.blockCoord=GetBlockCoord(task); d.actualBlockShape=GetActualBlockShape(d.blockCoord);
    }
};
using Scheduler = std::conditional_t<bool(AFTER),TailScheduler,OrdinaryScheduler>;
using Policy = Gemm::MmadAtlasA2Streamk<true,true>;
using OutputTile = GemmShape<128,256,256>;
using Block = Gemm::Block::BlockMmad<Policy,OutputTile,GemmShape<128,256,64>,
    Gemm::GemmType<half,layout::RowMajor>,Gemm::GemmType<half,layout::RowMajor>,
    Gemm::GemmType<half,layout::RowMajor>>;
using Reduce = Gemm::Kernel::StreamkReduceAdd<Arch::AtlasA2,Scheduler,float,half,49152>;
using Kernel = Gemm::Kernel::StreamkMatmul<Block,void,Scheduler,Reduce>;
#if AFTER
#define ENTRY w003_tail_streamk_after
#else
#define ENTRY w003_tail_streamk_before
#endif
extern "C" CATLASS_GLOBAL void ENTRY(Kernel::Params params,uint64_t sync) {
    AscendC::SetSyncBaseAddr(sync); Kernel op; op(params);
}
static uint32_t m_,n_,k_,cores_;
static uint64_t sync_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || m%16 || n%16 || k%256 || cores!=20) return 1;
    m_=m;n_=n;k_=k;cores_=cores;
    return aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync_));
}
extern "C" uint32_t catlass_tail_after() { return AFTER; }
extern "C" uint64_t catlass_tail_workspace() {
    Kernel::Arguments args{GemmCoord(m_,n_,k_),cores_,nullptr,nullptr,nullptr};
    return Kernel::GetWorkspaceSize(args);
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR1(x) #x
#define STR(x) STR1(x)
extern "C" const char *catlass_bridge_kernel_names() { return "[\"" STR(ENTRY) "\"]"; }
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=4 || !cores_ || !sync_ || !p[0] || !p[1] || !p[2] || !p[3]) return 2;
    Kernel::Arguments args{GemmCoord(m_,n_,k_),cores_,(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[3]};
    auto params=Kernel::ToUnderlyingArguments(args,(uint8_t*)p[2]);
    ENTRY<<<cores_,nullptr,stream>>>(params,sync_); return 0;
}
