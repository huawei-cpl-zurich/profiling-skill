/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from pinned Catlass 40aced927fda; license retained in CANN_LICENSE.
#define K_MAX_SHAPE_DIM 0
#include "acl/acl.h"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201 || (FAMILY != 6 && FAMILY != 7)
#error This example requires Atlas A2/A3 and family 6 or 7
#endif
using namespace Catlass;
using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::RowMajor>;
using L1TileShape = GemmShape<128,256,256>;
using L0TileShape = GemmShape<128,256,64>;
#if FAMILY == 6 && AFTER
using CType = Gemm::GemmType<float, layout::RowMajor>;
using Policy = Gemm::MmadAtlasA2Pingpong<true>;
#define ENTRY wave1_splitk_after
#else
using CType = Gemm::GemmType<half, layout::RowMajor>;
#if FAMILY == 7
constexpr uint32_t stages = AFTER ? 2 : 1;
using Policy = Gemm::MmadPingpong<Arch::AtlasA2,true,false,1,false,stages,stages,stages,stages>;
#if AFTER
#define ENTRY wave1_buffering_after
#else
#define ENTRY wave1_buffering_before
#endif
#else
using Policy = Gemm::MmadAtlasA2Pingpong<true>;
#define ENTRY wave1_splitk_before
#endif
#endif
using BlockMmadType = Gemm::Block::BlockMmad<Policy,L1TileShape,L0TileShape,AType,BType,CType>;
#if FAMILY == 6 && AFTER
using ReduceAdd = Gemm::Kernel::SplitkReduceAdd<Arch::AtlasA2,float,half,2,16384>;
using MatmulKernel = Gemm::Kernel::SplitkMatmul<BlockMmadType,void,
    Gemm::Block::SplitkGemmIdentityBlockSwizzle<3,0>,ReduceAdd>;
#else
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmadType,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
#endif
extern "C" CATLASS_GLOBAL void ENTRY(MatmulKernel::Params params) { MatmulKernel op; op(params); }
static uint32_t m_, n_, k_, cores_;
static uint64_t sync_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16) return 1;
    m_=m; n_=n; k_=k; cores_=cores;
#if FAMILY == 6 && AFTER
    return aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&sync_));
#else
    return 0;
#endif
}
extern "C" uint32_t catlass_parallelism_family() { return FAMILY; }
extern "C" uint32_t catlass_parallelism_factor() {
#if FAMILY == 6 && AFTER
    return MatmulKernel::GetSplitkFactor(m_,n_,k_,cores_);
#else
    return 1;
#endif
}
extern "C" uint64_t catlass_parallelism_workspace() {
    return uint64_t(FAMILY == 6 && AFTER) * sizeof(float) * m_ * n_ * catlass_parallelism_factor();
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR_INNER(x) #x
#define STR(x) STR_INNER(x)
extern "C" const char *catlass_bridge_kernel_names() { return "[\"" STR(ENTRY) "\"]"; }
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if (count!=4 || !cores_ || !p[0] || !p[1] || !p[3]) return 2;
#if FAMILY == 6 && AFTER
    if (!p[2] || !sync_) return 3;
    MatmulKernel::Arguments args{GemmCoord(m_,n_,k_),cores_,sizeof(float),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[3]};
    auto params=MatmulKernel::ToUnderlyingArguments(args,(uint8_t*)p[2]);
    ENTRY<<<cores_, nullptr, stream>>>(params, sync_);
#else
    MatmulKernel::Arguments args{GemmCoord(m_,n_,k_),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[3]};
    auto params=MatmulKernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_, nullptr, stream>>>(params);
#endif
    return 0;
}
