/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// Source-derived prepared launch only; upstream example main is not executed.
#define K_MAX_SHAPE_DIM 0
#include "catlass/gemm/kernel/matmul_full_loadA.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include <cstddef>
#include <new>
using namespace Catlass;
using AType=Gemm::GemmType<half,layout::RowMajor>;
using BType=Gemm::GemmType<half,layout::RowMajor>;
using CType=Gemm::GemmType<half,layout::RowMajor>;
using Block=Gemm::Block::BlockMmad<Gemm::MmadAtlasA2FullLoadA<true>,GemmShape<128,256,256>,GemmShape<128,256,64>,AType,BType,CType>;
using Kernel=Gemm::Kernel::MatmulFullLoadA<Block,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
using Device=Gemm::Device::DeviceGemm<Kernel>;
static_assert(std::is_same_v<Kernel::ElementAccumulator,float>);
static_assert(Block::STAGES==2 && Block::ENABLE_UNIT_FLAG);
static_assert(Block::L1A_SIZE==256*1024 && Block::L1B_SIZE*Block::STAGES==256*1024);
struct Handle { Device gemm; };
static bool shape(uint32_t m,uint32_t n,uint32_t k) {return m>0&&m<=128&&n>0&&n%16==0&&k>0&&k<=1024&&k%16==0;}
extern "C" int r12_inspect(uint32_t m,uint32_t n,uint32_t k,uint64_t *out,uint32_t count) {
    if(!shape(m,n,k)||!out||count!=16)return 1;
    auto a=Block::LayoutAInL1::MakeLayout<half>(128,k);
    Kernel::Arguments args{GemmCoord(m,n,k),nullptr,nullptr,nullptr};
    uint64_t values[]={sizeof(Kernel::Params),alignof(Kernel::Params),offsetof(Kernel::Params,problemShape),offsetof(Kernel::Params,ptrA),offsetof(Kernel::Params,layoutA),offsetof(Kernel::Params,ptrB),offsetof(Kernel::Params,layoutB),offsetof(Kernel::Params,ptrC),offsetof(Kernel::Params,layoutC),uint64_t(a.Capacity())*sizeof(half),Block::L1B_SIZE*Block::STAGES,Block::L0A_TILE_SIZE*Block::STAGES,Block::L0B_TILE_SIZE*Block::STAGES,Block::L0C_TILE_SIZE,sizeof(Kernel::ElementAccumulator),Device::GetWorkspaceSize(args)};
    if(values[9]>Block::L1A_SIZE||values[10]>Arch::AtlasA2::L1_SIZE-Block::L1A_SIZE)return 2;
    for(uint32_t i=0;i<count;i++)out[i]=values[i];return 0;
}
extern "C" int r12_create(uint32_t m,uint32_t n,uint32_t k,void *a,void *b,void *c,void *workspace,void *stream,void **result) {
    if(!shape(m,n,k)||!a||!b||!c||!result)return 1;
    uint64_t proof[16];if(r12_inspect(m,n,k,proof,16)||proof[15]!=0||workspace)return 2;
    auto handle=new(std::nothrow) Handle;if(!handle)return 3;
    Kernel::Arguments args{GemmCoord(m,n,k),(uint8_t*)a,(uint8_t*)b,(uint8_t*)c};
    if(Device::CanImplement(args)!=Status::kSuccess||handle->gemm.Initialize(args,nullptr,stream)!=Status::kSuccess){delete handle;return 4;}
    *result=handle;return 0;
}
extern "C" int r12_launch(void *context,void *stream) {
    if(!context)return 1;
    return ((Handle*)context)->gemm(stream,24)==Status::kSuccess?0:2;
}
extern "C" void r12_destroy(void *context) {delete (Handle*)context;}
