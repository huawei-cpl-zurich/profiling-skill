/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "catlass/arch/resource.hpp"
namespace Catlass::Gemm::Kernel {
template <bool DIRECT, class BlockMmad_>
class SingleStageKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        CATLASS_HOST_DEVICE
        Params()
        {}
        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_)
        {}
    };
    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };
    static bool CanImplement(const Arguments& args)
    {
        return true;
    }
    static size_t GetWorkspaceSize(const Arguments& args)
    {
        return 0;
    }
    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(args.problemShape.m(), args.problemShape.n());
        Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC};
        return params;
    }
    CATLASS_DEVICE
    SingleStageKernel()
    {}
    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        uint32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;
        uint32_t nLoops = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        uint32_t totalLoops = mLoops * nLoops;
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < totalLoops;
             loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord{loopIdx / nLoops, loopIdx % nLoops, 0};
            uint32_t mActual = (blockCoord.m() == (mLoops - 1)) ?
                                   (params.problemShape.m() - blockCoord.m() * L1TileShape::M) :
                                   L1TileShape::M;
            uint32_t nActual = (blockCoord.n() == (nLoops - 1)) ?
                                   (params.problemShape.n() - blockCoord.n() * L1TileShape::N) :
                                   L1TileShape::N;
            GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, 0U};
            MatrixCoord offsetB{0U, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            blockMmad(
                gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC,
                actualBlockShape);
            if constexpr (DIRECT) break;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Gemm::Kernel
