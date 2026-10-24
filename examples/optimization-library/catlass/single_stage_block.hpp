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
#include "catlass/gemm/block/block_mmad.hpp"
namespace Catlass::Gemm::Block {
template <bool DIRECT, class Base> struct SingleStageBlock : Base {
public:
    using Base::Base;
    using Base::l1ATensorList;
    using Base::l1BTensorList;
    using Base::l0ATensorList;
    using Base::l0BTensorList;
    using Base::l0CTensor;
    using Base::l1AEventList;
    using Base::l1BEventList;
    using Base::l0AEventList;
    using Base::l0BEventList;
    using Base::l1ListId;
    using Base::l0AListId;
    using Base::l0BListId;
    using Base::tileMmad;
    using Base::copyGmToL1A;
    using Base::copyGmToL1B;
    using Base::copyL1ToL0A;
    using Base::copyL1ToL0B;
    using Base::copyL0CToGm;
    using DispatchPolicy = typename Base::DispatchPolicy;
    using ArchTag = typename Base::ArchTag;
    using L1TileShape = typename Base::L1TileShape;
    using L0TileShape = typename Base::L0TileShape;
    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;
    using LayoutAInL1 = typename Base::LayoutAInL1;
    using LayoutBInL1 = typename Base::LayoutBInL1;
    using LayoutAInL0 = typename Base::LayoutAInL0;
    using LayoutBInL0 = typename Base::LayoutBInL0;
    using LayoutCInL0 = typename Base::LayoutCInL0;
    using L1AAlignHelper = typename Base::L1AAlignHelper;
    using L1BAlignHelper = typename Base::L1BAlignHelper;
    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr uint32_t STAGES = Base::STAGES;
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const& gmA, LayoutA const& layoutA, AscendC::GlobalTensor<ElementB> const& gmB,
        LayoutB const& layoutB, AscendC::GlobalTensor<ElementC> const& gmC, LayoutC const& layoutC,
        GemmCoord const& actualShape)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t outerLoops = DIRECT ? 1 : CeilDiv<L1TileShape::K>(actualShape.k());
        for (uint32_t outer = 0; outer < outerLoops; ++outer) {
            uint32_t kStart = outer * L1TileShape::K;
            uint32_t kActual = Min(L1TileShape::K, actualShape.k() - kStart);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmA[layoutA.GetOffset(MakeCoord(0U, kStart))], layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmB[layoutB.GetOffset(MakeCoord(kStart, 0U))], layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);
            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                uint32_t kPartActual =
                    (kPartIdx < kPartLoop - 1) ? L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);
                auto l0ATile = l0ATensorList[l0AListId];
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kPartActual);
                MatrixCoord l1AOffset{0, kPartIdx * L0TileShape::K};
                auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);
                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                }
                auto l0BTile = l0BTensorList[l0BListId];
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nRound);
                MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                }
                copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);
                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                auto l0CTile = l0CTensor;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                bool initC = (outer == 0 && kPartIdx == 0);
                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    if (kPartIdx == kPartLoop - 1) {
                        unitFlag = 0b11;
                    } else {
                        unitFlag = 0b10;
                    }
                }
                tileMmad(l0CTile, l0ATile, l0BTile, mRound, nRound, kPartActual, initC, unitFlag);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
            }
            l1ListId = (l1ListId + 1) % STAGES;
        }
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }
};
} // namespace Catlass::Gemm::Block
