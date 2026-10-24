/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted only from the source-qualified TopK sort/filter/merge routines.
// CANN_LICENSE is retained beside this file. The host contract rejects K>1024.
#include "kernel_operator.h"
using namespace AscendC;
#ifndef AFTER
#define AFTER 0
#endif
#if AFTER
#define ENTRY wave003_topk_after
#else
#define ENTRY wave003_topk_before
#endif
constexpr uint32_t F03_TOPK_MAX=1024, F03_SORT32_PER_LEN=2048, F03_SORT32_MAX_LOOP=4;
constexpr uint32_t F03_MRGSORT_PER_NUM=4, F03_MRGSORT_PER_NUM_LAST=2;
constexpr uint32_t F03_SORT32_PERGROUP_LEN=2048;
constexpr uint32_t F03_NUM_ZERO=0,F03_NUM_ONE=1,F03_NUM_TWO=2,F03_NUM_THREE=3;
constexpr uint32_t F03_SIXTEEN=16,F03_THIRTY_TWO=32,F03_SIXTY_FOUR=64,F03_EIGHT=8;
constexpr float F03_FLOAT_MIN=-3.402823466e38f;
__aicore__ inline uint32_t F03_Ceil(uint32_t a,uint32_t b) {return (a+b-1)/b;}
__aicore__ inline uint32_t F03_Align(uint32_t a,uint32_t b) {return F03_Ceil(a,b)*b;}
template<HardEvent event>
__aicore__ inline void F03_SetWaitFlag(HardEvent evt) {
    auto id=GetTPipePtr()->FetchEventID(evt);SetFlag<event>(id);WaitFlag<event>(id);
}
class ThresholdTopK {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf0,buf1,buf2,buf3,buf4,buf5;
    uint32_t queLenGlobal[2]{};
public:
    __aicore__ inline void Run(GM_ADDR x,GM_ADDR indices,GM_ADDR values,uint32_t rows,uint32_t vocab,uint32_t k) {
        const uint32_t row=GetBlockIdx();
        // AIV launch/sub-block capacity can exceed the logical row payload.
        if (g_coreType == AIC || row >= rows) return;
        GlobalTensor<half> input,output;
        GlobalTensor<uint32_t> outIndex;
        input.SetGlobalBuffer((__gm__ half*)x+row*vocab,vocab);
        output.SetGlobalBuffer((__gm__ half*)values+row*k,k);
        outIndex.SetGlobalBuffer((__gm__ uint32_t*)indices+row*k,k);
        pipe.InitBuffer(buf0,32768);pipe.InitBuffer(buf1,32768);pipe.InitBuffer(buf2,32768);
        pipe.InitBuffer(buf3,32768);pipe.InitBuffer(buf4,32768);pipe.InitBuffer(buf5,32768);
        auto halfValues=buf0.Get<half>(); auto srcIndex=buf1.Get<uint32_t>();
        auto castValues=buf3.Get<float>(); auto sorted=buf2.Get<float>();
        for(uint32_t offset=0;offset<vocab;offset+=8192) {
            uint32_t count=vocab-offset<8192?vocab-offset:8192;
            // CreateVecIndex seeds buf1 with scalar stores before its vector ramp.
            // Wait for the previous vector merge to release that aliased buffer.
            F03_SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            CreateVecIndex(srcIndex.ReinterpretCast<int32_t>(),(int32_t)offset,count);
            PipeBarrier<PIPE_V>();
            // buf0 held previous vector sort scratch: fence its reuse by MTE2.
            F03_SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopyPad(halfValues,input[offset],{1,(uint16_t)(2*count),0,0},{false,0,0,0});
            F03_SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Cast(castValues,halfValues,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
            if(offset==0) Sort32Block(castValues,srcIndex,count,false);
            else {
#if AFTER
                Sort32BlockNext(castValues,srcIndex,count,sorted,k);
#else
                Sort32Block(castValues,srcIndex,count,true);
#endif
            }
        }
        auto vals=buf0.Get<float>(); auto ids=buf1.Get<uint32_t>();
        Extract(vals,ids,sorted,F03_TOPK_MAX/32);PipeBarrier<PIPE_V>();
        auto halfOut=buf3.Get<half>();Cast(halfOut,vals,RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
        F03_SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(output,halfOut,{1,(uint16_t)(2*k),0,0});
        DataCopyPad(outIndex,ids,{1,(uint16_t)(4*k),0,0});
        // Full arrays must leave UB before the next native invocation can reuse it.
        F03_SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        pipe.Destroy();
    }
private:
    __aicore__ inline void SortOneTime(LocalTensor<float>&& srcData, LocalTensor<uint32_t>&& srcIndex, uint32_t dataLen,
                                       LocalTensor<float>& sortedLocal)
    {
        LocalTensor<float> concatLocal;
        LocalTensor<float> sortTmpLocal = buf5.Get<float>();
        Concat(concatLocal, srcData, sortTmpLocal, dataLen / F03_SIXTEEN);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sortedLocal, concatLocal, srcIndex.ReinterpretCast<uint32_t>(), sortTmpLocal,
                          dataLen / F03_THIRTY_TWO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Sort32Block(LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex, uint32_t dataLen,
                                       bool mrg)
    {
        auto sort32LoopNum = F03_Ceil(dataLen, F03_SORT32_PER_LEN);
        auto sort32LoopTailNum = dataLen % F03_SORT32_PER_LEN;
        auto compareLen = F03_SORT32_PER_LEN;
        auto compareLenPad = F03_SORT32_PER_LEN;
        LocalTensor<float> msgSortPart1 = buf0.Get<float>();
        LocalTensor<float> msgSortPart2 = buf4.Get<float>();
        LocalTensor<float> msgSortLocal[F03_SORT32_MAX_LOOP] = {msgSortPart1, msgSortPart1[F03_SORT32_PER_LEN * F03_NUM_TWO],
                                                            msgSortPart2, msgSortPart2[F03_SORT32_PER_LEN * F03_NUM_TWO]};
        uint32_t queLen[F03_SORT32_MAX_LOOP] = {0};

        for (uint32_t sort32LoopCount = 0; sort32LoopCount < sort32LoopNum; ++sort32LoopCount) {
            uint32_t sort32BlkOfs = sort32LoopCount * F03_SORT32_PER_LEN;
            if (sort32LoopTailNum > 0 && sort32LoopCount == (sort32LoopNum - 1)) {
                compareLen = sort32LoopTailNum;
                compareLenPad = F03_Ceil(compareLen, F03_THIRTY_TWO) * F03_THIRTY_TWO;
                if (compareLenPad > compareLen) {
                    for (uint32_t i = compareLen; i < compareLenPad; ++i) {
                        srcData[sort32BlkOfs].SetValue(i, F03_FLOAT_MIN);
                    }
                    F03_SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
                }
            }
            SortOneTime(srcData[sort32BlkOfs], srcIndex[sort32BlkOfs], compareLenPad, msgSortLocal[sort32LoopCount]);
            queLen[sort32LoopCount] = compareLen;
        }

        uint16_t elementLengths[F03_MRGSORT_PER_NUM] = {F03_TOPK_MAX, F03_TOPK_MAX, F03_TOPK_MAX, F03_TOPK_MAX};
        uint16_t valueBits[5] = {0b11, 0b11, 0b11, 0b111, 0b1111};
        MrgSort4Info localParams = {};
        LocalTensor<float> msgSortLocalResult[3] = {buf2.Get<float>(), buf3.Get<float>(), buf1.Get<float>()};
        LocalTensor<float> msgSortLocalResultTemp = {};
        uint32_t qid = 0;
        if (mrg) {
            qid = 1;
            msgSortLocalResultTemp = msgSortLocalResult[F03_NUM_ONE];
        } else {
            qid = 0;
            msgSortLocalResultTemp = msgSortLocalResult[F03_NUM_ZERO];
        }
        if (sort32LoopNum <= 1) {
            // DataCopy demands 32bytes alignment, immediate call using calCount aligned to 32bytes is valid ONLY WHEN
            // msgSortLocal HAS BEEN ALIGNED to 32bytes yet.
            DataCopy(msgSortLocalResultTemp, msgSortLocal[F03_NUM_ZERO],
                     static_cast<uint32_t>(F03_Align(queLen[F03_NUM_ZERO] * F03_NUM_TWO, F03_EIGHT)));
            PipeBarrier<PIPE_V>();
            this->queLenGlobal[qid] = queLen[F03_NUM_ZERO];
        } else {
            this->queLenGlobal[qid] = F03_TOPK_MAX;
            for (uint32_t i = 0; i < sort32LoopNum; i++) {
                elementLengths[i] = queLen[i] > F03_TOPK_MAX ? F03_TOPK_MAX : queLen[i];
            }
            localParams = {elementLengths, false, valueBits[sort32LoopNum], 1};
            MrgSort<float>(
                msgSortLocalResultTemp,
                {msgSortLocal[F03_NUM_ZERO], msgSortLocal[F03_NUM_ONE], msgSortLocal[F03_NUM_TWO], msgSortLocal[F03_NUM_THREE]},
                localParams);
            PipeBarrier<PIPE_V>();
        }

        if (mrg) {
            elementLengths[F03_NUM_ZERO] = static_cast<uint16_t>(this->queLenGlobal[F03_NUM_ZERO]);
            elementLengths[F03_NUM_ONE] = static_cast<uint16_t>(this->queLenGlobal[F03_NUM_ONE]);
            localParams = {elementLengths, false, valueBits[F03_MRGSORT_PER_NUM_LAST], 1};
            MrgSort<float>(msgSortLocalResult[F03_NUM_TWO],
                           {msgSortLocalResult[F03_NUM_ZERO], msgSortLocalResult[F03_NUM_ONE], msgSortLocalResult[F03_NUM_ONE],
                            msgSortLocalResult[F03_NUM_ONE]},
                           localParams);
            PipeBarrier<PIPE_V>();
            DataCopy(msgSortLocalResult[F03_NUM_ZERO], msgSortLocalResult[F03_NUM_TWO], F03_SORT32_PERGROUP_LEN);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Sort32BlockNext(LocalTensor<float>& srcData, LocalTensor<uint32_t>& srcIndex,
                                           uint32_t dataLen, LocalTensor<float>& mrgSortResult, uint32_t kCount)
    {
        uint32_t firstTopkIndex = (kCount - 1) * F03_NUM_TWO;
        uint64_t rsvdCnt = 0;
        uint64_t oriRsvdCnt = 0;
        GatherMaskParams gatherMaskParams{1, 1, 8, 8};
        LocalTensor<uint8_t> compareIndexMask = buf5.Get<uint8_t>();

        F03_SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float firstTopk = mrgSortResult.GetValue(firstTopkIndex);
        CompareScalar(compareIndexMask, srcData, firstTopk, CMPMODE::GT, F03_Ceil(dataLen, F03_SIXTY_FOUR) * F03_SIXTY_FOUR);
        PipeBarrier<PIPE_V>();
        GatherMask(srcData, srcData, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams,
                   rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMask(srcIndex, srcIndex, compareIndexMask.ReinterpretCast<uint32_t>(), true, dataLen, gatherMaskParams,
                   rsvdCnt);
        PipeBarrier<PIPE_V>();

        if (rsvdCnt > 0) {
            uint64_t oriRsvCnt = rsvdCnt;
            rsvdCnt = F03_Ceil(rsvdCnt, F03_THIRTY_TWO) * F03_THIRTY_TWO;
            if (rsvdCnt > oriRsvCnt) {
                for (uint32_t i = oriRsvCnt; i < rsvdCnt; ++i) {
                    srcData.SetValue(i, F03_FLOAT_MIN);
                }
                F03_SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            }
            Sort32Block(srcData, srcIndex, rsvdCnt, true);
        }
    }

};
extern "C" __global__ __aicore__ void ENTRY(GM_ADDR x,GM_ADDR indices,GM_ADDR values,uint32_t rows,uint32_t vocab,uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ThresholdTopK op;op.Run(x,indices,values,rows,vocab,k);
}
static uint32_t rows_,vocab_,k_;
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){
#if AFTER
return "[\"wave003_topk_after\"]";
#else
return "[\"wave003_topk_before\"]";
#endif
}
extern "C" int threshold_configure(uint32_t rows,uint32_t vocab,uint32_t k){
    if(!rows||!vocab||!k||k>1024||k>vocab)return 1;
    rows_=rows;vocab_=vocab;k_=k;return 0;
}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
    if(count!=3||!rows_)return 1;
    ENTRY<<<rows_,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2],rows_,vocab_,k_);
    return 0;
}
