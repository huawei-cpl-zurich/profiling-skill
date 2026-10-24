// Source mechanism: certified prefix with full-sort fallback, N04.
// Independently implemented heap ordering; no upstream implementation copied.
#include "kernel_operator.h"
using namespace AscendC;
#ifndef AFTER
#define AFTER 0
#endif
#if AFTER
#define ENTRY wave003_topp_after
#else
#define ENTRY wave003_topp_before
#endif
class TopP {
    TPipe pipe;
    TBuf<TPosition::VECCALC> halfBuffer, floatBuffer;
    GlobalTensor<half> input;
    GlobalTensor<float> probability;
    GlobalTensor<int32_t> indices, counts, tags;
    uint32_t vocab;
    __aicore__ inline float Score(uint32_t pos) { return (float)input.GetValue(indices.GetValue(pos)); }
    __aicore__ inline void Swap(uint32_t a,uint32_t b) {
        int32_t saved=indices.GetValue(a);
        indices.SetValue(a,indices.GetValue(b));indices.SetValue(b,saved);
    }
    // A minimum heap retains the highest K; extraction leaves descending order.
    __aicore__ inline void Sift(uint32_t pos,uint32_t size) {
        while(2*pos+1<size) {
            uint32_t child=2*pos+1;
            if(child+1<size && Score(child+1)<Score(child)) ++child;
            if(Score(pos)<=Score(child)) break;
            Swap(pos,child);pos=child;
        }
    }
    __aicore__ inline void Order(uint32_t limit) {
        for(uint32_t i=0;i<limit;++i) indices.SetValue(i,i);
        for(int32_t i=(int32_t)limit/2-1;i>=0;--i) Sift(i,limit);
        for(uint32_t i=limit;i<vocab;++i) {
            if((float)input.GetValue(i)>Score(0)) {
                indices.SetValue(0,i);Sift(0,limit);
            }
        }
        for(uint32_t end=limit-1;end>0;--end) { Swap(0,end);Sift(0,end); }
    }
    __aicore__ inline uint32_t Crossing(uint32_t limit,float p,float &previous,float &mass) {
        mass=0;previous=0;
        for(uint32_t i=0;i<limit;++i) {
            previous=mass;
            // Scalar FP32 addition specifies left-to-right accumulation.
            mass=mass+probability.GetValue(indices.GetValue(i));
            if(mass>p) return i+1;
        }
        return 0;
    }
public:
    __aicore__ inline void Run(GM_ADDR x,GM_ADDR probs,GM_ADDR ids,GM_ADDR count,GM_ADDR tag,
                              uint32_t rows,uint32_t width,uint32_t prefix,float p) {
        uint32_t row=GetBlockIdx();
        if(g_coreType==AIC || row>=rows) return;
        vocab=width;
        input.SetGlobalBuffer((__gm__ half*)x+row*vocab,vocab);
        const uint32_t stride=(vocab+31)/32*32;
        probability.SetGlobalBuffer((__gm__ float*)probs+row*stride,vocab);
        indices.SetGlobalBuffer((__gm__ int32_t*)ids+row*stride,vocab);
        counts.SetGlobalBuffer((__gm__ int32_t*)count+row*32,1);
        tags.SetGlobalBuffer((__gm__ int32_t*)tag+row*32,1);
        pipe.InitBuffer(halfBuffer,2048);pipe.InitBuffer(floatBuffer,4096);
        auto h=halfBuffer.Get<half>();auto f=floatBuffer.Get<float>();
        float maximum=-3.402823466e38f;
        for(uint32_t i=0;i<vocab;++i) {float v=(float)input.GetValue(i);if(v>maximum) maximum=v;}
        float denominator=0;
        for(uint32_t offset=0;offset<vocab;offset+=1024) {
            uint32_t n=vocab-offset<1024?vocab-offset:1024;
            DataCopyPad(h,input[offset],{1,(uint16_t)(2*n),0,0},{false,0,0,0});
            auto e=pipe.FetchEventID(HardEvent::MTE2_V);SetFlag<HardEvent::MTE2_V>(e);WaitFlag<HardEvent::MTE2_V>(e);
            Cast(f,h,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
            Adds(f,f,-maximum,n);PipeBarrier<PIPE_V>();Exp(f,f,n);PipeBarrier<PIPE_V>();
            e=pipe.FetchEventID(HardEvent::V_S);SetFlag<HardEvent::V_S>(e);WaitFlag<HardEvent::V_S>(e);
            for(uint32_t i=0;i<n;++i) {float v=f.GetValue(i);denominator=denominator+v;probability.SetValue(offset+i,v);}
            e=pipe.FetchEventID(HardEvent::V_MTE2);SetFlag<HardEvent::V_MTE2>(e);WaitFlag<HardEvent::V_MTE2>(e);
        }
        for(uint32_t i=0;i<vocab;++i) probability.SetValue(i,probability.GetValue(i)/denominator);
        uint32_t size=0;float previous=0,mass=0;int32_t dispatch=0;
#if AFTER
        Order(prefix);size=Crossing(prefix,p,previous,mass);
        if(size && p-previous>1e-5f && mass-p>1e-5f) dispatch=1;
        else
#endif
        {Order(vocab);size=Crossing(vocab,p,previous,mass);}
        counts.SetValue(0,size?size:vocab);tags.SetValue(0,dispatch);
        // Scalar GM stores live in the scalar cache; publish complete cache lines.
        // Row outputs have 128-byte strides to prevent cross-core false sharing.
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(indices);
        pipe.Destroy();
    }
};
extern "C" __global__ __aicore__ void ENTRY(GM_ADDR x,GM_ADDR probs,GM_ADDR ids,GM_ADDR counts,GM_ADDR tags,
                                         uint32_t rows,uint32_t vocab,uint32_t prefix,float p) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TopP op;op.Run(x,probs,ids,counts,tags,rows,vocab,prefix,p);
}
static uint32_t rows_,vocab_,prefix_;static float p_;
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){
#if AFTER
return "[\"wave003_topp_after\"]";
#else
return "[\"wave003_topp_before\"]";
#endif
}
extern "C" int topp_configure(uint32_t r,uint32_t v,uint32_t k,float p){
    if(!r||!v||!k||k>v||!(p>0&&p<1))return 1;
    rows_=r;vocab_=v;prefix_=k;p_=p;return 0;
}
extern "C" int catlass_bridge_launch(void** p,uint64_t n,void* stream){
    if(n!=5||!rows_)return 1;
    ENTRY<<<rows_,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2],(uint8_t*)p[3],(uint8_t*)p[4],rows_,vocab_,prefix_,p_);
    return 0;
}
