// Prepared source-backed component; vendor source and full CANN OSLA2 retained.
#include <cstddef>
#include <cstdint>
#if BEFORE
#include "sger_product.cpp"
#else
#include "vendor/sger_kernel.cpp"
#endif
static_assert(sizeof(SgerTilingData)==56 && alignof(SgerTilingData)==8);
static_assert(offsetof(SgerTilingData,m)==24 && offsetof(SgerTilingData,alpha)==52);
extern "C" __global__ __aicore__ void sger_reset(GM_ADDR original,GM_ADDR working,uint32_t size,uint32_t cores){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,32768);auto local=buf.Get<float>();
    GlobalTensor<float> a,b;a.SetGlobalBuffer((__gm__ float*)original);b.SetGlobalBuffer((__gm__ float*)working);
    for(uint32_t start=GetBlockIdx()*8192;start<size;start+=cores*8192){
        uint32_t count=size-start<8192?size-start:8192;DataCopyExtParams p{1,count*4,0,0,0};
        DataCopyPadExtParams<float> pad{true,0,static_cast<uint8_t>((8-count%8)%8),0};DataCopyPad(local,a[start],p,pad);
        event_t id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));SetFlag<HardEvent::MTE2_MTE3>(id);WaitFlag<HardEvent::MTE2_MTE3>(id);
        DataCopyPad(b[start],local,p);id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));SetFlag<HardEvent::MTE3_MTE2>(id);WaitFlag<HardEvent::MTE3_MTE2>(id);
    }
}
#if BEFORE
extern "C" __global__ __aicore__ void sger_add(GM_ADDR product,const SgerTilingData t){
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;TBuf<TPosition::VECCALC> aa,pp,oo;pipe.InitBuffer(aa,32768);pipe.InitBuffer(pp,32768);pipe.InitBuffer(oo,32768);
    auto a=aa.Get<float>(),p=pp.Get<float>(),o=oo.Get<float>();GlobalTensor<float> work,prod;work.SetGlobalBuffer((__gm__ float*)t.A);prod.SetGlobalBuffer((__gm__ float*)product);
    uint32_t start=GetBlockIdx()*t.colsPerBlock;
    uint32_t count=start>=t.n?0:GetBlockIdx()==t.useCoreNum-1?t.n-start:t.colsPerBlock;
    for(uint32_t col=start;col<start+count;col++)for(uint32_t row=0;row<t.m;row+=8192){
        uint32_t n=t.m-row<8192?t.m-row:8192;uint64_t off=(uint64_t)col*t.lda+row;
        DataCopyExtParams copy{1,n*4,0,0,0};DataCopyPadExtParams<float> pad{true,0,static_cast<uint8_t>((8-n%8)%8),0};
        DataCopyPad(a,work[off],copy,pad);DataCopyPad(p,prod[off],copy,pad);
        event_t id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));SetFlag<HardEvent::MTE2_V>(id);WaitFlag<HardEvent::MTE2_V>(id);
        Add(o,a,p,n);id=static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));SetFlag<HardEvent::V_MTE3>(id);WaitFlag<HardEvent::V_MTE3>(id);
        DataCopyPad(work[off],o,copy);id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));SetFlag<HardEvent::MTE3_MTE2>(id);WaitFlag<HardEvent::MTE3_MTE2>(id);
    }
}
#endif
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){
#if BEFORE
    return "[\"sger_reset\",\"_Z12sger_product14SgerTilingData\",\"sger_add\"]";
#else
    return "[\"sger_reset\",\"_Z11sger_kernel14SgerTilingData\"]";
#endif
}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
    if(count!=6)return 1;for(uint32_t i=0;i<count;i++)if(!p[i])return 2;
    const auto&t=*reinterpret_cast<const SgerTilingData*>(p[5]);
    if(t.A!=(uint64_t)p[3]||t.x!=(uint64_t)p[1]||t.y!=(uint64_t)p[2]||!t.useCoreNum)return 3;
    sger_reset<<<t.useCoreNum,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[3],t.lda*t.n,t.useCoreNum);
#if BEFORE
    auto producer=t;producer.A=(uint64_t)p[4];sger_product_do(producer,t.useCoreNum,stream);
    sger_add<<<t.useCoreNum,nullptr,stream>>>((uint8_t*)p[4],t);
#else
    sger_kernel_do(t,t.useCoreNum,stream);
#endif
    return 0;
}
