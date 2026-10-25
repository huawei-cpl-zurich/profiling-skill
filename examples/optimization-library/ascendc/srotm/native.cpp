// Prepared source derivative; original Huawei OSLA2 notices in vendor/ and LICENSE.
#include <cstddef>
#include <cstdint>
#if BEFORE
#include "vendor/srotm_kernel.cpp"
#else
#include "srotm_dma.cpp"
#endif
static_assert(sizeof(SrotmTilingData)==80 && alignof(SrotmTilingData)==8);
static_assert(offsetof(SrotmTilingData,h12)==64 && offsetof(SrotmTilingData,h21)==68);
extern "C" __global__ __aicore__ void srotm_reset(GM_ADDR ox,GM_ADDR oy,GM_ADDR x,GM_ADDR y,uint32_t size) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,32768);auto local=buf.Get<float>();
    GlobalTensor<float> original[2],work[2];
    original[0].SetGlobalBuffer((__gm__ float*)ox);original[1].SetGlobalBuffer((__gm__ float*)oy);
    work[0].SetGlobalBuffer((__gm__ float*)x);work[1].SetGlobalBuffer((__gm__ float*)y);
    for(uint32_t start=GetBlockIdx()*8192;start<size;start+=8*8192){
        uint32_t count=size-start<8192?size-start:8192;DataCopyExtParams p{1,count*4,0,0,0};
        DataCopyPadExtParams<float> pad{true,0,static_cast<uint8_t>((8-count%8)%8),0};
        for(uint32_t v=0;v<2;v++){
            DataCopyPad(local,original[v][start],p,pad);
            event_t id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(id);WaitFlag<HardEvent::MTE2_MTE3>(id);
            DataCopyPad(work[v][start],local,p);
            id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(id);WaitFlag<HardEvent::MTE3_MTE2>(id);
        }
    }
}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){
#if BEFORE
    return "[\"srotm_reset\",\"_Z12srotm_kernel15SrotmTilingData\"]";
#else
    return "[\"srotm_reset\",\"_Z16srotm_kernel_dma15SrotmTilingData\"]";
#endif
}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
    if(count!=5)return 1;for(uint32_t i=0;i<count;i++)if(!p[i])return 2;
    const auto& t=*reinterpret_cast<const SrotmTilingData*>(p[4]);
    if(t.x!=(uint64_t)p[2] || t.y!=(uint64_t)p[3] || t.xStorageSize!=t.yStorageSize)return 3;
    srotm_reset<<<8,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2],(uint8_t*)p[3],t.xStorageSize);
#if BEFORE
    srotm_kernel_do(t,8,stream);
#else
    srotm_kernel_dma_do(t,8,stream);
#endif
    return 0;
}
