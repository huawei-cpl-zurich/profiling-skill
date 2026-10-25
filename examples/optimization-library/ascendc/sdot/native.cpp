// Source-derived educational comparison. Exact source and OSLA2 notices in vendor/ and LICENSE.
#include <cstdint>
#if BEFORE
#include "sdot_reduce.cpp"
extern "C" __global__ __aicore__ __vector__ void sdot_materialize(GM_ADDR x, GM_ADDR y, GM_ADDR p, GM_ADDR t) {
    AscendC::SetMaskNorm(); SetVectorMask<float>((uint64_t)-1, (uint64_t)-1); SetAtomicnone();
    uint32_t core=AscendC::GetBlockIdx(); auto q=reinterpret_cast<__gm__ uint32_t*>(t);
    uint32_t offset=q[3+core], count=q[43+core];
    AscendC::GlobalTensor<float> gx,gy,gp;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));gy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto ux=buf.GetBuffer<BufferType::ASCEND_UB,float>(0);auto uy=buf.GetBuffer<BufferType::ASCEND_UB,float>(46*1024);
    for(uint32_t start=0;start<count;start+=11776){
        uint32_t len=count-start<11776?count-start:11776;uint32_t repeats=(len+63)/64;
        AscendC::Duplicate<float,false>(ux,0.f,64,repeats,1,8);AscendC::Duplicate<float,false>(uy,0.f,64,repeats,1,8);
        SET_FLAG(V,MTE2,EVENT_ID0);WAIT_FLAG(V,MTE2,EVENT_ID0);
        copy_vec_gm2ub(ux,gx[offset+start],len);copy_vec_gm2ub(uy,gy[offset+start],len);
        SET_FLAG(MTE2,V,EVENT_ID0);WAIT_FLAG(MTE2,V,EVENT_ID0);
        mul_v<ArchType::ASCEND_V220,float>(ux,ux,uy,repeats,1,1,1,8,8,8);
        SET_FLAG(V,MTE3,EVENT_ID0);WAIT_FLAG(V,MTE3,EVENT_ID0);
        copy_vec_ub2gm(gp[offset+start],ux,len);PIPE_BARRIER(ALL);
    }
    SetAtomicnone();
}
#else
#include "vendor/blas/dot/arch22/sdot_kernel.cpp"
#endif
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){
#if BEFORE
    return "[\"sdot_materialize\",\"_Z17sdot_materializedPhS_S_S_S_\"]";
#else
    return "[\"_Z4sdotPhS_S_S_S_\"]";
#endif
}
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream){
    if(count!=6)return 1;for(uint32_t i=0;i<count;i++)if(!p[i])return 2;
#if BEFORE
    sdot_materialize<<<8,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[4],(uint8_t*)p[3]);
    sdot_materialized_kernel_do((uint8_t*)p[4],(uint8_t*)p[1],(uint8_t*)p[5],(uint8_t*)p[2],(uint8_t*)p[3],8,stream);
#else
    sdot_kernel_do((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[5],(uint8_t*)p[2],(uint8_t*)p[3],8,stream);
#endif
    return 0;
}
