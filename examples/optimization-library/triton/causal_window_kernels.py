# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# MIT; source7acb9c40a6ca827e885d902b7eaee5d45457b230.
import triton
import triton.language as tl

@triton.jit
def _direct_state_window(x_ptr,conv_state_ptr,weight_ptr,bias_ptr,conv_state_indices_ptr,out_ptr,pad_slot_id,
                        batch:tl.constexpr,dim:tl.constexpr,align_val:tl.constexpr,state_len:tl.constexpr,
                        seq_len:tl.constexpr,width:tl.constexpr,out_len:tl.constexpr,x_batch_stride:tl.constexpr,
                        conv_batch_stride:tl.constexpr,out_batch_stride:tl.constexpr,DIM_BLOCK:tl.constexpr,
                        HAS_BIAS:tl.constexpr,SILU_ACTIVATION:tl.constexpr,IS_CONTINUOUS_BATCHING:tl.constexpr,USE_PAD_SLOT:tl.constexpr):
    pid=tl.program_id(0)
    if IS_CONTINUOUS_BATCHING:
        slot=tl.load(conv_state_indices_ptr+pid)
    else:
        slot=pid
    if USE_PAD_SLOT:
        if slot==pad_slot_id:
            return
    for doffs in range(0,dim,DIM_BLOCK):
        c=doffs+tl.arange(0,DIM_BLOCK)
        old0=tl.load(conv_state_ptr+slot*conv_batch_stride+c*3)
        old1=tl.load(conv_state_ptr+slot*conv_batch_stride+c*3+1)
        old2=tl.load(conv_state_ptr+slot*conv_batch_stride+c*3+2)
        x=tl.load(x_ptr+pid*x_batch_stride+c)
        tap=tl.arange(0,4)[:,None]
        window=tl.where(tap==0,old0[None,:],tl.where(tap==1,old1[None,:],tl.where(tap==2,old2[None,:],x[None,:])))
        weight=tl.load(weight_ptr+c[None,:]*4+tap)
        tl.store(conv_state_ptr+slot*conv_batch_stride+c*3,old1)
        tl.store(conv_state_ptr+slot*conv_batch_stride+c*3+1,old2)
        tl.store(conv_state_ptr+slot*conv_batch_stride+c*3+2,x)
        if HAS_BIAS:
            bias=tl.load(bias_ptr+c)
        else:
            bias=0
        result=tl.sum(window.to(tl.float32)*weight,0)+bias
        if SILU_ACTIVATION:
            result=result/(1+tl.exp(-result))
        tl.store(out_ptr+pid*out_batch_stride+c*out_len,result)
