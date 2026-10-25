# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM/Huawei mean_batch_invariant.py; see mean_inner_source.py.
import triton
import triton.language as tl


@triton.jit
def _mean_inner(input_ptr,output_ptr,s0,s1,s2,o0,o1,M,N,K,BLOCK_SIZE:tl.constexpr,K_TILE:tl.constexpr):
    m=tl.program_id(0);k=tl.program_id(1)*K_TILE+tl.arange(0,K_TILE)
    acc=tl.full((K_TILE,),0.,tl.float32)
    for n_start in range(0,N,BLOCK_SIZE):
        n=n_start+tl.arange(0,BLOCK_SIZE)
        vals=tl.load(input_ptr+m*s0+n[:,None]*s1+k[None,:]*s2,(n[:,None]<N)&(k[None,:]<K),other=0.)
        acc+=tl.sum(vals,axis=0)
    mean_val=acc/N
    tl.store(output_ptr+m*o0+k*o1,mean_val,k<K)
