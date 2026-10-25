# Copyright © 2026 Huawei Technologies Co., Ltd.
# Source-derived educational materialization ablation; see dsa_indexer_LICENSE.
import triton
import triton.language as tl


@triton.jit
def dsa_indexer_qk(Q,K,S,B,H:tl.constexpr,M,N,D:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0);batch=row//M
    h=tl.arange(0,H);d=tl.arange(0,D);n=tl.arange(0,BN)
    q=tl.load(Q+row*H*D+h[:,None]*D+d[None,:])
    for i in range(tl.cdiv(N,BN)):
        mask=i*BN+n<N
        k=tl.load(K+batch*N*D+(i*BN+n[:,None])*D+d[None,:],mask=mask[:,None],other=0.0)
        score=tl.dot(q,tl.trans(k))
        tl.store(S+row*H*N+h[:,None]*N+i*BN+n[None,:],score,mask=mask[None,:])


@triton.jit
def dsa_indexer_activate(S,W,A,H:tl.constexpr,N,BN:tl.constexpr):
    row=tl.program_id(0);h=tl.arange(0,H);n=tl.arange(0,BN)
    weights=tl.load(W+row*H+h)
    for i in range(tl.cdiv(N,BN)):
        ptr=row*H*N+h[:,None]*N+i*BN+n[None,:];mask=i*BN+n<N
        score=tl.load(S+ptr,mask=mask[None,:],other=0.0)
        tl.store(A+ptr,tl.maximum(score,0.0)*weights[:,None],mask=mask[None,:])


@triton.jit
def dsa_indexer_headsum(A,O,H:tl.constexpr,N,BN:tl.constexpr):
    row=tl.program_id(0);h=tl.arange(0,H);n=tl.arange(0,BN)
    for i in range(tl.cdiv(N,BN)):
        mask=i*BN+n<N
        value=tl.load(A+row*H*N+h[:,None]*N+i*BN+n[None,:],mask=mask[None,:],other=0.0)
        tl.store(O+row*N+i*BN+n,tl.sum(value,axis=0),mask=mask)
