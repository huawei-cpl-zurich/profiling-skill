"""Deliberately naive row-wise RainFusionAttention Triton starting point."""
import torch
import triton
import triton.language as tl


@triton.jit
def _row(Q,K,V,I,N,O, QOFF:tl.constexpr,KOFF:tl.constexpr,BOFF:tl.constexpr,
         QLEN:tl.constexpr,KLEN:tl.constexpr,H:tl.constexpr,KH:tl.constexpr,
         D:tl.constexpr,SX:tl.constexpr,SY:tl.constexpr,SLOTS:tl.constexpr,
         SCALE:tl.constexpr,BD:tl.constexpr,BT:tl.constexpr):
    r=tl.program_id(0)
    h=tl.program_id(1)
    d=tl.arange(0,BD)
    t=tl.arange(0,BT)
    qb=BOFF+r//SX
    count=tl.load(N+qb*H+h)
    q=tl.load(Q+((QOFF+r)*H+h)*D+d,d<D,0).to(tl.float32)
    m=tl.full((),float('-inf'),tl.float32)
    z=tl.full((),0,tl.float32)
    acc=tl.full((BD,),0,tl.float32)
    for slot in range(count):
        block=tl.load(I+(qb*H+h)*SLOTS+slot)
        valid=(block>=0)&(block<tl.cdiv(KLEN,SY))
        for tile in range(tl.cdiv(SY,BT)):
            j=block*SY+tile*BT+t
            mask=valid&(tile*BT+t<SY)&(j<KLEN)
            kh=h//(H//KH)
            k=tl.load(K+((KOFF+j[:,None])*KH+kh)*D+d[None,:],mask[:,None]&(d[None,:]<D),0).to(tl.float32)
            v=tl.load(V+((KOFF+j[:,None])*KH+kh)*D+d[None,:],mask[:,None]&(d[None,:]<D),0).to(tl.float32)
            score=tl.sum(k*q[None,:],1)*SCALE
            score=tl.where(mask,score,float('-inf'))
            nm=tl.maximum(m,tl.max(score,0))
            safe=tl.where(nm==float('-inf'),0,nm)
            alpha=tl.exp(m-safe)
            p=tl.exp(score-safe)
            acc=acc*alpha+tl.sum(v*p[:,None],0)
            z=z*alpha+tl.sum(p,0)
            m=nm
    out=acc/tl.where(z>0,z,1)
    tl.store(O+((QOFF+r)*H+h)*D+d,out,d<D)


def forward(query,key,value,select_idx,select_num_idx,block_shape,
            q_seqlen_list,kv_seqlen_list,scale_value,q_input_layout='TND',
            kv_input_layout='TND',inner_precise=0):
    if q_input_layout!='TND' or kv_input_layout!='TND':
        raise NotImplementedError('TND only')
    if not all(x.is_contiguous() for x in (query,key,value,select_idx,select_num_idx)):
        raise ValueError('starter requires contiguous tensors')
    scale=float(scale_value)
    if inner_precise==1:
        scale=float(torch.tensor(scale,dtype=torch.float16).item())
    out=torch.empty_like(query,dtype=torch.float32)
    qo=ko=bo=0
    sx,sy=map(int,block_shape)
    h,d=query.shape[1:]
    for nq,nk in zip(q_seqlen_list,kv_seqlen_list):
        nq,nk=int(nq),int(nk)
        if nq:
            _row[(nq,h)](query,key,value,select_idx,select_num_idx,out,
                qo,ko,bo,nq,nk,h,key.shape[1],d,sx,sy,select_idx.shape[2],scale,
                triton.next_power_of_2(d),16)
        qo+=nq;ko+=nk;bo+=(nq+sx-1)//sx
    return out
