"""Exact quantized projections and full source-epsilon Compressor CPU oracle."""
from pathlib import Path
import torch

EPS, ATOL, RTOL = 1e-20, .03125, .015625


def dimensions(spec):
    p=spec.get('parameters',spec)
    m,c,r,start,seed=(p.get('M'),p.get('cache_size'),p.get('ratio'),p.get('start_pos'),spec.get('seed'))
    if any(type(x) is not int for x in (m,c,r,start,seed)) or seed<0:
        raise ValueError('integer dimensions and nonnegative seed required')
    if r not in (2,4) or not r<m<=256 or not 1<=c<=64 or not 0<=start<r:
        raise ValueError('bounded full BSND: ratio2/4 < M <=256, cache1..64, start0..ratio-1')
    if type(p.get('batch',1)) is not int or p.get('batch',1)!=1 or type(p.get('used_len',m)) is not int or p.get('used_len',m)!=m:
        raise ValueError('uniform B1 BSND and used_len=M required')
    table=p.get('block_table',[0])
    if not isinstance(table,list) or len(table)!=1 or type(table[0]) is not int or table!=[0]:raise ValueError('unique mapped private cache row0 required')
    return m,c,r,start,seed


def cache_writers(m,c,start):
    """Final contiguous tail maps injectively to cyclic slots; no padded lane."""
    return [(token,(start+token)%c) for token in range(max(0,m-c),m)]


def projection(x,w):
    # q/128 inputs: every integer partial sum is <=5120*256<2^24.
    qx=(x.float()*128).to(torch.int64);qw=(w.float()*128).to(torch.int64)
    return ((qx@qw.T).float()/16384).contiguous()


def reference(spec,x,wkv,wgate,norm_w,cache):
    m,c,r,start,_=dimensions(spec)
    kv=projection(x.view(m,5120),wkv);score=projection(x.view(m,5120),wgate)
    output=torch.zeros((1,(m+r-1)//r,512),dtype=torch.bfloat16)
    for group in range((start+m)//r):
        values=[];scores=[]
        for pos in range(group*r,(group+1)*r):
            row=pos-start
            values.append(kv[row] if row>=0 else cache[0,pos%c,:512])
            scores.append(score[row] if row>=0 else cache[0,pos%c,512:])
        values=torch.stack(values).double();scores=torch.stack(scores).double()
        pooled=(torch.softmax(scores,dim=0)*values).sum(0).bfloat16().double()
        output[0,group]=(pooled*torch.rsqrt(pooled.square().mean()+EPS)*norm_w.double()).bfloat16()
    updated=cache.clone()
    for token,slot in cache_writers(m,c,start):
        updated[0,slot,:512]=kv[token];updated[0,slot,512:]=score[token]
    return dict(output=output,cache=updated,kv=kv,score=score)


def make_case(spec):
    m,c,_,_,seed=dimensions(spec);rng=torch.Generator().manual_seed(seed)
    quant=lambda size:(torch.randint(-16,17,size,generator=rng).float()/128).bfloat16()
    x=quant((1,m,5120));wkv=quant((512,5120));wgate=quant((512,5120))
    norm_w=(.5+torch.randint(0,65,(512,),generator=rng).float()/128).bfloat16()
    cache=torch.randint(-128,129,(1,c,1024),generator=rng).float()/128
    return dict(x=x,wkv=wkv,wgate=wgate,norm_w=norm_w,cache=cache,
                expected=reference(spec,x,wkv,wgate,norm_w,cache))


def check_fixture(spec,data):
    m,c,r,_,_=dimensions(spec)
    if set(data)!={'x','wkv','wgate','norm_w','cache','expected'}:raise ValueError('exact fixture arity required')
    inputs=dict(x=((1,m,5120),torch.bfloat16),wkv=((512,5120),torch.bfloat16),wgate=((512,5120),torch.bfloat16),
                norm_w=((512,),torch.bfloat16),cache=((1,c,1024),torch.float32))
    expected=dict(output=((1,(m+r-1)//r,512),torch.bfloat16),cache=((1,c,1024),torch.float32),
                  kv=((m,512),torch.float32),score=((m,512),torch.float32))
    if not isinstance(data['expected'],dict) or set(data['expected'])!=set(expected):raise ValueError('full output/cache/projection expectations required')
    ptrs=[]
    for source,items in [(data,inputs),(data['expected'],expected)]:
        for name,(shape,dtype) in items.items():
            x=source[name]
            if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape or not x.is_contiguous() or not torch.isfinite(x).all():
                raise ValueError('finite contiguous CPU shape/dtype mismatch: '+name)
            ptrs.append(x.untyped_storage().data_ptr())
    if len(set(ptrs))!=len(ptrs):raise ValueError('all fixture buffers must be nonaliasing')
    for name in ('x','wkv','wgate'):
        x=data[name].float()*128
        if not (x.abs()<=16).all() or not torch.equal(x,x.round()):raise ValueError('bounded exact q/128 input required')
    w=data['norm_w'].float()*128
    if not ((w>=64)&(w<=128)).all() or not torch.equal(w,w.round()):raise ValueError('bounded dyadic norm weight required')
    cache=data['cache']*128
    if not (cache.abs()<=128).all() or not torch.equal(cache,cache.round()):raise ValueError('bounded dyadic cache required')
    return m,c,r


CASE_SOURCES=[Path(__file__)]
