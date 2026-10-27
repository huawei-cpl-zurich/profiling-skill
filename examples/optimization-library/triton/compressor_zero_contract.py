"""Independent full Compressor reference; exact q/128 projection certificate."""
from pathlib import Path
import torch
EPS,ATOL,RTOL=1e-20,.03125,.015625


def dimensions(spec):
    p=spec.get('parameters',spec);b,s,r,seed=(p.get('B'),p.get('S'),p.get('ratio'),spec.get('seed'))
    if any(type(x) is not int for x in (b,s,r,seed)) or seed<0 or not 1<=b<=4 or r not in (2,4,8) or not r<s<=2048:
        raise ValueError('bounded B1..4, ratio2/4/8 < S <=2048 and nonnegative integer seed required')
    packed=p.get('packed',False);start=p.get('start_pos',0);used=p.get('used_len',s);table=p.get('block_table',list(range(b)))
    if type(packed) is not bool or (packed and b!=1) or type(start) is not int or not 0<=start<r or type(used) is not int or not r<used<=s:
        raise ValueError('packed B1, start0..ratio-1 and ratio < used_len <= S required')
    if not isinstance(table,list) or len(table)!=b or any(type(x) is not int for x in table) or sorted(table)!=list(range(b)):
        raise ValueError('distinct private cache block mapping required')
    return b,s,r,seed,packed,start,used,table


def complete_write(spec):
    _,s,r,_,packed,start,used,_=dimensions(spec)
    return not packed and used==s and start%r==0 and s%r==0


def output_shape(spec):
    b,s,r,_,packed,_,_,_=dimensions(spec)
    return (min(b*s,b*s//r+b),512) if packed else (b,(s+r-1)//r,512)


def projection(x,w):
    # All partial integer dot sums <=5120*16*16<2^24, and scale is 2^-14.
    return (((x.reshape(-1,5120).float()*128).to(torch.int64) @
             (w.float()*128).to(torch.int64).T).float()/16384).contiguous()


def reference(spec,x,wkv,wgate,norm_w,cache):
    b,s,r,_,packed,start,used,table=dimensions(spec)
    kv=projection(x,wkv);score=projection(x,wgate);updated=cache.clone()
    output=torch.zeros(output_shape(spec),dtype=torch.bfloat16);flat=output.view(-1,512)
    groups=(start+used)//r;rows=(s+r-1)//r
    for batch in range(b):
        for group in range(groups):
            values=[];scores=[]
            for pos in range(group*r,(group+1)*r):
                token=pos-start
                values.append(kv[batch*s+token] if token>=0 else cache[table[batch],pos%16,:512])
                scores.append(score[batch*s+token] if token>=0 else cache[table[batch],pos%16,512:])
            pooled=(torch.softmax(torch.stack(scores).double(),dim=0)*torch.stack(values).double()).sum(0).bfloat16().double()
            flat[(batch*groups if packed else batch*rows)+group]=(pooled*torch.rsqrt(pooled.square().mean()+EPS)*norm_w.double()).bfloat16()
        for token in range(max(0,used-16),used):
            updated[table[batch],(start+token)%16,:512]=kv[batch*s+token]
            updated[table[batch],(start+token)%16,512:]=score[batch*s+token]
    return dict(output=output,cache=updated,kv=kv,score=score)


def make_case(spec):
    b,s,_,seed,packed,_,_,_=dimensions(spec);rng=torch.Generator().manual_seed(seed)
    quant=lambda shape:(torch.randint(-16,17,shape,generator=rng).float()/128).bfloat16()
    x=quant((b*s,5120) if packed else (b,s,5120));wkv=quant((512,5120));wgate=quant((512,5120))
    norm_w=(.5+torch.randint(0,65,(512,),generator=rng).float()/128).bfloat16()
    cache=torch.randint(-128,129,(b,16,1024),generator=rng).float()/128
    return dict(x=x,wkv=wkv,wgate=wgate,norm_w=norm_w,cache=cache,expected=reference(spec,x,wkv,wgate,norm_w,cache))


def check_fixture(spec,data):
    b,s,_,_,packed,_,_,_=dimensions(spec)
    if not isinstance(data,dict) or set(data)!={'x','wkv','wgate','norm_w','cache','expected'}:raise ValueError('exact fixture arity required')
    inputs=dict(x=((b*s,5120) if packed else (b,s,5120),torch.bfloat16),wkv=((512,5120),torch.bfloat16),wgate=((512,5120),torch.bfloat16),norm_w=((512,),torch.bfloat16),cache=((b,16,1024),torch.float32))
    expected=dict(output=(output_shape(spec),torch.bfloat16),cache=((b,16,1024),torch.float32),kv=((b*s,512),torch.float32),score=((b*s,512),torch.float32))
    if not isinstance(data['expected'],dict) or set(data['expected'])!=set(expected):raise ValueError('all expectations required')
    ptrs=[]
    for source,items in ((data,inputs),(data['expected'],expected)):
        for name,(shape,dtype) in items.items():
            x=source[name]
            if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape or not x.is_contiguous() or not torch.isfinite(x).all():raise ValueError('finite contiguous CPU shape/dtype required: '+name)
            ptrs.append(x.untyped_storage().data_ptr())
    if len(set(ptrs))!=len(ptrs):raise ValueError('fixture storage must not alias')
    for name,bound in [('x',16),('wkv',16),('wgate',16),('cache',128)]:
        q=data[name].float()*128
        if not (q.abs()<=bound).all() or not torch.equal(q,q.round()):raise ValueError('bounded dyadic input required')
    q=data['norm_w'].float()*128
    if not ((q>=64)&(q<=128)).all() or not torch.equal(q,q.round()):raise ValueError('bounded norm weight required')
    return dimensions(spec)


CASE_SOURCES=[Path(__file__)]
