"""Independent INT64 dot oracle with ordered FP32 scales and postcast bias."""
from pathlib import Path
import torch
from scaled_index_bounds import index_audit


def dimensions(spec):
    p=spec.get('parameters',spec);values=tuple(p.get(k) for k in ('M','N','K'));seed=spec.get('seed')
    if any(type(x) is not int for x in (*values,seed)) or seed<0:raise ValueError('integer dimensions and nonnegative seed required')
    m,n,k=values
    if not 1<=m<=128 or not 1<=n<=256 or not 0<k<=512 or k%128:raise ValueError('M1..128,N1..256,K multiple128 <=512 required')
    for key in ('transposed_b','metadata_wide_stride','bias'):
        if type(p.get(key,True if key=='bias' else False)) is not bool:raise ValueError('boolean '+key+' required')
    return m,n,k,seed,p


def strides(spec):
    m,n,k,_,p=dimensions(spec)
    return (k,1),((1,k) if p.get('transposed_b') else (n,1)),(n,1)


def dispatch_audit(spec):
    m,n,k,_,p=dimensions(spec);a,b,c=strides(spec)
    actual=index_audit(m,n,k,a,b,c)
    synthetic=index_audit(m,n,k,(1<<31,1),b,c) if p.get('metadata_wide_stride') else None
    force64=bool(synthetic is not None)
    return dict(actual=actual,synthetic=synthetic,force64_audit=force64,
                candidate_bits=32 if actual['fits_int32'] and not force64 else 64,
                wide_guard_scope='Synthetic metadata only; actual device allocations remain small, both arms explicitly64.' if force64 else None)


def reference(a,b,sa,sb,bias):
    value=sa*(a.to(torch.int64)@b.to(torch.int64)).float()
    value=sb*value
    return value if bias is None else value+bias


def make_case(spec):
    m,n,k,seed,p=dimensions(spec);rng=torch.Generator().manual_seed(seed)
    a=torch.randint(-8,9,(m,k),dtype=torch.int8,generator=rng)
    b=(torch.randint(-8,9,(n,k),dtype=torch.int8,generator=rng).T if p.get('transposed_b')
       else torch.randint(-8,9,(k,n),dtype=torch.int8,generator=rng))
    sa=torch.tensor([.25]);sb=torch.tensor([.5])
    bias=torch.randint(-8,9,(n,),generator=rng).float()/2 if p.get('bias',True) else None
    return dict(a=a,b=b,sa=sa,sb=sb,bias=bias,expected=reference(a,b,sa,sb,bias))


def check_fixture(spec,data):
    m,n,k,_,p=dimensions(spec)
    if set(data)!={'a','b','sa','sb','bias','expected'}:raise ValueError('exact fixture arity required')
    shapes=dict(a=((m,k),torch.int8),b=((k,n),torch.int8),sa=((1,),torch.float32),sb=((1,),torch.float32),expected=((m,n),torch.float32))
    if p.get('bias',True):shapes['bias']=((n,),torch.float32)
    elif data['bias'] is not None:raise ValueError('unexpected bias')
    pointers=[]
    for name,(shape,dtype) in shapes.items():
        x=data[name]
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape:
            raise ValueError('CPU shape/dtype mismatch: '+name)
        required=strides(spec)[1] if name=='b' else ((k,1) if name=='a' else None)
        if required is not None:
            if tuple(x.stride())!=required or x.storage_offset()!=0:raise ValueError('physical layout mismatch: '+name)
        elif not x.is_contiguous():raise ValueError('contiguous scalar/bias/output required')
        pointers.append(x.untyped_storage().data_ptr())
        if name in ('a','b'):
            if not ((x>=-8)&(x<=8)).all():raise ValueError('INT8 values[-8,8] required')
        elif not torch.isfinite(x).all():raise ValueError('finite FP32 required')
    if len(set(pointers))!=len(pointers):raise ValueError('independent nonaliasing buffers required')
    if not torch.equal(data['sa'],torch.tensor([.25])) or not torch.equal(data['sb'],torch.tensor([.5])):raise ValueError('fixed scalar scales .25/.5 required')
    if data['bias'] is not None and not torch.equal(data['bias']*2,(data['bias']*2).round()):raise ValueError('dyadic half bias required')
    return dispatch_audit(spec)


CASE_SOURCES=[Path(__file__),Path(__file__).with_name('scaled_index_bounds.py'),Path(__file__).resolve().parents[3]/'experiments/optimization-wave008/T10-fixture-defaults.json']
