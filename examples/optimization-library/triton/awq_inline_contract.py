"""Independent integer AWQ decoder and both prospectively declared typed models."""
from pathlib import Path
import torch
CASE_SOURCES=[Path(__file__)]
MODELS=('fp32','fp16')

def dimensions(spec):
    p=spec.get('parameters',spec);m,k,n,g=(p.get(x) for x in ('M','K','N','group'))
    if (any(type(x) is not int or x<=0 for x in (m,k,n,g)) or k>512 or n%8 or k%g
            or g not in (32,64,128,k) or (g!=k and g%32)
            or type(spec.get('seed')) is not int or spec['seed']<0):
        raise ValueError('positive M/K<=512/Nmultiple8, full groups32/64/128 or K required')
    if max(((m+15)//16*16)*max(k,n),((k+31)//32*32)*((n+127)//128*128))*4>=2**31:
        raise ValueError('padded pointer bytes exceed int32')
    return m,k,n,g

def decode(v):
    u=v.long()&0xffffffff
    return torch.stack([(u>>(4*i))&15 for i in (0,4,1,5,2,6,3,7)],-1).reshape(v.shape[0],-1)

def validate_inputs(v,spec):
    m,k,n,g=dimensions(spec);shapes={'A':(m,k),'Q':(k,n//8),'Z':(k//g,n//8),'S':(k//g,n)}
    if not isinstance(v,dict) or set(v)!=set(shapes):raise ValueError('A/Q/Z/S required')
    for name,x in v.items():
        dtype=torch.int32 if name in ('Q','Z') else torch.float16
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shapes[name]
                or not x.is_contiguous() or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=x.numel()*x.element_size()):
            raise ValueError('owned contiguous CPU FP16 A/S and int32 Q/Z required')
    if len({x.untyped_storage().data_ptr() for x in v.values()})!=4:raise ValueError('input alias')
    for name,unit,lo,hi in (('A',64,-8,8),('S',256,4,16)):
        q=v[name].float()*unit
        if not bool(torch.isfinite(q).all() and ((q>=lo)&(q<=hi)&(q==q.trunc())).all()):raise ValueError('fixed exact dyadic input domain required')

def make_case(spec):
    m,k,n,g=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed'])
    v=dict(A=torch.randint(-8,9,(m,k),generator=rng).half()/64,
        Q=torch.randint(-2**31,2**31,(k,n//8),dtype=torch.int32,generator=rng),
        Z=torch.randint(-2**31,2**31,(k//g,n//8),dtype=torch.int32,generator=rng),
        S=torch.randint(4,17,(k//g,n),generator=rng).half()/256)
    # Guarantee all nibble values and signed words independent of random coverage.
    words=torch.tensor([int(''.join(format((j+i)%16,'x') for j in range(8)),16) for i in range(16)],dtype=torch.int64).int()
    for name in ('Q','Z'):
        x=v[name];x.view(-1)[:min(16,x.numel())]=words[:min(16,x.numel())]
    validate_inputs(v,spec)
    b=(decode(v['Q'])-decode(v['Z']).repeat_interleave(g,0)).double()*v['S'].double().repeat_interleave(g,0)
    a=v['A'].double();full=a@b;carry=torch.zeros((m,n),dtype=torch.float16)
    for start in range(0,k,32):carry=(carry.double()+a[:,start:start+32]@b[start:start+32]).half()
    steps=(k+31)//32;u=2**-11;absolute=a.abs()@b.abs()
    return dict(inputs=v,expected=dict(fp32=full.half(),fp16=carry,fp64=full,
        fp32_rounding_bound=absolute*u,fp16_rounding_bound=absolute*(steps*u/(1-steps*u))))

def check_output(actual,expected,model):
    if model not in MODELS:raise ValueError('compiler-proven model required')
    ref=expected[model]
    if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float16 or actual.shape!=ref.shape or not actual.is_contiguous():
        return dict(passed=False,reason='one contiguous FP16 C required'),None
    got=actual.detach().cpu()
    same=(got.view(torch.int16)==ref.view(torch.int16))|((got==0)&(ref==0));bad=~same;count=int(bad.sum());first=None
    if count:
        index=tuple(int(i) for i in bad.nonzero()[0]);first=dict(index=list(index),actual=float(got[index]),expected=float(ref[index]))
    finite=bool(torch.isfinite(got).all())
    return dict(passed=finite and count==0,model=model,mismatch_count=count,first_mismatch=first,
        nonfinite_count=int((~torch.isfinite(got)).sum()),max_error=float((got.double()-ref.double()).abs().max()),
        fp64_max_error=float((got.double()-expected['fp64']).abs().max()),
        zero_sign_differences=int(((got==0)&(ref==0)&(got.view(torch.int16)!=ref.view(torch.int16))).sum())),got
