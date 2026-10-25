"""Independent CPU integer AWQ decoder; source-format zeros have NO +1."""
from pathlib import Path
import torch

PERM=(0,4,1,5,2,6,3,7)
CASE_SOURCES=[Path(__file__)]


def dimensions(spec):
    p=spec.get('parameters',spec);k,n,g=(p.get(x) for x in ('K','N','group'))
    if (any(type(x) is not int for x in (k,n,g,spec.get('seed')))
            or min(k,n,g)<=0 or spec['seed']<0 or n%8 or k%g
            or g not in (32,64,128,k) or (g!=k and g%16)
            or type(p.get('all_nibbles_signed_words',False)) is not bool):
        raise ValueError('positive K,Nmultiple8 and aligned full groups32/64/128 or K required')
    # Include masked lanes and bytes, not just valid addresses.
    if max(((k+15)//16*16)*n,((n+127)//128*128)*max(k,16))*4>=2**31:
        raise ValueError('padded address bytes exceed signed32 domain')
    return k,n,g


def decode(words):
    """uint32 logical shifts via int64, independent of signed NPU shifts."""
    raw=words.to(torch.int64)&0xffffffff
    return torch.stack([(raw>>(4*i))&15 for i in PERM],dim=-1).reshape(words.shape[0],-1)


def validate_inputs(inputs,spec):
    k,n,g=dimensions(spec)
    if not isinstance(inputs,dict) or set(inputs)!={'Q','Z','S'}:raise ValueError('Q,Z,S inputs required')
    shapes={'Q':(k,n//8),'Z':(k//g,n//8),'S':(k//g,n)}
    for name,v in inputs.items():
        dtype=torch.float16 if name=='S' else torch.int32
        if (not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=dtype
                or tuple(v.shape)!=shapes[name] or not v.is_contiguous() or v.storage_offset()!=0
                or v.untyped_storage().nbytes()!=v.numel()*v.element_size()):
            raise ValueError('owned contiguous CPU Q/Z int32 and S FP16 required')
    s=inputs['S'].float()*256
    if not bool(torch.isfinite(s).all() and ((s>=4)&(s<=64)&(s==s.trunc())).all()):
        raise ValueError('scales must be exact q/256 for integer q in [4,64]')
    if len({v.untyped_storage().data_ptr() for v in inputs.values()})!=3:raise ValueError('input alias')


def make_case(spec):
    k,n,g=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed'])
    q=torch.randint(-2**31,2**31,(k,n//8),dtype=torch.int32,generator=rng)
    z=torch.randint(-2**31,2**31,(k//g,n//8),dtype=torch.int32,generator=rng)
    if spec.get('parameters',spec).get('all_nibbles_signed_words'):
        words=torch.tensor([int(''.join(format((j+i)%16,'x') for j in range(8)),16) for i in range(16)],dtype=torch.int64)
        words=words.to(torch.int32)
        for v in (q,z):v.copy_(words.repeat((v.numel()+15)//16)[:v.numel()].reshape_as(v))
    scales=torch.randint(4,65,(k//g,n),generator=rng).to(torch.float16)/256
    inputs=dict(Q=q,Z=z,S=scales);validate_inputs(inputs,spec)
    # Integer product magnitude <=15*64=960, exactly representable after /256 in FP16.
    difference=decode(q)-decode(z).repeat_interleave(g,dim=0)
    product=difference*(scales.float()*256).to(torch.int64).repeat_interleave(g,dim=0)
    expected=(product.double()/256).to(torch.float16)
    return dict(inputs=inputs,expected=expected)


def check_output(actual,expected):
    if (not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float16
            or actual.shape!=expected.shape or not actual.is_contiguous()):
        return dict(passed=False,reason='contiguous FP16 output shape required')
    got=actual.detach().cpu()
    same=torch.equal(got.view(torch.int16),expected.view(torch.int16))
    return dict(passed=same,exact_fp16_bits=same,max_abs_error=float((got.float()-expected.float()).abs().max()))
