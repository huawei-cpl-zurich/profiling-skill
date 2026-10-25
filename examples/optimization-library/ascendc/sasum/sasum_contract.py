"""Independent dyadic FP64 absolute-sum oracle and exact source-host tiling."""
import struct
import torch


def dimensions(spec):
    p=spec.get('parameters',spec);n=p.get('n');inc=p.get('incx',1)
    if type(n) is not int or not 1<=n<=2097155:raise ValueError('n in [1,2097155] required')
    if type(inc) is not int or inc not in (1,2):raise ValueError('incx 1 or2 required')
    if p.get('negative_zero',False) and n!=1:raise ValueError('negative-zero singleton required')
    return n,inc


def tiling(n,inc):
    dimensions(dict(n=n,incx=inc));offset=[0]*40;count=[0]*40;blocks,rem=divmod(n,8)
    if blocks==0:count[0]=rem;cores=1
    elif blocks<=8:
        cores=blocks
        for i in range(cores):offset[i]=8*i;count[i]=8
        count[cores-1]+=rem
    else:
        cores=8;q,r=divmod(blocks,8);start=0
        for i in range(8):offset[i]=start;count[i]=8*(q+(i<r));start+=count[i]
        count[7]+=rem
    return struct.pack('<Ii81I',n,inc,cores,*offset,*count)


def full(t):return t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)


def check_fixture(spec,fixture):
    n,inc=dimensions(spec);span=(n-1)*inc+1
    if not isinstance(fixture,dict) or set(fixture)!={'x','expected'}:raise ValueError('x/expected required')
    t=fixture['x'];e=fixture['expected']
    if (not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.device.type!='cpu' or t.shape!=(n,)
            or t.stride()!=(inc,) or t.storage_offset()!=32 or full(t).numel()!=span+64 or t.requires_grad):raise ValueError('owned FP32 stride/halo required')
    if not bool(torch.isfinite(t).all() and (t.abs()<=4/256).all() and (t*256==torch.round(t*256)).all()):raise ValueError('finite q/256, |q|<=4 required')
    backing=full(t)
    if not bool((backing[:32]==12345).all() and (backing[-32:]==12345).all()):raise ValueError('halo canary')
    if inc==2 and not bool((backing[33:32+span:2]==12345).all()):raise ValueError('stride hole canary')
    if not isinstance(e,torch.Tensor) or e.shape!=(1,) or e.dtype!=torch.float32 or e.device.type!='cpu' or not bool(torch.isfinite(e).all()):raise ValueError('scalar FP32 expected required')
    if t.untyped_storage().data_ptr()==e.untyped_storage().data_ptr():raise ValueError('fixture alias')


def make_case(spec):
    n,inc=dimensions(spec);p=spec.get('parameters',spec);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',p.get('seed',0)))
    store=torch.full(((n-1)*inc+65,),12345.,dtype=torch.float32,device='cpu');x=store.as_strided((n,),(inc,),32)
    x.copy_(torch.randint(-4,5,(n,),generator=rng,device='cpu').to(torch.float32)/256)
    if p.get('negative_zero',False):x.fill_(-0.)
    data=dict(x=x,expected=x.double().abs().sum().to(torch.float32).view(1));check_fixture(spec,data);return data
