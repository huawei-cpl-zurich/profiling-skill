"""Independent exact dyadic FP64 dot reference; native source-host tiling."""
import struct
import torch


def dimensions(spec):
    p=spec.get('parameters',spec);n=p.get('n')
    if type(n) is not int or not 1<=n<=1048576:raise ValueError('integer n in [1,1048576] required')
    for k in ('incx','incy'):
        if p.get(k,1)!=1 or type(p.get(k,1)) is not int:raise ValueError('unit strides required')
    if p.get('isConj',0)!=0:raise ValueError('real unconjugated dot required')
    return n


def tiling(n):
    dimensions({'n':n});offset=[0]*40;count=[0]*40;q,r=divmod(n,8);start=0
    for i in range(8):
        count[i]=q+(i<r);offset[i]=start if count[i] else 0;start+=count[i]
    assert start==n
    intervals=[(o,o+c) for o,c in zip(offset[:8],count[:8]) if c]
    assert intervals[0][0]==0 and intervals[-1][1]==n
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    return struct.pack('<83I',n,min(n,8),0,*offset,*count)


def full(t):return t.as_strided((t.untyped_storage().nbytes()//t.element_size(),),(1,),0)


def check_fixture(spec,fixture):
    n=dimensions(spec)
    if not isinstance(fixture,dict) or set(fixture)!={'x','y','expected'}:raise ValueError('x/y/expected required')
    for key in ('x','y'):
        t=fixture[key]
        if (not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.device.type!='cpu' or t.shape!=(n,)
                or not t.is_contiguous() or t.storage_offset()!=32 or full(t).numel()!=n+64 or t.requires_grad):raise ValueError('owned aligned-halo FP32 input required')
        if not torch.isfinite(t).all():raise ValueError('finite inputs required')
        if n==1 and spec.get('parameters',spec).get('signed_zero',False):
            if not bool((t==0).all()) if key=='x' else not bool((t==.5).all()):raise ValueError('signed-zero guard domain')
        elif not bool((t.abs()<=4/256).all() and (t*256==torch.round(t*256)).all()):raise ValueError('q/256, |q|<=4 required')
    e=fixture['expected']
    if not isinstance(e,torch.Tensor) or e.shape!=(1,) or e.dtype!=torch.float32 or e.device.type!='cpu' or not torch.isfinite(e).all():raise ValueError('scalar FP32 expected required')
    if len({fixture[k].untyped_storage().data_ptr() for k in ('x','y','expected')})!=3:raise ValueError('fixture storage alias')


def make_case(spec):
    n=dimensions(spec);tiling(n);rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0));p=spec.get('parameters',spec);data={}
    for key in ('x','y'):
        store=torch.full((n+64,),12345.,dtype=torch.float32,device='cpu');data[key]=store[32:-32]
        data[key].copy_(torch.randint(-4,5,(n,),generator=rng,device='cpu').float()/256)
    if p.get('signed_zero',False):data['x'].fill_(-0.);data['y'].fill_(.5)
    expected=(data['x'].double()*data['y'].double()).sum().float().view(1)
    data['expected']=expected;check_fixture(spec,data);return data
