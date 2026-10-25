"""Independent CPU FP64 reference and bounded, stride-preserving fixtures."""
import torch


def dimensions(spec):
    p=spec.get('parameters',spec)
    for name,limit in [('rows',256),('cols',32768)]:
        if type(p.get(name)) is not int or not 1<=p[name]<=limit:raise ValueError('invalid '+name)
    if type(p.get('row_padding',0)) is not int or not 0<=p.get('row_padding',0)<=7:raise ValueError('invalid padding')
    for flag in ('uniform_row','extreme_finite'):
        if type(p.get(flag,False)) is not bool:raise ValueError('invalid '+flag)
    return p['rows'],p['cols'],p.get('row_padding',0)


def backing(x):return x.as_strided((x.untyped_storage().nbytes()//x.element_size(),),(1,),0)


def check_inputs(spec,inputs):
    r,c,pad=dimensions(spec)
    if not isinstance(inputs,dict) or set(inputs)!={'x'}:raise ValueError('one x required')
    x=inputs['x']
    if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.layout!=torch.strided
            or tuple(x.shape)!=(r,c) or x.stride()!=(c+pad,1) or x.storage_offset()!=0
            or x.requires_grad or backing(x).numel()!=r*(c+pad)):raise ValueError('invalid CPU input layout/dtype')
    if not torch.isfinite(x).all() or not (x.abs()<=80).all():raise ValueError('finite abs<=80 required')
    return x


def make_case(spec):
    r,c,pad=dimensions(spec);p=spec.get('parameters',spec)
    rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0))
    store=torch.full((r,c+pad),1234.,dtype=torch.float32,device='cpu');x=store[:,:c]
    if p.get('uniform_row',False):x.copy_((torch.arange(r,device='cpu')%3-1).float()[:,None]*8)
    elif p.get('extreme_finite',False):
        x.copy_(torch.where(torch.arange(r*c,device='cpu').reshape(r,c)%2==0,80.,-80.))
    else:
        chunks=(c+2047)//2048
        centers=torch.linspace(-24,24,chunks,dtype=torch.float32,device='cpu')
        center=centers[(torch.arange(c,device='cpu')//2048)]
        jitter=torch.randint(-32,33,(r,c),generator=rng,device='cpu').float()/64
        x.copy_(jitter+torch.where(torch.arange(r,device='cpu')[:,None]%2==0,center,-center))
        if r>=2:x[1].copy_(x[0])
    check_inputs(spec,{'x':x});v=x.double();m=v.max(dim=1,keepdim=True).values
    expected=(v-m)-torch.log(torch.exp(v-m).sum(dim=1,keepdim=True))
    return dict(inputs={'x':x},expected=expected)


def check_fixture(spec,fixture):
    if not isinstance(fixture,dict) or set(fixture)!={'inputs','expected'}:raise ValueError('fixture fields')
    x=check_inputs(spec,fixture['inputs']);e=fixture['expected']
    if (not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=torch.float64 or e.shape!=x.shape
            or not e.is_contiguous() or not torch.isfinite(e).all()
            or e.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()):raise ValueError('invalid independent expected')


def diagnostics(a,e):
    error=(a.double()-e).abs();finite=torch.isfinite(a);bad=~finite|(error>4e-5+2e-6*e.abs())
    coords=bad.nonzero()[:8].tolist()
    norm=(torch.exp(a.double()).sum(dim=1)-1).abs()
    return dict(passed=not bool(bad.any()) and bool((norm<=2e-5).all()),atol=4e-5,rtol=2e-6,
        mismatch_count=int(bad.sum()),first_mismatches=[dict(coordinates=q,expected=float(e[tuple(q)]),
        actual=float(a[tuple(q)])) for q in coords],nonfinite_count=int((~finite).sum()),
        max_absolute_error=float(error.max()),max_normalization_error=float(norm.max()))
