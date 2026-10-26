"""Dense final-dA contract and an independent exact dyadic CPU oracle."""
import torch
CANARY=1000.


def dimensions(spec):
    p=spec.get('parameters',spec);b=p.get('B');t=p.get('T');h=p.get('H');bt=p.get('BT')
    if any(type(v)is not int for v in (b,t,h,bt)) or not(1<=b<=2 and 1<=t<=2048 and 1<=h<=8 and bt in(64,128)):raise ValueError('bounded B/T/H/BT required')
    if p.get('dense')is not True or p.get('dtype_A','float16')!='float16' or p.get('dtype_P','float32')!='float32':raise ValueError('dense F16 A / F32 P component only')
    return b,t,h,bt


def backing(x):return x.as_strided((x.untyped_storage().nbytes()//x.element_size(),),(1,),0)
def guarded(x):
    pad=128//x.element_size();raw=torch.full((x.numel()+2*pad,),CANARY,dtype=x.dtype,device='cpu');out=raw[pad:-pad].view(x.shape);out.copy_(x);return out


def check_inputs(spec,z):
    shape=dimensions(spec)
    if not isinstance(z,dict) or set(z)!={'A','P'}:raise ValueError('complete A/P required')
    for name,dtype in [('A',torch.float16),('P',torch.float32)]:
        x=z[name]
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape or not x.is_contiguous() or x.requires_grad:raise ValueError('owned contiguous CPU typed tensors required')
        pad=128//x.element_size();raw=backing(x)
        if x.storage_offset()!=pad or raw.numel()!=x.numel()+2*pad or not bool(torch.isfinite(x).all() and(raw[:pad]==CANARY).all()and(raw[-pad:]==CANARY).all()):raise ValueError('finite complete input with 128B halos required')
    if z['A'].untyped_storage().data_ptr()==z['P'].untyped_storage().data_ptr():raise ValueError('A/P must not alias')
    b,t,h,bt=shape;rows=torch.arange(t,device='cpu').remainder(bt).view(1,t,1,1);cols=torch.arange(bt,device='cpu').view(1,1,1,bt);lower=cols<rows;diag=cols==rows
    a=z['A'].float();p=z['P'];off=a-torch.where(diag,1.,0.)
    if not bool((torch.where(lower,off,0.)==off).all()and(off.abs()<=2/256).all()and(off*256==torch.round(off*256)).all()):raise ValueError('causal unit-lower A dyadic domain')
    if not bool((torch.where(lower,p,0.)==p).all()and(p.abs()<=4/16).all()and(p*16==torch.round(p*16)).all()):raise ValueError('strict-lower P dyadic domain')


def check_fixture(spec,f):
    shape=dimensions(spec)
    if not isinstance(f,dict) or set(f)!={'inputs','expected','expected_fp64'}:raise ValueError('complete independent fixture required')
    check_inputs(spec,f['inputs']);all_tensors=list(f['inputs'].values())
    for name,dtype in [('expected',torch.float32),('expected_fp64',torch.float64)]:
        x=f[name]
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shape or not x.is_contiguous() or not bool(torch.isfinite(x).all()):raise ValueError('independent complete CPU expected required')
        all_tensors.append(x)
    if len({x.untyped_storage().data_ptr()for x in all_tensors})!=4:raise ValueError('fixture aliases')


def reference(z,shape):
    b,t,h,bt=shape;out=torch.empty(shape,dtype=torch.float64,device='cpu')
    for batch in range(b):
        for start in range(0,t,bt):
            n=min(bt,t-start)
            for head in range(h):
                a=torch.zeros((bt,bt),dtype=torch.float64,device='cpu');p=a.clone()
                a[:n]=z['A'][batch,start:start+n,head].double();p[:n]=z['P'][batch,start:start+n,head].half().double()
                tmp=(p@a.T).float().half().double();value=-(a.T@tmp)
                out[batch,start:start+n,head]=torch.tril(value,diagonal=-1)[:n]
    return out


def make_case(spec):
    shape=dimensions(spec);b,t,h,bt=shape;rng=torch.Generator(device='cpu').manual_seed(spec.get('seed',0));rows=torch.arange(t,device='cpu').remainder(bt).view(1,t,1,1);cols=torch.arange(bt,device='cpu').view(1,1,1,bt)
    a=torch.randint(-2,3,shape,generator=rng,device='cpu').double()/256;p=torch.randint(-4,5,shape,generator=rng,device='cpu').double()/16
    a=torch.where(cols<rows,a,0.)+torch.where(cols==rows,1.,0.);p=torch.where(cols<rows,p,0.)
    z=dict(A=guarded(a.half()),P=guarded(p.float()));e=reference(z,shape);f=dict(inputs=z,expected=e.float(),expected_fp64=e);check_fixture(spec,f);return f
