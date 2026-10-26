"""Independent exact dyadic CPU backward oracle; no Triton/NPU imports."""
import torch


def parameters(spec):
    p=spec.get('parameters',spec)
    for k,lo,hi in [('B',1,2),('S',1,128),('H',8,256),('half',8,32),('nope_dim',64,192)]:
        if type(p.get(k)) is not int or not lo<=p[k]<=hi:raise ValueError('bounded integer B/S/H/half/nope_dim required')
    if p['H']&(p['H']-1) or p['half']&(p['half']-1) or p['nope_dim']%64:raise ValueError('power-of-two H/half and positive64-aligned prefix required')
    return p


def shapes(p):
    b,s,h,r,n=(p[k] for k in ('B','S','H','half','nope_dim'))
    return dict(dy=(b,s,h,n+2*r),x_rope=(b,s,h,2*r),cos=(b,s,r),sin=(b,s,r))


def check_inputs(inputs,p):
    if not isinstance(inputs,dict) or set(inputs)!=set(shapes(p)):raise ValueError('dy/saved-pre-rotation x_rope/cos/sin required')
    for name,shape in shapes(p).items():
        t=inputs[name];dtype=torch.float16 if name in ('dy','x_rope') else torch.float32
        if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=dtype or t.shape!=shape or not t.is_contiguous() or t.requires_grad:raise ValueError('distinct contiguous CPU tensors with exact represented types/shapes required')
        if not torch.isfinite(t).all() or (t.abs()>.25).any() or not torch.equal(t.float()*16,(t.float()*16).round()):raise ValueError('finite q/16 integerq[-4,4] required')
    if len({t.untyped_storage().data_ptr() for t in inputs.values()})!=4:raise ValueError('input storage aliases forbidden')


def make_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative integer seed required')
    g=torch.Generator().manual_seed(seed);inputs={k:(torch.randint(-4,5,s,generator=g).float()/16).to(torch.float16 if k in ('dy','x_rope') else torch.float32) for k,s in shapes(p).items()};check_inputs(inputs,p)
    y=inputs['dy'].double();x=inputs['x_rope'].double();de,do=y[...,p['nope_dim']::2],y[...,p['nope_dim']+1::2];e,o=x[...,::2],x[...,1::2];c=inputs['cos'].double().unsqueeze(2);ss=inputs['sin'].double().unsqueeze(2)
    dx=y.clone();dx[...,p['nope_dim']::2]=de*c+do*ss;dx[...,p['nope_dim']+1::2]=-de*ss+do*c
    dc=(de*e+do*o).sum(2);ds=(-de*o+do*e).sum(2)
    return dict(inputs=inputs,expected=(dx.half(),dc.float(),ds.float()),expected_fp64=(dx,dc,ds))


def bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().view(torch.int16 if a.dtype==torch.float16 else torch.int32),b.detach().cpu().view(torch.int16 if b.dtype==torch.float16 else torch.int32))
