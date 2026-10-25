"""Independent exact-dyadic CPU oracle for the bounded prepared SGER component."""
import struct
import torch


def dimensions(spec):
    p=spec.get('parameters',{});m,n,lda,cores=(p.get(k) for k in ('m','n','lda','aiv_cores'))
    if (any(type(x) is not int for x in (m,n,lda,cores)) or not 1<=m<=2049 or not 1<=n<=480 or not m<=lda<=m+16
        or cores!=48 or type(spec.get('seed')) is not int or spec['seed']<0 or p.get('incx')!=1 or p.get('incy')!=1
        or type(p.get('alpha')) not in (int,float) or p['alpha'] not in (.5,-.25)):raise ValueError('unsupported prepared SGER domain')
    use=min(n,cores);cols=(n+use-1)//use
    for core in range(use):
        start=core*cols;count=0 if start>=n else n-start if core==use-1 else cols
        if start<n and start+count>n:raise ValueError('source column partition overrun')
    return m,n,lda,use,cols


def tiling(spec,a=0,x=0,y=0):
    m,n,lda,use,cols=dimensions(spec)
    return struct.pack('<QQQIIIIIiif',a,x,y,m,n,lda,use,cols,1,1,spec['parameters']['alpha'])


def check_inputs(spec,inputs):
    m,n,lda,_,_=dimensions(spec)
    if not isinstance(inputs,dict) or set(inputs)!={'A','X','Y'}:raise ValueError('A/X/Y required')
    for name,size in [('A',lda*n),('X',m),('Y',n)]:
        x=inputs[name]
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or tuple(x.shape)!=(size,)
            or not x.is_contiguous() or x.requires_grad or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=size*4):raise ValueError('owned contiguous FP32 full-storage vectors required')
        if not bool(torch.isfinite(x).all() and (x.abs()<=4).all() and (x*16==(x*16).trunc()).all()):raise ValueError('bounded integer/16 inputs required')
    if len({x.untyped_storage().data_ptr() for x in inputs.values()})!=3:raise ValueError('input alias')


def make_case(spec):
    m,n,lda,_,_=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed'])
    v={name:torch.randint(-64,65,(size,),generator=rng).to(torch.float32)/16 for name,size in [('A',lda*n),('X',m),('Y',n)]};check_inputs(spec,v)
    expected=v['A'].double().reshape(n,lda).clone()
    expected[:,:m]+=((spec['parameters']['alpha']*v['Y'].double())[:,None]*v['X'].double()[None,:])
    return dict(inputs=v,expected=expected.flatten().float())


def check_output(actual,expected):
    if not isinstance(actual,torch.Tensor) or actual.device.type!='cpu' or actual.dtype!=torch.float32 or actual.shape!=expected.shape or not actual.is_contiguous():return dict(passed=False,reason='CPU FP32 full-storage output required')
    finite=torch.isfinite(actual);bad=torch.where(expected!=0,actual.view(torch.int32)!=expected.view(torch.int32),actual!=0)|~finite;indices=bad.nonzero().flatten();first=None
    if indices.numel():
        i=int(indices[0]);first=dict(coordinates=[i],expected=float(expected[i]),actual=float(actual[i]) if bool(finite[i]) else str(actual[i].item()))
    return dict(passed=not bool(indices.numel()),mismatch_count=int(indices.numel()),first_mismatch=first,nonfinite_count=int((~finite).sum()),max_abs_error=float((actual[finite]-expected[finite]).abs().max()) if bool(finite.any()) else None,zero_sign_differences_secondary=int(((actual==0)&(expected==0)&(torch.signbit(actual)!=torch.signbit(expected))).sum()))
