"""Independent exact-dyadic CPU SROTM provider and native ABI metadata."""
import struct
import torch


def dimensions(spec):
    p=spec.get('parameters',spec);n=p.get('n');ix=p.get('incx');iy=p.get('incy');h=p.get('sparam')
    if (type(n) is not int or not 1<=n<=1048579 or type(ix) is not int or type(iy) is not int or ix!=iy or ix not in (1,2)
        or type(spec.get('seed')) is not int or spec['seed']<0 or not isinstance(h,list) or len(h)!=5
        or any(type(x) not in (int,float) for x in h) or h not in ([-1.,.75,-.25,.5,1.25],[0.,.75,-.25,.5,1.25])):raise ValueError('unsupported SROTM domain')
    return n,ix,1+(n-1)*ix


def tiling(spec,x=0,y=0):
    n,inc,size=dimensions(spec);h=spec.get('parameters',spec)['sparam'];flag,h11,h21,h12,h22=h
    if flag==0:h11=h22=1.
    return struct.pack('<QQQQIIqqfffff4x',x,y,size,size,n,min((n+2047)//2048,8) if inc==1 else 1,inc,inc,flag,h11,h12,h21,h22)


def ranges(spec):
    n,inc,_=dimensions(spec);use=min((n+2047)//2048,8) if inc==1 else 1;chunks,tail=divmod(n,128);active=min(use,chunks) if chunks else 1
    result=[]
    for core in range(8):
        if core>=active:result.append((0,0));continue
        if not chunks:result.append((0,n));continue
        base,extra=divmod(chunks,active);start=(core*base+min(core,extra))*128;count=(base+(core<extra))*128+(tail if core==active-1 else 0);result.append((start,count))
    return result


def check_inputs(spec,inputs):
    _,_,size=dimensions(spec)
    if not isinstance(inputs,dict) or set(inputs)!={'X','Y'}:raise ValueError('X/Y required')
    for x in inputs.values():
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or tuple(x.shape)!=(size,) or not x.is_contiguous()
            or x.storage_offset()!=0 or x.untyped_storage().nbytes()!=size*4):raise ValueError('owned contiguous FP32 CPU full-storage vectors required')
        if not bool(torch.isfinite(x).all() and (x.abs()<=4).all() and (x*16==(x*16).trunc()).all()):raise ValueError('bounded finite integer/16 required')
    if inputs['X'].untyped_storage().data_ptr()==inputs['Y'].untyped_storage().data_ptr():raise ValueError('input alias')


def make_case(spec):
    n,inc,size=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed']);v={k:torch.randint(-64,65,(size,),generator=rng).float()/16 for k in ('X','Y')};check_inputs(spec,v)
    x,y=v['X'].double(),v['Y'].double();a,b=x[::inc].clone(),y[::inc].clone();flag,h11,h21,h12,h22=spec.get('parameters',spec)['sparam']
    x[::inc]=(h11*a+h12*b) if flag<0 else a+h12*b
    y[::inc]=(h21*a+h22*b) if flag<0 else h21*a+b
    return dict(inputs=v,expected=(x.float(),y.float()))


def check_output(host,expected):
    if not isinstance(host,tuple) or len(host)!=2:return dict(passed=False,reason='two FP32 full-storage outputs required')
    rows=[]
    for name,a,e in zip(('X','Y'),host,expected):
        if not isinstance(a,torch.Tensor) or a.dtype!=torch.float32 or a.device.type!='cpu' or a.shape!=e.shape or not a.is_contiguous():return dict(passed=False,reason='output layout/dtype/shape')
        finite=torch.isfinite(a);nonzero=e!=0;bad=(torch.where(nonzero,a.view(torch.int32)!=e.view(torch.int32),a!=0))|~finite;indices=bad.nonzero().flatten();first=None
        if indices.numel():
            i=int(indices[0]);first=dict(coordinates=[i],expected=float(e[i]),actual=float(a[i]) if bool(finite[i]) else str(a[i].item()))
        rows.append(dict(name=name,mismatch_count=int(indices.numel()),first_mismatch=first,nonfinite_count=int((~finite).sum()),max_abs_error=float((a[finite]-e[finite]).abs().max()) if bool(finite.any()) else None,zero_sign_differences_secondary=int(((a==0)&(e==0)&(torch.signbit(a)!=torch.signbit(e))).sum())))
    return dict(passed=all(x['mismatch_count']==0 for x in rows),outputs=rows)
