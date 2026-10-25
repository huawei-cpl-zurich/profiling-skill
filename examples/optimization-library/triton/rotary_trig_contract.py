"""Independent FP64 rotary oracle and physical CPU admission."""
from pathlib import Path
import torch
CASE_SOURCES=[Path(__file__)]


def dimensions(spec):
    p=spec.get('parameters',spec);b,s,hq,hk,d=(p.get(x) for x in ('B','S','Hq','Hk','D'))
    if (any(type(x) is not int or x<=0 for x in (b,s,hq,hk,d)) or d not in (64,128)
            or p.get('scaling') not in (2.,4.) or type(p.get('scaling')) not in (int,float)
            or type(spec.get('seed')) is not int or spec['seed']<0
            or p.get('position_stride',1) not in (1,2) or type(p.get('position_stride',1)) is not int
            or type(p.get('zero_position',False)) is not bool):raise ValueError('unsupported rotary shape/scaling')
    if (s-1+32*(b-1)+spec['seed']%17>2048 and not p.get('zero_position')) or b*((s+31)//32*32)*max(hq,hk)*d*4>=2**31:
        raise ValueError('positions or padded signed32 addresses out of range')
    return b,s,hq,hk,d


def validate_inputs(v,spec):
    b,s,hq,hk,d=dimensions(spec);p=spec.get('parameters',spec);ps=p.get('position_stride',1)
    if not isinstance(v,dict) or set(v)!={'Q','K','P','I'}:raise ValueError('Q,K,P,I required')
    shapes={'Q':(b,s,hq,d),'K':(b,s,hk,d),'P':(b,s),'I':(d//2,)}
    for name,x in v.items():
        dtype=torch.int64 if name=='P' else torch.float32
        strides=(s*ps,ps) if name=='P' else None
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=dtype or tuple(x.shape)!=shapes[name]
                or x.storage_offset()!=0 or (strides and x.stride()!=strides) or (not strides and not x.is_contiguous())
                or x.untyped_storage().nbytes()!=x.numel()*x.element_size()*(ps if name=='P' else 1)):
            raise ValueError('owned CPU contiguous Q/K/I and exact declared P backing/strides required')
    if len({x.untyped_storage().data_ptr() for x in v.values()})!=4:raise ValueError('input alias')
    for name in ('Q','K'):
        x=v[name]
        if not bool(torch.isfinite(x).all() and (x.abs()<=1).all() and (x*128==(x*128).trunc()).all()):raise ValueError('finite dyadic Q/K in [-1,1] required')
    if not bool(((v['P']>=0)&(v['P']<=2048)).all()):raise ValueError('position range')
    if not torch.equal(v['I'],torch.arange(1,d//2+1,dtype=torch.float32)/1024):raise ValueError('fixed invfreq required')


def make_case(spec):
    b,s,hq,hk,d=dimensions(spec);p=spec.get('parameters',spec);rng=torch.Generator().manual_seed(spec['seed'])
    q=torch.randint(-128,129,(b,s,hq,d),generator=rng).float()/128
    k=torch.randint(-128,129,(b,s,hk,d),generator=rng).float()/128
    ps=p.get('position_stride',1);back=torch.empty((b,s*ps),dtype=torch.int64)
    position=torch.arange(s)[None,:]+32*torch.arange(b)[:,None]+spec['seed']%17
    if p.get('zero_position'):
        position.zero_()
        for x in (q,k):
            x.reshape(-1)[::4]=0.;x.reshape(-1)[1::4]=-0.;x.reshape(-1)[2::4]=.5;x.reshape(-1)[3::4]=-.5
    back[:,::ps]=position
    if ps==2:back[:,1::2]=2048-position
    inv=torch.arange(1,d//2+1,dtype=torch.float32)/1024
    inputs=dict(Q=q,K=k,P=back[:,::ps],I=inv);validate_inputs(inputs,spec)
    angle=inputs['P'].double()[:,:,None,None]/p['scaling']*inv.double()[None,None,None,:]
    cos,sin=angle.cos(),angle.sin()
    def rotate(x):
        a,c=x.double().chunk(2,dim=-1)
        return torch.cat((a*cos-c*sin,c*cos+a*sin),dim=-1)
    return dict(inputs=inputs,expected=(rotate(q),rotate(k)))


def check_output(actual,expected):
    if not isinstance(actual,tuple) or len(actual)!=2:return dict(passed=False,reason='two outputs required')
    for a,e in zip(actual,expected):
        if not isinstance(a,torch.Tensor) or a.dtype!=torch.float32 or a.shape!=e.shape or not a.is_contiguous():return dict(passed=False,reason='FP32 contiguous Q/K shapes required')
    got=tuple(a.detach().cpu().double() for a in actual)
    errors=[float((a-e).abs().max()) for a,e in zip(got,expected)]
    return dict(passed=all(bool(torch.isfinite(a).all() and torch.allclose(a,e,atol=3e-5,rtol=2e-5)) for a,e in zip(got,expected)),max_abs_errors=errors,atol=3e-5,rtol=2e-5)
