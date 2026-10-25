"""Independent fixed FP64 oracle for bounded clipped SwiGLU backward."""
import math
import torch


def check_spec(spec):
    p = spec.get('parameters', spec)
    for key in ('M', 'K'):
        if type(p.get(key)) is not int or not 1 <= p[key] <= 1024:
            raise ValueError('integer '+key+' in [1,1024] required')
    if p.get('two_k', 2*p['K']) != 2*p['K']:
        raise ValueError('even last dimension equal to 2K required')
    for name, expected in [('alpha',1.702),('limit',7.0)]:
        v = p.get(name,expected)
        if type(v) not in (int,float) or not math.isfinite(v) or v != expected:
            raise ValueError('unsupported '+name)
    if p.get('pattern','unsaturated') not in ('saturated','unsaturated','saturated_half_up_clipped',
            'mixed_every_tile','exact_clamp_boundaries','zeros_signed_cancellation'):
        raise ValueError('unknown pattern')
    return p


def make_case(spec):
    p=check_spec(spec);m,k=p['M'],p['K'];rng=torch.Generator().manual_seed(spec.get('seed',0))
    def quant(lo,hi):return torch.randint(lo,hi+1,(m,k),generator=rng).float()/64
    pattern=p.get('pattern','unsaturated');g=quant(-192,192);u=quant(-768,768);dy=quant(-64,64)
    pos=torch.arange(m*k).reshape(m,k)
    if pattern.startswith('saturated'):g=quant(512,768)
    if pattern=='saturated_half_up_clipped':
        u=quant(-320,320);u[pos%4==0]=9;u[pos%4==2]=-9
    if pattern=='mixed_every_tile':g=torch.where(pos%2==0,8.,-2.)
    if pattern=='exact_clamp_boundaries':
        seven=torch.tensor(7.);neg=-seven
        gs=torch.stack([torch.nextafter(seven,torch.tensor(float('inf'))),seven,
                        torch.nextafter(seven,torch.tensor(-float('inf')))])
        us=torch.stack([neg,torch.nextafter(neg,torch.tensor(-float('inf'))),
                       torch.nextafter(neg,torch.tensor(float('inf'))),seven,
                       torch.nextafter(seven,torch.tensor(float('inf'))),
                       torch.nextafter(seven,torch.tensor(-float('inf')))])
        g=gs[pos%3];u=us[pos%6]
    if pattern=='zeros_signed_cancellation':
        g=torch.where(pos%2==0,0.,-0.);u=torch.where(pos%3==0,-1.,0.);dy=torch.where(pos%2==0,1.,-1.)
    inputs=dict(gup=torch.cat((g,u),dim=1),dy=dy)
    gd,ud,dd=g.double(),u.double(),dy.double();gc=gd.clamp(max=7);uc=ud.clamp(-7,7)
    s=torch.sigmoid(gc*1.702);dg=dd*(uc+1)*(s+gc*1.702*s*(1-s));du=dd*(gc*s)
    dg=torch.where(gd<=7,dg,0.);du=torch.where((ud>=-7)&(ud<=7),du,0.)
    return dict(inputs=inputs,expected=torch.cat((dg,du),dim=1).float())


def check_inputs(inputs,p,device):
    if set(inputs)!={'gup','dy'}:raise ValueError('gup and dy required')
    m,k=p['M'],p['K']
    for name,t in inputs.items():
        if (not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.device.type!=device
                or tuple(t.shape)!=(m,2*k if name=='gup' else k) or not t.is_contiguous() or t.storage_offset()!=0):
            raise ValueError('invalid input shape/layout/dtype/device: '+name)
        if device=='cpu' and (not torch.isfinite(t).all() or not torch.all(t.abs()<=(12 if name=='gup' else 1))):
            raise ValueError('input outside finite bounded domain: '+name)
    if inputs['gup'].untyped_storage().data_ptr()==inputs['dy'].untyped_storage().data_ptr():
        raise ValueError('input storage alias')


def validate_actual(actual,fixture,inputs=None,buffer=None):
    e=fixture['expected']
    if (not isinstance(actual,torch.Tensor) or actual.shape!=e.shape or actual.dtype!=torch.float32
            or not actual.is_contiguous()):raise ValueError('output shape/dtype/layout')
    a=actual.detach().cpu()
    if not torch.isfinite(a).all():raise ValueError('nonfinite output')
    error=(a-e).abs();passed=bool(torch.all(error<=2e-5+2e-5*e.abs()))
    k=e.shape[1]//2;g,u=fixture['inputs']['gup'].split(k,dim=1)
    for value,mask in [(a[:,:k],g>7),(a[:,k:],u.abs()>7)]:
        if not torch.all(value.contiguous().view(torch.int32)[mask]==0):raise ValueError('strict-clipped gradient must be positive zero')
    if inputs is not None:
        if actual.device!=inputs['gup'].device:raise ValueError('output device')
        if len({actual.untyped_storage().data_ptr(),*(t.untyped_storage().data_ptr() for t in inputs.values())})!=3:
            raise ValueError('storage alias')
        for n,t in inputs.items():
            if not torch.equal(t.cpu(),fixture['inputs'][n]):raise ValueError('input mutation')
    if buffer is not None:
        b=buffer.cpu()
        if not torch.all(b[:32]==12345) or not torch.all(b[-32:]==12345):raise ValueError('output canary')
    return dict(passed=passed,max_absolute_error=float(error.max()),rtol=2e-5,atol=2e-5)
