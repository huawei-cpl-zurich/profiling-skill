"""Independent CPU fixtures and FP64 oracle for bounded clipped forward."""
import math
import torch


def check_spec(spec):
    p=spec.get('parameters',spec)
    for key in ('M','K'):
        if type(p.get(key)) is not int or not 1<=p[key]<=1024:raise ValueError('integer '+key+' in1..1024 required')
    if p.get('two_k',2*p['K'])!=2*p['K']:raise ValueError('even last dimension2K required')
    if type(p.get('limit',0)) not in (int,float) or not math.isfinite(p.get('limit',0)) or p.get('limit',0) not in (0,7):raise ValueError('limit0 or7 required')
    if p.get('alpha',1.702)!=1.702:raise ValueError('alpha1.702 required')
    return p


def check_input(t,p,device):
    if (not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.device.type!=device or tuple(t.shape)!=(p['M'],2*p['K'])
            or not t.is_contiguous() or t.storage_offset()!=0):raise ValueError('invalidGUP shape/dtype/layout/device')
    if device=='cpu':
        q=t.detach()*64
        if not torch.all(torch.isfinite(q)&(q.abs()<=768)&(q==q.round())):raise ValueError('finite1/64 grid in[-12,12] required includingUP')


def make_case(spec):
    p=check_spec(spec);rng=torch.Generator().manual_seed(spec.get('seed',0));m,k=p['M'],p['K']
    gup=torch.randint(-768,769,(m,2*k),generator=rng).float()/64
    if p.get('signed_zero'):
        gup[0,:k]=torch.where(torch.arange(k)%2==0,0.,-0.)
    check_input(gup,p,'cpu')  # Validate before oracle creation and generic cache publication.
    g,u=gup.double().chunk(2,dim=1);limit=p.get('limit',0);g=g.clamp(max=limit);u=u.clamp(-limit,limit)
    expected=((u+1)*(g*torch.sigmoid(g*1.702))).float()
    return dict(inputs=dict(gup=gup),expected=expected)


def validate_actual(actual,fixture,device_input=None,buffer=None):
    e=fixture['expected']
    if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32 or actual.shape!=e.shape or not actual.is_contiguous():raise ValueError('output shape/dtype/layout')
    a=actual.detach().cpu()
    if not torch.isfinite(a).all():raise ValueError('nonfinite output')
    delta=(a-e).abs();passed=bool(torch.all(delta<=3e-5+2e-5*e.abs()))
    zero=e==0
    if not torch.equal(a.view(torch.int32)[zero],e.view(torch.int32)[zero]):raise ValueError('signed zero differs from independent expected')
    if device_input is not None:
        if actual.device!=device_input.device or actual.untyped_storage().data_ptr()==device_input.untyped_storage().data_ptr():raise ValueError('output device/alias')
        if not torch.equal(device_input.cpu(),fixture['inputs']['gup']):raise ValueError('input mutation')
    if buffer is not None:
        b=buffer.cpu()
        if not torch.all(b[:32]==12345) or not torch.all(b[-32:]==12345):raise ValueError('canary changed')
    return dict(passed=passed,max_absolute_error=float(delta.max()),rtol=2e-5,atol=3e-5,signed_zero_exact=True)
