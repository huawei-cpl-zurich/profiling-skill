"""Independent CPU generation and given-FP32-saved-state L2 derivative."""
import torch


def check_spec(spec):
    p=spec.get('parameters',spec)
    if type(p.get('rows')) is not int or not 1<=p['rows']<=512:raise ValueError('rows must be1..512')
    if type(p.get('D')) is not int or not 513<=p['D']<=4096:raise ValueError('only D513..4096 given-state branch supported')
    return p


def make_case(spec):
    p=check_spec(spec);rows,d=p['rows'],p['D'];rng=torch.Generator().manual_seed(spec.get('seed',0))
    x=torch.randint(-128,129,(rows,d),generator=rng).float()/128
    dy=torch.randint(-128,129,(rows,d),generator=rng).float()/128
    if p.get('zero_parallel_cancel'):
        x[0].zero_()
        if rows>=3:
            x[2]=torch.where(torch.arange(d)%2==0,1.,-1.)
            if d%2:x[2,-1]=0
            dy[2].fill_(1)
    r=(x.double().square().sum(1)+1e-6).rsqrt().float()
    y=(x.double()*r.double()[:,None]).float()
    if p.get('zero_parallel_cancel') and rows>=2:dy[1]=y[1]
    yd,dd,rd=y.double(),dy.double(),r.double()[:,None]
    expected=(rd*(dd-yd*(dd*yd).sum(1,keepdim=True))).float()
    return dict(inputs=dict(y=y,rstd=r,dy=dy),source_x=x,expected=expected)


def check_inputs(inputs,p,device):
    if set(inputs)!={'y','rstd','dy'}:raise ValueError('y,rstd,dy required')
    for name,t in inputs.items():
        shape=(p['rows'],) if name=='rstd' else (p['rows'],p['D'])
        if (not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or t.device.type!=device
                or tuple(t.shape)!=shape or not t.is_contiguous() or t.storage_offset()!=0):raise ValueError('invalid input shape/dtype/layout/device: '+name)
        if device=='cpu':
            bound=1000.001 if name=='rstd' else (1.0 if name=='dy' else 1.000001)
            if not torch.isfinite(t).all() or not torch.all(t.abs()<=bound) or (name=='rstd' and not torch.all(t>0)):
                raise ValueError('saved input outside finite bounded domain: '+name)
    pointers=[t.untyped_storage().data_ptr() for t in inputs.values()]
    if len(set(pointers))!=3:raise ValueError('input storage alias')


def validate_actual(actual,fixture,inputs=None,buffer=None):
    e=fixture['expected']
    if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32 or actual.shape!=e.shape or not actual.is_contiguous():
        raise ValueError('output shape/dtype/layout')
    a=actual.detach().cpu()
    if not torch.isfinite(a).all():raise ValueError('nonfinite output')
    delta=(a-e).abs();passed=bool(torch.all(delta<=2e-6+2e-5*e.abs()))
    if inputs is not None:
        if actual.device!=inputs['y'].device:raise ValueError('output device')
        if len({actual.untyped_storage().data_ptr(),*(v.untyped_storage().data_ptr() for v in inputs.values())})!=4:raise ValueError('output alias')
        for name,t in inputs.items():
            if not torch.equal(t.cpu(),fixture['inputs'][name]):raise ValueError('input mutation')
    if buffer is not None:
        b=buffer.cpu()
        if not torch.all(b[:32]==12345) or not torch.all(b[-32:]==12345):raise ValueError('canary changed')
    return dict(passed=passed,max_absolute_error=float(delta.max()),rtol=2e-5,atol=2e-6)
