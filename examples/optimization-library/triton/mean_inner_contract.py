"""Independent FP64 CPU middle-axis mean and bounded represented-input provider."""
import torch


def parameters(spec):
    p=dict(spec.get('parameters',spec))
    if any(type(p.get(k)) is not int or not 1<=p[k]<=v for k,v in [('M',64),('N',4096),('K',128)]):raise ValueError('positive bounded M/N/K required')
    if p.get('physical_layout','MNK') not in ('MNK','MKN-backing'):raise ValueError('unknown physical layout')
    return p


def strides(p):
    m,n,k=(p[x] for x in ('M','N','K'))
    return (k*n,1,n) if p.get('physical_layout')=='MKN-backing' else (n*k,k,1)


def check_inputs(inputs,p):
    if set(inputs)!= {'x'}:raise ValueError('exact input x required')
    x=inputs['x'];shape=tuple(p[k] for k in ('M','N','K'))
    if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32 or tuple(x.shape)!=shape or x.stride()!=strides(p) or x.requires_grad:raise ValueError('CPU FP32 physical shape/stride required')
    if not torch.isfinite(x).all() or (x.abs()>16.0001221).any():raise ValueError('finite bounded input required')


def input_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative seed required')
    shape=tuple(p[k] for k in ('M','N','K'));g=torch.Generator().manual_seed(seed)
    palette=torch.tensor([-16.,16.,-1/16,1/16],dtype=torch.float32)
    x=palette[torch.randint(0,4,shape,generator=g)]+torch.randint(-4,5,shape,generator=g).to(torch.float32)*2**-15
    if p['M']>=2:x[1].copy_(x[0])
    if p.get('physical_layout')=='MKN-backing':x=x.transpose(1,2).contiguous().transpose(1,2)
    check_inputs(dict(x=x),p);return dict(x=x)


def make_case(spec):
    inputs=input_case(spec)
    return dict(inputs=inputs,expected=inputs['x'].double().sum(dim=1)/parameters(spec)['N'])


def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().view(torch.int32),b.detach().cpu().view(torch.int32))
