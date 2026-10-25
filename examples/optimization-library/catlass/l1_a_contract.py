"""Independent dyadic FP64 matmul, quantized only once to FP16 RNE on CPU."""
import torch


def parameters(spec):
    p=spec.get('parameters',spec)
    if any(type(p.get(k)) is not int or p[k]<=0 for k in ('M','N','K')) or p['M']>128 or p['N']>24576 or p['N']%16 or p['K']>1024 or p['K']%16:raise ValueError('M1..128, positive N<=24576/K<=1024 multiples16 required')
    return dict(p)


def check_inputs(inputs,p):
    if set(inputs)!={'a','b'}:raise ValueError('exact immutable input pair required')
    for key,shape in [('a',(p['M'],p['K'])),('b',(p['K'],p['N']))]:
        x=inputs[key]
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float16 or x.device.type!='cpu' or tuple(x.shape)!=shape or not x.is_contiguous() or x.requires_grad:raise ValueError('contiguous CPU FP16 row-major input shape required')
        if not torch.isfinite(x).all() or (x.abs()>4/64).any() or not torch.equal(x.float()*64,(x.float()*64).round()):raise ValueError('finite integer/64 inputs q[-4,4] required')
    if inputs['a'].untyped_storage().data_ptr()==inputs['b'].untyped_storage().data_ptr():raise ValueError('input storage alias forbidden')


def make_case(spec):
    p=parameters(spec);seed=spec.get('seed',p.get('seed'))
    if type(seed) is not int or seed<0:raise ValueError('nonnegative seed required')
    g=torch.Generator().manual_seed(seed)
    inputs={key:(torch.randint(-4,5,shape,generator=g).to(torch.float32)/64).to(torch.float16) for key,shape in [('a',(p['M'],p['K'])),('b',(p['K'],p['N']))]}
    check_inputs(inputs,p);exact=inputs['a'].double()@inputs['b'].double()
    # K*max|qA*qB|<=16384<2^24; all partial dyadic sums fit FP32 exactly.
    return dict(inputs=inputs,expected=(exact,exact.to(torch.float16)))


def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().view(torch.int16),b.detach().cpu().view(torch.int16))
