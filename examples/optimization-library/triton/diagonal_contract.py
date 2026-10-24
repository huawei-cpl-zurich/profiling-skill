"""W10's separately qualified FP32 B=1 exact-structure domain."""
from pathlib import Path
import torch
from eigenvalues_contract import diagnostics


def parameters(spec):
    p = spec.get('parameters', spec)
    n = p.get('N')
    if (type(n) is not int or n % 2 or not 4 <= n <= 128 or
            p.get('batch') != 1 or p.get('sweeps') != 5 or p.get('want_vectors') is not False):
        raise ValueError('W10 requires B=1, even N in 4..128, five sweeps, values only')
    if p.get('structure') not in ('diagonal','near-diagonal-dense',
            'one-tiny-symmetric-offdiagonal','unsorted-repeated-diagonal'):
        raise ValueError('unsupported frozen structure')
    return 1, n, False


def make_case(spec):
    batch,n,_ = parameters(spec)
    structure = spec['parameters']['structure']
    g = torch.Generator(device='cpu').manual_seed(spec['seed'])
    if structure == 'diagonal':
        d = torch.rand(n,generator=g,dtype=torch.float32,device='cpu')*4-2
        a = torch.diag(d).unsqueeze(0)
    elif structure == 'unsorted-repeated-diagonal':
        d = torch.tensor([2.,-1.,1.,0.],dtype=torch.float32,device='cpu').repeat((n+3)//4)[:n]
        a = torch.diag(d).unsqueeze(0)
    else:
        raw = torch.rand((batch,n,n),generator=g,dtype=torch.float32,device='cpu')*2-1
        a = (raw+raw.transpose(-1,-2))/(2*64*n)
        if structure == 'one-tiny-symmetric-offdiagonal':
            a.zero_();a[0,0,1]=a[0,1,0]=2**-20
        a.diagonal(dim1=-2,dim2=-1).copy_(1+torch.arange(n,dtype=torch.float32,device='cpu')/n)
    return dict(a=a.contiguous(),expected=torch.linalg.eigvalsh(a.double()))


def validate_input(spec, fixture):
    shape = parameters(spec)
    a = fixture['a']
    if (a.device.type!='cpu' or a.dtype!=torch.float32 or a.shape!=(1,shape[1],shape[1])
            or not a.is_contiguous() or not torch.isfinite(a).all()
            or not torch.equal(a,a.transpose(-1,-2))):
        raise ValueError('finite symmetric contiguous CPU FP32 [1,N,N] required')
    # This is an admission bound, never a shortcut predicate or launch decision.
    off=a.clone();off.diagonal(dim1=-2,dim2=-1).zero_()
    if off.abs().max()>1/(64*shape[1]) or a.abs().max()>2:
        raise ValueError('outside bounded W10 convergence domain')
    return shape


CASE_SOURCES=[Path(__file__),Path(__file__).with_name('eigenvalues_contract.py'),
              Path(__file__).with_name('jacobi_contract.py')]
