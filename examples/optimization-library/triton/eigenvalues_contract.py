"""Frozen near-diagonal FP32 eigvalsh domain; independent CPU FP64 oracle."""
from pathlib import Path
import torch
from jacobi_contract import round_pairs


def parameters(spec):
    p = spec.get('parameters', spec)
    n, batch = p.get('N'), p.get('batch')
    if type(n) is not int or n % 2 or not 4 <= n <= 96:
        raise ValueError('even N in 4..96 required')
    if type(batch) is not int or batch < 1 or p.get('sweeps') != 5:
        raise ValueError('positive batch and five complete sweeps required')
    if type(p.get('want_vectors')) is not bool:
        raise ValueError('want_vectors must be boolean')
    if p.get('distribution') not in ('near-diagonal-spread','repeated-diagonal'):
        raise ValueError('unsupported bounded distribution')
    return batch, n, p['want_vectors']


def make_case(spec):
    batch, n, _ = parameters(spec)
    if spec['parameters']['distribution'] == 'repeated-diagonal':
        a = torch.eye(n, dtype=torch.float32).expand(batch,n,n).clone()*1.25
    else:
        g = torch.Generator().manual_seed(spec['seed'])
        raw = torch.rand((batch,n,n), generator=g, dtype=torch.float32)*2-1
        a = (raw+raw.transpose(-1,-2))/(2*64*n)
        a.diagonal(dim1=-2,dim2=-1).copy_(1+torch.arange(n,dtype=torch.float32)/n)
    return dict(a=a.contiguous(), expected=torch.linalg.eigvalsh(a.double()))


def validate_input(spec, fixture):
    batch, n, vectors = parameters(spec)
    a = fixture['a']
    if (a.device.type != 'cpu' or a.dtype != torch.float32 or a.shape != (batch,n,n)
            or not a.is_contiguous() or not torch.isfinite(a).all()
            or not torch.equal(a,a.transpose(-1,-2))):
        raise ValueError('finite symmetric contiguous CPU FP32 [B,N,N] required')
    # Reject matrices outside the fixed convergence claim before any NPU import.
    off = a.clone(); off.diagonal(dim1=-2,dim2=-1).zero_()
    diag = a.diagonal(dim1=-2,dim2=-1)
    expected = (torch.full((n,),1.25,dtype=torch.float32) if
        spec['parameters']['distribution']=='repeated-diagonal' else
        1+torch.arange(n,dtype=torch.float32)/n)
    if not torch.equal(diag,expected.expand(batch,n)) or off.abs().max()>1/(64*n):
        raise ValueError('matrix outside frozen near-diagonal envelope')
    return batch, n, vectors


def diagnostics(actual, fixture, vectors):
    values, v = actual if vectors else (actual,None)
    values = values.detach().cpu()
    expected = fixture['expected']
    if values.dtype != torch.float32 or values.shape != expected.shape or not torch.isfinite(values).all():
        return dict(passed=False,reason='shape/dtype/nonfinite values')
    err = (values.double()-expected).abs()
    ok = bool((err<=2e-4+2e-4*expected.abs()).all()) and bool((values[:,1:]>=values[:,:-1]).all())
    result = dict(passed=ok,max_absolute_error=float(err.max()),atol=2e-4,rtol=2e-4)
    if vectors:
        v = v.detach().cpu()
        if v.dtype != torch.float32 or v.shape != fixture['a'].shape or not torch.isfinite(v).all():
            return dict(passed=False,reason='shape/dtype/nonfinite vectors')
        a, v = fixture['a'].double(), v.double()
        residual = torch.linalg.matrix_norm(a@v-v*values.double().unsqueeze(-2))/torch.linalg.matrix_norm(a)
        orth = torch.linalg.matrix_norm(v.transpose(-1,-2)@v-torch.eye(v.shape[-1],dtype=torch.float64),ord=float('inf'))
        result.update(residual_relative_frobenius=float(residual.max()),orthogonality_inf=float(orth.max()))
        result['passed'] &= bool((residual<=2e-3).all() and (orth<=2e-3).all())
    return result


CASE_SOURCES = [Path(__file__), Path(__file__).with_name('jacobi_contract.py')]
