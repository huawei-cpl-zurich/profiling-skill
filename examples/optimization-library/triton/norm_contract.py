"""CPU contracts for L2 row batching and complete unweighted RMS input VJP.

E07 returns the forward output and the gradient with respect to X for the
supplied cotangent. It has no trainable weight or omitted weight gradient.
"""
from pathlib import Path
import math
import torch

RTOL = ATOL = 3e-5
CASE_SOURCES = [Path(__file__)]


def check_spec(spec):
    family = spec.get('family')
    if family not in ('E01', 'E07'):
        raise ValueError('normalization family must be E01 or E07')
    rows, width, seed = (spec.get(k) for k in ('rows', 'width', 'seed'))
    if any(type(v) is not int for v in (rows, width, seed)) or not (
            1 <= rows <= 65536 and 1 <= width <= (2048 if family == 'E01' else 4096)
            and rows * width <= 8388608 and 0 <= seed < 2**64):
        raise ValueError('positive bounded row/width dimensions and uint64 seed required')
    eps = spec.get('eps', 1e-6)
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 0 < eps <= .1:
        raise ValueError('finite epsilon in (0,.1] required')
    return rows, width, seed, float(eps)


def make_case(spec):
    rows, width, seed, eps = check_spec(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(rows, width, generator=rng, dtype=torch.float32)
    x[::17].zero_()  # Include the epsilon-dominated domain in every fixture.
    inputs = dict(x=x)
    precise = x.double()
    if spec['family'] == 'E01':
        inverse = torch.rsqrt(precise.square().sum(-1) + eps)
        expected = ((precise * inverse[:, None]).float(), inverse.float())
    else:
        dy = torch.randn(rows, width, generator=rng, dtype=torch.float32)
        inputs['dy'] = dy
        precise.requires_grad_(True)
        y = precise * torch.rsqrt(precise.square().mean(-1, keepdim=True) + eps)
        dx, = torch.autograd.grad(y, precise, dy.double())
        expected = (y.detach().float(), dx.detach().float())
    return dict(inputs=inputs, expected=expected)


def diagnostics(actual, fixture):
    if not isinstance(actual, (list, tuple)) or len(actual) != 2:
        return dict(passed=False, reason='two output tensors required')
    records = []
    for got, expected in zip(actual, fixture['expected']):
        if not isinstance(got, torch.Tensor) or got.device.type != 'cpu' or (
                got.shape != expected.shape or got.dtype != torch.float32):
            return dict(passed=False, reason='CPU FP32 outputs with exact shapes required')
        if not torch.isfinite(got).all():
            return dict(passed=False, reason='nonfinite output')
        error = (got - expected).abs()
        allowed = ATOL + RTOL * expected.abs()
        records.append(dict(passed=bool((error <= allowed).all()),
                            max_absolute_error=error.max().item(),
                            max_tolerance_fraction=(error / allowed).max().item()))
    return dict(passed=all(r['passed'] for r in records), outputs=records,
                rtol=RTOL, atol=ATOL)
