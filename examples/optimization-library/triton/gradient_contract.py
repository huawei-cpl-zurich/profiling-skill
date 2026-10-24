"""Independent FP64 oracle for the complete weighted RMS input/weight VJP."""
from pathlib import Path
import math
import torch

RTOL, ATOL = 2e-4, 2e-3
CASE_SOURCES = [Path(__file__)]


def check_spec(spec):
    rows, width, seed = (spec.get(key) for key in ('rows', 'width', 'seed'))
    if spec.get('family') != 'E08' or any(type(v) is not int for v in (rows, width, seed)):
        raise ValueError('E08 integer rows, width and seed required')
    if not (1 <= rows <= 16384 and 1 <= width <= 4096 and rows * width <= 2097152
            and 0 <= seed < 2**64):
        raise ValueError('positive bounded matrix dimensions and uint64 seed required')
    eps = spec.get('eps', 1e-5)
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 0 < eps <= .1:
        raise ValueError('finite epsilon in (0,.1] required')
    return rows, width, seed, float(eps)


def make_case(spec):
    rows, width, seed, eps = check_spec(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(rows, width, generator=rng, dtype=torch.float32)
    x[::17].zero_()
    w = torch.randn(width, generator=rng, dtype=torch.float32)
    dy = torch.randn(rows, width, generator=rng, dtype=torch.float32)
    precise_x, precise_w = x.double().requires_grad_(), w.double().requires_grad_()
    y = precise_x * torch.rsqrt(precise_x.square().mean(-1, keepdim=True) + eps) * precise_w
    dx, dw = torch.autograd.grad(y, (precise_x, precise_w), dy.double())
    return dict(inputs=dict(x=x, w=w, dy=dy), expected=(dx.float(), dw.float()))


def diagnostics(actual, fixture):
    if not isinstance(actual, (list, tuple)) or len(actual) != 2:
        return dict(passed=False, reason='both input and weight gradients required')
    records = []
    for got, expected in zip(actual, fixture['expected']):
        if not isinstance(got, torch.Tensor) or got.device.type != 'cpu' or (
                got.shape != expected.shape or got.dtype != torch.float32):
            return dict(passed=False, reason='exact CPU FP32 gradient shapes required')
        if not torch.isfinite(got).all():
            return dict(passed=False, reason='nonfinite gradient')
        error = (got - expected).abs()
        allowed = ATOL + RTOL * expected.abs()
        records.append(dict(passed=bool((error <= allowed).all()),
                            max_absolute_error=error.max().item(),
                            max_tolerance_fraction=(error / allowed).max().item()))
    return dict(passed=all(item['passed'] for item in records), outputs=records,
                rtol=RTOL, atol=ATOL)
