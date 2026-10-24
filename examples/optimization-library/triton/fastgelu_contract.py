"""Immutable finite FP32 FastGELU contract and independent FP64 reference."""
from pathlib import Path
import math
import torch
A = float(torch.tensor(1.702, dtype=torch.float32))
HALF = float(torch.tensor(.851, dtype=torch.float32))
assert A == 2 * HALF


def parameters(spec):
    p = spec['parameters']
    n, seed = p['elements'], spec['seed']
    if type(n) is not int or n < 1 or type(seed) is not int or seed < 0:
        raise ValueError('positive integer elements and nonnegative integer seed required')
    distribution = p.get('distribution', 'uniform')
    if distribution not in ('uniform', 'signedzero', 'finite-range-boundaries'):
        raise ValueError('unsupported distribution')
    low, high = p.get('low', -4), p.get('high', 4)
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (low, high)) or not -80 <= low < high <= 80:
        raise ValueError('finite -80 <= low < high <= 80 required')
    if distribution == 'signedzero' and n != 8 or distribution == 'finite-range-boundaries' and n != 16:
        raise ValueError('guard distribution has a fixed size')
    return n, seed, distribution, low, high


def reference(x):
    x = x.to(torch.float64)
    return x / (1 + torch.exp(-A * x))


def make_case(spec):
    n, seed, mode, low, high = parameters(spec)
    if mode == 'signedzero':
        x = torch.tensor([0., -0., 1., -1., 2., -2., 4., -4.], dtype=torch.float32, device='cpu')
    elif mode == 'finite-range-boundaries':
        x = torch.tensor([-80,-75,-65,-60,-55,-52.2,-52,-50,0,1,20,40,50,60,75,80], dtype=torch.float32, device='cpu')
    else:
        rng = torch.Generator(device='cpu').manual_seed(seed)
        x = torch.rand(n, generator=rng, dtype=torch.float32, device='cpu') * (high-low) + low
    return dict(x=x, expected=reference(x))


def diagnostics(actual, fixture):
    expected, x = fixture['expected'], fixture['x']
    if actual.device.type != 'cpu' or actual.dtype != torch.float32 or actual.shape != expected.shape:
        return dict(passed=False, reason='output shape, dtype or device')
    error = (actual.double() - expected).abs()
    tiny = (expected != 0) & (expected.abs() < 1e-30)
    zero = actual == 0
    exp_before = -A * x
    exp_after = HALF * (x - x.abs())
    def count(mask):
        return int(mask.sum())
    return dict(passed=bool(torch.isfinite(actual).all() and torch.allclose(actual.double(), expected, atol=2e-5, rtol=2e-4)),
        max_absolute_error=float(error.max()), atol=2e-5, rtol=2e-4,
        tiny_tail=dict(elements=count(tiny), max_absolute_error=float(error[tiny].max()) if bool(tiny.any()) else None,
            actual_nonzero=count(tiny & ~zero), reference_nonzero=count(tiny),
            actual_subnormal=count(tiny & ~zero & (actual.abs() < torch.finfo(torch.float32).tiny)),
            reference_FP32_normal_range=count(tiny & (expected.abs() >= torch.finfo(torch.float32).tiny)),
            actual_negative_zero=count(tiny & zero & torch.signbit(actual))),
        zeros=dict(negative=count(zero & torch.signbit(actual)), positive=count(zero & ~torch.signbit(actual))),
        expected_intermediate_ranges=dict(before_exp_argument_min=float(exp_before.min()),
            before_exp_argument_max=float(exp_before.max()),
            before_exp_FP32_overflow_arguments=count(exp_before > math.log(torch.finfo(torch.float32).max)),
            after_first_exp_argument_max=float((-A*x.abs()).max()),
            after_second_exp_argument_min=float(exp_after.min()), after_second_exp_argument_max=float(exp_after.max())),
        range_note='Expected source exponent ranges; no device intermediate instrumentation. Subnormal/zero observations come from this sole correctness output.')


CASE_SOURCES = [Path(__file__)]
