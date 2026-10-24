"""Explicit approximation permission and immutable FP32 GELU fixtures."""
from pathlib import Path
import math
import torch


def parameters(spec):
    p = spec['parameters']
    n, seed = p['elements'], spec['seed']
    if type(n) is not int or n < 1 or type(seed) is not int or seed < 0:
        raise ValueError('positive integer elements and nonnegative seed required')
    allowed = p.get('allow_approx', True)
    if type(allowed) is not bool:
        raise ValueError('allow_approx must be bool')
    mode = p.get('distribution', 'uniform')
    if mode not in ('uniform', 'endpoints-nearzero'):
        raise ValueError('unsupported distribution')
    low, high = p.get('low', -4), p.get('high', 4)
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (low, high)) or not -8 <= low < high <= 8:
        raise ValueError('finite -8 <= low < high <= 8 required')
    if mode == 'endpoints-nearzero' and n != 16:
        raise ValueError('endpoint guard requires 16 elements')
    return n, seed, allowed, mode, low, high


def reference(x):
    x = x.to(torch.float64)
    return .5 * x * (1 + torch.erf(x / math.sqrt(2)))


def approximation(x):
    x = x.to(torch.float64)
    return .5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + .044715*x*x*x)))


def grid_check():
    x = torch.linspace(-8, 8, 160001, dtype=torch.float64, device='cpu')
    error = (reference(x) - approximation(x)).abs()
    maximum, index = error.max(dim=0)
    return dict(passed=bool(maximum <= 6e-4), points=x.numel(), low=-8, high=8,
                maximum_absolute_error=float(maximum), maximum_at_x=float(x[index]),
                budget=6e-4, interpretation='Empirical finite grid, not a global proof.')


def make_case(spec):
    n, seed, _, mode, low, high = parameters(spec)
    if mode == 'endpoints-nearzero':
        x = torch.tensor([-8,-6,-4,-2,-1,-2**-10,-2**-20,-0.,0.,2**-20,2**-10,1,2,4,6,8],
                         dtype=torch.float32, device='cpu')
    else:
        rng = torch.Generator(device='cpu').manual_seed(seed)
        x = torch.rand(n, generator=rng, dtype=torch.float32, device='cpu')*(high-low)+low
    return dict(x=x, expected=reference(x))


def diagnostics(actual, fixture, arm, allowed):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.dtype != torch.float32 or actual.shape != expected.shape:
        return dict(passed=False, reason='output shape, dtype or device')
    exact = arm == 'before' or not allowed
    atol, rtol = (2e-5, 2e-4) if exact else (6.2e-4, 0)
    error = (actual.double() - expected).abs()
    return dict(passed=bool(torch.isfinite(actual).all() and torch.allclose(actual.double(), expected, atol=atol, rtol=rtol)),
                max_absolute_error=float(error.max()), atol=atol, rtol=rtol,
                dispatch='erf' if exact else 'permitted-tanh-approximation',
                approximation_budget=0 if exact else 6e-4, arithmetic_allowance=2e-5)


CASE_SOURCES = [Path(__file__)]
