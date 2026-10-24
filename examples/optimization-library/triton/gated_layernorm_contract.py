"""Frozen full-row FP32 gated LayerNorm study; independent FP64 oracle."""
from pathlib import Path
import math
import torch

INPUTS = ('x', 'z', 'gamma', 'beta')


def parameters(spec):
    p = spec['parameters']
    r, n, seed = p['R'], p['N'], spec['seed']
    if any(type(v) is not int for v in (r, n, seed)):
        raise ValueError('R, N and seed must be integers')
    if not 1 <= r <= 2048 or not 2 <= n <= 2048 or seed < 0:
        raise ValueError('R in 1..2048, N in 2..2048 and nonnegative seed required')
    first, eps = p['gate_first'], p.get('eps', 1e-5)
    if type(first) is not bool or type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('boolean gate_first and positive finite eps required')
    distribution = p.get('distribution', 'bounded-uniform')
    if distribution not in ('bounded-uniform', 'constant-x-one-nonconstant-z'):
        raise ValueError('unsupported distribution')
    return r, n, seed, first, eps, distribution


def reference(x, z, gamma, beta, gate_first, eps):
    x, z, gamma, beta = (v.double() for v in (x, z, gamma, beta))
    u = z * torch.sigmoid(z)
    v = x * u if gate_first else x
    mean = v.mean(-1)
    centered = v - mean[:, None]
    rstd = (centered.square().mean(-1) + eps).rsqrt()
    y = centered * rstd[:, None] * gamma + beta
    return y if gate_first else y * u, mean, rstd


def make_case(spec):
    r, n, seed, first, eps, distribution = parameters(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    def uniform(shape, lo, hi):
        return torch.rand(shape, generator=rng, dtype=torch.float32, device='cpu') * (hi-lo) + lo
    x, z = uniform((r, n), -2, 2), uniform((r, n), -2, 2)
    gamma, beta = uniform((n,), .5, 1.5), uniform((n,), -.25, .25)
    if distribution == 'constant-x-one-nonconstant-z':
        x.fill_(1)
    expected = reference(x, z, gamma, beta, first, eps)
    return dict(x=x, z=z, gamma=gamma, beta=beta, expected=expected)


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 3:
        return dict(passed=False, reason='expected y, mean, rstd')
    metrics = {}
    for name, value, expected in zip(('y', 'mean', 'rstd'), actual, fixture['expected']):
        valid = (value.device.type == 'cpu' and value.dtype == torch.float32
                 and value.shape == expected.shape and bool(torch.isfinite(value).all()))
        if not valid:
            return dict(passed=False, reason='invalid ' + name)
        error = (value.double() - expected).abs()
        metrics[name] = dict(max_abs_error=float(error.max()),
                            passed=bool(torch.all(error <= 5e-5 + 3e-4 * expected.abs())))
    return dict(passed=all(v['passed'] for v in metrics.values()), metrics=metrics,
                tolerance=dict(atol=5e-5, rtol=3e-4), oracle='CPU FP64')


CASE_SOURCES = [Path(__file__)]
