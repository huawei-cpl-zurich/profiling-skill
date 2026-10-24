"""Bounded immutable FP32 SwiGLU forward and formal partial-derivative contract."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec['parameters']
    rows, width, seed = p['R'], p['D'], spec['seed']
    if any(type(x) is not int or x < 0 for x in (rows, width, seed)) or not rows or not width:
        raise ValueError('positive integer R,D and nonnegative integer seed required')
    mode = p.get('distribution', 'bounded-uniform')
    if mode not in ('bounded-uniform', 'zero-and-symmetric-a'):
        raise ValueError('unsupported distribution')
    if type(p.get('share_ab_storage', False)) is not bool:
        raise ValueError('share_ab_storage must be boolean')
    return rows, width, seed, mode


def reference(a, b, dc):
    a, b, dc = (x.to(torch.float64) for x in (a, b, dc))
    s = torch.sigmoid(a)
    return (a * s) * b, dc * b * (s + a * s * (1 - s)), dc * a * s


def make_case(spec):
    rows, width, seed, mode = dimensions(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    a, b, dc = (torch.rand((rows, width), generator=rng, dtype=torch.float32, device='cpu') * 8 - 4
                for _ in range(3))
    if mode == 'zero-and-symmetric-a':
        a.copy_((torch.arange(rows * width, dtype=torch.float32, device='cpu') % 9 - 4).reshape(rows, width))
    if spec['parameters'].get('share_ab_storage', False):
        b = a
    return dict(a=a, b=b, dc=dc, expected=reference(a, b, dc))


def diagnostics(actual, fixture):
    if not isinstance(actual, (list, tuple)) or len(actual) != 3:
        return dict(passed=False, reason='expected three outputs')
    errors = []
    for value, expected in zip(actual, fixture['expected']):
        if value.device.type != 'cpu' or value.dtype != torch.float32 or value.shape != expected.shape:
            return dict(passed=False, reason='output dtype, shape or device')
        errors.append(dict(max_abs=float((value.double() - expected).abs().max()),
                           passed=bool(torch.isfinite(value).all() and torch.allclose(
                               value.double(), expected, atol=2e-5, rtol=2e-4))))
    return dict(passed=all(x['passed'] for x in errors), outputs=errors, atol=2e-5, rtol=2e-4)


CASE_SOURCES = [Path(__file__)]
