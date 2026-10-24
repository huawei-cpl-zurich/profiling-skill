"""FP32 positive reciprocal-root domain and independent CPU FP64 oracle."""
from pathlib import Path
import torch

ATOL, RTOL = 2e-6, 2e-5
LOW, HIGH = 2.0 ** -8, 2.0 ** 8


def dimensions(spec):
    n, seed = spec.get('N'), spec.get('seed')
    if type(n) is not int or not 1 <= n <= 1048576:
        raise ValueError('N must be an integer in [1,1048576]')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    distribution = spec.get('distribution')
    if distribution not in ('loguniform', 'powers-of-four', 'powers-endpoints-and-neighbors'):
        raise ValueError('unsupported distribution')
    if distribution == 'loguniform':
        lo, hi = spec.get('exp_low'), spec.get('exp_high')
        if any(type(v) is not int for v in (lo, hi)) or not -8 <= lo < hi <= 8:
            raise ValueError('loguniform exponents must satisfy -8 <= low < high <= 8')
    return n, seed


def validate_input(x, n):
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.dtype != torch.float32:
        raise ValueError('CPU FP32 input required')
    if x.shape != (n,) or not x.is_contiguous():
        raise ValueError('contiguous input with declared shape required')
    if not bool((torch.isfinite(x) & (x >= LOW) & (x <= HIGH)).all()):
        raise ValueError('input must be finite and in [2^-8,2^8]')


def reference(x):
    validate_input(x, x.numel())
    return torch.reciprocal(torch.sqrt(x.to(dtype=torch.float64, device='cpu')))


def make_case(spec):
    n, seed = dimensions(spec)
    if spec['distribution'] == 'loguniform':
        rng = torch.Generator(device='cpu').manual_seed(seed)
        exponent = torch.rand(n, generator=rng, dtype=torch.float64, device='cpu')
        exponent = spec['exp_low'] + exponent * (spec['exp_high'] - spec['exp_low'])
        x = torch.exp2(exponent).to(torch.float32)
    else:
        powers = torch.tensor([4.0 ** i for i in range(-4, 5)], dtype=torch.float32, device='cpu')
        values = powers
        if spec['distribution'] == 'powers-endpoints-and-neighbors':
            values = torch.cat((powers, torch.nextafter(powers, torch.zeros_like(powers)),
                                torch.nextafter(powers, torch.full_like(powers, float('inf')))))
            values = values[(values >= LOW) & (values <= HIGH)]
        x = values.repeat((n + len(values) - 1) // len(values))[:n].clone()
    return dict(x=x, expected=reference(x))


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.dtype != torch.float32 or actual.shape != expected.shape:
        return dict(passed=False, reason='CPU FP32 output with declared shape required')
    error = (actual.double() - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool((torch.isfinite(actual) & (error <= allowed)).all()), atol=ATOL, rtol=RTOL,
                max_absolute_error=error.max().item(), max_tolerance_fraction=(error / allowed).max().item())


CASE_SOURCES = [Path(__file__)]
