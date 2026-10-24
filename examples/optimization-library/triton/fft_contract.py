"""Frozen W004 complex FFT fixtures and FP64 mathematical constants."""
from pathlib import Path
import math
import torch


def dimensions(spec):
    b, n = spec.get('batch'), spec.get('N')
    if type(b) is not int or type(n) is not int or not 1 <= b <= 64 or not 1 <= n <= 512:
        raise ValueError('batch must be 1..64 and N must be 1..512')
    return b, n


def factors(n):
    if 32 <= n <= 512 and n & (n - 1) == 0:
        a = 1 << ((n.bit_length() - 1) // 2)
        return a, n // a
    return None


def coefficients(n):
    angle = torch.arange(n, dtype=torch.float64) * (-2 * math.pi / n)
    return angle.cos().float(), angle.sin().float()


def make_case(spec):
    b, n = dimensions(spec)
    gen = torch.Generator().manual_seed(spec['seed'])
    real, imag = [torch.rand((b, n), generator=gen) * .25 - .125 for _ in range(2)]
    if spec.get('distribution') == 'ordering':
        real.zero_(); imag.zero_()
        real[0, 3] = .125
        angle = torch.arange(n, dtype=torch.float64) * (2 * math.pi * 7 / n)
        real[1] = (.125 * angle.cos()).float()
        imag[1] = (.125 * angle.sin()).float()
    x = torch.complex(real.double(), imag.double())
    return dict(inputs=dict(real=real, imag=imag), expected=torch.fft.fft(x), row_l1=x.abs().sum(1))


def validate_inputs(inputs, spec):
    shape = dimensions(spec)
    for key in ('real', 'imag'):
        x = inputs[key]
        if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != shape or not x.is_contiguous() or not torch.isfinite(x).all() or (x.abs() > .125).any():
            raise ValueError('expected bounded contiguous CPU FP32 complex components')


def diagnostics(actual, fixture):
    if not isinstance(actual, tuple) or len(actual) != 2:
        return dict(passed=False, reason='two FP32 output components required')
    real, imag = [v.detach().cpu() for v in actual]
    expected = fixture['expected']
    if any(v.dtype != torch.float32 or v.shape != expected.shape or not torch.isfinite(v).all() for v in (real, imag)):
        return dict(passed=False, reason='output shape/dtype/finite mismatch')
    error = (torch.complex(real.double(), imag.double()) - expected).abs()
    allowed = 2e-6 * fixture['row_l1'][:, None] + 2e-5 * expected.abs()
    return dict(passed=bool((error <= allowed).all()), max_absolute_error=float(error.max()),
                max_tolerance_fraction=float((error / allowed).max()))


CASE_SOURCES = [Path(__file__)]
