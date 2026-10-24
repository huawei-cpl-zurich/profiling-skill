"""Frozen W03 real-input contract; independent CPU complex128 oracle."""
from pathlib import Path
import torch


def normalize_case(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get('parameters', {}), dict):
        raise ValueError('case and parameters must be objects')
    parameters = spec.get('parameters', {})
    if any(k in spec and spec[k] != v for k, v in parameters.items()):
        raise ValueError('conflicting flat and nested parameters')
    result = {k: v for k, v in spec.items() if k != 'parameters'}
    result.update(parameters)
    distribution = result.setdefault('distribution', 'bounded-uniform')
    if distribution not in ('bounded-uniform', 'dc-constant-and-alternating-nyquist'):
        raise ValueError('unknown distribution')
    return result


def dimensions(spec):
    spec = normalize_case(spec)
    b, n = spec.get('batch'), spec.get('N')
    if type(b) is not int or type(n) is not int or not 1 <= b <= 64 or not 1 <= n <= 1024:
        raise ValueError('batch must be 1..64 and N must be 1..1024')
    if spec['distribution'] != 'bounded-uniform' and (b != 2 or n % 2):
        raise ValueError('DC/Nyquist guard requires two rows and even N')
    algorithm = 'two-stage' if n >= 64 and n & (n-1) == 0 else 'direct-dft'
    if spec.get('common_complex_algorithm', algorithm) != algorithm:
        raise ValueError('common complex algorithm disagrees with fixed dispatch')
    return b, n


def case_inputs(spec):
    """Build only CPU FP32 inputs, independently of ambient Torch defaults."""
    spec = normalize_case(spec)
    b, n = dimensions(spec)
    x = torch.rand((b, n), dtype=torch.float32, device='cpu',
                   generator=torch.Generator(device='cpu').manual_seed(spec['seed'])) * .25 - .125
    if spec['distribution'] != 'bounded-uniform':
        x[0].fill_(.125)
        x[1, ::2] = .125
        x[1, 1::2] = -.125
    return dict(real=x)


def make_case(spec):
    inputs = case_inputs(spec)
    x = inputs['real']
    return dict(inputs=inputs, expected=torch.fft.rfft(x.double()), row_l1=x.double().abs().sum(1), N=x.shape[1])


def validate_inputs(inputs, spec):
    x = inputs['real']
    if (x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != dimensions(spec)
            or not x.is_contiguous() or not torch.isfinite(x).all() or (x.abs() > .125).any()):
        raise ValueError('bounded contiguous CPU FP32[B,N] required')


def diagnostics(actual, fixture):
    if not isinstance(actual, tuple) or len(actual) != 2:
        return dict(passed=False, reason='two FP32 output components required')
    real, imag = [v.detach().cpu() for v in actual]
    expected = fixture['expected']
    if any(v.dtype != torch.float32 or v.shape != expected.shape or not torch.isfinite(v).all() for v in (real, imag)):
        return dict(passed=False, reason='output shape/dtype/finite mismatch')
    error = (torch.complex(real.double(), imag.double()) - expected).abs()
    allowed = 2e-6 * fixture['row_l1'][:, None] + 2e-5 * expected.abs()
    endpoints = bool((imag[:, 0] == 0).all() and (fixture['N'] % 2 or (imag[:, -1] == 0).all()))
    return dict(passed=bool((error <= allowed).all()) and endpoints, endpoints_exact_zero=endpoints,
                max_absolute_error=float(error.max()), max_tolerance_fraction=float((error / allowed.clamp_min(1e-300)).max()))


CASE_SOURCES = [Path(__file__)]
