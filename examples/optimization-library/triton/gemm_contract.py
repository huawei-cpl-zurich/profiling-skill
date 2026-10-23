"""CPU-only GEMM fixtures shared by Triton and native adapters."""
import math

import torch


def check_spec(spec):
    if spec.get('family') not in ('02', '05', '12'):
        raise ValueError('family must be 02, 05 or 12')
    for key in ('m', 'n', 'k'):
        if type(spec.get(key)) is not int or not 0 < spec[key] <= 32768:
            raise ValueError('dimensions must be positive integers at most 32768')
    if spec.get('b_layout', 'row') not in ('row', 'column'):
        raise ValueError('B layout must be row or column')
    if type(spec.get('seed', 0)) is not int:
        raise ValueError('seed must be an integer')


def make_case(spec):
    """Generate FP16 operands once; FP32 CPU accumulation is independent of JIT."""
    check_spec(spec)
    m, n, k = (spec[key] for key in ('m', 'n', 'k'))
    generator = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    # Scaling bounds error magnitude without hiding zeros or cancellation.
    a = (torch.randn(m, k, generator=generator) / math.sqrt(k)).half()
    b = torch.randn(k, n, generator=generator).half()
    if spec.get('b_layout', 'row') == 'column':
        b = b.T.contiguous().T
    expected = a.float() @ b.float().contiguous()
    inputs = dict(a=a, b=b)
    if spec['family'] == '02':
        inputs['z'] = torch.randn(m, n, generator=generator)
        expected = expected + inputs['z']
    else:
        expected = expected.half()
    return dict(inputs=inputs, expected=expected)


def validate_actual(actual, fixture, device_inputs=None):
    """Inspect outputs and inputs only after the sole correctness forward."""
    expected = fixture['expected']
    result = actual.detach().cpu()
    if result.shape != expected.shape or result.dtype != expected.dtype:
        raise ValueError('output shape or dtype differs from contract')
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite output')
    delta = (result.float() - expected.float()).abs()
    passed = bool(torch.all(delta <= 1e-3 + 1e-3 * expected.float().abs()))
    if device_inputs is not None:
        for name, original in fixture['inputs'].items():
            if not torch.equal(device_inputs[name].detach().cpu(), original):
                raise ValueError('input mutated: ' + name)
    return dict(passed=passed, max_absolute_error=float(delta.max()),
                max_relative_error=float((delta / expected.float().abs().clamp_min(1e-30)).max()),
                rtol=1e-3, atol=1e-3)
