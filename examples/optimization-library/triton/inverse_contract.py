"""CPU contract for the W004 inverse adjacency ownership experiment."""
from pathlib import Path
import torch

DISTRIBUTIONS = ('half-to-destination-zero-remaining-uniform',
                 'one-edge-per-destination-permutation',
                 'duplicates-and-empty-destinations', 'seededpermuted-edge-order')


def dimensions(spec):
    p = spec.get('parameters', spec)
    if any(type(p.get(k)) is not int for k in ('E', 'N', 'D')) or type(spec.get('seed')) is not int:
        raise ValueError('integer E,N,D and seed required')
    e, n, d = (p[k] for k in ('E', 'N', 'D'))
    if not (1 <= e <= 16384 and 1 <= n <= 1024 and d in (4, 8, 16, 32)) or spec['seed'] < 0:
        raise ValueError('unsupported dimensions or seed')
    distribution = p.get('distribution')
    if distribution not in DISTRIBUTIONS:
        raise ValueError('unsupported distribution')
    if distribution == DISTRIBUTIONS[1] and e != n:
        raise ValueError('one edge per destination requires E=N')
    return e, n, d, distribution


def reference(inputs, n):
    # Independent sequential FP64 accumulation counts each duplicate edge.
    result = torch.zeros((n, inputs['values'].shape[1]), dtype=torch.float64)
    for dest, value in zip(inputs['dest'].tolist(), inputs['values'].double()):
        result[dest] += value
    return result.float()


def make_case(spec):
    e, n, d, distribution = dimensions(spec)
    g = torch.Generator().manual_seed(spec['seed'])
    if distribution == DISTRIBUTIONS[1]:
        dest = torch.randperm(n, generator=g)
    elif distribution == DISTRIBUTIONS[2]:
        dest = torch.arange(e) % max(1, n // 2)
    else:
        dest = torch.randint(n, (e,), generator=g)
        if distribution == DISTRIBUTIONS[0]:
            dest[:e // 2] = 0
        else:
            dest = dest[torch.randperm(e, generator=g)]
    inputs = dict(dest=dest.to(torch.int32), values=torch.rand((e, d), generator=g) * .25 - .125)
    return dict(inputs=inputs, expected=reference(inputs, n))


def validate_inputs(inputs, spec):
    e, n, d, _ = dimensions(spec)
    for key, shape, dtype in [('dest', (e,), torch.int32), ('values', (e, d), torch.float32)]:
        value = inputs[key]
        if value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError('invalid contiguous CPU input: ' + key)
    if not ((inputs['dest'] >= 0) & (inputs['dest'] < n)).all():
        raise ValueError('destination outside [0,N)')
    if not torch.isfinite(inputs['values']).all() or not (inputs['values'].abs() <= .125).all():
        raise ValueError('values outside finite FP32 envelope')


def diagnostics(actual, fixture):
    value = actual.detach().cpu()
    expected = fixture['expected']
    if value.shape != expected.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
        return dict(passed=False, reason='output shape/dtype/nonfinite')
    error = (value - expected).abs()
    occupied = torch.zeros(expected.shape[0], dtype=torch.bool)
    occupied[fixture['inputs']['dest'].long()] = True
    empty_exact = bool((value[~occupied] == 0).all())
    return dict(passed=bool(torch.allclose(value, expected, atol=.002, rtol=.002)) and empty_exact,
                empty_rows_exact_zero=empty_exact, max_absolute_error=float(error.max()), atol=.002, rtol=.002)


CASE_SOURCES = [Path(__file__)]
