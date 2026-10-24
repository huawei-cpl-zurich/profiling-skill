"""W07 compact LU contract and independent CPU FP64 elimination oracle."""
from pathlib import Path
import torch


def dimensions(spec):
    params = spec.get('parameters', spec)
    n, panel = params.get('N'), params.get('panel')
    if type(n) is not int or not 1 <= n <= 256 or panel != 16:
        raise ValueError('W07 supports square N=1..256 with panel=16')
    if type(spec.get('seed')) is not int or spec['seed'] < 0:
        raise ValueError('nonnegative integer seed required')
    return n, panel


def dense_lu(a, pivot=False):
    """Scalar CPU elimination; permutation indexes rows of the original input."""
    lu = a.double().clone()
    n = len(lu)
    p = torch.arange(n, dtype=torch.int64)
    for k in range(n):
        row = k + int(lu[k:, k].abs().argmax()) if pivot else k
        if float(lu[row, k]) == 0:
            raise ValueError('singular matrix')
        if row != k:
            lu[[k, row]] = lu[[row, k]]
            p[[k, row]] = p[[row, k]]
        lu[k+1:, k] /= lu[k, k]
        lu[k+1:, k+1:] -= lu[k+1:, k:k+1] * lu[k:k+1, k+1:]
    return lu, p


def admitted(a):
    diagonal = a.diagonal()
    off = a.abs().sum(1) - diagonal.abs()
    return bool((diagonal > off).all() and (diagonal > 0).all())


def make_case(spec):
    n, _ = dimensions(spec)
    distribution = spec.get('parameters', spec).get('distribution', 'strictdominance')
    if distribution == 'nonsingular-zero-first-pivot':
        if n != 4 or spec.get('role') != 'guard':
            raise ValueError('pivot fixture is a correctness-only N4 guard')
        a = torch.tensor([[0., 1., .1, 0.], [1., 2., 0., .1],
                          [.1, 0., 2., .1], [0., .1, .1, 2.]])
    elif distribution == 'strictdominance':
        generator = torch.Generator().manual_seed(spec['seed'])
        a = torch.rand((n, n), generator=generator) * 2 - 1
        a.fill_diagonal_(0)
        if n > 1:
            a *= .125 / a.abs().sum(1, keepdim=True)
        a.fill_diagonal_(1.125)
    else:
        raise ValueError('unsupported distribution')
    no_pivot = admitted(a)
    return dict(inputs=dict(a=a), expected=dense_lu(a, pivot=not no_pivot), no_pivot=no_pivot)


def validate_inputs(fixture, spec):
    n, _ = dimensions(spec)
    a = fixture['inputs']['a']
    if a.device.type != 'cpu' or a.dtype != torch.float32 or a.shape != (n, n) or not a.is_contiguous() or not torch.isfinite(a).all():
        raise ValueError('finite contiguous CPU square FP32 input required')
    # Admission is CPU-only and precedes all forwards; no data-dependent host
    # inspection or dispatch occurs in an NPU profiled forward.
    no_pivot = admitted(a)
    if no_pivot != fixture['no_pivot']:
        raise ValueError('fixture admission changed')
    if not no_pivot and spec.get('role') != 'guard':
        raise ValueError('partial pivot fallback is correctness-only')
    return no_pivot


def diagnostics(actual, fixture, spec):
    n, _ = dimensions(spec)
    lu, p = (x.detach().cpu() for x in actual)
    expected, expected_p = fixture['expected']
    if lu.shape != (n, n) or lu.dtype != torch.float32 or p.shape != (n,) or p.dtype != torch.int64 or not torch.isfinite(lu).all():
        return dict(passed=False, reason='shape, dtype or nonfinite output')
    error = (lu.double() - expected).abs()
    tolerance = 2e-5 + 2e-4 * expected.abs()
    if not torch.equal(p, expected_p):
        return dict(passed=False, reason='permutation mismatch')
    lower = torch.tril(lu.double(), -1) + torch.eye(n, dtype=torch.float64)
    upper = torch.triu(lu.double())
    a = fixture['inputs']['a'].double()
    residual = float((a[p] - lower @ upper).abs().sum(1).max() / a.abs().sum(1).max())
    return dict(passed=bool((error <= tolerance).all()) and residual <= 2e-5,
                max_absolute_error=float(error.max()), relative_infinity_residual=residual,
                atol=2e-5, rtol=2e-4, residual_limit=2e-5)


CASE_SOURCES = [Path(__file__)]
