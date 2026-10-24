"""One prescribed Jacobi round; block-size-one adaptation, not an eigensolver."""
from pathlib import Path
import torch


def parameters(spec):
    p = spec.get('parameters', spec)
    n, r = p.get('N'), p.get('round_index', 0)
    if type(n) is not int or not 2 <= n <= 128 or type(r) is not int:
        raise ValueError('N must be 2..128 and round_index an integer')
    if p.get('block_size', 1) != 1 or not 0 <= r < n + n % 2 - 1:
        raise ValueError('unsupported block size or round')
    return n, r


def round_pairs(n, r):
    ring = n + n % 2
    def rotate(slot):
        return 0 if slot == 0 else 1 + (slot - 1 + r) % (ring - 1)
    return [sorted((rotate(i), rotate(ring - 1 - i)))
            for i in range(ring // 2) if max(rotate(i), rotate(ring - 1 - i)) < n]


def pair_map(spec):
    n, r = parameters(spec)
    expected = round_pairs(n, r)
    p = spec.get('parameters', spec)
    pairs = p.get('pairs', expected)
    if p.get('invalid_duplicate_pair_map'):
        pairs = expected + expected[:1]
    if pairs != expected:
        raise ValueError('pair map must be the prescribed disjoint round')
    mate, sign = list(range(n)), [0] * n
    for left, right in pairs:
        mate[left], mate[right] = right, left
        sign[left], sign[right] = -1, 1
    return pairs, mate, sign


def reference(a, pairs):
    """Independent dense FP64 product of embedded orthogonal blocks."""
    a = a.double()
    j = torch.eye(a.shape[0], dtype=torch.float64)
    for p, q in pairs:
        apq = float(a[p, q])
        t = 0.
        if apq != 0:
            tau = float((a[q, q] - a[p, p]) / (2 * apq))
            t = (1. if tau >= 0 else -1.) / (abs(tau) + (1 + tau*tau)**.5)
        c = (1 + t*t)**-.5
        j[p,p] = j[q,q] = c
        j[p,q], j[q,p] = t*c, -t*c
    return j.T @ a @ j


def make_case(spec):
    n, r = parameters(spec)
    # The host-negative guard still has a sixth CPU oracle, never a device run.
    pairs = round_pairs(n, r)
    g = torch.Generator().manual_seed(spec['seed'])
    raw = torch.rand((n, n), generator=g) * .25 - .125
    a = ((raw + raw.T) / 2).contiguous()
    a.diagonal().copy_(1 + torch.rand(n, generator=g) * .25 - .125)
    return dict(a=a, expected=reference(a, pairs))


def validate_input(spec, fixture):
    pairs, mate, sign = pair_map(spec)  # Metadata rejection precedes device imports.
    n, _ = parameters(spec)
    a = fixture['a']
    if (a.device.type != 'cpu' or a.dtype != torch.float32 or a.shape != (n,n)
            or not a.is_contiguous() or not torch.isfinite(a).all() or not torch.equal(a,a.T)):
        raise ValueError('finite symmetric contiguous CPU FP32 matrix required')
    if (a.diagonal() < .875).any() or (a.diagonal() > 1.125).any():
        raise ValueError('diagonal outside bounded fixture envelope')
    off = a.clone(); off.diagonal().zero_()
    if off.abs().max() > .125:
        raise ValueError('off-diagonal outside bounded fixture envelope')
    return pairs, mate, sign


def diagnostics(actual, fixture):
    v = actual.detach().cpu()
    a, expected = fixture['a'].double(), fixture['expected']
    if v.dtype != torch.float32 or v.shape != expected.shape or not torch.isfinite(v).all():
        return dict(passed=False, reason='shape/dtype/nonfinite output')
    v = v.double()
    err = (v - expected).abs()
    norm = float(abs(v.norm() - a.norm()) / a.norm())
    symmetry = float((v-v.T).abs().max())
    trace = float(abs(v.trace() - a.trace()))
    return dict(passed=bool((err <= 2e-5 + 2e-4*expected.abs()).all()) and norm <= 2e-5
                and symmetry <= 2e-5 and trace <= 2e-5*a.shape[0],
                max_absolute_error=float(err.max()), relative_frobenius_error=norm,
                symmetry_max_absolute=symmetry, trace_absolute_error=trace, atol=2e-5, rtol=2e-4)


CASE_SOURCES = [Path(__file__)]
