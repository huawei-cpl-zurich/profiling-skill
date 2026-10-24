"""E06 scoped causal depthwise convolution: no bias/activation, immutable state.

X[B,T,D], W[D,K], initial[B,D,K-1] and both results are dense FP16.
Weights are ordered from oldest to newest sample. FP32 products/sum precede
FP16 output. Final state is the last K-1 samples of (initial, X).
"""
from pathlib import Path
import torch
import torch.nn.functional as F

CASE_SOURCES = [Path(__file__)]
RTOL = ATOL = 2e-3


def check_spec(spec):
    values = tuple(spec.get(k) for k in ('batch', 'tokens', 'width', 'kernel', 'seed'))
    if spec.get('family') != 'E06' or any(type(v) is not int for v in values):
        raise ValueError('E06 integer dimensions and seed required')
    b, t, d, k, seed = values
    if not (1 <= b <= 32 and 1 <= t <= 128 and 1 <= d <= 4096
            and 2 <= k <= 8 and b*t*d <= 4194304 and 0 <= seed < 2**64):
        raise ValueError('positive bounded dimensions, K in [2,8], uint64 seed required')
    return values


def make_case(spec):
    b, t, d, k, seed = check_spec(spec)
    rng = torch.Generator().manual_seed(seed)
    x = (torch.randn(b, t, d, generator=rng) * .25).half()
    w = (torch.randn(d, k, generator=rng) * .25).half()
    # Strictly nonzero state prevents an omitted prefix from accidentally passing.
    initial = (torch.rand(b, d, k-1, generator=rng) * .5 + .125).half()
    stream = torch.cat((initial.float(), x.float().transpose(1, 2)), dim=2)
    y = F.conv1d(stream, w.float().unsqueeze(1), groups=d).transpose(1, 2).half().contiguous()
    final = stream[:, :, -(k-1):].half().contiguous()
    return dict(inputs=dict(x=x, w=w, initial=initial), expected=(y, final))


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return dict(passed=False, reason='output and final state required')
    records = []
    for i, (got, expected) in enumerate(zip(actual, fixture['expected'])):
        if not isinstance(got, torch.Tensor) or got.device.type != 'cpu' or (
                got.shape != expected.shape or got.dtype != torch.float16):
            return dict(passed=False, reason='CPU FP16 outputs with exact shapes required')
        error = (got.float() - expected.float()).abs()
        allowed = ATOL + RTOL * expected.float().abs()
        passed = torch.equal(got, expected) if i == 1 else bool(
            torch.isfinite(got).all() and (error <= allowed).all())
        records.append(dict(passed=passed, max_absolute_error=error.max().item(),
                            exact_state=i == 1))
    return dict(passed=all(r['passed'] for r in records), outputs=records,
                rtol=RTOL, atol=ATOL)
