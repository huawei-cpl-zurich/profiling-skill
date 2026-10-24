"""E09 complete split-half Q/K rotary with identical FP16 coefficient rounding."""
from pathlib import Path
import math
import torch

CASE_SOURCES = [Path(__file__)]
RTOL = ATOL = 2e-3


def check_spec(spec):
    values = tuple(spec.get(k) for k in ('batch', 'tokens', 'q_heads', 'k_heads', 'width', 'seed'))
    if spec.get('family') != 'E09' or any(type(v) is not int for v in values):
        raise ValueError('E09 integer dimensions and seed required')
    b, t, hq, hk, d, seed = values
    scale = spec.get('scale', 2.0)
    if (not 1 <= b <= 8 or not 1 <= t <= 8192 or not 1 <= hq <= 64
            or not 1 <= hk <= 16 or d not in (32, 64, 128, 256)
            or b*t*(hq+hk)*d > 33554432 or not 0 <= seed < 2**64
            or type(scale) not in (int, float) or not math.isfinite(scale) or not .125 <= scale <= 128.):
        raise ValueError('bounded positive dimensions, even power-of-two width and finite scale required')
    return (*values, float(scale))


def make_case(spec):
    b, t, hq, hk, d, seed, scale = check_spec(spec)
    rng = torch.Generator().manual_seed(seed)
    q = (torch.randn(b, t, hq, d, generator=rng)*.25).half()
    k = (torch.randn(b, t, hk, d, generator=rng)*.25).half()
    positions = torch.randint(0, 2048, (b, t), generator=rng, dtype=torch.int32)
    freq = torch.pow(10000., -2*torch.arange(d//2, dtype=torch.float32)/d)
    angles = positions.float()[:, :, None] / scale * freq[None, None, :]
    # Coefficient cast occurs once before FP32 products, identically under both arms.
    cos, sin = angles.cos().half().float()[:, :, None, :], angles.sin().half().float()[:, :, None, :]
    results = []
    for x in (q, k):
        lo, hi = x.float().chunk(2, dim=-1)
        results.append(torch.cat((lo*cos-hi*sin, hi*cos+lo*sin), dim=-1).half())
    return dict(inputs=dict(q=q, k=k, positions=positions, freq=freq), expected=tuple(results))


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return dict(passed=False, reason='complete Q and K outputs required')
    rows = []
    for got, expected in zip(actual, fixture['expected']):
        if not isinstance(got, torch.Tensor) or got.device.type != 'cpu' or (
                got.dtype != torch.float16 or got.shape != expected.shape):
            return dict(passed=False, reason='CPU FP16 outputs with exact shapes required')
        error = (got.float()-expected.float()).abs()
        rows.append(dict(passed=bool(torch.isfinite(got).all()
            and (error <= ATOL+RTOL*expected.float().abs()).all()),
            max_absolute_error=error.max().item()))
    return dict(passed=all(row['passed'] for row in rows), outputs=rows, rtol=RTOL, atol=ATOL)
