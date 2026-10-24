"""Independent FP32 CPU attention contract for the query-blocking gallery."""
from pathlib import Path
import math

import torch

RTOL = ATOL = 1e-3


def dimensions(spec):
    keys = ('b', 'h', 'q', 'k', 'd', 'seed')
    values = tuple(spec.get(key) for key in keys)
    if any(type(value) is not int for value in values):
        raise ValueError('b, h, q, k, d and seed must be integers')
    b, h, q, k, d, seed = values
    if b != 1 or h not in (4, 8) or not 1 <= q <= 1536 or not 1 <= k <= 1536 or d not in (64, 128) or not 0 <= seed < 2**64:
        raise ValueError('supported envelope: B=1, H=4/8, Q/K=1..1536, D=64/128, 0<=seed<2**64')
    return values


def reference(q, k, v):
    """No candidate blocking, probability conversion or online recurrence."""
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def make_case(spec):
    b, h, nq, nk, d, seed = dimensions(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    q = (torch.randn(b, h, nq, d, generator=rng) * 0.5).half()
    k = (torch.randn(b, h, nk, d, generator=rng) * 0.5).half()
    v = torch.randn(b, h, nk, d, generator=rng).half()
    return dict(q=q, k=k, v=v, expected=reference(q, k, v))


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.shape != expected.shape or actual.dtype != torch.float32:
        return dict(passed=False, reason='expected CPU FP32 output with declared shape')
    if not torch.isfinite(actual).all():
        return dict(passed=False, reason='nonfinite output')
    error = (actual - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool((error <= allowed).all()), rtol=RTOL, atol=ATOL,
                max_absolute_error=error.max().item(),
                max_relative_error=(error / expected.abs().clamp_min(1e-12)).max().item(),
                max_tolerance_fraction=(error / allowed).max().item())


CASE_SOURCES = [Path(__file__)]
