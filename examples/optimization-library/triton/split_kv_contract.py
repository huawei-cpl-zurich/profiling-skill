"""Frozen F07 decode attention fixtures and independent dense FP64 oracle."""
from pathlib import Path
import torch

ATOL = RTOL = 2e-3
CASE_SOURCES = [Path(__file__)]


def dimensions(spec):
    p = spec['parameters']
    keys = ('B', 'Hq', 'Hkv', 'Q', 'KV', 'D', 'Dv', 'split', 'blockN')
    if any(type(p[k]) is not int for k in keys):
        raise ValueError('integer dimensions required')
    b, hq, hkv, q, kv, d, dv, split, bn = (p[k] for k in keys)
    if not (b > 0 and hkv > 0 and hq % hkv == 0 and hq // hkv <= 16
            and q == 1 and kv >= 0 and d in (32, 64) and dv in (32, 64)
            and split == 4 and bn == 64):
        raise ValueError('requires contiguous decode, GQA group<=16, D/Dv32/64, split4, BN64')
    return b, hq, hkv, kv, d, dv


def reference(q, k, v):
    if k.shape[-2] == 0:
        return torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=torch.float64)
    group = q.shape[1] // k.shape[1]
    keys = k.double().repeat_interleave(group, dim=1)
    values = v.double().repeat_interleave(group, dim=1)
    score = q.double() @ keys.transpose(-1, -2) * q.shape[-1] ** -0.5
    return score.softmax(-1) @ values


def make_case(spec):
    b, hq, hkv, kv, d, dv = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    def sample(shape):
        return (torch.rand(shape, generator=rng) * .5 - .25).half()
    q = sample((b, hq, 1, d))
    k = sample((b, hkv, kv, d))
    v = sample((b, hkv, kv, dv))
    return dict(q=q, k=k, v=v, expected=reference(q, k, v))


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.shape != expected.shape or actual.dtype != torch.float16:
        return dict(passed=False, reason='expected FP16 decode output')
    error = (actual.double() - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool(torch.isfinite(actual).all() and (error <= allowed).all()),
                atol=ATOL, rtol=RTOL, max_absolute_error=error.max().item(),
                max_tolerance_fraction=(error / allowed).max().item())
