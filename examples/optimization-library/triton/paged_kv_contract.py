"""CPU FP64 oracle for causal grouped-query attention over shuffled pages."""
from pathlib import Path
import math
import torch

RTOL = ATOL = 1e-3
KEYS = ('B', 'Hq', 'Hkv', 'Q', 'KV', 'D', 'Dv', 'page', 'seed')


def dimensions(spec):
    values = tuple(spec.get(k) for k in KEYS)
    if any(type(v) is not int for v in values):
        raise ValueError('integer dimensions and seed required')
    b, hq, hkv, q, kv, d, dv, page, seed = values
    if not (1 <= b <= 8 and 1 <= hkv <= hq <= 16 and hq % hkv == 0
            and 1 <= q <= 128 and 0 <= kv <= 1024 and (kv == 0 or q <= kv)
            and d in (32, 64) and dv in (32, 64) and page == 16 and 0 <= seed < 2**63):
        raise ValueError('unsupported paged attention envelope')
    return values


def reference(q, k_pages, v_pages, table, kv):
    b, hq, nq, d = q.shape
    dv, hkv = v_pages.shape[-1], k_pages.shape[-2]
    expected = torch.zeros((b, hq, nq, dv), dtype=torch.float64)
    if kv == 0:
        return expected
    for batch in range(b):
        k = k_pages[table[batch].long()].reshape(-1, hkv, d)[:kv].double()
        v = v_pages[table[batch].long()].reshape(-1, hkv, dv)[:kv].double()
        for head in range(hq):
            kh = head // (hq // hkv)
            for row in range(nq):
                end = kv - nq + row + 1
                scores = k[:end, kh] @ q[batch, head, row].double() / math.sqrt(d)
                expected[batch, head, row] = torch.softmax(scores, 0) @ v[:end, kh]
    return expected


def make_case(spec):
    b, hq, hkv, q, kv, d, dv, page, seed = dimensions(spec)
    rng = torch.Generator().manual_seed(seed)
    pages = max(1, (kv + page - 1) // page)
    def values(shape):
        return (torch.rand(shape, generator=rng) * 0.5 - 0.25).half()
    query = values((b, hq, q, d))
    key = values((b * pages, page, hkv, d))
    value = values((b * pages, page, hkv, dv))
    table = torch.randperm(b * pages, generator=rng).reshape(b, pages).int()
    return dict(q=query, k=key, v=value, table=table,
                expected=reference(query, key, value, table, kv))


def check_inputs(spec, fixture):
    b, hq, hkv, q, kv, d, dv, page, _ = dimensions(spec)
    pages = max(1, (kv + page - 1) // page)
    for name, shape, dtype in (
        ('q', (b, hq, q, d), torch.float16),
        ('k', (b * pages, page, hkv, d), torch.float16),
        ('v', (b * pages, page, hkv, dv), torch.float16),
        ('table', (b, pages), torch.int32)):
        value = fixture[name]
        if value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError('invalid contiguous CPU fixture: ' + name)
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite input: ' + name)
    table = fixture['table'].flatten()
    if not torch.equal(table.sort().values, torch.arange(b * pages, dtype=torch.int32)):
        raise ValueError('page table must permute fully allocated pages')


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.device.type != 'cpu' or actual.shape != expected.shape or actual.dtype != torch.float32:
        return dict(passed=False, reason='expected CPU FP32 output with declared shape')
    if not torch.isfinite(actual).all():
        return dict(passed=False, reason='nonfinite or unwritten output')
    error = (actual.double() - expected).abs()
    allowed = ATOL + RTOL * expected.abs()
    return dict(passed=bool((error <= allowed).all()), atol=ATOL, rtol=RTOL,
                max_absolute_error=error.max().item(), max_tolerance_fraction=(error / allowed).max().item())


CASE_SOURCES = [Path(__file__)]
