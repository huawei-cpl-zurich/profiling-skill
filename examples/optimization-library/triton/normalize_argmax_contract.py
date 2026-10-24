"""Frozen W004-W01 dyadic probability contract; no arbitrary FP32 ordering claim."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec['parameters']
    rows, width, seed = p['rows'], p['V'], spec['seed']
    if any(type(x) is not int for x in (rows, width, seed)):
        raise ValueError('integer rows, V and seed required')
    if rows < 1 or width < 2 or width & (width - 1) or width > 16384 or seed < 0:
        raise ValueError('positive rows, power-of-two V in 2..16384 and nonnegative seed required')
    mode = p.get('distribution', 'unique-winner')
    if mode not in ('unique-winner', 'target-equals-draft', 'dyadic-residual-q-all-one'):
        raise ValueError('unsupported distribution')
    return rows, width, seed, mode


def reference(target, draft, q, normalize=True):
    residual = (target - draft).clamp_min(0)
    if normalize:
        total = residual.sum(-1, keepdim=True)
        residual = residual / torch.where(total > 0, total, torch.ones_like(total))
    return (residual / q).argmax(-1).to(torch.int32)


def make_case(spec):
    rows, width, seed, mode = dimensions(spec)
    rng = torch.Generator().manual_seed(seed)
    target = torch.full((rows, width), 1 / width, dtype=torch.float32)
    draft = torch.full_like(target, 2 / width)
    q = torch.ones_like(target)
    for row in range(rows):
        active = torch.randperm(width, generator=rng)[:width // 2]
        draft[row, active] = 0
        if mode == 'unique-winner':
            q[row, active[torch.randint(len(active), (), generator=rng)]] = .5
    if mode == 'target-equals-draft':
        draft.copy_(target)
    expected = reference(target, draft, q)
    assert torch.equal(expected, reference(target, draft, q, normalize=False))
    return dict(target=target, draft=draft, q=q, expected=expected)


def diagnostics(actual, fixture):
    expected = fixture['expected']
    passed = (actual.device.type == 'cpu' and actual.dtype == torch.int32
              and actual.shape == expected.shape and torch.equal(actual, expected))
    return dict(passed=passed, comparison='bitwise int32 lowest-index argmax')


CASE_SOURCES = [Path(__file__)]
