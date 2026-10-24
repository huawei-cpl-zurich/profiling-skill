"""Frozen FP64 scientific contract; contains no device implementation."""
from pathlib import Path
import torch
ATOL = RTOL = 3e-5
CASE_SOURCES = [Path(__file__)]


def dimensions(spec):
    p = spec.get('parameters', spec)
    rows, width, seed = p.get('rows'), p.get('width'), spec.get('seed')
    if any(type(x) is not int or x <= 0 for x in (rows, width)) or type(seed) is not int or seed < 0:
        raise ValueError('positive integer rows/width and nonnegative integer seed required')
    contexts = p.get('batch_contexts')
    if contexts is not None and (contexts != [1, 7, 32] or rows != 1):
        raise ValueError('compound contexts must be 1,7,32 with one target row')
    return rows, width, seed, contexts


def reference(x):
    x = x.double()
    centered = x - x.max(dim=-1, keepdim=True).values
    return (centered - centered.exp().sum(dim=-1, keepdim=True).log()).float()


def make_case(spec):
    rows, width, seed, contexts = dimensions(spec)
    rng = torch.Generator().manual_seed(seed)
    def sample(n):
        return torch.rand((n, width), generator=rng) * 40 - 20
    if contexts:
        target = 15 + torch.rand((1, width), generator=rng) * 0.0001
        inputs = []
        for n in contexts:
            x = sample(n)
            x[0] = target[0]
            inputs.append(x)
        inputs = tuple(inputs)
    else:
        inputs = (sample(rows),)
    return dict(inputs=inputs, expected=tuple(reference(x) for x in inputs))


def diagnostics(actual, fixture):
    if len(actual) != len(fixture['expected']):
        return dict(passed=False, reason='component count mismatch')
    errors = []
    for out, expected in zip(actual, fixture['expected']):
        if out.device.type != 'cpu' or out.dtype != torch.float32 or out.shape != expected.shape:
            return dict(passed=False, reason='CPU FP32 output shape required')
        error = (out - expected).abs()
        if not torch.isfinite(out).all() or not (error <= ATOL + RTOL * expected.abs()).all():
            return dict(passed=False, reason='fixed FP64 accuracy gate failed')
        errors.append(error.max().item())
    invariant = all(torch.equal(actual[0][0].view(torch.int32), out[0].view(torch.int32)) for out in actual[1:])
    return dict(passed=invariant, batch_invariant_bits=invariant,
                max_absolute_errors=errors, atol=ATOL, rtol=RTOL)
