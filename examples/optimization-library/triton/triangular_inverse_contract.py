"""Bounded unit-lower inverse contract, independent FP64 CPU solve oracle."""
from pathlib import Path
import torch


def dimensions(spec, *, allow_unsupported=False):
    if any(type(spec.get(k)) is not int for k in ('B', 'T', 'H', 'BT', 'seed')):
        raise ValueError('B/T/H/BT/seed must be integers')
    b, t, h, bt = (spec[k] for k in ('B', 'T', 'H', 'BT'))
    if min(b, t, h) < 1 or spec['seed'] < 0 or bt < 1:
        raise ValueError('positive dimensions and nonnegative seed required')
    if not allow_unsupported and bt not in (16, 32, 64, 128):
        raise ValueError('unsupported BT; expected 16/32/64/128')
    bounds = spec.get('segments', [0, t])
    if not isinstance(bounds, list) or len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != t or any(type(x) is not int for x in bounds) or any(y <= x for x, y in zip(bounds, bounds[1:])):
        raise ValueError('segments must partition [0,T] strictly')
    return b, t, h, bt, bounds


def chunks(spec, *, allow_unsupported=False):
    b, t, h, bt, bounds = dimensions(spec, allow_unsupported=allow_unsupported)
    return [(batch, start, min(bt, end-start)) for batch in range(b)
            for begin, end in zip(bounds, bounds[1:]) for start in range(begin, end, bt)]


def make_case(spec):
    # Rejection guards also own a deterministic CPU payload; only their interface
    # rejection, rather than a candidate inverse, is admitted.
    reject = spec.get('expect_rejection', False)
    b, t, h, bt, _ = dimensions(spec, allow_unsupported=reject)
    a = torch.zeros((b, t, h, bt), dtype=torch.float16)
    expected = torch.zeros((b, t, h, bt), dtype=torch.float32)
    generator = torch.Generator().manual_seed(spec['seed'])
    for batch, start, length in chunks(spec, allow_unsupported=reject):
        for head in range(h):
            lower = torch.tril(torch.rand(length, length, generator=generator)*2-1, diagonal=-1)
            # Leave rounding margin so quantization cannot violate <= 0.25.
            lower *= .249 / lower.abs().sum(-1, keepdim=True).clamp_min(1.)
            lower = lower.half()
            a[batch, start:start+length, head, :length] = lower
            expected[batch, start:start+length, head, :length] = torch.linalg.solve(
                torch.eye(length, dtype=torch.float64)+lower.double(),
                torch.eye(length, dtype=torch.float64)).float()
    return dict(inputs=dict(a=a), expected=expected)


def validate_inputs(inputs, spec):
    b, t, h, bt, _ = dimensions(spec)
    a = inputs['a']
    if a.device.type != 'cpu' or a.dtype != torch.float16 or tuple(a.shape) != (b,t,h,bt) or not a.is_contiguous() or not torch.isfinite(a).all():
        raise ValueError('A must be contiguous finite CPU FP16 [B,T,H,BT]')
    for batch, start, length in chunks(spec):
        block = a[batch,start:start+length]
        for head in range(h):
            local = block[:,head,:length].float()
            if torch.count_nonzero(torch.triu(local)) or torch.count_nonzero(block[:,head,length:]) or (local.abs().sum(-1) > .25).any():
                raise ValueError('A must be strict lower with row absolute sum <= .25')


def diagnostics(actual, fixture, spec):
    value = actual.detach().cpu()
    expected = fixture['expected']
    if value.dtype != torch.float32 or value.shape != expected.shape or not torch.isfinite(value).all():
        return dict(passed=False, reason='FP32 finite output with exact shape required')
    error = (value-expected).abs()
    tolerance = .001 + .001*expected.abs()
    residual, structure = 0., True
    for batch, start, length in chunks(spec):
        for head in range(spec['H']):
            x = value[batch,start:start+length,head,:length]
            a = fixture['inputs']['a'][batch,start:start+length,head,:length].double()
            identity = torch.eye(length,dtype=torch.float64)
            residual = max(residual, float(((identity+a)@x.double()-identity).abs().sum(-1).max()))
            structure &= bool(torch.equal(x.diagonal(), torch.ones(length)) and torch.count_nonzero(torch.triu(x,1)) == 0 and torch.count_nonzero(value[batch,start:start+length,head,length:]) == 0)
    return dict(passed=bool((error <= tolerance).all() and residual <= .002 and structure),
                max_absolute_error=float(error.max()), residual_infinity=residual,
                exact_structure=structure, atol=.001, rtol=.001, residual_limit=.002)


CASE_SOURCES = [Path(__file__)]
