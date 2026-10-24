"""W04 dyadic chunk-local inclusive prefixes; oracle uses CPU FP64."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec['parameters']
    values = tuple(p[k] for k in ('B', 'T', 'H', 'chunk'))
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('B,T,H,chunk must be positive integers')
    b, t, h, chunk = values
    if b > 8 or t > 2048 or h not in (1, 2, 4, 8) or chunk not in (64, 128):
        raise ValueError('shape outside supported study envelope')
    return values


def reference(x, chunk):
    result = torch.empty_like(x)
    for start in range(0, x.shape[1], chunk):
        result[:, start:start+chunk] = x[:, start:start+chunk].double().cumsum(1).float()
    return result


def make_case(spec):
    b, t, h, chunk = dimensions(spec)
    generator = torch.Generator().manual_seed(spec['seed'])
    x = torch.randint(-8, 9, (b, t, h), generator=generator).float() / 1024
    if spec['parameters'].get('distribution') == 'firstchunk+1/1024,nextchunk-1/1024':
        x[:, :chunk] = 1 / 1024
        x[:, chunk:] = -1 / 1024
    return dict(inputs=x, expected=reference(x, chunk))


def validate_inputs(x, spec):
    b, t, h, _ = dimensions(spec)
    if x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != (b,t,h) or not x.is_contiguous():
        raise ValueError('expected contiguous CPU FP32[B,T,H]')
    scaled = x * 1024
    if not torch.isfinite(x).all() or not ((scaled >= -8) & (scaled <= 8) & (scaled == scaled.round())).all():
        raise ValueError('input must contain integers[-8,8]/1024')


def diagnostics(actual, expected):
    actual = actual.detach().cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return dict(passed=False, reason='shape or dtype mismatch')
    return dict(passed=bool(torch.equal(actual, expected)),
                max_absolute_error=float((actual-expected).abs().max()),
                equality='exact numerical FP32; signed zero equal')


CASE_SOURCES = [Path(__file__)]
