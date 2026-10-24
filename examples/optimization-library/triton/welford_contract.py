"""Frozen FP64 oracle and FP32 population-statistics contract for F08."""
from pathlib import Path
import torch

TOLERANCES = ((2e-5, 2e-6), (1e-7, 2e-3))  # (atol, rtol): mean, variance


def dimensions(spec):
    values = tuple(spec.get(k) for k in ('N', 'C', 'H', 'W', 'seed'))
    if any(type(v) is not int for v in values):
        raise ValueError('N, C, H, W and seed must be integers')
    n, c, h, w, seed = values
    if min(n, c, h, w) <= 0:
        raise ValueError('empty population or channel dimension is unsupported')
    if n*h*w > 32768 or c > 256 or seed < 0:
        raise ValueError('supported envelope: population <=32768, C<=256, seed>=0')
    if spec.get('distribution') not in ('bounded-iid', 'constant-plus-small-noise', 'constant'):
        raise ValueError('unknown distribution')
    return n, c, h, w


def reference(x):
    """Independent centered FP64 oracle on the represented input values."""
    x = x.double()
    mean = x.mean((0, 2, 3))
    variance = ((x-mean[None, :, None, None])**2).mean((0, 2, 3))
    return mean, variance


def validate_inputs(x, spec):
    shape = dimensions(spec)
    if (x.device.type != 'cpu' or tuple(x.shape) != shape or x.dtype != torch.float32
            or not x.is_contiguous() or not torch.isfinite(x).all()):
        raise ValueError('input must be finite contiguous CPU FP32 NCHW')
    distribution = spec['distribution']
    bound = 1 if distribution == 'bounded-iid' else 16+2**-8
    if not (x.abs() <= bound).all():
        raise ValueError('input exceeds bounded contract')


def make_case(spec):
    shape = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    if spec['distribution'] == 'constant':
        x = torch.full(shape, 16., dtype=torch.float32)
    else:
        x = torch.rand(shape, generator=rng, dtype=torch.float32)*2-1
        if spec['distribution'] == 'constant-plus-small-noise':
            x = 16+x*2**-8
    validate_inputs(x, spec)
    return dict(x=x, expected=reference(x))


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return dict(passed=False, reason='mean and population variance required')
    rows = []
    for value, expected, (atol, rtol) in zip(actual, fixture['expected'], TOLERANCES):
        value = value.detach().cpu()
        if value.shape != expected.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            return dict(passed=False, reason='invalid output shape/dtype/finiteness')
        error = (value.double()-expected).abs()
        allowed = atol+rtol*expected.abs()
        rows.append(dict(passed=bool((error <= allowed).all()), atol=atol, rtol=rtol,
                         max_absolute_error=error.max().item(),
                         max_tolerance_fraction=(error/allowed).max().item()))
    # Constant populations require exact zero, even though the general gate has an atol.
    constant = bool((fixture['x'] == fixture['x'].flatten()[0]).all())
    zero = not constant or bool((actual[1].detach().cpu() == 0).all())
    return dict(passed=all(r['passed'] for r in rows) and zero,
                mean=rows[0], variance=rows[1], constant_zero=zero)


def host_statistics(x, streaming, block=256):
    """FP32 executable model for lane updates, weighted tails and repeat-call tests."""
    if x.ndim != 4 or min(x.shape) <= 0 or block <= 0:
        raise ValueError('nonempty NCHW and positive block required')
    rows = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1).float()
    if not streaming:
        mean = rows.mean(1)
        return mean, ((rows-mean[:, None])**2).mean(1)
    count = torch.zeros(block)
    mean = torch.zeros(rows.shape[0], block)
    m2 = torch.zeros_like(mean)
    for start in range(0, rows.shape[1], block):
        size = min(block, rows.shape[1]-start)
        count[:size] += 1
        delta = rows[:, start:start+size]-mean[:, :size]
        mean[:, :size] += delta/count[:size]
        m2[:, :size] += delta*delta*((count[:size]-1)/count[:size])
    merged = (mean*count).sum(1)/rows.shape[1]
    variance = (m2+count*(mean-merged[:, None])**2).sum(1)/rows.shape[1]
    return merged, variance


CASE_SOURCES = [Path(__file__)]
