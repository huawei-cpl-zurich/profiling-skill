"""Bounded FP32 gated recurrence with an independent FP64 CPU oracle."""
from pathlib import Path
import torch

NAMES = ('q', 'k', 'v', 'g', 'beta', 'h0')
ATOL, RTOL = 1e-4, 2e-4


def dimensions(spec):
    p = spec.get('parameters', spec)
    allowed = {'B', 'T', 'H', 'HV', 'K', 'V', 'zeros_and_epsilon'}
    if 'parameters' in spec and any(k in spec and spec[k] != p.get(k) for k in allowed):
        raise ValueError('conflicting flat/nested parameters')
    if any(k not in allowed for k in p if k not in {'seed', 'case_id', 'role'}):
        raise ValueError('unsupported optional recurrence mode')
    values = tuple(p.get(k) for k in ('B', 'T', 'H', 'HV', 'K', 'V'))
    if any(type(x) is not int for x in values) or type(spec.get('seed')) is not int or spec['seed'] < 0:
        raise ValueError('positive integer dimensions and nonnegative seed required')
    b, t, h, hv, k, v = values
    if not (1 <= b <= 4 and 1 <= t <= 8 and 1 <= h <= hv <= 4 and hv % h == 0 and 1 <= k <= 64 and 1 <= v <= 512):
        raise ValueError('unsupported bounded recurrence shape')
    if type(p.get('zeros_and_epsilon', False)) is not bool:
        raise ValueError('zeros_and_epsilon must be bool')
    return values


def reference(inputs):
    q, k, v, g, beta, h0 = (inputs[n].double() for n in NAMES)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    b, t, hv, _ = v.shape
    mapping = torch.arange(hv) // (hv // q.shape[2])
    state = h0.clone()
    out = torch.empty_like(v)
    for token in range(t):
        key = k[:, token, mapping]
        state = state * g[:, token].exp()[..., None, None]
        residual = (v[:, token] - (state * key[..., None]).sum(-2)) * beta[:, token, :, None]
        state = state + key[..., None] * residual[..., None, :]
        out[:, token] = (state * (q[:, token, mapping] * q.shape[-1] ** -.5)[..., None]).sum(-2)
    return out, state


def make_case(spec):
    b, t, h, hv, k, v = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    def uniform(shape, low, high):
        return torch.rand(shape, generator=rng) * (high-low) + low
    inputs = dict(q=uniform((b,t,h,k),-.1,.1), k=uniform((b,t,h,k),-.1,.1),
                  v=uniform((b,t,hv,v),-.1,.1), g=uniform((b,t,hv),-2,-.1),
                  beta=uniform((b,t,hv),.1,.9), h0=uniform((b,hv,k,v),-.01,.01))
    if spec.get('parameters', spec).get('zeros_and_epsilon'):
        for name in ('q', 'k'):
            rows = inputs[name].reshape(-1,k)
            rows[0].zero_()
            if len(rows) > 1:
                rows[1] = torch.tensor([(-1)**i * (i+1)*1e-4 for i in range(k)])
    return dict(inputs=inputs, expected=reference(inputs))


def validate_inputs(inputs, spec):
    b,t,h,hv,k,v = dimensions(spec)
    shapes = ((b,t,h,k),(b,t,h,k),(b,t,hv,v),(b,t,hv),(b,t,hv),(b,hv,k,v))
    bounds = ((-.1,.1),(-.1,.1),(-.1,.1),(-2,-.1),(.1,.9),(-.01,.01))
    if set(inputs) != set(NAMES):
        raise ValueError('all six inputs including non-None h0 required')
    for n, shape, (lo,hi) in zip(NAMES,shapes,bounds):
        x = inputs[n]
        if not isinstance(x,torch.Tensor) or x.device.type != 'cpu' or x.dtype != torch.float32 or tuple(x.shape) != shape or not x.is_contiguous() or not torch.isfinite(x).all() or not ((x >= lo)&(x <= hi)).all():
            raise ValueError('invalid contiguous bounded CPU FP32 input: '+n)
    if len({x.untyped_storage().data_ptr() for x in inputs.values()}) != len(NAMES):
        raise ValueError('inputs must not alias')


def diagnostics(actual, fixture):
    if not isinstance(actual,(tuple,list)) or len(actual) != 2 or not isinstance(fixture.get('expected'),(tuple,list)) or len(fixture['expected']) != 2:
        return dict(passed=False, reason='exactly output and final state required')
    rows = []
    for x, expected in zip(actual,fixture['expected']):
        if not isinstance(x,torch.Tensor) or x.dtype != torch.float32 or x.shape != expected.shape or not torch.isfinite(x).all():
            return dict(passed=False, reason='invalid output shape/dtype/finiteness')
        error = (x.detach().cpu().double()-expected).abs()
        allowed = ATOL + RTOL*expected.abs()
        rows.append(dict(passed=bool((error<=allowed).all()), max_absolute_error=error.max().item(),
                         max_tolerance_fraction=(error/allowed).max().item()))
    return dict(passed=all(r['passed'] for r in rows), outputs=rows, atol=ATOL, rtol=RTOL)


CASE_SOURCES = [Path(__file__)]
