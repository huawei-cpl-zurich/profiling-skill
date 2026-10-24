"""Independent sequential CPU contract for packed gated delta recurrence."""
from pathlib import Path
import math
import torch


def dimensions(spec):
    keys = ('t', 'hk', 'hv', 'dk', 'dv', 'seed')
    if any(type(spec.get(k)) is not int for k in keys):
        raise ValueError('t, hk, hv, dk, dv and seed must be integers')
    t, hk, hv, dk, dv, seed = (spec[k] for k in keys)
    bounds = spec.get('segments')
    if not 1 <= t <= 4096 or not 1 <= hk <= hv <= 4 or hv % hk or dk not in (64, 128) or dv not in (64, 128) or seed < 0:
        raise ValueError('unsupported recurrence shape')
    if not isinstance(bounds, list) or len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != t or any(type(x) is not int for x in bounds) or any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError('segments must strictly partition [0,T]')
    return t, hk, hv, dk, dv, bounds


def reference(inputs, segments):
    """FP32 CPU token recurrence; no chunk formulation or candidate conversion."""
    q, k, v, beta, g, initial = (inputs[n].float() for n in ('q', 'k', 'v', 'beta', 'g', 'initial'))
    t, hv, dv = v.shape
    hk = k.shape[1]
    output = torch.empty((t, hv, dv), dtype=torch.float32)
    final = torch.empty_like(initial)
    mapping = torch.arange(hv) // (hv // hk)
    for segment, (start, end) in enumerate(zip(segments, segments[1:])):
        state = initial[segment].clone()
        for token in range(start, end):
            key = k[token, mapping]
            state = state * g[token].exp()[:, None, None]
            prediction = (state * key[:, :, None]).sum(1)
            delta = beta[token, :, None] * (v[token] - prediction)
            state = state + key[:, :, None] * delta[:, None, :]
            output[token] = (state * q[token, mapping, :, None]).sum(1)
        final[segment] = state
    return output.half(), final


def make_case(spec):
    t, hk, hv, dk, dv, segments = dimensions(spec)
    generator = torch.Generator().manual_seed(spec['seed'])
    k = torch.randn(t, hk, dk, generator=generator)
    k = (k / k.norm(dim=-1, keepdim=True)).half()
    inputs = dict(q=(torch.randn(t, hk, dk, generator=generator) / math.sqrt(dk)).half(),
                  k=k, v=(torch.randn(t, hv, dv, generator=generator) * .1).half(),
                  beta=torch.rand(t, hv, generator=generator) * .2,
                  g=-torch.rand(t, hv, generator=generator) * .01,
                  initial=torch.randn(len(segments)-1, hv, dk, dv, generator=generator) * .02)
    return dict(inputs=inputs, expected=reference(inputs, segments))


def validate_inputs(inputs, spec):
    t, hk, hv, dk, dv, segments = dimensions(spec)
    shapes = dict(q=(t,hk,dk), k=(t,hk,dk), v=(t,hv,dv), beta=(t,hv), g=(t,hv), initial=(len(segments)-1,hv,dk,dv))
    for key, shape in shapes.items():
        value = inputs[key]
        dtype = torch.float16 if key in ('q','k','v') else torch.float32
        if value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError('invalid contiguous finite CPU input: '+key)
    if not ((inputs['beta'] >= 0) & (inputs['beta'] <= .2)).all() or not ((inputs['g'] >= -.01) & (inputs['g'] <= 0)).all():
        raise ValueError('beta/g outside declared fixture envelope')
    if not torch.allclose(inputs['k'].float().norm(dim=-1), torch.ones((t,hk)), rtol=.002, atol=.002):
        raise ValueError('K rows must be normalized')


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple,list)) or len(actual) != 2:
        return dict(passed=False, reason='output and final state both required')
    rows = []
    for value, expected in zip(actual, fixture['expected']):
        value = value.detach().cpu()
        if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            return dict(passed=False, reason='shape/dtype/nonfinite output or state')
        error = (value.float()-expected.float()).abs()
        allowed = .02 + .02 * expected.float().abs()
        rows.append(dict(passed=bool((error <= allowed).all()), max_absolute_error=float(error.max()),
                         max_relative_error=float((error/expected.float().abs().clamp_min(1e-12)).max()),
                         max_tolerance_fraction=float((error/allowed).max())))
    return dict(passed=all(row['passed'] for row in rows), output=rows[0], final_state=rows[1], rtol=.02, atol=.02)


CASE_SOURCES = [Path(__file__)]
