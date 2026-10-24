"""Independent restricted linear-loss CPU oracle; no candidate chunk algebra."""
from pathlib import Path
import math
import torch


def check_spec(spec):
    if spec.get('family') != 'E11':
        raise ValueError('family must be E11')
    keys = ('tokens', 'hidden', 'vocab', 'chunk', 'seed')
    if any(type(spec.get(k)) is not int for k in keys):
        raise ValueError('integer dimensions, chunk and seed required')
    t, d, v, c, seed = (spec[k] for k in keys)
    if not (1 <= t <= 2048 and 16 <= d <= 256 and 17 <= v <= 4096 and
            1 <= c <= t and seed >= 0):
        raise ValueError('unsupported linear-loss domain')
    return t, d, v, c


def validate_inputs(inputs, spec):
    t, d, v, _ = check_spec(spec)
    for name, shape, dtype in (('x', (t,d), torch.float16), ('w', (v,d), torch.float16),
                                ('target', (t,), torch.int32)):
        value = inputs[name]
        if value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError('invalid dense CPU input: ' + name)
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite input: ' + name)
    if not ((inputs['target'] >= 0) & (inputs['target'] < v)).all():
        raise ValueError('target outside vocabulary')


def reference(inputs):
    x, w, target = inputs['x'].float(), inputs['w'].float(), inputs['target'].long()
    logits = x @ w.T
    losses = torch.logsumexp(logits, dim=1) - logits[torch.arange(x.shape[0]), target]
    gradient = logits.softmax(dim=1)
    gradient[torch.arange(x.shape[0]), target] -= 1
    # This explicit common rounding is part of the operator contract, not a fast-arm shortcut.
    gradient = gradient.half().float()
    return losses.sum().reshape(1), gradient @ w, gradient.T @ x


def make_case(spec):
    t, d, v, _ = check_spec(spec)
    generator = torch.Generator().manual_seed(spec['seed'])
    inputs = dict(x=(torch.randn(t,d,generator=generator)/math.sqrt(d)).half(),
                  w=(torch.randn(v,d,generator=generator)*.5).half(),
                  target=torch.randint(v,(t,),generator=generator,dtype=torch.int32))
    validate_inputs(inputs, spec)
    return dict(inputs=inputs, expected=reference(inputs))


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple,list)) or len(actual) != 3:
        return dict(passed=False, reason='loss, dX and dW all required')
    rows = []
    for value, expected in zip(actual, fixture['expected']):
        value = value.detach().cpu()
        if value.shape != expected.shape or value.dtype != torch.float32 or not value.is_contiguous() or not torch.isfinite(value).all():
            return dict(passed=False, reason='output shape/dtype/layout/finite')
        error = (value-expected).abs()
        allowed = .002 + .002 * expected.abs()
        rows.append(dict(passed=bool((error <= allowed).all()), max_absolute_error=float(error.max()),
                         max_tolerance_fraction=float((error/allowed).max())))
    return dict(passed=all(row['passed'] for row in rows), outputs=rows, rtol=.002, atol=.002)


CASE_SOURCES = [Path(__file__)]
