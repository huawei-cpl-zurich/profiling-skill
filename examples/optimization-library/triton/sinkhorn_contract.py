"""Independent FP64 oracle aligned with the pinned source test, not old docs."""
from pathlib import Path
import math
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    bs, iters, eps = (p.get(k) for k in ('BS', 'iters', 'eps'))
    if (type(bs) is not int or bs <= 0 or type(iters) is not int or not 1 <= iters <= 20
            or type(eps) not in (int, float) or not math.isfinite(eps) or not 1e-6 <= eps <= 1e-3
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported Sinkhorn case: positive BS, iters 1..20, eps 1e-6..1e-3 required')
    return p, bs, iters, eps


def reference(inputs, iters, eps):
    mixes, scale, base = (inputs[k].double() for k in ('mixes', 'scale', 'base'))
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    logits = (mixes[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = torch.softmax(logits, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return tuple(v.float() for v in (pre, post, comb))


def validate_inputs(inputs, spec):
    _, bs, _, _ = dimensions(spec)
    shapes = dict(mixes=(bs, 24), scale=(3,), base=(24,))
    if set(inputs) != set(shapes):
        raise ValueError('inputs must be mixes, scale, base')
    pointers = set()
    for k, shape in shapes.items():
        v = inputs[k]
        if (v.device.type != 'cpu' or v.dtype != torch.float32 or tuple(v.shape) != shape
                or not v.is_contiguous() or v.storage_offset() != 0
                or v.untyped_storage().nbytes() != v.numel()*4 or not torch.isfinite(v).all()
                or not (v.abs() <= (3 if k == 'mixes' else 1)).all()):
            raise ValueError('owned finite contiguous FP32 input required: ' + k)
        pointer = v.untyped_storage().data_ptr()
        if pointer in pointers:
            raise ValueError('aliased input storage')
        pointers.add(pointer)


def make_case(spec):
    p, bs, iters, eps = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: (torch.rand(shape, generator=rng)*2-1)*(3 if k == 'mixes' else 1)
              for k, shape in dict(mixes=(bs,24), scale=(3,), base=(24,)).items()}
    if p.get('zero_scale_equal_logits'):
        inputs['scale'].zero_(); inputs['base'].zero_()
    if p.get('nontrivial_scales'):
        inputs['scale'][:] = torch.tensor([-.75, .5, .25])
    validate_inputs(inputs, spec)
    return dict(inputs=inputs, expected=reference(inputs, iters, eps))


def check_outputs(actual, expected):
    if not isinstance(actual, tuple) or len(actual) != 3 or not isinstance(expected, tuple) or len(expected) != 3:
        return dict(passed=False, reason='output must be tuple (pre, post, comb)')
    rows = []
    for got, wanted in zip(actual, expected):
        if not isinstance(got, torch.Tensor) or got.shape != wanted.shape or got.dtype != torch.float32:
            return dict(passed=False, reason='output shape/dtype mismatch')
        got = got.detach().cpu()
        rows.append(dict(passed=bool(torch.allclose(got, wanted, atol=3e-6, rtol=3e-6)),
                         max_abs_error=float((got-wanted).abs().max())))
    return dict(passed=all(x['passed'] for x in rows), outputs=rows, atol=3e-6, rtol=3e-6)


CASE_SOURCES = [Path(__file__)]
