"""Independent FP64 fixtures and admission for flattened MHC pre backward."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    bs, d, tile = (p.get(k) for k in ('BS', 'D', 'tile_D'))
    if (any(type(x) is not int for x in (bs, d, tile)) or bs <= 0
            or not 1 <= d <= 4096 or d & (d - 1)
            or not 1 <= tile <= d or tile & (tile - 1)
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported MHC pre case')
    if p.get('zero_cancelled_and_signedzero') and (bs != 3 or d < 2):
        raise ValueError('zero guard requires three rows and even D')
    return p, bs, d, tile


def reference(inputs):
    h, x, dy = (inputs[k].double() for k in ('H', 'X', 'dY'))
    return ((x * dy[:, None, :]).sum(-1).float(),
            (dy[:, None, :] * h[:, :, None]).float())


def validate_inputs(inputs, spec):
    _, bs, d, _ = dimensions(spec)
    shapes = dict(H=(bs, 4), X=(bs, 4, d), dY=(bs, d))
    if set(inputs) != set(shapes):
        raise ValueError('input keys must be H, X, dY')
    storages = set()
    for key, shape in shapes.items():
        value = inputs[key]
        if (value.device.type != 'cpu' or value.dtype != torch.float32
                or tuple(value.shape) != shape or not value.is_contiguous()
                or value.storage_offset() != 0
                or value.untyped_storage().nbytes() != value.numel() * 4
                or not torch.isfinite(value).all() or not (value.abs() <= 1).all()):
            raise ValueError('invalid owned contiguous finite FP32 input: ' + key)
        pointer = value.untyped_storage().data_ptr()
        if pointer in storages:
            raise ValueError('aliased input storage')
        storages.add(pointer)


def make_case(spec):
    p, bs, d, _ = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: torch.rand(shape, generator=rng) * 2 - 1
              for k, shape in dict(H=(bs, 4), X=(bs, 4, d), dY=(bs, d)).items()}
    if p.get('zero_cancelled_and_signedzero'):
        inputs['dY'][0].zero_()
        inputs['dY'][1].fill_(1)
        inputs['X'][1, :, ::2] = 1
        inputs['X'][1, :, 1::2] = -1
        inputs['dY'][2].fill_(-0.)
        inputs['H'][2] = torch.tensor([0., -0., 1., -1.])
        inputs['X'][2].fill_(1.)
    validate_inputs(inputs, spec)  # Bounds checked before generic fixture caching.
    return dict(inputs=inputs, expected=reference(inputs))


def check_outputs(actual, expected):
    if (not isinstance(actual, tuple) or len(actual) != 2
            or not isinstance(expected, tuple) or len(expected) != 2):
        return dict(passed=False, reason='output must be (dH, dX) tuple')
    rows = []
    for got, wanted in zip(actual, expected):
        if not isinstance(got, torch.Tensor) or got.shape != wanted.shape or got.dtype != torch.float32:
            return dict(passed=False, reason='output shape or dtype mismatch')
        got = got.detach().cpu()
        rows.append(dict(passed=bool(torch.allclose(got, wanted, atol=2e-5, rtol=2e-5)),
                         max_abs_error=float((got - wanted).abs().max()),
                         zero_count=int((got == 0).sum()),
                         negative_zero_count=int(((got == 0) & torch.signbit(got)).sum())))
    return dict(passed=all(x['passed'] for x in rows), outputs=rows,
                signed_zero_bitwise_guarantee=False)


CASE_SOURCES = [Path(__file__)]
