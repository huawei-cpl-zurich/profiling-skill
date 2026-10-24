"""Independent frozen FP64 MHC post provider; no device or Triton imports."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    allowed = {'BS', 'D', 'cancellation_and_rounding', 'nonsymmetric_transpose'}
    metadata = {'seed', 'case_id', 'role', 'expected', 'scenario_designation', 'hypothesis'}
    if set(p) - allowed - (metadata if 'parameters' not in spec else set()):
        raise ValueError('unsupported algorithm parameters')
    bs, d = p.get('BS'), p.get('D')
    if (type(bs) is not int or bs <= 0 or type(d) is not int or not 1 <= d <= 4096
            or d & (d - 1) or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported MHC post dimensions')
    for key in allowed - {'BS', 'D'}:
        if key in p and type(p[key]) is not bool:
            raise ValueError('guard flags must be Boolean')
    return p, bs, d


def validate_inputs(inputs, spec):
    _, bs, d = dimensions(spec)
    shapes = dict(x=(bs, d), R=(bs, 4, d), hp=(bs, 4), H=(bs, 4, 4))
    if set(inputs) != set(shapes):
        raise ValueError('input keys must be x, R, hp, H')
    pointers = set()
    for key, shape in shapes.items():
        v = inputs[key]
        if (not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != torch.float32
                or tuple(v.shape) != shape or not v.is_contiguous() or v.storage_offset() != 0
                or v.untyped_storage().nbytes() != v.numel() * 4 or v.requires_grad
                or not torch.isfinite(v).all() or not (v.abs() <= 1).all()):
            raise ValueError('owned finite contiguous FP32 input required: ' + key)
        pointer = v.untyped_storage().data_ptr()
        if pointer in pointers:
            raise ValueError('aliased inputs')
        pointers.add(pointer)


def reference(inputs):
    x, r, hp, h = (inputs[k].double() for k in ('x', 'R', 'hp', 'H'))
    # User H[j,i], with an independent CPU transpose; one final FP32 cast.
    return (hp[:, :, None] * x[:, None, :] + torch.einsum('bji,bjd->bid', h, r)).float()


def make_case(spec):
    p, bs, d = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: torch.rand(shape, generator=rng) * 2 - 1 for k, shape in
              dict(x=(bs, d), R=(bs, 4, d), hp=(bs, 4), H=(bs, 4, 4)).items()}
    if p.get('cancellation_and_rounding'):
        pattern = torch.tensor([1., -1., 2**-12, -2**-12, .5, -.5, 2**-24, -2**-24])
        inputs['R'].copy_(pattern.repeat((inputs['R'].numel()+7)//8)[:inputs['R'].numel()].view_as(inputs['R']))
        # Dyadic 1±2^-12 normalized into finite [-1,1].
        h = torch.tensor([1., 1.-2**-12, -1., -(1.-2**-12), .5, -.5, .25, -.25,
                          .75, -.75, 2**-12, -2**-12, .125, -.125, .625, -.625])
        inputs['H'].copy_(h.view(1,4,4).expand(bs,-1,-1))
    if p.get('nonsymmetric_transpose'):
        inputs['H'].copy_(torch.arange(bs*16).reshape(bs,4,4).float() / (bs*16) * 2 - 1)
    validate_inputs(inputs, spec)
    return dict(inputs=inputs, expected=reference(inputs))


def check_output(actual, expected):
    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != torch.float32):
        return dict(passed=False, reason='expected one FP32 tensor with exact output shape')
    got = actual.detach().cpu()
    return dict(passed=bool(torch.allclose(got, expected, atol=3e-6, rtol=3e-6)),
                max_abs_error=float((got-expected).abs().max()), atol=3e-6, rtol=3e-6)


CASE_SOURCES = [Path(__file__)]
