"""CPU contract for natural-order unsigned INT4 grouped-weight GEMM.

This encoding is deliberately independent of AWQ's nibble permutation. Each
int32 packs eight consecutive N values, low nibble first. Grouping is along K.
"""
import math

import torch


def check_spec(spec):
    if spec.get('family') != 'int4-consumer':
        raise ValueError('family must be int4-consumer')
    for key, limit in [('m', 4096), ('n', 8192), ('k', 8192)]:
        if type(spec.get(key)) is not int or not 0 < spec[key] <= limit:
            raise ValueError('dimensions must be positive integers within the declared envelope')
    group = spec.get('group_size')
    if type(group) is not int or group not in (32, 64, 128) or spec['k'] % group or spec['n'] % 8:
        raise ValueError('N must be divisible by eight and K by group_size=32/64/128')
    seed = spec.get('seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')


def pack8(values):
    if (values.device.type != 'cpu' or values.ndim != 2 or values.shape[1] % 8
            or values.dtype not in (torch.int32, torch.int64) or (values < 0).any() or (values > 15).any()):
        raise ValueError('pack8 requires a CPU integer matrix of unsigned nibbles')
    groups = values.long().reshape(values.shape[0], -1, 8)
    return (groups << (torch.arange(8) * 4)).sum(-1).to(torch.int32)


def unpack8(words):
    if words.device.type != 'cpu' or words.ndim != 2 or words.dtype != torch.int32:
        raise ValueError('unpack8 requires a CPU int32 matrix')
    return ((words.long()[..., None] >> (torch.arange(8) * 4)) & 15).reshape(words.shape[0], -1)


def dequantize(words, zeros, scales, group_size):
    q, z = unpack8(words), unpack8(zeros)
    if (scales.device.type != 'cpu' or scales.dtype != torch.float16 or scales.shape != z.shape
            or words.shape[0] != z.shape[0] * group_size):
        raise ValueError('quantization groups and FP16 scale matrix differ')
    groups = torch.arange(words.shape[0]) // group_size
    # The FP16 conversion is part of the mathematical contract in BOTH arms.
    return ((q.float() - z[groups].float()) * scales[groups].float()).half()


def make_case(spec):
    check_spec(spec)
    m, n, k, group = (spec[key] for key in ('m', 'n', 'k', 'group_size'))
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    a = (torch.randn(m, k, generator=rng) / math.sqrt(k)).half()
    packed = pack8(torch.randint(16, (k, n), generator=rng))
    zeros = pack8(torch.randint(16, (k // group, n), generator=rng))
    scales = (torch.rand(k // group, n, generator=rng) * .06 + .01).half()
    weight = dequantize(packed, zeros, scales, group)
    return dict(inputs=dict(a=a, packed=packed, zeros=zeros, scales=scales),
                expected=(a.float() @ weight.float()).half())


def validate_actual(actual, fixture, device_inputs=None):
    value, target = actual.detach().cpu(), fixture['expected']
    if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
        raise ValueError('output shape/dtype/finite contract violated')
    error = (value.float() - target.float()).abs()
    allowed = .001 + .001 * target.float().abs()
    if device_inputs is not None:
        for name, original in fixture['inputs'].items():
            if not torch.equal(device_inputs[name].detach().cpu(), original):
                raise ValueError('input mutated: ' + name)
    return dict(passed=bool((error <= allowed).all()), max_absolute_error=float(error.max()),
                max_tolerance_fraction=float((error / allowed).max()), rtol=.001, atol=.001)
