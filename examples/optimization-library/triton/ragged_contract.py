"""Independent CPU contract: packed tokens -> padded [batch, max_length, width]."""
import math

import torch


def check_spec(spec):
    if spec.get('family') not in ('prefix-offsets', 'padding-store'):
        raise ValueError('family must be prefix-offsets or padding-store')
    for key, limit in [('batch', 1024), ('length', 2048), ('width', 1024)]:
        if type(spec.get(key)) is not int or not 1 <= spec[key] <= limit:
            raise ValueError(f'{key} must be a positive integer at most {limit}')
    if spec['batch'] * spec['length'] * spec['width'] > 2**25:
        raise ValueError('output is limited to 2**25 elements')
    if spec.get('dtype', 'float32') not in ('float16', 'float32'):
        raise ValueError('dtype must be float16 or float32')
    seed = spec.get('seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    pad = spec.get('pad', -3.25)
    if type(pad) not in (float, int) or not math.isfinite(pad):
        raise ValueError('padding must be finite')
    dtype = getattr(torch, spec.get('dtype', 'float32'))
    if not torch.isfinite(torch.tensor(pad, dtype=dtype)):
        raise ValueError('padding must remain finite in output dtype')
    if spec.get('pattern', 'mixed') not in ('mixed', 'empty', 'full'):
        raise ValueError('pattern must be mixed, empty or full')
    lengths = spec.get('lengths')
    if lengths is not None and (not isinstance(lengths, list) or len(lengths) != spec['batch']
            or any(type(x) is not int or not 0 <= x <= spec['length'] for x in lengths)):
        raise ValueError('explicit lengths must contain batch integers within [0, length]')


def reference(x, lengths, maximum, padding):
    """Sequential slicing/copy oracle; no candidate scan or address implementation."""
    if x.device.type != 'cpu' or lengths.device.type != 'cpu':
        raise ValueError('reference inputs must be CPU tensors')
    if x.ndim != 2 or lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ValueError('reference needs x[N,D] and int32 lengths[B]')
    if any(not 0 <= n <= maximum for n in lengths.tolist()) or int(lengths.sum()) != x.shape[0]:
        raise ValueError('lengths do not partition the packed input')
    result = torch.full((lengths.numel(), maximum, x.shape[1]), padding, dtype=x.dtype)
    cursor = 0
    for batch, count in enumerate(lengths.tolist()):
        result[batch, :count] = x[cursor:cursor + count]
        cursor += count
    return result


def make_case(spec):
    check_spec(spec)
    batch, maximum, width = (spec[k] for k in ('batch', 'length', 'width'))
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    lengths = torch.randint(maximum + 1, (batch,), generator=rng, dtype=torch.int32)
    pattern = spec.get('pattern', 'mixed')
    if pattern == 'empty':
        lengths.zero_()
    elif pattern == 'full':
        lengths.fill_(maximum)
    else:
        lengths[0] = 0
        lengths[-1] = maximum
    if 'lengths' in spec:
        lengths = torch.tensor(spec['lengths'], dtype=torch.int32)
    dtype = getattr(torch, spec.get('dtype', 'float32'))
    x = torch.randn((int(lengths.sum()), width), generator=rng).to(dtype)
    # Quantize once outside forwards, exactly as the output dtype requires.
    padding = float(torch.tensor(spec.get('pad', -3.25), dtype=dtype))
    return dict(inputs=dict(x=x, lengths=lengths), padding=padding,
                expected=reference(x, lengths, maximum, padding))


def validate_actual(actual, fixture, inputs=None):
    result = actual.detach().cpu()
    expected = fixture['expected']
    if result.shape != expected.shape or result.dtype != expected.dtype:
        raise ValueError('output shape or dtype differs from contract')
    if inputs is not None:
        for name, original in fixture['inputs'].items():
            if not torch.equal(inputs[name].detach().cpu(), original):
                raise ValueError('input mutated: ' + name)
    passed = torch.equal(result, expected)
    return dict(passed=passed, max_absolute_error=float((result.float() - expected.float()).abs().max()),
                rtol=0.0, atol=0.0)
