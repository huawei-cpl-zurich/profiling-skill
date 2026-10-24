"""CPU ragged-copy oracle and pre-cache validation for the pack study."""
from pathlib import Path
import math
CASE_SOURCES = [Path(__file__)]


def dimensions(spec):
    p = dict(spec.get('parameters', spec))
    lengths, d = p.get('lengths'), p.get('D')
    if not isinstance(lengths, list) or any(type(n) is not int or n < 0 or n > 2**31-1 for n in lengths):
        raise ValueError('nonnegative int32 lengths required')
    if type(d) is not int or d <= 0 or type(spec.get('seed', 0)) is not int or spec.get('seed', 0) < 0:
        raise ValueError('positive D and nonnegative integer seed required')
    if sum(lengths)*d >= 2**31 or len(lengths)*max(lengths, default=0)*d >= 2**31:
        raise ValueError('signed address range exceeded')
    try: pad = float(p.get('pad_value', '-inf'))
    except (ValueError, TypeError): raise ValueError('invalid padding') from None
    if math.isnan(pad): raise ValueError('NaN padding unsupported')
    return dict(p, lengths=lengths, D=d, pad_value=pad)


def checked_inputs(fixture, p):
    import torch
    x, lengths = fixture['x'], fixture['lengths']
    if x.device.type != 'cpu' or lengths.device.type != 'cpu' or x.dtype != torch.float32 or lengths.dtype != torch.int32:
        raise ValueError('CPU FP32 input and int32 lengths required')
    if not x.is_contiguous() or not lengths.is_contiguous() or tuple(x.shape) != (sum(p['lengths']),p['D']) or tuple(lengths.shape) != (len(p['lengths']),):
        raise ValueError('contiguous input shape mismatch')
    if lengths.tolist() != p['lengths']: raise ValueError('length metadata mismatch')


def reference(x, lengths, pad):
    import torch
    out = torch.full((len(lengths),max(lengths,default=0),x.shape[1]),pad,dtype=torch.float32)
    start = 0
    for b, length in enumerate(lengths):
        out[b,:length].copy_(x[start:start+length]); start += length
    return out


def make_case(spec):
    import torch
    p = dimensions(spec)
    x = torch.randn((sum(p['lengths']),p['D']),generator=torch.Generator().manual_seed(spec.get('seed',0)))
    values = [0.,-0.,float('inf'),-float('inf')] if p.get('signedzero_and_infinity_input') else [0.,-0.]
    for i,v in enumerate(values[:x.numel()]): x.flatten()[i] = v
    result = dict(x=x,lengths=torch.tensor(p['lengths'],dtype=torch.int32))
    result['expected'] = reference(x,p['lengths'],p['pad_value'])
    if p.get('also_empty_batch'):
        result['expected'] = (result['expected'],torch.empty((0,0,p['D']),dtype=torch.float32))
    return result
