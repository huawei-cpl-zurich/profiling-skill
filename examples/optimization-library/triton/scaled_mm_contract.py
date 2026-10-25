"""Frozen bounded INT8 matmul domain and independent CPU oracle."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    vals = tuple(p.get(k) for k in ('M', 'N', 'K'))
    seed = spec.get('seed')
    if any(type(v) is not int for v in (*vals, seed)) or seed < 0:
        raise ValueError('integer M/N/K and nonnegative seed required')
    m, n, k = vals
    if not (1 <= m <= 128 and 1 <= n <= 128 and 0 < k <= 512 and k % 128 == 0):
        raise ValueError('bounded M/N 1..128, K multiple of 128 up to 512 required')
    if p.get('scale_mode') not in ('scalar', 'nonpower_scalar', 'row_col') or type(p.get('bias')) is not bool:
        raise ValueError('explicit scale mode and boolean bias required')
    if p['scale_mode'] == 'scalar':
        if any(type(p.get(x)) is not int or not -8 <= p[x] <= 8 for x in ('exp_a', 'exp_b')):
            raise ValueError('power exponents must be integers in [-8,8]')
    if p['scale_mode'] == 'nonpower_scalar':
        if any(type(p.get(x)) not in (int, float) or not 2**-8 <= p[x] <= 2**8 for x in ('scale_a', 'scale_b')):
            raise ValueError('finite positive scales in bounded interval required')
    return m, n, k, seed, p


def reference(a, b, sa, sb, bias):
    dot = a.to(torch.int64) @ b.to(torch.int64)
    value = sa.reshape(-1, 1) * dot.to(torch.float32)
    value = sb.reshape(1, -1) * value
    return value if bias is None else value + bias


def make_case(spec):
    m, n, k, seed, p = dimensions(spec)
    rng = torch.Generator().manual_seed(seed)
    a = torch.randint(-8, 9, (m, k), dtype=torch.int8, generator=rng)
    b = torch.randint(-8, 9, (k, n), dtype=torch.int8, generator=rng)
    if p['scale_mode'] == 'row_col':
        sa = .25 + torch.arange(m, dtype=torch.float32) / 64
        sb = .5 + torch.arange(n, dtype=torch.float32) / 128
    elif p['scale_mode'] == 'scalar':
        sa, sb = (torch.tensor([2.0**p[key]], dtype=torch.float32) for key in ('exp_a', 'exp_b'))
    else:
        sa, sb = (torch.tensor([p[key]], dtype=torch.float32) for key in ('scale_a', 'scale_b'))
    bias = torch.randint(-8, 9, (n,), generator=rng).float() / 2 if p['bias'] else None
    return dict(a=a, b=b, sa=sa, sb=sb, bias=bias, expected=reference(a,b,sa,sb,bias))


def check_fixture(spec, fixture):
    m, n, k, _, p = dimensions(spec)
    if set(fixture) != {'a','b','sa','sb','bias','expected'}:
        raise ValueError('exact fixture arity required')
    shapes = {'a':(m,k), 'b':(k,n), 'sa':(m if p['scale_mode']=='row_col' else 1,),
              'sb':(n if p['scale_mode']=='row_col' else 1,), 'expected':(m,n)}
    if p['bias']: shapes['bias'] = (n,)
    elif fixture['bias'] is not None: raise ValueError('unexpected bias')
    pointers = []
    for name, shape in shapes.items():
        x = fixture[name]
        dtype = torch.int8 if name in ('a','b') else torch.float32
        if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.dtype != dtype or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('CPU contiguous shape/dtype mismatch: '+name)
        pointers.append(x.untyped_storage().data_ptr())
        if name in ('a','b'):
            if not ((x >= -8) & (x <= 8)).all(): raise ValueError('bounded INT8 values required')
        elif not torch.isfinite(x).all(): raise ValueError('finite values required')
        if name in ('sa','sb') and not ((x >= 2**-8) & (x <= 2**8)).all():
            raise ValueError('bounded positive scales required')
    if len(set(pointers)) != len(pointers): raise ValueError('fixture aliases unsupported')
    return m,n,k,p


CASE_SOURCES = [Path(__file__)]
