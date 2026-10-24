"""Last-axis FP32 mean and an explicit cross-batch numerical contract."""
from pathlib import Path
import torch


def parameters(spec):
    p = spec['parameters']
    r, n, seed = p['R'], p['N'], spec['seed']
    if any(type(x) is not int or x <= 0 for x in (r, n)) or type(seed) is not int or seed < 0:
        raise ValueError('positive integer R,N and nonnegative seed required')
    if type(p.get('dim', -1)) is not int or p.get('dim', -1) not in (-1, 1):
        raise ValueError('only last-axis dim=-1 or1 supported')
    if type(p.get('keepdim', False)) is not bool:
        raise ValueError('keepdim must be boolean')
    mode = p.get('distribution', 'bounded-uniform')
    if mode not in ('bounded-uniform', 'bounded-cancellation'):
        raise ValueError('unsupported distribution')
    if mode == 'bounded-cancellation' and (n != 1024 or p.get('context_rows') != [1, 3, 129]
                                           or p.get('context_positions') != [0, 1, 128]):
        raise ValueError('cancellation guard requires its fixed context contract')
    return r, n, seed, p.get('keepdim', False), mode


def make_case(spec):
    r, n, seed, keepdim, mode = parameters(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    sizes = [1, 3, 129] if mode == 'bounded-cancellation' else [r]
    xs = tuple(torch.rand((rows, n), generator=rng, dtype=torch.float32, device='cpu') * 2 - 1
               for rows in sizes)
    if mode == 'bounded-cancellation':
        row = torch.tensor([1, 2**-24, -1, 2**-24], dtype=torch.float32, device='cpu').repeat(256)
        for x, pos in zip(xs, [0, 1, 128]):
            x[pos].copy_(row)
    expected = tuple(x.double().mean(dim=1, keepdim=keepdim) for x in xs)
    return dict(xs=xs, expected=expected)


def diagnostics(actual, fixture, spec, arm):
    if not isinstance(actual, (tuple, list)) or len(actual) != len(fixture['expected']):
        return dict(passed=False, reason='output count')
    errors = []
    for y, ref in zip(actual, fixture['expected']):
        if y.device.type != 'cpu' or y.dtype != torch.float32 or y.shape != ref.shape:
            return dict(passed=False, reason='output shape/dtype/device')
        errors.append(dict(max_absolute_error=float((y.double() - ref).abs().max()),
                           passed=bool(torch.isfinite(y).all() and torch.allclose(
                               y.double(), ref, atol=2e-5, rtol=2e-4))))
    result = dict(passed=all(e['passed'] for e in errors), outputs=errors, atol=2e-5, rtol=2e-4)
    if len(actual) == 3:
        values = [y.reshape(-1)[pos:pos+1].view(torch.int32) for y, pos in zip(actual, [0, 1, 128])]
        same = all(torch.equal(values[0], v) for v in values[1:])
        result.update(designated_row_bitwise_equal=same, designated_row_bits=[int(v.item()) for v in values])
        if arm == 'after' and not same:
            result.update(passed=False, reason='fixed row changed across unrelated contexts')
    return result


CASE_SOURCES = [Path(__file__)]
