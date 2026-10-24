"""Frozen F11 MSD arithmetic, independent CPU oracles and dense error gate."""
import torch


def check_spec(spec):
    p = spec['parameters']
    if any(type(p.get(k)) is not int or p[k] <= 0 for k in ('M', 'N', 'K', 'weight_max')):
        raise ValueError('positive dimensions and signed code bound required')
    if p['weight_max'] > 127 or p.get('amax') != 1. or p.get('scale') != 2**-10:
        raise ValueError('fixed activation/scale contract required')
    if p['M'] * p['K'] + p['K'] * p['N'] > 2**24:
        raise ValueError('resource limit exceeded')
    return p


def eligible(spec):
    p = check_spec(spec)
    return p['K'] * 127 * p['weight_max'] < 2**31


def components(x):
    amax = x.float().abs().amax(1, keepdim=True)
    inverse = 127. / torch.where(amax > 0, amax, torch.ones_like(amax))
    t = x.float() * inverse
    a1 = t.round()
    residual = (t - a1) * 254.
    a2 = residual.round()
    return amax, a1.to(torch.int8), a2.to(torch.int8), t, residual


def reconstruct(c1, c2, amax, scale):
    # Each operation rounds to FP32, matching separate source Muls/Add/Mul.
    residual = c2.float() * torch.tensor(1. / 254, dtype=torch.float32)
    combined = c1.float() + residual
    rowscale = amax / 127.
    return ((combined * rowscale) * scale).half()


def make_case(spec):
    p = check_spec(spec)
    m, n, k = (p[x] for x in ('M', 'N', 'K'))
    gen = torch.Generator().manual_seed(spec['seed'])
    x = (torch.rand((m, k), generator=gen) * 2 - 1).half()
    x[:, 0] = 1
    # Deterministic rejection from the same fixed RNG before any timing.
    for _ in range(100):
        amax, a1, a2, t, residual = components(x)
        distance = lambda v: ((v - v.floor()) - .5).abs()
        bad = (distance(t) < 1e-5) | (distance(residual) < 1e-5)
        if not bad.any():
            break
        x[bad] = (torch.rand((int(bad.sum()),), generator=gen) * 2 - 1).half()
    else:
        raise ValueError('rounding margin generator failed')
    if spec.get('distribution') == 'zero-row':
        x[0] = 0
    amax, a1, a2, t, residual = components(x)
    w = torch.randint(-p['weight_max'], p['weight_max'] + 1, (k, n), generator=gen, dtype=torch.int8)
    scale = torch.full((n,), p['scale'], dtype=torch.float32)
    dense = (x.double() @ w.double()) * scale.double()
    bound = w.double().abs().sum(0)[None, :] * amax.double() * scale.double().abs() / (127 * 508) + 2**-8
    c1, c2 = a1.long() @ w.long(), a2.long() @ w.long()
    expected = reconstruct(c1, c2, amax, scale) if eligible(spec) else dense.half()
    return dict(inputs=dict(x=x, w=w, scale=scale), expected=expected, dense=dense,
                budget=bound, a1=a1, a2=a2,
                rounding_margin=float(torch.minimum(distance(t), distance(residual)).min()))


def validate_actual(actual, fixture, inputs, arm, codes=None):
    y = actual.detach().cpu()
    if y.shape != fixture['expected'].shape or y.dtype != torch.float16:
        raise ValueError('output shape/dtype differs')
    error = (y.double() - fixture['dense']).abs()
    passed = bool(torch.isfinite(y).all() and (error <= fixture['budget']).all())
    diagnostics = dict(passed=passed, dense_max_absolute_error=float(error.max()),
                       dense_max_budget_ratio=float((error / fixture['budget']).max()))
    if arm == 'after':
        exact = bool(torch.equal(y, fixture['expected']))  # signed zeros compare equal
        if codes is not None:
            exact = exact and all(torch.equal(c.detach().cpu(), fixture[n]) for c, n in zip(codes, ('a1', 'a2')))
        diagnostics.update(passed=passed and exact, exact_reconstructed_fp16_and_codes=exact)
    for name, original in fixture['inputs'].items():
        if not torch.equal(inputs[name].detach().cpu(), original):
            raise ValueError('input mutated: ' + name)
    return diagnostics
