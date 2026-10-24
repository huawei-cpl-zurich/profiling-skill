"""F09 finite independent FP16 grouped-M GEMMs and descriptor validation."""
import torch


def check_spec(spec):
    p = spec['parameters']
    ms = p['Ms']
    if not isinstance(ms, list) or not 1 <= len(ms) <= 64:
        raise ValueError('one to 64 groups required')
    if any(type(m) is not int or m < 0 for m in ms) or not sum(ms):
        raise ValueError('nonnegative group rows and nonempty union required')
    if any(type(p[x]) is not int or not 0 < p[x] <= 2048 for x in ('N', 'K')):
        raise ValueError('positive bounded N/K required')
    if sum(ms) > 8192:
        raise ValueError('group union exceeds example limit')
    return p


def check_prefix(prefix, groups, total):
    if len(prefix) != groups + 1 or prefix[0] != 0 or prefix[-1] != total:
        raise ValueError('prefix extents differ')
    if any(type(x) is not int for x in prefix) or any(b < a for a, b in zip(prefix, prefix[1:])):
        raise ValueError('prefix must be monotonically nondecreasing')
    return [b - a for a, b in zip(prefix, prefix[1:])]


def make_case(spec):
    p = check_spec(spec)
    ms, n, k = p['Ms'], p['N'], p['K']
    rng = torch.Generator().manual_seed(spec['seed'])
    a = ((torch.rand(sum(ms), k, generator=rng) * 2 - 1) * .125).half()
    b = ((torch.rand(len(ms), k, n, generator=rng) * 2 - 1) * .125).half()
    expected = torch.empty((sum(ms), n), dtype=torch.float32)
    offset = 0
    for group, m in enumerate(ms):
        expected[offset:offset+m] = (a[offset:offset+m].double() @ b[group].double()).float()
        offset += m
    return dict(inputs=dict(a=a, b=b, ms=torch.tensor(ms, dtype=torch.int32)), expected=expected)


def validate_actual(actual, fixture, inputs):
    if actual is True:
        return dict(passed=True, rejected_decreasing_prefix=True)
    value = actual.detach().cpu()
    target = fixture['expected']
    if value.shape != target.shape or value.dtype != torch.float32:
        raise ValueError('output shape/dtype differs')
    error = (value - target).abs()
    passed = bool(torch.isfinite(value).all() and torch.all(error <= .003 + .003 * target.abs()))
    for name, original in fixture['inputs'].items():
        if not torch.equal(inputs[name].cpu(), original):
            raise ValueError('input mutated: ' + name)
    return dict(passed=passed, atol=.003, rtol=.003, max_absolute_error=float(error.max()))
