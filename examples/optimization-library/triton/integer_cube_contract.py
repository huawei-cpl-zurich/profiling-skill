"""Bounded prequantized scaled integer GEMM; no quantization is performed."""
import torch


def check_spec(spec):
    p = spec['parameters']
    if any(type(p.get(k)) is not int or p[k] <= 0 for k in ('M', 'N', 'K', 'code_max')):
        raise ValueError('positive integer dimensions and code bound required')
    if p['code_max'] > 127 or p.get('output') != 'fp32':
        raise ValueError('signed int8 codes and FP32 output required')
    if p['M'] * p['N'] > 1048576 or p['K'] > 65536:
        raise ValueError('example resource limit exceeded')
    return p


def integer_eligible(spec):
    p = check_spec(spec)
    bound = p['K'] * p['code_max'] ** 2
    return bound <= 2**24 and bound < 2**31


def make_case(spec):
    p = check_spec(spec)
    m, n, k, bound = (p[x] for x in ('M', 'N', 'K', 'code_max'))
    g = torch.Generator().manual_seed(spec['seed'])
    a = torch.randint(-bound, bound + 1, (m, k), generator=g, dtype=torch.int8)
    b = torch.randint(-bound, bound + 1, (k, n), generator=g, dtype=torch.int8)
    if spec.get('distribution') == 'positive-bound':
        a.fill_(bound)
        b.fill_(bound)
    sa = torch.pow(2., torch.randint(-5, 0, (m, 1), generator=g).float())
    sb = torch.pow(2., torch.randint(-5, 0, (n, 1), generator=g).float())
    # Independent int64 accumulation precedes the exact specified FP32 order.
    dot = a.to(torch.int64) @ b.to(torch.int64)
    expected = sb.T * (sa * dot.float())
    return dict(inputs=dict(a=a, b=b, sa=sa, sb=sb), expected=expected)


def validate_actual(actual, fixture, inputs):
    result = actual.detach().cpu()
    expected = fixture['expected']
    if result.shape != expected.shape or result.dtype != torch.float32:
        raise ValueError('output shape/dtype differs')
    # Compare representations as the frozen gate requires bitwise equality.
    passed = bool(torch.equal(result.view(torch.int32), expected.view(torch.int32)))
    for name, original in fixture['inputs'].items():
        if not torch.equal(inputs[name].detach().cpu(), original):
            raise ValueError('input mutated: ' + name)
    return dict(passed=passed, bitwise=True, max_absolute_error=float((result - expected).abs().max()))
