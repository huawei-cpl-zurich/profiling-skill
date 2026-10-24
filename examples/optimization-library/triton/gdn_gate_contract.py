"""Standalone FP32 dual-output GDN gates with an independent CPU FP64 oracle."""
from pathlib import Path
import torch

INPUTS = ('a', 'b', 'A_log', 'dt_bias')


def parameters(spec):
    p = spec['parameters']
    r, h, seed = p['R'], p['H'], spec['seed']
    if any(type(v) is not int for v in (r, h, seed)):
        raise ValueError('R, H and seed must be integers')
    if not 1 <= r <= 8192 or not 1 <= h <= 128 or seed < 0:
        raise ValueError('R in 1..8192, H in 1..128 and nonnegative seed required')
    distribution = p.get('distribution', 'bounded-uniform')
    if distribution not in ('bounded-uniform', 'threshold20', 'symmetric-b'):
        raise ValueError('unsupported distribution')
    if distribution == 'threshold20' and (r, h) != (2, 3):
        raise ValueError('threshold20 fixture requires R2 H3')
    return r, h, seed, distribution


def reference(a, b, A_log, dt_bias):
    a, b, A_log, dt_bias = (v.double() for v in (a, b, A_log, dt_bias))
    x = a + dt_bias
    softplus = torch.where(x > 20, x, torch.log1p(torch.exp(x)))
    return (-torch.exp(A_log) * softplus).unsqueeze(0), torch.sigmoid(b).unsqueeze(0)


def make_case(spec):
    r, h, seed, distribution = parameters(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    def uniform(shape, low, high):
        return torch.rand(shape, generator=rng, dtype=torch.float32, device='cpu') * (high-low) + low
    a, b = uniform((r,h), -4, 4), uniform((r,h), -4, 4)
    A_log, dt_bias = uniform((h,), -2, 2), uniform((h,), -1, 1)
    if distribution == 'threshold20':
        dt_bias.zero_()
        a[:] = torch.tensor([20-1/1024, 20, 20+1/1024], dtype=torch.float32, device='cpu')
    if distribution == 'symmetric-b':
        b[0] = -torch.linspace(0, 4, h, dtype=torch.float32, device='cpu')
        b[-1] = -b[0]
    return dict(a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                expected=reference(a,b,A_log,dt_bias))


def checked_inputs(spec, fixture):
    r, h, _, distribution = parameters(spec)
    for key, shape, limit in zip(INPUTS, ((r,h),(r,h),(h,),(h,)), (21 if distribution=='threshold20' else 4,4,2,1)):
        value = fixture[key]
        if (value.dtype != torch.float32 or value.shape != shape or not value.is_contiguous()
                or not bool(torch.isfinite(value).all()) or bool((value.abs() > limit).any())):
            raise ValueError('contiguous finite bounded FP32 input required: ' + key)


def diagnostics(actual, fixture):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        return dict(passed=False, reason='expected g and beta')
    metrics = {}
    for name, value, expected in zip(('g','beta'), actual, fixture['expected']):
        if (value.dtype != torch.float32 or value.device.type != 'cpu'
                or value.shape != expected.shape or not bool(torch.isfinite(value).all())):
            return dict(passed=False, reason='invalid ' + name)
        error = (value.double()-expected).abs()
        metrics[name] = dict(max_abs_error=float(error.max()),
            passed=bool(torch.all(error <= 2e-5 + 2e-4 * expected.abs())))
    return dict(passed=all(v['passed'] for v in metrics.values()), metrics=metrics,
                tolerance=dict(atol=2e-5, rtol=2e-4), oracle='CPU FP64 represented inputs')


CASE_SOURCES = [Path(__file__)]


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)


if __name__ == '__main__': main()
