"""Standalone external-gate LayerNorm: strict FP32 inputs and FP64 oracle."""
from pathlib import Path
import torch

INPUTS = ('x', 'U', 'gamma', 'beta')


def parameters(spec):
    p = spec['parameters']
    a, r, seed = p['A'], p['R'], spec['seed']
    if any(type(v) is not int for v in (a, r, seed)) or not 1 <= a <= 2048 or not 2 <= r <= 2048 or seed < 0:
        raise ValueError('positive bounded A/R and nonnegative integer seed required')
    if p.get('distribution', 'bounded-uniform') not in ('bounded-uniform', 'zero-negative-nonuniform-U'):
        raise ValueError('unsupported distribution')
    return a, r, seed


def checked_input(spec, fixture):
    a, r, _ = parameters(spec)
    if spec['parameters'].get('malformed_U_shape'):
        raise ValueError('U shape must equal x shape; broadcasting is unsupported')
    for key, shape in zip(INPUTS, ((a,r), (a,r), (r,), (r,))):
        value = fixture[key]
        if value.dtype != torch.float32 or value.shape != shape or not value.is_contiguous() or not bool(torch.isfinite(value).all()):
            raise ValueError('finite contiguous FP32 input with exact shape required: ' + key)
    return a, r


def reference(x, U, gamma, beta):
    x, U, gamma, beta = (v.double() for v in (x, U, gamma, beta))
    centered = x - x.mean(-1, keepdim=True)
    return (centered * (centered.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * gamma + beta) * U


def make_case(spec):
    a, r, seed = parameters(spec)
    if spec['parameters'].get('malformed_U_shape'):
        raise ValueError('U shape must equal x shape; broadcasting is unsupported')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    def uniform(shape, lo, hi):
        return torch.rand(shape, generator=rng, dtype=torch.float32, device='cpu') * (hi-lo) + lo
    x, U = uniform((a,r), -1, 1), uniform((a,r), -1, 1)
    gamma, beta = uniform((r,), .5, 1.5), uniform((r,), -.25, .25)
    if spec['parameters'].get('distribution') == 'zero-negative-nonuniform-U':
        U.flatten()[0::3] = 0
        U.flatten()[1::3] = -1
        beta.fill_(.125)
    return dict(x=x, U=U, gamma=gamma, beta=beta, expected=reference(x,U,gamma,beta))


def diagnostics(actual, fixture):
    expected = fixture['expected']
    if actual.dtype != torch.float32 or actual.shape != expected.shape or not bool(torch.isfinite(actual).all()):
        return dict(passed=False, reason='invalid output shape/dtype/finiteness')
    error = (actual.double() - expected).abs()
    return dict(passed=bool(torch.all(error <= 1e-4 + 1e-4*expected.abs())),
                max_absolute_error=float(error.max()), atol=1e-4, rtol=1e-4, oracle='CPU FP64')


CASE_SOURCES = [Path(__file__)]


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)


if __name__ == '__main__':
    main()
