"""Standalone FP32 residual token-mixing contract and independent FP64 oracle."""
from pathlib import Path
import torch

INPUTS = ('x', 'gamma', 'beta')


def parameters(spec):
    p = spec['parameters']
    b, s, seed = p['B'], p['S'], spec['seed']
    if any(type(v) is not int for v in (b, s, seed)) or not 1 <= b <= 8 or not 2 <= s <= 512 or seed < 0:
        raise ValueError('B in 1..8, S in 2..512 and nonnegative integer seed required')
    if p.get('malformed_nonsquare', False):
        raise ValueError('square token/channel dimensions required')
    distribution = p.get('distribution', 'bounded-uniform')
    if distribution not in ('bounded-uniform', 'constant-and-symmetric'):
        raise ValueError('unsupported distribution')
    return b, s, seed, distribution


def reference(x, gamma, beta):
    z = x.double() + x.double().transpose(-1, -2)
    centered = z - z.mean(-1, keepdim=True)
    return centered * (centered.square().mean(-1, keepdim=True) + 1e-5).rsqrt() * gamma.double() + beta.double()


def make_case(spec):
    b, s, seed, distribution = parameters(spec)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    def uniform(shape, lo, hi):
        return torch.rand(shape, generator=rng, dtype=torch.float32, device='cpu') * (hi-lo) + lo
    x = uniform((b,s,s), -1, 1)
    gamma, beta = uniform((s,), .5, 1.5), uniform((s,), -.25, .25)
    if distribution == 'constant-and-symmetric':
        x[0].fill_(.5)
        if b > 1:
            x[1] = (x[1] + x[1].T) * .5
    return dict(x=x, gamma=gamma, beta=beta, expected=reference(x,gamma,beta))


def check_inputs(fixture, b, s):
    for key, shape in zip(INPUTS, ((b,s,s), (s,), (s,))):
        value = fixture[key]
        lo, hi = {'x':(-1,1),'gamma':(.5,1.5),'beta':(-.25,.25)}[key]
        if value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous() or not bool(torch.isfinite(value).all()) or not bool(((value>=lo)&(value<=hi)).all()):
            raise ValueError('finite contiguous FP32 input required: ' + key)


CASE_SOURCES = [Path(__file__)]


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)
