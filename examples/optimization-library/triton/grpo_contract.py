"""Forward-only GRPO objective: one global policy sum plus row KL."""
from pathlib import Path
import math
import torch

KEYS = ('logp', 'old', 'ref', 'adv')

def parameters(spec):
    p = dict(spec.get('parameters', spec))
    for key in ('T', 'V'):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError('positive integer ' + key + ' required')
    if p.get('kl_type') not in ('kl', 'unbias', 'mse'):
        raise ValueError('unsupported KL type')
    p.setdefault('kl_coef', .1)
    if type(p['kl_coef']) not in (int, float) or not math.isfinite(p['kl_coef']) or p['kl_coef'] < 0:
        raise ValueError('finite nonnegative KL coefficient required')
    if p.get('distribution', 'uniform') not in ('uniform', 'advantage-sign-clip-boundaries'):
        raise ValueError('unsupported distribution')
    return p


def checked_inputs(f, p):
    for key in KEYS:
        x = f[key]
        shape = (p['T'],) if key == 'adv' else (p['T'], p['V'])
        if x.dtype != torch.float32 or x.device.type != 'cpu' or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('contiguous CPU FP32 shape required: ' + key)
        if not bool(torch.isfinite(x).all()) or not bool((x.abs() <= 1).all()):
            raise ValueError('finite inputs in [-1,1] required: ' + key)


def reference(f, p):
    checked_inputs(f, p)
    logp, old, ref, adv = (f[k].double() for k in KEYS)
    ratio = (logp-old).exp()
    policy = torch.maximum(-adv[:, None]*ratio, -adv[:, None]*ratio.clamp(.8, 1.2))
    k = ref-logp
    kl = k if p['kl_type'] == 'kl' else k.exp()-k-1 if p['kl_type'] == 'unbias' else k.square()/2
    return (policy.sum() + p['kl_coef']*kl.sum(dim=1))/(p['T']*p['V'])


def make_case(spec):
    p = parameters(spec)
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    f = {k: torch.rand((p['T'],) if k == 'adv' else (p['T'], p['V']), generator=rng,
                      dtype=torch.float32, device='cpu')*2-1 for k in KEYS}
    if p.get('distribution') == 'advantage-sign-clip-boundaries':
        if p['T'] != 3:
            raise ValueError('sign guard requires T3')
        values = torch.tensor([math.log(.8), 0., math.log(1.2), -.5, .5, 0., math.log(.8), math.log(1.2)], dtype=torch.float32, device='cpu')
        f['logp'][:] = values.repeat((p['V']+7)//8)[:p['V']]
        f['old'].zero_()
        f['adv'][:] = torch.tensor([-1., 0., 1.], dtype=torch.float32, device='cpu')
        f['ref'][:] = torch.tensor([-.25, 0., .25], dtype=torch.float32, device='cpu')[:, None]
    f['expected'] = reference(f, p)
    return f


def diagnostics(actual, f):
    cpu = actual.cpu()
    valid = cpu.dtype == torch.float32 and cpu.shape == f['expected'].shape
    finite = bool(torch.isfinite(cpu).all())
    error = float((cpu.double()-f['expected']).abs().max()) if valid else None
    return dict(passed=valid and finite and torch.allclose(cpu.double(), f['expected'], atol=2e-5, rtol=2e-4),
                max_absolute_error=error, finite=finite, atol=2e-5, rtol=2e-4)

CASE_SOURCES = [Path(__file__)]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)

if __name__ == '__main__': main()
