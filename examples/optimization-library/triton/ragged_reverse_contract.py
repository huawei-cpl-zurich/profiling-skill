# SPDX-License-Identifier: Apache-2.0
"""Standalone FP32 ragged reverse with explicit length clamping."""
from pathlib import Path
import torch

PATTERNS = ('mostly-full', 'all-zero', 'mixed-long', 'short-zero', 'clamp-extremes', 'mixed-short')

def parameters(spec):
    p = dict(spec.get('parameters', spec))
    for key in ('B', 'T', 'D'):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError('positive integer ' + key + ' required')
    if p['B'] * p['T'] * p['D'] >= 2**31:
        raise ValueError('int32 indexing scope exceeded')
    if p.get('length_pattern') not in PATTERNS:
        raise ValueError('unsupported length pattern')
    if p['length_pattern'] == 'clamp-extremes' and p['B'] != 4:
        raise ValueError('clamp guard requires four batches')
    if p['length_pattern'] == 'mixed-short' and p['B'] != 3:
        raise ValueError('suffix guard requires three batches')
    return p


def checked_inputs(f, p):
    for key, shape, dtype in (('x', (p['B'], p['T'], p['D']), torch.float32), ('lengths', (p['B'],), torch.int32)):
        x = f[key]
        if x.device.type != 'cpu' or x.dtype != dtype or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('CPU contiguous dtype/shape required: ' + key)
    if not bool(torch.isfinite(f['x']).all()):
        raise ValueError('finite x required')


def reference(x, lengths):
    # Integer gather mapping, rather than clone-and-overwrite implementation.
    b, t, d = x.shape
    i = torch.arange(t, dtype=torch.int64, device='cpu')[None, :].expand(b, t)
    length = lengths.to(torch.int64).clamp(0, t)[:, None]
    source = torch.where(i < length, length - 1 - i, i)
    return x.gather(1, source[:, :, None].expand(b, t, d))


def make_case(spec):
    p = parameters(spec); b, t, d = (p[k] for k in ('B', 'T', 'D'))
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    x = torch.rand((b, t, d), generator=rng, dtype=torch.float32, device='cpu') * 2 - 1
    pattern = p['length_pattern']
    if pattern == 'mostly-full':
        lengths = t - torch.arange(b, dtype=torch.int32, device='cpu').remainder(max(1, t//4))
    elif pattern == 'all-zero':
        lengths = torch.zeros(b, dtype=torch.int32, device='cpu')
    elif pattern == 'mixed-long':
        lengths = torch.randint(max(1, t//2), t+1, (b,), generator=rng, dtype=torch.int32, device='cpu')
    elif pattern == 'short-zero':
        lengths = torch.arange(b, dtype=torch.int32, device='cpu').remainder(2) * min(3, t)
    else:
        values = [-3, 0, t, t+5] if pattern == 'clamp-extremes' else [1, min(7, t), 0]
        lengths = torch.tensor(values, dtype=torch.int32, device='cpu')
    f = dict(x=x, lengths=lengths)
    checked_inputs(f, p)
    f['one_reverse'] = reference(x, lengths)
    f['expected'] = x.clone() if pattern == 'clamp-extremes' else f['one_reverse'].clone()
    return f

CASE_SOURCES = [Path(__file__)]


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)

if __name__ == '__main__':
    main()
