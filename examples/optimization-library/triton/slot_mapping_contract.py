"""Exact int32 contiguous request-local to global slot mapping contract."""
from pathlib import Path
import torch

KEYS = ('tok', 'request_id', 'prefill_id', 'prefill_mask', 'block_table', 'starts')

def parameters(spec):
    p = dict(spec.get('parameters', spec))
    for key in ('T', 'K', 'requests', 'block_size'):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError('positive integer ' + key + ' required')
    if p['block_size'] & (p['block_size'] - 1) or p['block_size'] > 4096:
        raise ValueError('block_size must be power of two, <=4096')
    if p.get('block_table_width', 128) != 128 or p.get('mode', 'decode') not in ('decode', 'mixed'):
        raise ValueError('unsupported table width or mode')
    p.setdefault('mode', 'decode')
    return p


def checked_inputs(fixture, p):
    t, k, r = (p[x] for x in ('T', 'K', 'requests'))
    shapes = ((t, k), (t,), (t,), (t,), (r, 128), (r,))
    for key, shape in zip(KEYS, shapes):
        x = fixture[key]
        if x.device.type != 'cpu' or x.dtype != torch.int32 or tuple(x.shape) != shape or not x.is_contiguous():
            raise ValueError('CPU contiguous int32 shape required: ' + key)
    for key in ('request_id', 'prefill_id'):
        if not bool(((fixture[key] >= 0) & (fixture[key] < r)).all()):
            raise ValueError('invalid request id')
    if not bool(((fixture['prefill_mask'] == 0) | (fixture['prefill_mask'] == 1)).all()):
        raise ValueError('prefill mask must be zero or one')
    # Conservative bounds certify every possible selected arithmetic result fits int32.
    if int(fixture['block_table'].min()) < 0 or int(fixture['block_table'].max()) * p['block_size'] + p['block_size'] - 1 > 2147483647:
        raise ValueError('block base overflow or negative base')
    if int(fixture['starts'].min()) < 0 or int(fixture['starts'].max()) + 128 * p['block_size'] - 1 > 2147483647:
        raise ValueError('workspace start overflow or negative start')


def reference(fixture, p):
    checked_inputs(fixture, p)
    tok = fixture['tok'].to(torch.int64)
    bid = torch.div(tok, p['block_size'], rounding_mode='floor')
    valid = (tok >= 0) & (bid < 128)
    req = fixture['request_id'].to(torch.int64)[:, None]
    base = fixture['block_table'].to(torch.int64)[req, bid.clamp(0, 127)]
    decode = base * p['block_size'] + tok.remainder(p['block_size'])
    prefill = fixture['starts'].to(torch.int64)[fixture['prefill_id'].long()][:, None] + tok
    return torch.where(valid, torch.where(fixture['prefill_mask'][:, None].bool(), prefill, decode), -1).to(torch.int32)


def make_case(spec):
    p = parameters(spec)
    t, k, r, b = (p[x] for x in ('T', 'K', 'requests', 'block_size'))
    rng = torch.Generator(device='cpu').manual_seed(spec.get('seed', 0))
    def random(high, shape):
        return torch.randint(high, shape, generator=rng, dtype=torch.int32, device='cpu')
    f = dict(tok=random(128*b, (t, k)), request_id=random(r, (t,)), prefill_id=random(r, (t,)),
             prefill_mask=torch.arange(t, dtype=torch.int32, device='cpu').remainder(2),
             block_table=random(4096, (r, 128)), starts=random(4096, (r,)) + 131072)
    f['tok'][:, 0] = -1
    if k > 1:
        f['tok'][:, 1] = 128*b
    if p['mode'] == 'decode':
        f['prefill_mask'].zero_()
    if p.get('distribution') == 'boundaries-and-large-exact-base':
        if k != 8:
            raise ValueError('boundary guard requires K8')
        f['tok'][:] = torch.tensor([-2, -1, 0, b-1, b, 128*b-1, 128*b, 1], dtype=torch.int32, device='cpu')
        f['request_id'][0] = 0
        f['block_table'][0, 0] = 524289
    f['expected'] = reference(f, p)
    return f

CASE_SOURCES = [Path(__file__)]


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.save(make_case(json.loads(args.case.read_text())), args.output)

if __name__ == '__main__':
    main()
