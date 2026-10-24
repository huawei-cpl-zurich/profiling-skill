"""E06 CPU payload provider, with one trajectory for compound snapshots."""
from pathlib import Path
import torch
import sinkhorn_contract as parent


def dimensions(spec):
    p, bs, iters, eps = parent.dimensions(spec)
    counts = p.get('counts_sequence')
    if counts is not None and (not isinstance(counts, list) or len(counts) != 2
            or any(type(n) is not int or not 1 <= n <= 20 for n in counts)
            or counts[0] != iters or counts[0] >= counts[1]):
        raise ValueError('counts_sequence requires two increasing counts, starting at iters, in 1..20')
    return p, bs, iters, eps


def make_inputs(spec):
    p, bs, _, _ = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    inputs = {k: (torch.rand(shape, generator=rng)*2-1)*(3 if k == 'mixes' else 1)
              for k, shape in dict(mixes=(bs,24), scale=(3,), base=(24,)).items()}
    if p.get('zero_scale_equal_logits'):
        inputs['scale'].zero_(); inputs['base'].zero_()
    parent.validate_inputs(inputs, spec)
    return inputs


def make_case(spec):
    p, bs, iters, eps = dimensions(spec)
    counts = p.get('counts_sequence')
    if counts is None:
        return parent.make_case(spec)
    inputs = make_inputs(spec)
    mixes, scale, base = (inputs[k].double() for k in ('mixes','scale','base'))
    pre = (torch.sigmoid(mixes[:,:4]*scale[0]+base[:4])+eps).float()
    post = (2*torch.sigmoid(mixes[:,4:8]*scale[1]+base[4:8])).float()
    logits = (mixes[:,8:]*scale[2]+base[8:]).reshape(bs,4,4)
    comb = torch.softmax(logits,-1)+eps
    comb = comb/(comb.sum(-2,keepdim=True)+eps)
    snapshots = {}
    for count in range(1, counts[-1]+1):
        if count in counts:
            snapshots[count] = (pre,post,comb.float())
        if count != counts[-1]:
            comb = comb/(comb.sum(-1,keepdim=True)+eps)
            comb = comb/(comb.sum(-2,keepdim=True)+eps)
    return dict(inputs=inputs,expected=tuple(snapshots[n] for n in counts))


CASE_SOURCES = [Path(__file__), Path(parent.__file__)]
