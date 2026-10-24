"""Recovered-token sampler fixtures: exact integer argmax, FP64 CPU scoring."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    lengths, vocab = p.get('lengths'), p.get('vocab')
    if (not isinstance(lengths, list) or not lengths or len(lengths) > 128
            or any(type(x) is not int or not 0 <= x <= 16 for x in lengths)
            or sum(lengths) == 0 or sum(lengths) > 256
            or type(vocab) is not int or not 1 <= vocab <= 65536
            or type(p.get('draft_probs_present')) is not bool
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported sampler case')
    return p, lengths, vocab


def reference(inputs):
    target = inputs['target'].double().clone()
    ends, ids = inputs['ends'], inputs['ids']
    draft = inputs.get('draft')
    if draft is None:
        target[torch.arange(len(ids)), ids.long()] = 0
    else:
        target = (target - draft.double()).clamp_min(0)
    lengths = torch.diff(torch.cat((torch.zeros(1, dtype=ends.dtype), ends)))
    q = torch.repeat_interleave(inputs['q'].double(), lengths.long(), dim=0)
    return (target / q).argmax(-1).to(torch.int32)


def qualify_ranking(inputs, expected):
    """Reject FP32 score ordering that disagrees with the trusted FP64 winner.

    The FP64 oracle remains authoritative. Cached expected tensors are reused;
    this admission check does not evaluate that oracle again during preparation.
    """
    target = inputs['target'].clone()
    draft = inputs.get('draft')
    if draft is None:
        target[torch.arange(len(inputs['ids'])), inputs['ids'].long()] = 0
    else:
        target = (target - draft).clamp_min(0)
    ends = inputs['ends']
    lengths = torch.diff(torch.cat((torch.zeros(1, dtype=ends.dtype), ends)))
    q = torch.repeat_interleave(inputs['q'], lengths.long(), dim=0)
    modeled = (target / q).argmax(-1).to(torch.int32)
    if (expected.device.type != 'cpu' or expected.dtype != torch.int32
            or expected.shape != modeled.shape or not torch.equal(modeled, expected)):
        raise ValueError('unsupported sampler ranking: FP32 scores disagree with FP64 oracle')


def make_case(spec):
    p, lengths, vocab = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    n, batch = sum(lengths), len(lengths)
    def probability():
        weights = torch.randint(1, 256, (n, vocab), generator=rng).double()
        return (weights / weights.sum(-1, keepdim=True)).float()
    inputs = dict(target=probability(), q=2. ** torch.randint(-1, 3, (batch, vocab), generator=rng).float(),
                  ids=torch.randint(vocab, (n,), generator=rng, dtype=torch.int32),
                  ends=torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32))
    if p['draft_probs_present']:
        inputs['draft'] = probability()
    if p.get('tie_and_allzero'):
        if n != 3 or vocab < 4097 or p['draft_probs_present']:
            raise ValueError('tie guard needs three rows and vocab >=4097')
        inputs['target'].zero_()
        inputs['target'][0, 4095:4097] = .5
        inputs['target'][1, [0, 4095]] = .5
        inputs['q'].fill_(1)
        inputs['ids'] = torch.tensor(p.get('draft_ids', [0, 4096, 4095]), dtype=torch.int32)
    expected = reference(inputs)
    qualify_ranking(inputs, expected)  # Before the generic cache writes a payload.
    return dict(inputs=inputs, expected=expected)


def validate_inputs(inputs, spec):
    p, lengths, vocab = dimensions(spec)
    n, batch = sum(lengths), len(lengths)
    shapes = dict(target=(n, vocab), q=(batch, vocab), ids=(n,), ends=(batch,))
    if p['draft_probs_present']:
        shapes['draft'] = (n, vocab)
    if set(inputs) != set(shapes):
        raise ValueError('fixture input keys disagree with draft-probability mode')
    storages = set()
    for name, shape in shapes.items():
        value = inputs[name]
        dtype = torch.int32 if name in ('ids', 'ends') else torch.float32
        if (value.device.type != 'cpu' or tuple(value.shape) != shape or value.dtype != dtype
                or not value.is_contiguous() or value.storage_offset() != 0
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()
                or not torch.isfinite(value).all()):
            raise ValueError('invalid owned contiguous CPU input: ' + name)
        pointer = value.untyped_storage().data_ptr()
        if pointer in storages:
            raise ValueError('aliased input storage unsupported')
        storages.add(pointer)
    if not torch.equal(inputs['ends'], torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)):
        raise ValueError('cumulative endpoints disagree with lengths')
    if not ((inputs['ids'] >= 0) & (inputs['ids'] < vocab)).all() or not (inputs['q'] > 0).all():
        raise ValueError('invalid IDs or nonpositive q')
    for key in ('target', 'draft'):
        if key in inputs and not (inputs[key] >= 0).all():
            raise ValueError('probabilities must be nonnegative')


CASE_SOURCES = [Path(__file__)]
