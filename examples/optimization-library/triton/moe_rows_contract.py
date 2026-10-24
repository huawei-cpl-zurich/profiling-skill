"""CPU-only supplied-map admission and independent bit-preserving oracle."""
from pathlib import Path
import torch


def dimensions(spec):
    p = spec.get('parameters', spec)
    n, d = p.get('N'), p.get('D')
    if (type(n) is not int or not 0 <= n < 2**24 or type(d) is not int
            or not 0 < d <= 4096 or d & (d-1) or n*d >= 2**31
            or type(p.get('forward')) is not bool or type(p.get('with_probs')) is not bool
            or type(spec.get('seed')) is not int or spec['seed'] < 0):
        raise ValueError('unsupported supplied-map case')
    return p, n, d


def validate_inputs(inputs, spec):
    p, n, d = dimensions(spec)
    shapes = dict(x=((n,d), torch.float32), row_map=((n,), torch.int32))
    if p['with_probs']: shapes['probs'] = ((n,), torch.float32)
    if set(inputs) != set(shapes): raise ValueError('input keys disagree with probability mode')
    pointers = set()
    for key, (shape, dtype) in shapes.items():
        v = inputs[key]
        if (not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != dtype
                or tuple(v.shape) != shape or not v.is_contiguous()
                or v.storage_offset() != 0 or v.untyped_storage().nbytes() != v.numel()*v.element_size()):
            raise ValueError('owned contiguous CPU input required: '+key)
        ptr = v.untyped_storage().data_ptr()
        if v.numel() and ptr in pointers: raise ValueError('aliased inputs')
        if v.numel(): pointers.add(ptr)
    if not torch.equal(torch.sort(inputs['row_map']).values, torch.arange(n, dtype=torch.int32)):
        raise ValueError('row map must be a bijection on [0,N)')


def reference(inputs, forward):
    mapping = inputs['row_map'].long()
    def move(value):
        bits = value.view(torch.int32)
        if forward:
            output = torch.empty_like(bits); output[mapping] = bits
        else: output = bits[mapping].clone()
        return output.view(torch.float32)
    return (move(inputs['x']), move(inputs['probs']) if 'probs' in inputs else None)


def make_case(spec):
    p, n, d = dimensions(spec)
    rng = torch.Generator().manual_seed(spec['seed'])
    chunks = p.get('chunks', 1)
    if type(chunks) is not int or chunks <= 0: raise ValueError('positive chunk count required')
    kind = p.get('map_kind', 'identity')
    if kind not in ('identity','empty','shuffled_chunks','skew_zero_chunks','reverse_chunks'):
        raise ValueError('unsupported map generation')
    if kind == 'empty' and n: raise ValueError('empty map requires N=0')
    sizes = [n//chunks + (i < n%chunks) for i in range(chunks)]
    if kind == 'skew_zero_chunks': sizes = [0]*(chunks-1)+[n]
    order = list(range(chunks))
    if kind in ('shuffled_chunks','skew_zero_chunks'): order = torch.randperm(chunks,generator=rng).tolist()
    if kind == 'reverse_chunks': order.reverse()
    source_chunks = torch.arange(n,dtype=torch.int32).split(sizes)
    ordered = torch.cat([source_chunks[i] for i in order])
    row_map = torch.empty(n,dtype=torch.int32); row_map[ordered.long()] = torch.arange(n,dtype=torch.int32)
    inputs = dict(x=torch.randn((n,d),generator=rng), row_map=row_map)
    if p['with_probs']: inputs['probs'] = torch.randn(n,generator=rng)
    for key in ('x','probs'):
        if key not in inputs: continue
        flat = inputs[key].view(torch.int32).flatten()
        payload = [0, -2147483648, 2139095040, -8388608, 2143289345, -4194302] if p.get('ieee_payloads') else [0,-2147483648]
        for i,v in enumerate(payload[:flat.numel()]): flat[i] = v
    validate_inputs(inputs,spec)  # Admission precedes generic cache creation.
    return dict(inputs=inputs, expected=reference(inputs,p['forward']))


CASE_SOURCES = [Path(__file__)]
