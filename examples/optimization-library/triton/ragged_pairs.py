"""Two independent ragged-pack pairs; no device execution in preparation.

Source: ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L24-L51
Prefix reuse is an independent scan+consumer adaptation. Store elimination
may already be performed by a compiler; actual measurements decide its value.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from ragged_contract import check_spec, make_case, validate_actual


def configuration(arm, spec):
    check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    cached = spec['family'] == 'prefix-offsets' and arm == 'after'
    return dict(cached_offsets=cached,
                single_store=spec['family'] == 'prefix-offsets' or arm == 'after',
                block_t=32, block_d=64,
                scan_width=1 << (spec['batch'] - 1).bit_length(),
                grid=(spec['batch'], (spec['length'] + 31) // 32, (spec['width'] + 63) // 64))


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def ragged_prefix(Lengths, Starts, B: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.arange(0, BLOCK)
        count = tl.load(Lengths + row, row < B, other=0)
        exclusive = tl.cumsum(count, 0) - count
        tl.store(Starts + row, exclusive, row < B)

    @triton.jit
    def ragged_pack(X, Lengths, Starts, Out, L: tl.constexpr, D: tl.constexpr,
                    PAD: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                    CACHED: tl.constexpr, SINGLE: tl.constexpr):
        batch = tl.program_id(0)
        time = tl.program_id(1) * BT + tl.arange(0, BT)
        feature = tl.program_id(2) * BD + tl.arange(0, BD)
        start = tl.full((), 0, tl.int32)
        if CACHED:
            start = tl.load(Starts + batch)
        else:
            for previous in range(batch):
                start += tl.load(Lengths + previous)
        count = tl.load(Lengths + batch)
        bounds = (time[:, None] < L) & (feature[None, :] < D)
        valid = bounds & (time[:, None] < count)
        value = tl.load(X + (start + time[:, None]) * D + feature[None, :], valid, other=0)
        destination = Out + (batch * L + time[:, None]) * D + feature[None, :]
        if SINGLE:
            tl.store(destination, tl.where(time[:, None] < count, value, PAD), bounds)
        else:
            tl.store(destination, tl.full((BT, BD), PAD, tl.float32), bounds)
            tl.store(destination, value, valid)

    return ragged_prefix, ragged_pack


def compiled_identity(compiled):
    return [dict(name=k.name, hash=str(k.hash), binaries={name: hashlib.sha256(value).hexdigest()
            for name, value in k.asm.items() if isinstance(value, bytes)}) for k in compiled.values()]


class Adapter:
    case_sources = [Path(__file__).with_name('ragged_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu  # Backend registration; no forward.
        config = configuration(arm, spec)
        prefix, pack = kernels()
        inputs = {name: value.to(device='npu:0', memory_format=torch.preserve_format)
                  for name, value in fixture['inputs'].items()}
        for name, value in inputs.items():
            if value.stride() != fixture['inputs'][name].stride():
                raise ValueError('device transfer changed input strides')
        self.inputs[arm] = inputs
        result = torch.empty_like(fixture['expected'], device='npu:0')
        starts = (torch.empty(spec['batch'], dtype=torch.int32, device='npu:0')
                  if config['cached_offsets'] else inputs['lengths'])
        compiled = {}

        def run():
            if config['cached_offsets']:
                compiled['prefix'] = prefix[(1,)](inputs['lengths'], starts, spec['batch'], config['scan_width'])
            compiled['pack'] = pack[config['grid']](inputs['x'], inputs['lengths'], starts, result,
                spec['length'], spec['width'], fixture['padding'], config['block_t'], config['block_d'],
                config['cached_offsets'], config['single_store'])
            return result

        return SimpleNamespace(run=run,
            expected_kernels=([prefix.fn.__name__] if config['cached_offsets'] else []) + [pack.fn.__name__],
            sources=[__file__, *self.case_sources], compilation_state=lambda: compiled_identity(compiled),
            metadata=dict(backend='triton', family=spec['family'], configuration=config,
                          workspace_bytes=spec['batch'] * 4 if config['cached_offsets'] else 0,
                          allocation_boundary='output and optional offsets preallocated; scan included in each forward',
                          input_strides={name: list(t.stride()) for name, t in inputs.items()}))

    def validate(self, arm, actual, fixture, spec):
        return validate_actual(actual, fixture, self.inputs[arm])


def factory():
    return Adapter()
