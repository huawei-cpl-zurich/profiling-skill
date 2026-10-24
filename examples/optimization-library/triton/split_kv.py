"""Contiguous grouped decode attention: single KV range versus four plus merge.

Adapted from DeepLink/OpenMMLab/lmdeploy paged_attention.py (MIT), revision
7acb9c40a6ca, split kernel lines132-221 and merge225-272. Both arms use identical
head grouping, FP16 dot operands, FP32 state and online softmax. No page gather,
query blocking, sliding window or softcap transformation is included.
"""
from pathlib import Path
from types import SimpleNamespace
from attention import compiled_identity
from split_kv_contract import CASE_SOURCES, diagnostics, dimensions, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave003_decode_ranges(Q, K, V, P, O, HQ: tl.constexpr, HKV: tl.constexpr,
                              NK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                              NS: tl.constexpr, BN: tl.constexpr, BH: tl.constexpr):
        kvhead, part, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        group: tl.constexpr = HQ // HKV
        heads = kvhead * group + tl.arange(0, BH)
        valid_head = tl.arange(0, BH) < group
        dd = tl.arange(0, D)
        vv = tl.arange(0, DV)
        nn = tl.arange(0, BN)
        query = tl.load(Q + (batch * HQ + heads[:, None]) * D + dd[None, :],
                        valid_head[:, None], 0)
        maximum = tl.full((BH,), -float('inf'), tl.float32)
        denominator = tl.full((BH,), 0., tl.float32)
        numerator = tl.full((BH, DV), 0., tl.float32)
        extent: tl.constexpr = triton.cdiv(triton.cdiv(NK, BN), NS) * BN
        begin = part * extent
        end = tl.minimum(begin + extent, NK)
        for start in range(begin, end, BN):
            indices = start + nn
            key = tl.load(K + ((batch * HKV + kvhead) * NK + indices[None, :]) * D
                          + dd[:, None], indices[None, :] < NK, 0)
            value = tl.load(V + ((batch * HKV + kvhead) * NK + indices[:, None]) * DV
                            + vv[None, :], indices[:, None] < NK, 0)
            scores = tl.dot(query, key) * (D ** -0.5)
            scores = tl.where(indices[None, :] < NK, scores, -float('inf'))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            probability = tl.exp(scores - next_max[:, None])
            alpha = tl.exp(maximum - next_max)
            numerator = numerator * alpha[:, None] + tl.dot(probability.to(tl.float16), value)
            denominator = denominator * alpha + tl.sum(probability, 1)
            maximum = next_max
        if NS == 1:
            result = numerator / tl.where(denominator > 0, denominator, 1.)[:, None]
            tl.store(O + (batch * HQ + heads[:, None]) * DV + vv[None, :], result,
                     valid_head[:, None])
        else:
            base = ((batch * HQ + heads) * NS + part) * (DV + 2)
            tl.store(P + base[:, None] + vv[None, :], numerator, valid_head[:, None])
            tl.store(P + base + DV, maximum, valid_head)
            tl.store(P + base + DV + 1, denominator, valid_head)

    @triton.jit
    def wave003_decode_merge(P, O, DV: tl.constexpr, NS: tl.constexpr):
        head = tl.program_id(0)
        parts = tl.arange(0, NS)
        channels = tl.arange(0, DV)
        base = (head * NS + parts) * (DV + 2)
        maximum = tl.load(P + base + DV)
        denominator = tl.load(P + base + DV + 1)
        numerator = tl.load(P + base[:, None] + channels[None, :])
        merged_max = tl.max(maximum, 0)
        occupied = denominator > 0
        safe_max = tl.where(tl.sum(occupied.to(tl.int32), 0) > 0, merged_max, 0.)
        alpha = tl.exp(tl.where(occupied, maximum - safe_max, -float('inf')))
        total = tl.sum(denominator * alpha, 0)
        weighted = tl.sum(numerator * alpha[:, None], 0)
        result = weighted / tl.where(total > 0, total, 1.)
        tl.store(O + head * DV + channels, result)

    return wave003_decode_ranges, wave003_decode_merge


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        import torch_npu
        import triton
        if arm not in ('before', 'after'):
            raise ValueError('unknown arm')
        b, hq, hkv, nk, d, dv = dimensions(spec)
        group = hq // hkv
        for name, shape in zip(('q', 'k', 'v'), ((b,hq,1,d), (b,hkv,nk,d), (b,hkv,nk,dv))):
            value = fixture[name]
            if value.shape != shape or value.dtype != torch.float16 or not value.is_contiguous():
                raise ValueError('contiguous FP16 payload required')
            if not torch.isfinite(value).all() or (value.abs() > .25).any():
                raise ValueError('finite bounded FP16 inputs required')
        inputs = tuple(fixture[name].npu() for name in ('q', 'k', 'v'))
        out = torch.empty((b,hq,1,dv), device=inputs[0].device, dtype=torch.float16)
        splits = 4 if arm == 'after' and triton.cdiv(nk, 64) >= 4 else 1
        partial = torch.empty((b,hq,splits,dv+2), device=out.device, dtype=torch.float32)
        ranges, merge = kernels()
        args = (*inputs, partial, out, hq, hkv, nk, d, dv, splits, 64, 16)
        grid = (hkv, splits, b)
        # Compile-only preparation does not dispatch or initialize device data.
        compiled = [ranges.warmup(*args, grid=grid)]
        if splits > 1:
            compiled.append(merge.warmup(partial, out, dv, splits, grid=(b*hq,)))
        def run():
            ranges[grid](*args)
            if splits > 1:
                merge[(b*hq,)](partial, out, dv, splits)
            return out
        item = SimpleNamespace(run=run, inputs=inputs,
            sources=[Path(__file__), *CASE_SOURCES],
            expected_kernels=[kernel.name for kernel in compiled],
            compilation_state=lambda: [compiled_identity(kernel) for kernel in compiled],
            metadata=dict(arm=arm, dtype='fp16', state_dtype='fp32', query_tile=1,
                          head_tile=16, valid_heads_per_program=group, blockN=64,
                          split=splits, grid=list(grid), workspace_bytes=partial.numel()*4,
                          preallocation='output and fully overwritten partial state outside forward',
                          full_api='range kernel; all partial stores; nonlinear merge if split4',
                          empty_policy='unsplit zero numerator divided by safe unit denominator',
                          torch_version=torch.__version__, triton_version=triton.__version__))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.detach().cpu(), fixture)
        for value, name in zip(self.prepared[arm].inputs, ('q','k','v')):
            if not torch.equal(value.cpu(), fixture[name]):
                result.update(passed=False, reason='input mutation: '+name)
        return result
