"""Gather-then-attend versus direct paged addressing; one shared attention core.

The direct page mapping follows the paged-attention mechanism documented in
triton-ascend-kernels (7acb9c40a6ca), originating in DeepLink/OpenMMLab lmdeploy.
This controlled pair holds query blocking, causal masking and arithmetic fixed.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from paged_kv_contract import CASE_SOURCES, check_inputs, diagnostics, dimensions, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave003_paged_gather(K, V, Table, KC, VC, KV: tl.constexpr, HK: tl.constexpr,
                             D: tl.constexpr, DV: tl.constexpr, NP: tl.constexpr,
                             PAGE: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(1)
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        kt, kc = x // (HK * D), x % (HK * D)
        kp = tl.load(Table + b * NP + kt // PAGE, kt < KV, 0)
        kval = tl.load(K + (kp * PAGE + kt % PAGE) * HK * D + kc, kt < KV, 0)
        tl.store(KC + b * KV * HK * D + x, kval, kt < KV)
        vt, vc = x // (HK * DV), x % (HK * DV)
        vp = tl.load(Table + b * NP + vt // PAGE, vt < KV, 0)
        vval = tl.load(V + (vp * PAGE + vt % PAGE) * HK * DV + vc, vt < KV, 0)
        tl.store(VC + b * KV * HK * DV + x, vval, vt < KV)

    @triton.jit
    def wave003_paged_attention(Q, K, V, Table, O, HQ: tl.constexpr, HK: tl.constexpr,
                                NQ: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
                                DV: tl.constexpr, NP: tl.constexpr, PAGE: tl.constexpr,
                                DIRECT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        bh = tl.program_id(1)
        b, h = bh // HQ, bh % HQ
        kh = h // (HQ // HK)
        d, dv = tl.arange(0, D), tl.arange(0, DV)
        query = tl.load(Q + (bh * NQ + rows[:, None]) * D + d[None, :], rows[:, None] < NQ, 0)
        acc = tl.full((BM, DV), 0, tl.float32)
        if KV > 0:
            maximum = tl.full((BM,), -float('inf'), tl.float32)
            denominator = tl.full((BM,), 0, tl.float32)
            for start in range(0, KV, BN):
                tokens = start + tl.arange(0, BN)
                if DIRECT:
                    pages = tl.load(Table + b * NP + tokens // PAGE, tokens < KV, 0)
                    slots = pages * PAGE + tokens % PAGE
                else:
                    slots = b * KV + tokens
                key = tl.load(K + (slots[None, :] * HK + kh) * D + d[:, None], tokens[None, :] < KV, 0)
                value = tl.load(V + (slots[:, None] * HK + kh) * DV + dv[None, :], tokens[:, None] < KV, 0)
                score = tl.dot(query, key) * (D ** -0.5)
                valid = (tokens[None, :] < KV) & (tokens[None, :] <= KV - NQ + rows[:, None])
                score = tl.where(valid, score, -float('inf'))
                new_maximum = tl.maximum(maximum, tl.max(score, 1))
                alpha = tl.exp(maximum - new_maximum)
                probability = tl.exp(score - new_maximum[:, None])
                acc = acc * alpha[:, None] + tl.dot(probability.to(tl.float16), value)
                denominator = denominator * alpha + tl.sum(probability, 1)
                maximum = new_maximum
            acc = acc / denominator[:, None]
        tl.store(O + (bh * NQ + rows[:, None]) * DV + dv[None, :], acc, rows[:, None] < NQ)

    return wave003_paged_gather, wave003_paged_attention


def compiled_identity(artifact):
    artifacts = {}
    for key, value in getattr(artifact, 'asm', {}).items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            artifacts[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=getattr(artifact, 'hash', None), artifacts=artifacts)


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
            raise ValueError('arm must be before or after')
        check_inputs(spec, fixture)
        b, hq, hk, nq, kv, d, dv, page, _ = dimensions(spec)
        npages = max(1, triton.cdiv(kv, page))
        inputs = tuple(fixture[name].npu() for name in ('q', 'k', 'v', 'table'))
        query, key, value, table = inputs
        out = torch.full((b, hq, nq, dv), float('nan'), dtype=torch.float32, device=query.device)
        kc, vc = key, value
        if arm == 'before':
            kc = torch.empty((b, kv, hk, d), dtype=torch.float16, device=query.device)
            vc = torch.empty((b, kv, hk, dv), dtype=torch.float16, device=query.device)
        gather, attention = kernels()
        compiled = {}
        def retain(artifact):
            compiled[id(artifact)] = artifact
        def run():
            if arm == 'before' and kv:
                retain(gather[(triton.cdiv(kv * hk * max(d, dv), 256), b)](
                    key, value, table, kc, vc, kv, hk, d, dv, npages, page, 256))
            retain(attention[(triton.cdiv(nq, 16), b * hq)](
                query, kc, vc, table, out, hq, hk, nq, kv, d, dv, npages, page,
                arm == 'after', 16, 32))
            return out
        names = ([gather.fn.__name__] if arm == 'before' and kv else []) + [attention.fn.__name__]
        item = SimpleNamespace(run=run, expected_kernels=names, inputs=inputs,
            sources=[Path(__file__), *CASE_SOURCES],
            compilation_state=lambda: [compiled_identity(x) for x in compiled.values()],
            metadata=dict(language='triton', arm=arm, torch_version=torch.__version__,
                triton_version=triton.__version__, query_tile=16, key_tile=32,
                causal_rule='key_position <= KV-Q+query_position', probability_dtype='fp16',
                gqa_group=hq // hk, stages_per_forward=len(names), num_autotune_trials=0,
                workspace_bytes=2*b*kv*hk*(d+dv) if arm == 'before' else 0,
                empty_sequence='shared attention kernel explicitly writes zeros'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.detach().cpu(), fixture)
        for value, name in zip(self.prepared[arm].inputs, ('q', 'k', 'v', 'table')):
            if not torch.equal(value.cpu(), fixture[name]):
                result.update(passed=False, reason='input mutated: ' + name)
        return result


def factory():
    return Adapter()
