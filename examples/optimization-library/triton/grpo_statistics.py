# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025, DeepLink.
# Independent forward-only adaptation of grpo_loss.py at
# triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230 (MIT).
"""Materialized clipped policy versus tile sufficient statistics."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from grpo_contract import CASE_SOURCES, KEYS, checked_inputs, diagnostics, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_grpo_tiles(Logp, Old, Ref, Adv, Policy, Partial, KL, V: tl.constexpr,
                          TILES: tl.constexpr, KIND: tl.constexpr, MATERIALIZE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        col = tile*BLOCK + tl.arange(0, BLOCK)
        logp = tl.load(Logp+row*V+col, col < V, 0)
        old = tl.load(Old+row*V+col, col < V, 0)
        ref = tl.load(Ref+row*V+col, col < V, 0)
        adv = tl.load(Adv+row)
        ratio = tl.exp(logp-old)
        p = tl.maximum(-adv*ratio, -adv*tl.minimum(tl.maximum(ratio, .8), 1.2))
        k = ref-logp
        if KIND == 0:
            kl = k
        elif KIND == 1:
            kl = (tl.exp(k)-k)-1.
        else:
            kl = (k*k)*.5
        tl.store(KL+row*TILES+tile, tl.sum(tl.where(col < V, kl, 0), 0))
        if MATERIALIZE:
            tl.store(Policy+row*V+col, p, col < V)
        else:
            tl.store(Partial+row*TILES+tile, tl.sum(tl.where(col < V, p, 0), 0))

    @triton.jit
    def wave005_grpo_policy_reduce(Policy, Partial, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        col = tile*BLOCK + tl.arange(0, BLOCK)
        p = tl.load(Policy+row*V+col, col < V, 0)
        tl.store(Partial+row*TILES+tile, tl.sum(p, 0))

    @triton.jit
    def wave005_grpo_finish(Partial, KL, Out, T: tl.constexpr, V: tl.constexpr, TILES: tl.constexpr,
                           COEF: tl.constexpr, GLOBAL: tl.constexpr, ROW: tl.constexpr):
        row = tl.program_id(0)
        i = tl.arange(0, GLOBAL)
        policy = tl.sum(tl.load(Partial+i, i < T*TILES, 0), 0)
        j = tl.arange(0, ROW)
        kl = tl.sum(tl.load(KL+row*TILES+j, j < TILES, 0), 0)
        tl.store(Out+row, (policy+COEF*kl)*(1./(T*V)))
    return wave005_grpo_tiles, wave005_grpo_policy_reduce, wave005_grpo_finish


def identity(compiled):
    binaries = {}
    for key, value in compiled.asm.items():
        if isinstance(value, str): value = value.encode()
        if isinstance(value, (bytes, bytearray)): binaries[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self): self.prepared = {}

    def prepare(self, arm, spec, f, output):
        import torch
        import torch_npu
        import triton
        if arm not in ('before', 'after'): raise ValueError('unknown arm')
        p = parameters(spec)
        checked_inputs(f, p)
        inputs = tuple(f[k].npu() for k in KEYS)
        logp, old, ref, adv = inputs
        t, v = p['T'], p['V']
        tiles = triton.cdiv(v, 256)
        options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        policy = torch.empty((t, v), dtype=torch.float32, device=logp.device) if arm == 'before' else None
        partial = torch.empty((t, tiles), dtype=torch.float32, device=logp.device)
        kl = torch.empty_like(partial)
        out = torch.full((t,), float('nan'), dtype=torch.float32, device=logp.device)
        produce, reduce, finish = kernels()
        kind = ('kl', 'unbias', 'mse').index(p['kl_type'])
        launches = [(produce, (t, tiles), (logp, old, ref, adv, policy if policy is not None else partial,
                    partial, kl, v, tiles, kind, arm == 'before', 256))]
        if arm == 'before': launches.append((reduce, (t, tiles), (policy, partial, v, tiles, 256)))
        launches.append((finish, (t,), (partial, kl, out, t, v, tiles, p['kl_coef'],
                        triton.next_power_of_2(t*tiles), triton.next_power_of_2(tiles))))
        compiled = [fn.warmup(*args, grid=grid, **options) for fn, grid, args in launches]
        artifacts = []
        for index, item in enumerate(compiled):
            for key, data in item.asm.items():
                if isinstance(data, str): data = data.encode()
                if isinstance(data, (bytes, bytearray)):
                    path = output/f'kernel-{index}.{key}'
                    path.write_bytes(data)
                    artifacts.append(path)

        def run():
            for fn, grid, args in launches: fn[grid](*args, **options)
            return out

        item = SimpleNamespace(run=run, inputs=inputs, workspace=(policy, partial, kl),
            expected_kernels=[fn.fn.__name__ for fn, _, _ in launches],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(c) for c in compiled],
            metadata=dict(arm=arm, parameters=p, block=256, language='triton', torch_version=torch.__version__,
                triton_version=triton.__version__, logical_workspace_bytes=8*t*tiles+(4*t*v if arm=='before' else 0),
                preallocation='Inputs, output and scratch allocated before forward; all required scratch overwritten per API.',
                reduction='256-column tiles; row-major global policy reduction; row KL reduction; same tree in both arms.',
                objective='loss_factor*(global_policy_sum + kl_coef*row_kl_sum); forward only'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        result['inputs_immutable'] = all(torch.equal(x.cpu(), fixture[k]) for x, k in zip(self.prepared[arm].inputs, KEYS))
        result['passed'] = bool(result['passed'] and result['inputs_immutable'])
        return result
