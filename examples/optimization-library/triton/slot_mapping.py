# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright contributors to the vLLM project.
# Independent integer-pipeline adaptation of triton-ascend-kernels convert_index.py,
# commit 7acb9c40a6ca827e885d902b7eaee5d45457b230 (MIT).
"""Three materializing stages versus one fused stage; identical integer mapping."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from slot_mapping_contract import KEYS, CASE_SOURCES, checked_inputs, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_slot_decompose(Tok, Bid, Offset, Valid, N: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tok = tl.load(Tok+i, i < N, 0)
        valid = (tok >= 0) & (tok < 128*B)
        tl.store(Bid+i, tl.minimum(tl.maximum(tok // B, 0), 127), i < N)
        tl.store(Offset+i, tok % B, i < N)
        tl.store(Valid+i, valid.to(tl.int32), i < N)

    @triton.jit
    def wave005_slot_gather(Req, Prefill, Mask, Table, Starts, Bid, Base, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = i // K
        req = tl.load(Req+row, i < N, 0)
        pre = tl.load(Prefill+row, i < N, 0)
        is_pre = tl.load(Mask+row, i < N, 0) != 0
        bid = tl.load(Bid+i, i < N, 0)
        base = tl.load(Table+req*128+bid, (i < N) & ~is_pre, 0)
        start = tl.load(Starts+pre, (i < N) & is_pre, 0)
        tl.store(Base+i, tl.where(is_pre, start, base), i < N)

    @triton.jit
    def wave005_slot_emit(Tok, Mask, Offset, Valid, Base, Out, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tok = tl.load(Tok+i, i < N, 0)
        valid = tl.load(Valid+i, i < N, 0) != 0
        offset = tl.load(Offset+i, i < N, 0)
        base = tl.load(Base+i, i < N, 0)
        is_pre = tl.load(Mask+i//K, i < N, 0) != 0
        slot = tl.where(is_pre, base+tok, base*B+offset)
        tl.store(Out+i, tl.where(valid, slot, -1), i < N)

    @triton.jit
    def wave005_slot_fused(Tok, Req, Prefill, Mask, Table, Starts, Out, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tok = tl.load(Tok+i, i < N, 0)
        valid = (tok >= 0) & (tok < 128*B)
        bid = tl.minimum(tl.maximum(tok // B, 0), 127)
        req = tl.load(Req+i//K, i < N, 0)
        pre = tl.load(Prefill+i//K, i < N, 0)
        is_pre = tl.load(Mask+i//K, i < N, 0) != 0
        base = tl.load(Table+req*128+bid, (i < N) & ~is_pre, 0)
        start = tl.load(Starts+pre, (i < N) & is_pre, 0)
        slot = tl.where(is_pre, start+tok, base*B+tok%B)
        tl.store(Out+i, tl.where(valid, slot, -1), i < N)

    return wave005_slot_decompose, wave005_slot_gather, wave005_slot_emit, wave005_slot_fused


def identity(compiled):
    binaries = {}
    for key, data in compiled.asm.items():
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, (bytes, bytearray)):
            binaries[key] = hashlib.sha256(data).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


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
        p = parameters(spec)
        checked_inputs(fixture, p)
        inputs = tuple(fixture[key].npu() for key in KEYS)
        tok, req, pre, mask, table, starts = inputs
        n, k, b = p['T']*p['K'], p['K'], p['block_size']
        out = torch.full((p['T'], k), -2147483648, dtype=torch.int32, device=tok.device)
        decompose, gather, emit, fused = kernels()
        workspace = tuple(torch.empty_like(out) for _ in range(4)) if arm == 'before' else ()
        if arm == 'before':
            bid, offset, valid, base = workspace
            launches = [(decompose, (tok, bid, offset, valid, n, b, 256)),
                        (gather, (req, pre, mask, table, starts, bid, base, n, k, 256)),
                        (emit, (tok, mask, offset, valid, base, out, n, k, b, 256))]
        else:
            launches = [(fused, (tok, req, pre, mask, table, starts, out, n, k, b, 256))]
        options = dict(num_stages=1, multibuffer=False)
        grid = (triton.cdiv(n, 256),)
        compiled = [fn.warmup(*args, grid=grid, **options) for fn, args in launches]
        artifacts = []
        for index, item in enumerate(compiled):
            for key, data in item.asm.items():
                if isinstance(data, str):
                    data = data.encode()
                if isinstance(data, (bytes, bytearray)):
                    path = output / f'kernel-{index}.{key}'
                    path.write_bytes(data)
                    artifacts.append(path)

        def run():
            for fn, args in launches:
                fn[grid](*args, **options)
            return out

        item = SimpleNamespace(run=run, inputs=inputs, workspace=workspace,
            expected_kernels=[fn.fn.__name__ for fn, _ in launches],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(x) for x in compiled],
            metadata=dict(arm=arm, parameters=p, language='triton', block=256,
                torch_version=torch.__version__, triton_version=triton.__version__, compile_only=True,
                logical_intermediate_bytes=16*n if arm == 'before' else 0,
                arithmetic='int32; unsafe block indices clamped before masked gather',
                preallocation='input copies, output and intermediates outside forward; every intermediate overwritten inside API'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        cpu = actual.cpu()
        passed = cpu.dtype == torch.int32 and cpu.shape == fixture['expected'].shape and torch.equal(cpu, fixture['expected'])
        immutable = all(torch.equal(x.cpu(), fixture[key]) for x, key in zip(self.prepared[arm].inputs, KEYS))
        return dict(passed=passed and immutable, exact_int32=passed, inputs_immutable=immutable,
                    sentinel_remaining=int((cpu == -2147483648).sum()))
