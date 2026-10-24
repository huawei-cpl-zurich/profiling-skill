# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Source contract: Ascend ops-rec e489c0bf6a2d5f2d05b3205fcacc01b62a908b10,
# reverse_sequence documentation and Torch plugin/example (Apache-2.0).
# Independent Triton implementation; upstream native device body was not retrieved.
"""Clone-and-overwrite versus a single output permutation."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from ragged_reverse_contract import CASE_SOURCES, checked_inputs, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_reverse_clone(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y+i, tl.load(X+i, i < N, 0), i < N)

    @triton.jit
    def wave005_reverse_prefix(X, Length, Y, N: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch = i // (T*D)
        time = (i // D) % T
        length = tl.minimum(tl.maximum(tl.load(Length+batch, i < N, 0), 0), T)
        valid = (i < N) & (time < length)
        source = batch*T*D + (length-1-time)*D + i%D
        value = tl.load(X+source, valid, 0)
        tl.store(Y+i, value, valid)

    @triton.jit
    def wave005_reverse_single(X, Length, Y, N: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        batch = i // (T*D)
        time = (i // D) % T
        length = tl.minimum(tl.maximum(tl.load(Length+batch, i < N, 0), 0), T)
        source_time = tl.where(time < length, length-1-time, time)
        source = batch*T*D + source_time*D + i%D
        tl.store(Y+i, tl.load(X+source, i < N, 0), i < N)

    return wave005_reverse_clone, wave005_reverse_prefix, wave005_reverse_single


def identity(compiled):
    binaries = {}
    for key, data in compiled.asm.items():
        if isinstance(data, str): data = data.encode()
        if isinstance(data, (bytes, bytearray)): binaries[key] = hashlib.sha256(data).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch, torch_npu, triton
        if arm not in ('before', 'after'): raise ValueError('unknown arm')
        p = parameters(spec); checked_inputs(fixture, p)
        x, lengths = fixture['x'].npu(), fixture['lengths'].npu()
        n, t, d = x.numel(), p['T'], p['D']
        out = torch.full(x.shape, float('nan'), dtype=torch.float32, device=x.device)
        twice = p['length_pattern'] == 'clamp-extremes'
        intermediate = torch.full_like(out, float('nan')) if twice else None
        clone, prefix, single = kernels()
        launches = []
        def operation(src, dst):
            if arm == 'before':
                launches.extend([(clone, (src, dst, n, 256)), (prefix, (src, lengths, dst, n, t, d, 256))])
            else:
                launches.append((single, (src, lengths, dst, n, t, d, 256)))
        if twice:
            operation(x, intermediate); operation(intermediate, out)
        else: operation(x, out)
        options = dict(num_stages=1, multibuffer=False)
        grid = (triton.cdiv(n, 256),)
        compiled = [fn.warmup(*args, grid=grid, **options) for fn, args in launches]
        artifacts = []
        for index, item in enumerate(compiled):
            for key, data in item.asm.items():
                if isinstance(data, str): data = data.encode()
                if isinstance(data, (bytes, bytearray)):
                    path = output / f'kernel-{index}.{key}'
                    path.write_bytes(data); artifacts.append(path)
        def run():
            for fn, args in launches: fn[grid](*args, **options)
            return out
        item = SimpleNamespace(run=run, inputs=(x, lengths), intermediate=intermediate,
            expected_kernels=[fn.fn.__name__ for fn, _ in launches],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(c) for c in compiled],
            metadata=dict(arm=arm, parameters=p, language='triton', block=256,
                torch_version=torch.__version__, triton_version=triton.__version__, compile_only=True,
                component_reversals_per_api=2 if twice else 1,
                kernel_launches_per_api=len(launches), clamp='[0,T] on device every operation',
                logical_full_copy_bytes=4*n if arm == 'before' else 0,
                preallocation='Inputs/output outside forward; clone and all required stores inside API'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]; cpu = actual.cpu()
        exact = cpu.dtype == torch.float32 and cpu.shape == fixture['expected'].shape and torch.equal(cpu.view(torch.int32), fixture['expected'].view(torch.int32))
        immutable = all(torch.equal(v.cpu().view(torch.int32), fixture[key].view(torch.int32)) for v, key in zip(item.inputs, ('x', 'lengths')))
        nonalias = actual.data_ptr() not in [v.data_ptr() for v in item.inputs]
        intermediate_exact = item.intermediate is None or torch.equal(item.intermediate.cpu().view(torch.int32), fixture['one_reverse'].view(torch.int32))
        return dict(passed=exact and immutable and nonalias and intermediate_exact, bitwise_permutation=exact,
            inputs_immutable=immutable, output_nonalias=nonalias, guard_intermediate_exact=intermediate_exact,
            nan_sentinel_remaining=int(torch.isnan(cpu).sum()), component_reversals_per_api=2 if item.intermediate is not None else 1)
