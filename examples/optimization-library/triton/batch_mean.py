# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2026 Huawei Technologies Co., Ltd.
# Independent adaptation of deterministic/mean_batch_invariant.py at
# triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230.
"""Controlled split-row baseline versus fixed per-output numerical decomposition."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from batch_mean_contract import CASE_SOURCES, parameters, make_case, diagnostics


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_mean_partial(X, P, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
        row, part = tl.program_id(0), tl.program_id(1)
        j = part * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + row * N + j, j < N, 0)
        tl.store(P + row * PARTS + part, tl.sum(value, 0))

    @triton.jit
    def wave005_mean_finish(P, Y, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        j = tl.arange(0, BLOCK)
        value = tl.load(P + row * PARTS + j, j < PARTS, 0)
        tl.store(Y + row, tl.sum(value, 0) / N)

    @triton.jit
    def wave005_mean_fixed(X, Y, N: tl.constexpr):
        row = tl.program_id(0)
        acc = 0.0
        for start in range(0, N, 1024):
            j = start + tl.arange(0, 1024)
            value = tl.load(X + row * N + j, j < N, 0)
            acc += tl.sum(value, 0)
        tl.store(Y + row, acc / N)

    return wave005_mean_partial, wave005_mean_finish, wave005_mean_fixed


def identity(compiled):
    return dict(hash=compiled.hash, binaries={key: hashlib.sha256(value.encode() if isinstance(value, str)
                else value).hexdigest() for key, value in compiled.asm.items()
                if isinstance(value, (str, bytes, bytearray))})


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        import torch_npu
        import triton
        _, n, _, keepdim, _ = parameters(spec)
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        for x in fixture['xs']:
            if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != n or not x.is_contiguous():
                raise ValueError('contiguous FP32 matrices required')
        xs = tuple(x.npu() for x in fixture['xs'])
        ys = tuple(torch.full((x.shape[0], 1) if keepdim else (x.shape[0],), float('nan'),
                              dtype=torch.float32, device=x.device) for x in xs)
        partial, finish, fixed = kernels()
        launches, compiled, artifacts, logical_bytes = [], [], [], 0
        options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        for x, y in zip(xs, ys):
            r = x.shape[0]
            if arm == 'before':
                block = 512 if r <= 2 else 1024
                parts = triton.cdiv(n, block)
                workspace = torch.empty((r, parts), dtype=torch.float32, device=x.device)
                logical_bytes += 4 * r * parts
                group = [(partial, (r, parts), (x, workspace, n, parts, block)),
                         (finish, (r,), (workspace, y, n, parts, triton.next_power_of_2(parts)))]
            else:
                group = [(fixed, (r,), (x, y, n))]
            for kernel, grid, args in group:
                binary = kernel.warmup(*args, grid=grid, **options)
                compiled.append(binary)
                launches.append((kernel, grid, args))
                for key, data in binary.asm.items():
                    if isinstance(data, str):
                        data = data.encode()
                    if isinstance(data, (bytes, bytearray)):
                        path = output / f'kernel-{len(compiled)}.{key}'
                        path.write_bytes(data)
                        artifacts.append(path)

        def run():
            for kernel, grid, args in launches:
                kernel[grid](*args, **options)
            return ys

        item = SimpleNamespace(run=run, inputs=xs, expected_kernels=[partial.fn.__name__, finish.fn.__name__]
            if arm == 'before' else [fixed.fn.__name__], sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(k) for k in compiled],
            metadata=dict(arm=arm, context_shapes=[list(x.shape) for x in xs], keepdim=keepdim,
                language='triton', partial_logical_bytes=logical_bytes, component_mean_calls=len(xs),
                task_launches_per_api=len(launches), fixed_block=1024,
                preallocation='inputs, separate outputs and partial workspace outside API; every partial recomputed inside'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(tuple(y.cpu() for y in actual), fixture, spec, arm)
        if any(not torch.equal(x.cpu(), ref) for x, ref in zip(item.inputs, fixture['xs'])):
            result.update(passed=False, reason='input mutation')
        pointers = [y.data_ptr() for y in actual]
        if len(set(pointers)) != len(pointers) or set(pointers) & {x.data_ptr() for x in item.inputs}:
            result.update(passed=False, reason='output alias')
        return result
