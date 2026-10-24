# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright contributors to the vLLM project
# Adapted from triton-ascend-kernels reject_sample_recovered_tokens.py,
# commit 7acb9c40a6ca827e885d902b7eaee5d45457b230, MIT license.
# Original upstream attribution:
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py
"""Remove a common normalization before argmax on the frozen dyadic domain.

Both arms use identical tiled reductions, ascending merge and launch options.
The BEFORE arm additionally sums all residuals then divides by that row sum.
All-zero rows use denominator one. No transform or residual is cached.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from normalize_argmax_contract import CASE_SOURCES, diagnostics, dimensions, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_normalize_argmax(Target, Draft, Q, Out, V: tl.constexpr,
                                 NORMALIZE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offset = tl.arange(0, BLOCK)
        total = 1.0
        if NORMALIZE:
            total = 0.0
            for start in range(tl.cdiv(V, BLOCK)):
                col = start * BLOCK + offset
                target = tl.load(Target + row * V + col, col < V, 0)
                draft = tl.load(Draft + row * V + col, col < V, 0)
                total += tl.sum(tl.maximum(target - draft, 0.0), 0)
            total = tl.where(total > 0, total, 1.0)
        best = -float('inf')
        winner = 0
        for start in range(tl.cdiv(V, BLOCK)):
            col = start * BLOCK + offset
            target = tl.load(Target + row * V + col, col < V, 0)
            draft = tl.load(Draft + row * V + col, col < V, 0)
            q = tl.load(Q + row * V + col, col < V, 1)
            residual = tl.maximum(target - draft, 0.0)
            if NORMALIZE:
                residual = residual / total
            score = tl.where(col < V, residual / q, -float('inf'))
            maximum = tl.max(score, 0)
            index = tl.min(tl.where(score == maximum, col, 2147483647), 0)
            winner = tl.where(maximum > best, index, winner)
            best = tl.maximum(best, maximum)
        tl.store(Out + row, winner)

    return wave004_normalize_argmax


def identity(compiled):
    artifacts = {}
    for key, value in compiled.asm.items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            artifacts[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=compiled.hash, artifacts=artifacts)


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
        rows, width, _, _ = dimensions(spec)
        inputs = tuple(fixture[key].npu() for key in ('target', 'draft', 'q'))
        out = torch.empty((rows,), dtype=torch.int32, device=inputs[0].device)
        kernel = kernels()
        arguments = (*inputs, out, width, arm == 'before', min(2048, width))
        options = dict(multibuffer=False)
        # warmup compiles only: never invokes the device kernel.
        compiled = kernel.warmup(*arguments, grid=(rows,), **options)
        artifacts = []
        for key, data in compiled.asm.items():
            if isinstance(data, str):
                data = data.encode()
            if isinstance(data, (bytes, bytearray)):
                path = output / ('kernel.' + key)
                path.write_bytes(data)
                artifacts.append(path)

        def run():
            kernel[(rows,)](*arguments, **options)
            return out

        item = SimpleNamespace(run=run, inputs=inputs, expected_kernels=[kernel.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(compiled)],
            metadata=dict(arm=arm, rows=rows, V=width, block=min(2048, width),
                language='triton', torch_version=torch.__version__, triton_version=triton.__version__,
                preallocation='three input device copies and output outside forward; no residual workspace',
                normalization='full-row sum and division in BEFORE only', compile_only=True))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.cpu(), fixture)
        for source, key in zip(self.prepared[arm].inputs, ('target', 'draft', 'q')):
            if not torch.equal(source.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        return result
