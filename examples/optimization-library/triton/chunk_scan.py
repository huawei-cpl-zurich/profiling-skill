# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 sgl-project
# Based on vLLM: https://github.com/vllm-project/vllm
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Original copied FLA code licensed MIT:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""W04 serial recurrence versus parallel inclusive scan in the same chunk grid.

Narrow adaptation of triton-ascend-kernels cumsum.py at 7acb9c40a6ca.
All calls include the full scan; no input-dependent preparation or scratch.
Both arms use the serial masked fallback when T is not chunk-divisible.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from chunk_scan_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, diagnostics


# Both arms use the same supported Ascend compiler scheduling configuration.
COMPILER_OPTIONS = dict(num_stages=1, multibuffer=False)

def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_chunk_serial(X, Y, T:tl.constexpr, H:tl.constexpr, C:tl.constexpr):
        start = tl.program_id(0) * C
        batch = tl.program_id(1)
        head = tl.arange(0, H)
        total = tl.full((H,), 0, tl.float32)
        for offset in range(C):
            token = start + offset
            value = tl.load(X + (batch*T+token)*H+head, token<T, 0)
            total = total + value
            tl.store(Y + (batch*T+token)*H+head, total, token<T)

    @triton.jit
    def wave004_chunk_parallel(X, Y, T:tl.constexpr, H:tl.constexpr, C:tl.constexpr):
        token = tl.program_id(0)*C + tl.arange(0, C)
        batch = tl.program_id(1)
        head = tl.arange(0, H)
        pointers = (batch*T+token[:,None])*H+head[None,:]
        values = tl.load(X+pointers, token[:,None]<T, 0)
        prefixes = tl.cumsum(values, axis=0)
        tl.store(Y+pointers, prefixes, token[:,None]<T)

    return wave004_chunk_serial, wave004_chunk_parallel


def selection(arm, spec):
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    _, t, _, chunk = dimensions(spec)
    return 0 if arm == 'before' or t % chunk else 1


def compile_case(arm, spec):
    """Compile only with mock pointers: never launch or execute a warmup."""
    import torch
    from triton.runtime.jit import MockTensor
    b,t,h,chunk = dimensions(spec)
    kernel = kernels()[selection(arm, spec)]
    return kernel.warmup(MockTensor(torch.float32), MockTensor(torch.float32),
                         t,h,chunk,grid=((t+chunk-1)//chunk,b), **COMPILER_OPTIONS)


def artifact_state(artifact):
    return dict(hash=artifact.hash, name=artifact.name,
                artifacts={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                           for k,v in artifact.asm.items() if isinstance(v,(str,bytes))})


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        import torch_npu
        b,t,h,chunk = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        x = fixture['inputs'].npu()
        y = torch.empty_like(x)
        kernel = kernels()[selection(arm,spec)]
        artifact = compile_case(arm,spec)
        def run():
            kernel[((t+chunk-1)//chunk,b)](x,y,t,h,chunk, **COMPILER_OPTIONS)
            return y
        item = SimpleNamespace(run=run, inputs=x, output_folder=Path(output), expected_kernels=[kernel.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES],
            compilation_state=lambda:[artifact_state(artifact)],
            metadata=dict(language='triton', arm=arm, grid=[(t+chunk-1)//chunk,b],
                          shape=[b,t,h], chunk=chunk, scale=1, dtype='FP32',
                          fallback=bool(t%chunk), workspace_bytes=0, compiler_options=COMPILER_OPTIONS,
                          allocation_scope='input transfer and output allocation before measured API',
                          measured_api='one full chunk-local scan kernel; no preprocessing',
                          triton_version=triton.__version__, torch_version=torch.__version__))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        host = actual.detach().cpu()
        torch.save(host, self.prepared[arm].output_folder / 'correctness-output.pt')
        result = diagnostics(host, fixture['expected'])
        if not result['passed']:
            mismatch = host != fixture['expected']
            indices = mismatch.nonzero()[:16]
            result['mismatches'] = [dict(index=i.tolist(), actual=float(host[tuple(i)]), expected=float(fixture['expected'][tuple(i)])) for i in indices]
        if not torch.equal(self.prepared[arm].inputs.cpu(), fixture['inputs']):
            result.update(passed=False, reason='input mutation')
        return result
