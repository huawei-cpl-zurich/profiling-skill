# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Independent Triton adaptation of ops-rec ln_mul, e489c0bf6a2d5f2d05b3205fcacc01b62a908b10.
"""Isolate external-gate fusion after centered-variance LayerNorm and affine."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from external_ln_contract import CASE_SOURCES, INPUTS, checked_input, diagnostics, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_external_norm(X, U, W, Bias, Y, R: tl.constexpr, B: tl.constexpr, FUSED: tl.constexpr):
        row, col = tl.program_id(0), tl.arange(0, B)
        mask = col < R
        offset = row*R + col
        x = tl.load(X + offset, mask, 0)
        mean = tl.sum(x, 0) / R
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, 0) / R
        rstd = tl.rsqrt(variance + 1e-5)
        gamma, beta = tl.load(W + col, mask, 0), tl.load(Bias + col, mask, 0)
        y = ((x - mean) * rstd) * gamma + beta
        if FUSED:
            y = y * tl.load(U + offset, mask, 0)
        tl.store(Y + offset, y, mask)

    @triton.jit
    def wave005_external_multiply(X, U, Y, R: tl.constexpr, B: tl.constexpr):
        col = tl.arange(0, B)
        offset = tl.program_id(0)*R + col
        x, u = tl.load(X + offset, col < R, 0), tl.load(U + offset, col < R, 0)
        tl.store(Y + offset, x * u, col < R)

    return wave005_external_norm, wave005_external_multiply


def identity(compiled):
    binaries = {}
    for key, value in compiled.asm.items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            binaries[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = CASE_SOURCES

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        a, r = checked_input(spec, fixture)
        if arm not in ('before','after'):
            raise ValueError('expected before or after')
        import torch_npu
        import triton
        inputs = tuple(fixture[key].npu() for key in INPUTS)
        x, U, gamma, beta = inputs
        y = torch.full((a,r), float('nan'), dtype=torch.float32, device=x.device)
        temporary = torch.full_like(y, float('nan'), dtype=torch.float32) if arm == 'before' else None
        norm, multiply = kernels()
        b = triton.next_power_of_2(r)
        options = dict(enable_fp_fusion=False, multibuffer=False, num_stages=1)
        target = temporary if arm == 'before' else y
        calls = [(norm, (x,U,gamma,beta,target,r,b,arm=='after'))]
        if arm == 'before':
            calls.append((multiply, (temporary,U,y,r,b)))
        compiled, artifacts = [], []
        for index, (kernel, arguments) in enumerate(calls):
            binary = kernel.warmup(*arguments, grid=(a,), **options)
            compiled.append(binary)
            for key, value in binary.asm.items():
                if isinstance(value, str):
                    value = value.encode()
                if isinstance(value, (bytes, bytearray)):
                    path = output / f'{index}.{key}'
                    path.write_bytes(value)
                    artifacts.append(path)

        def run():
            for kernel, arguments in calls:
                kernel[(a,)](*arguments, **options)
            return y

        item = SimpleNamespace(run=run, inputs=inputs,
            expected_kernels=list(dict.fromkeys(kernel.fn.__name__ for kernel,_ in calls)),
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(binary) for binary in compiled],
            metadata=dict(arm=arm,A=a,R=r,block=b,eps=1e-5,language='triton',
                torch_version=torch.__version__,triton_version=triton.__version__,
                workspace_logical_bytes=4*a*r if arm=='before' else 0,
                scope='preallocated buffers; complete LN and gate forwards; centered FP32 variance, FMA disabled'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.cpu(), fixture)
        for key, value in zip(INPUTS, self.prepared[arm].inputs):
            if not torch.equal(value.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: '+key)
        if any(actual.data_ptr()==value.data_ptr() for value in self.prepared[arm].inputs):
            result.update(passed=False, reason='output alias')
        return result
