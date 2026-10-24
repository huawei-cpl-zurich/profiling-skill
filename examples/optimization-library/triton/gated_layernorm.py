# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Adapted from triton-ascend-kernels norm/norm_gated.py, MIT license,
# commit 7acb9c40a6ca827e885d902b7eaee5d45457b230.
"""Isolate full-row gate/LayerNorm fusion with identical centered reductions."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from gated_layernorm_contract import CASE_SOURCES, INPUTS, diagnostics, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_gate(Z, U, N: tl.constexpr, B: tl.constexpr):
        col = tl.arange(0, B)
        z = tl.load(Z + tl.program_id(0)*N + col, col < N, 0)
        u = z * (1.0 / (1.0 + tl.exp(-z)))
        tl.store(U + tl.program_id(0)*N + col, u, col < N)

    @triton.jit
    def wave005_gate_multiply(X, U, Y, N: tl.constexpr, B: tl.constexpr):
        col = tl.arange(0, B)
        offset = tl.program_id(0)*N + col
        x, u = tl.load(X + offset, col < N, 0), tl.load(U + offset, col < N, 0)
        tl.store(Y + offset, x * u, col < N)

    @triton.jit
    def wave005_gated_layernorm(X, Z, W, Bias, Y, Mean, Rstd,
                               N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr,
                               FUSED: tl.constexpr, FIRST: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        mask = col < N
        x = tl.load(X + row*N + col, mask, 0)
        if FUSED:
            z = tl.load(Z + row*N + col, mask, 0)
            u = z * (1.0 / (1.0 + tl.exp(-z)))
            if FIRST:
                x = x * u
        mean = tl.sum(x, 0) / N
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, 0) / N
        rstd = tl.rsqrt(variance + EPS)
        weight = tl.load(W + col, mask, 0)
        bias = tl.load(Bias + col, mask, 0)
        y = ((x - mean) * rstd) * weight + bias
        if FUSED and not FIRST:
            y = y * u
        tl.store(Y + row*N + col, y, mask)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

    return wave005_gate, wave005_gate_multiply, wave005_gated_layernorm


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
        r, n, _, first, eps, _ = parameters(spec)
        if arm not in ('before', 'after'):
            raise ValueError('expected before or after')
        for key, shape in zip(INPUTS, ((r,n), (r,n), (n,), (n,))):
            value = fixture[key]
            if value.dtype != torch.float32 or value.shape != shape or not value.is_contiguous():
                raise ValueError('contiguous FP32 input required: ' + key)
        import torch_npu
        import triton
        inputs = tuple(fixture[key].npu() for key in INPUTS)
        x, z, gamma, beta = inputs
        def empty(shape):
            return torch.full(shape, float('nan'), dtype=torch.float32, device=x.device)
        y, mean, rstd = empty((r,n)), empty((r,)), empty((r,))
        gate, multiply, norm = kernels()
        b = triton.next_power_of_2(n)
        options = dict(enable_fp_fusion=False, multibuffer=False, num_stages=1)
        calls = []
        workspace = []
        if arm == 'before':
            u, intermediate = empty((r,n)), empty((r,n))
            workspace = [u, intermediate]
            calls.append((gate, (z, u, n, b)))
            if first:
                calls += [(multiply, (x, u, intermediate, n, b)),
                          (norm, (intermediate, z, gamma, beta, y, mean, rstd, n, eps, b, False, first))]
            else:
                calls += [(norm, (x, z, gamma, beta, intermediate, mean, rstd, n, eps, b, False, first)),
                          (multiply, (intermediate, u, y, n, b))]
        else:
            calls.append((norm, (x, z, gamma, beta, y, mean, rstd, n, eps, b, True, first)))
        compiled, artifacts = [], []
        for index, (kernel, arguments) in enumerate(calls):
            binary = kernel.warmup(*arguments, grid=(r,), **options)
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
                kernel[(r,)](*arguments, **options)
            return y, mean, rstd

        item = SimpleNamespace(run=run, inputs=inputs, workspace=workspace,
            expected_kernels=list(dict.fromkeys(kernel.fn.__name__ for kernel, _ in calls)),
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(binary) for binary in compiled],
            metadata=dict(arm=arm, R=r, N=n, block=b, gate_first=first, eps=eps,
                language='triton', torch_version=torch.__version__, triton_version=triton.__version__,
                workspace_logical_bytes=8*r*n if arm=='before' else 0,
                compile_only=True, full_row=True, arithmetic='FP32 centered variance; FMA disabled',
                scope='preallocated input/output/workspace; every gate and intermediate write is inside forward'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(tuple(value.cpu() for value in actual), fixture)
        for key, value in zip(INPUTS, self.prepared[arm].inputs):
            if not torch.equal(value.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        pointers = [value.data_ptr() for value in actual]
        if len(set(pointers)) != 3 or any(value.data_ptr() in pointers for value in self.prepared[arm].inputs):
            result.update(passed=False, reason='output alias')
        return result
