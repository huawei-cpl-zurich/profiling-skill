# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Independent fusion pair adapted from triton-ascend-kernels/moe/fused_gdn_gating.py,
# MIT license, commit 7acb9c40a6ca827e885d902b7eaee5d45457b230.
"""Four explicit gate stages versus a single dual-output gate kernel."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from gdn_gate_contract import CASE_SOURCES, INPUTS, checked_inputs, diagnostics, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_gdn_bias(A, Bias, X, TOTAL: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(A+i, i<TOTAL, 0) + tl.load(Bias+i%H, i<TOTAL, 0)
        tl.store(X+i, x, i<TOTAL)

    @triton.jit
    def wave005_gdn_softplus(X, S, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X+i, i<TOTAL, 0)
        s = tl.where(x > 20, x, tl.log(1.0 + tl.exp(x)))
        tl.store(S+i, s, i<TOTAL)

    @triton.jit
    def wave005_gdn_scale(S, ALog, G, TOTAL: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        s = tl.load(S+i, i<TOTAL, 0)
        alog = tl.load(ALog+i%H, i<TOTAL, 0)
        tl.store(G+i, (-tl.exp(alog)) * s, i<TOTAL)

    @triton.jit
    def wave005_gdn_beta(B, Beta, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        b = tl.load(B+i, i<TOTAL, 0)
        tl.store(Beta+i, 1.0 / (1.0 + tl.exp(-b)), i<TOTAL)

    @triton.jit
    def wave005_gdn_fused(A, B, ALog, Bias, G, Beta, TOTAL: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(A+i, i<TOTAL, 0) + tl.load(Bias+i%H, i<TOTAL, 0)
        s = tl.where(x > 20, x, tl.log(1.0 + tl.exp(x)))
        alog = tl.load(ALog+i%H, i<TOTAL, 0)
        b = tl.load(B+i, i<TOTAL, 0)
        tl.store(G+i, (-tl.exp(alog)) * s, i<TOTAL)
        tl.store(Beta+i, 1.0 / (1.0 + tl.exp(-b)), i<TOTAL)

    return wave005_gdn_bias, wave005_gdn_softplus, wave005_gdn_scale, wave005_gdn_beta, wave005_gdn_fused


def identity(binary):
    hashes = {}
    for key, value in binary.asm.items():
        if isinstance(value, str): value = value.encode()
        if isinstance(value, (bytes, bytearray)): hashes[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=binary.hash, binaries=hashes)


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = CASE_SOURCES

    def __init__(self): self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        r, h, _, _ = parameters(spec)
        if arm not in ('before','after'): raise ValueError('expected before or after')
        checked_inputs(spec, fixture)
        import torch_npu
        import triton
        inputs = tuple(fixture[key].npu() for key in INPUTS)
        a, b, alog, bias = inputs
        def empty():
            return torch.full((1,r,h), float('nan'), dtype=torch.float32, device=a.device)
        g, beta = empty(), empty()
        add, softplus, scale, sigmoid, fused = kernels()
        total, block = r*h, 256
        grid = (triton.cdiv(total, block),)
        workspace = []
        if arm == 'before':
            x, s = empty(), empty(); workspace = [x,s]
            calls = [(add,(a,bias,x,total,h,block)), (softplus,(x,s,total,block)),
                     (scale,(s,alog,g,total,h,block)), (sigmoid,(b,beta,total,block))]
        else:
            calls = [(fused,(a,b,alog,bias,g,beta,total,h,block))]
        options = dict(enable_fp_fusion=False, multibuffer=False, num_stages=1)
        binaries, artifacts = [], []
        for number, (kernel, args) in enumerate(calls):
            binary = kernel.warmup(*args, grid=grid, **options); binaries.append(binary)
            for key, value in binary.asm.items():
                if isinstance(value, str): value = value.encode()
                if isinstance(value, (bytes, bytearray)):
                    path = output/f'{number}.{key}'; path.write_bytes(value); artifacts.append(path)
        def run():
            for kernel, args in calls: kernel[grid](*args, **options)
            return g, beta
        item = SimpleNamespace(run=run, inputs=inputs, workspace=workspace,
            expected_kernels=[k.fn.__name__ for k, _ in calls],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(v) for v in binaries],
            metadata=dict(arm=arm, language='triton', R=r, H=h, grid=grid, block=block,
                torch_version=torch.__version__, triton_version=triton.__version__,
                workspace_logical_bytes=8*r*h if arm=='before' else 0,
                A_exponential='per output in both arms; no cross-row hoisting',
                arithmetic='FP32; FMA disabled; threshold20; beta1',
                scope='preallocated inputs/outputs/workspace; all four stages inside API'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(tuple(v.cpu() for v in actual), fixture)
        item = self.prepared[arm]
        for key, value in zip(INPUTS, item.inputs):
            if not torch.equal(value.cpu(), fixture[key]): result.update(passed=False, reason='input mutation: '+key)
        pointers = [v.data_ptr() for v in actual]
        if len(set(pointers)) != 2 or any(v.data_ptr() in pointers for v in item.inputs):
            result.update(passed=False, reason='output alias')
        return result
