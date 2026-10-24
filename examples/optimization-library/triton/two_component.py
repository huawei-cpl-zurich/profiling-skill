"""F11: dynamic row decomposition and two integer dots versus dense FP16 GEMM.

Independent implementation of the source's ordered FP32 MSD arithmetic. No
input-dependent transform is cached between complete API forwards.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from two_component_contract import check_spec, eligible, make_case, validate_actual


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def msd_decompose(X, A1, A2, AMAX, K: tl.constexpr, ROW: tl.constexpr):
        row = tl.program_id(0)
        kk = tl.arange(0, ROW)
        x = tl.load(X + row * K + kk, kk < K, other=0).to(tl.float32)
        amax = tl.max(tl.abs(x), 0)
        inv = 127. / tl.where(amax > 0, amax, 1.)
        t = x * inv
        a1 = tl.floor(t + .5)
        a2 = tl.floor((t - a1) * 254. + .5)
        tl.store(A1 + row * K + kk, a1.to(tl.int8), kk < K)
        tl.store(A2 + row * K + kk, a2.to(tl.int8), kk < K)
        tl.store(AMAX + row, amax)

    @triton.jit
    def msd_matmul(X, W, SCALE, A1, A2, AMAX, Y, M: tl.constexpr,
                   N: tl.constexpr, K: tl.constexpr, INTEGER: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
        rn = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        if INTEGER:
            c1 = tl.zeros((BM, BN), tl.int32)
            c2 = tl.zeros((BM, BN), tl.int32)
        else:
            dense = tl.zeros((BM, BN), tl.float32)
        scale = tl.load(SCALE + rn, rn < N, other=0)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + rk
            mask = (rm[:, None] < M) & (kk[None, :] < K)
            w = tl.load(W + kk[:, None] * N + rn[None, :],
                        (kk[:, None] < K) & (rn[None, :] < N), other=0)
            if INTEGER:
                a1 = tl.load(A1 + rm[:, None] * K + kk[None, :], mask, other=0)
                a2 = tl.load(A2 + rm[:, None] * K + kk[None, :], mask, other=0)
                c1 = tl.dot(a1, w, c1, out_dtype=tl.int32)
                c2 = tl.dot(a2, w, c2, out_dtype=tl.int32)
            else:
                x = tl.load(X + rm[:, None] * K + kk[None, :], mask, other=0)
                dequant = (w.to(tl.float32) * scale[None, :]).to(tl.float16)
                dense = tl.dot(x, dequant, dense, out_dtype=tl.float32)
        if INTEGER:
            amax = tl.load(AMAX + rm, rm < M, other=0)
            residual = c2.to(tl.float32) * (1. / 254.)
            combined = c1.to(tl.float32) + residual
            rowscale = amax / 127.
            out = (combined * rowscale[:, None]) * scale[None, :]
        else:
            out = dense
        tl.store(Y + rm[:, None] * N + rn[None, :], out.to(tl.float16),
                 (rm[:, None] < M) & (rn[None, :] < N))
    return msd_decompose, msd_matmul


def identity(compiled):
    return [dict(name=c.name, hash=str(c.hash), binaries={k: hashlib.sha256(v).hexdigest()
            for k, v in c.asm.items() if isinstance(v, bytes)}) for c in compiled]


class Adapter:
    case_sources = [Path(__file__).with_name('two_component_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.states = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        import triton
        if arm not in ('before', 'after'):
            raise ValueError('unknown arm')
        p = check_spec(spec)
        m, n, k = (p[x] for x in ('M', 'N', 'K'))
        fallback = not eligible(spec)
        if fallback and spec.get('role') != 'guard':
            raise ValueError('unsafe int32 scope is correctness-only; require role=guard')
        inputs = {name: value.to('npu:0') for name, value in fixture['inputs'].items()}
        y = torch.empty((m, n), device='npu:0', dtype=torch.float16)
        # Storage preallocated, but every element is recomputed each forward.
        a1 = torch.empty((m, k), device='npu:0', dtype=torch.int8)
        a2 = torch.empty_like(a1)
        amax = torch.empty((m,), device='npu:0', dtype=torch.float32)
        self.states[arm] = (inputs, None if fallback else (a1, a2))
        decompose, matmul = kernels()
        quant = arm == 'after'
        dargs = [inputs['x'], a1, a2, amax, k, triton.next_power_of_2(k)]
        args = [inputs['x'], inputs['w'], inputs['scale'], a1, a2, amax, y,
                m, n, k, quant, 16, 64, 128]
        grid = ((m + 15) // 16 * ((n + 63) // 64),)
        compiled = []
        def compile_only():
            compiled.clear()
            if not fallback:
                if quant:
                    compiled.append(decompose.warmup(*dargs, grid=(m,), enable_fp_fusion=False))
                compiled.append(matmul.warmup(*args, grid=grid, enable_fp_fusion=False))
            output.mkdir(parents=True, exist_ok=True)
            for index, c in enumerate(compiled):
                for name, value in c.asm.items():
                    if isinstance(value, (str, bytes)):
                        (output / f'compiled-{index}.{name}').write_bytes(value.encode() if isinstance(value, str) else value)
            return identity(compiled)
        def run():
            if fallback:
                x = fixture['inputs']
                y.copy_(((x['x'].double() @ x['w'].double()) * x['scale'].double()).half())
            else:
                active = []
                if quant:
                    active.append(decompose[(m,)](*dargs, enable_fp_fusion=False))
                active.append(matmul[grid](*args, enable_fp_fusion=False))
                compiled[:] = active
            return y
        return SimpleNamespace(run=run, compile_only=compile_only, compilation_state=lambda: identity(compiled),
            sources=[__file__, self.case_sources[0]], expected_kernels=[decompose.fn.__name__, matmul.fn.__name__] if quant else [matmul.fn.__name__],
            metadata=dict(backend='triton', dispatch='safe-dense-cpu-fp64' if fallback else 'two-int8-dots' if quant else 'dense-fp16',
                torch=torch.__version__, triton=triton.__version__, tile=[16,64,128], fp_fusion=False,
                workspace_bytes=2*m*k+4*m, device_dot_calls_per_forward=0 if fallback else 2 if quant else 1,
                allocation_boundary='inputs/scales, output and workspace preallocated; row amax, decomposition, '
                    'both dot accumulator initializations, conversions and ordered FP32 reconstruction inside after forward; '
                    'weight dequantization and floating dot inside before forward; no cached input transforms'))

    def validate(self, arm, actual, fixture, spec):
        inputs, codes = self.states[arm]
        return validate_actual(actual, fixture, inputs, arm, codes if arm == 'after' else None)
