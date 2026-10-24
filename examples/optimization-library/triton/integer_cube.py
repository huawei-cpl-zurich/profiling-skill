"""F10: IEEE float dot versus int8/int32 Cube, with identical scale order.

Independent implementation informed by the scaled-mm source cited in the card.
The floating arm converts loaded codes inside the measured kernel. No input
quantization, cached conversion, autotuning, warmup or output inspection occurs.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from integer_cube_contract import check_spec, integer_eligible, make_case, validate_actual


@lru_cache(maxsize=1)
def kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def integer_cube_scaled(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr,
                            K: tl.constexpr, INTEGER: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rm = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
        rn = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        if INTEGER:
            acc = tl.zeros((BM, BN), tl.int32)
        else:
            acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + rk
            a = tl.load(A + rm[:, None] * K + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(B + kk[:, None] * N + rn[None, :],
                        (kk[:, None] < K) & (rn[None, :] < N), other=0)
            if INTEGER:
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            else:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                             input_precision='ieee', out_dtype=tl.float32)
        sa = tl.load(SA + rm, rm < M, other=0)
        sb = tl.load(SB + rn, rn < N, other=0)
        out = sb[None, :] * (sa[:, None] * acc.to(tl.float32))
        tl.store(C + rm[:, None] * N + rn[None, :], out,
                 (rm[:, None] < M) & (rn[None, :] < N))

    return integer_cube_scaled


def identity(compiled):
    if compiled is None:
        return []
    return [dict(name=compiled.name, hash=str(compiled.hash),
                 binaries={k: hashlib.sha256(v).hexdigest() for k, v in compiled.asm.items()
                           if isinstance(v, bytes)})]


class Adapter:
    case_sources = [Path(__file__).with_name('integer_cube_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        import triton
        if arm not in ('before', 'after'):
            raise ValueError('unknown arm')
        p = check_spec(spec)
        m, n, k = (p[x] for x in ('M', 'N', 'K'))
        integer = arm == 'after' and integer_eligible(spec)
        fallback = not integer_eligible(spec)
        if fallback and spec.get('role') != 'guard':
            raise ValueError('out-of-bound host reference is correctness-only; require role=guard')
        inputs = {name: value.to('npu:0') for name, value in fixture['inputs'].items()}
        self.inputs[arm] = inputs
        result = torch.empty((m, n), device='npu:0', dtype=torch.float32)
        args = [inputs[x] for x in ('a', 'b', 'sa', 'sb')] + [result, m, n, k, integer, 16, 64, 128]
        grid = ((m + 15) // 16 * ((n + 63) // 64),)
        fn = kernel()
        compiled = [None]

        def compile_only():
            # JIT warmup compiles without launching. It is not a device warmup.
            compiled[0] = fn.warmup(*args, grid=grid)
            output.mkdir(parents=True, exist_ok=True)
            for name, value in compiled[0].asm.items():
                if isinstance(value, (str, bytes)):
                    path = output / ('compiled.' + name)
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
            return identity(compiled[0])

        def run():
            if fallback:
                # Out-of-scope correctness-only guard: independent FP64 CPU
                # floating reference avoids FP32 intermediate integer rounding.
                x = fixture['inputs']
                safe = x['sb'].T * (x['sa'] * (x['a'].double() @ x['b'].double()).float())
                result.copy_(safe)
                return result
            compiled[0] = fn[grid](*args)
            return result

        return SimpleNamespace(run=run, compile_only=compile_only,
            expected_kernels=[fn.fn.__name__], sources=[__file__, self.case_sources[0]],
            compilation_state=lambda: identity(compiled[0]),
            metadata=dict(backend='triton', integer=integer,
                dispatch='safe-floating-cpu-fp64' if fallback else 'integer' if integer else 'floating',
                accumulator='cpu-fp64-then-fp32' if fallback else 'int32' if integer else 'fp32-ieee',
                host_reference_and_h2d_inside_forward=fallback,
                device_dot_calls_per_forward=0 if fallback else 1, torch=torch.__version__,
                triton=triton.__version__, tile=[16, 64, 128], workspace_bytes=0,
                allocation_boundary='int8 inputs/scales transferred and FP32 output preallocated; '
                    'all float conversion, zero accumulator, dot and ordered scaling inside forward'))

    def validate(self, arm, actual, fixture, spec):
        return validate_actual(actual, fixture, self.inputs[arm])
