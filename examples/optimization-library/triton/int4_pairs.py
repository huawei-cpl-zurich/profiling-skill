"""Full dequant+GEMM versus the same GEMM with consumer-tile INT4 decoding.

Mechanism: ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/awq_triton.py:L12-L105
and L170-L208. Independent natural nibble encoding; not AWQ compatibility.
No runtime gain, physical-traffic saving or resource fit is claimed until A3 validation.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from int4_contract import check_spec, make_case, validate_actual


def configuration(arm, spec):
    check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    return dict(bm=32, bn=64, bk=32, fused=arm == 'after')


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def int4_dequantize(Q, Z, S, D, K: tl.constexpr, N: tl.constexpr,
                        G: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        k, n = i // N, i % N
        mask = i < K * N
        shift = (n % 8) * 4
        q = tl.load(Q + k * (N // 8) + n // 8, mask, 0).to(tl.uint32)
        z = tl.load(Z + (k // G) * (N // 8) + n // 8, mask, 0).to(tl.uint32)
        scale = tl.load(S + (k // G) * N + n, mask, 0).to(tl.float32)
        q = ((q >> shift) & 15).to(tl.int32)
        z = ((z >> shift) & 15).to(tl.int32)
        value = ((q - z).to(tl.float32) * scale).to(tl.float16)
        tl.store(D + i, value, mask)

    @triton.jit
    def int4_gemm(A, Q, Z, S, D, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  FUSED: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.cdiv(N, BN)
        m = (pid // cols) * BM + tl.arange(0, BM)
        n = (pid % cols) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for stage in range(tl.cdiv(K, BK)):
            k = stage * BK + rk
            a = tl.load(A + m[:, None] * K + k[None, :],
                        (m[:, None] < M) & (k[None, :] < K), 0)
            valid = (k[:, None] < K) & (n[None, :] < N)
            if FUSED:
                shift = (n[None, :] % 8) * 4
                q = tl.load(Q + k[:, None] * (N // 8) + n[None, :] // 8, valid, 0).to(tl.uint32)
                z = tl.load(Z + (k[:, None] // G) * (N // 8) + n[None, :] // 8, valid, 0).to(tl.uint32)
                scale = tl.load(S + (k[:, None] // G) * N + n[None, :], valid, 0).to(tl.float32)
                q = ((q >> shift) & 15).to(tl.int32)
                z = ((z >> shift) & 15).to(tl.int32)
                b = ((q - z).to(tl.float32) * scale).to(tl.float16)
            else:
                b = tl.load(D + k[:, None] * N + n[None, :], valid, 0)
            acc += tl.dot(a, b)
        tl.store(C + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))

    return int4_dequantize, int4_gemm


def compiled_identity(compiled):
    return [dict(name=k.name, hash=str(k.hash), binaries={name: hashlib.sha256(value).hexdigest()
            for name, value in k.asm.items() if isinstance(value, bytes)}) for k in compiled.values()]


class Adapter:
    case_sources = [Path(__file__).with_name('int4_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu  # Register backend before transfer, without a forward.
        import triton
        config = configuration(arm, spec)
        dequant, gemm = kernels()
        values = {name: value.to(device='npu:0', memory_format=torch.preserve_format)
                  for name, value in fixture['inputs'].items()}
        if any(t.stride() != fixture['inputs'][name].stride() for name, t in values.items()):
            raise ValueError('device transfer changed input layout')
        formats = {name: torch_npu.get_npu_format(value) for name, value in values.items()}
        if any(value != 2 for value in formats.values()):
            raise ValueError('linear addressing requires audited ND storage for every input')
        self.inputs[arm] = values
        m, n, k, group = (spec[key] for key in ('m', 'n', 'k', 'group_size'))
        result = torch.empty_like(fixture['expected'], device='npu:0')
        workspace = result if config['fused'] else torch.empty((k, n), dtype=torch.float16, device='npu:0')
        compiled = {}

        def run():
            if not config['fused']:
                compiled['dequant'] = dequant[(triton.cdiv(k * n, 1024),)](
                    values['packed'], values['zeros'], values['scales'], workspace, k, n, group, 1024)
            compiled['gemm'] = gemm[(triton.cdiv(m, config['bm']) * triton.cdiv(n, config['bn']),)](
                values['a'], values['packed'], values['zeros'], values['scales'], workspace, result,
                m, n, k, group, config['bm'], config['bn'], config['bk'], config['fused'])
            return result

        return SimpleNamespace(run=run, sources=[__file__, *self.case_sources],
            expected_kernels=([dequant.fn.__name__] if not config['fused'] else []) + [gemm.fn.__name__],
            compilation_state=lambda: compiled_identity(compiled),
            metadata=dict(backend='triton', family='int4-consumer', configuration=config,
                          torch=torch.__version__, triton=triton.__version__, input_formats=formats,
                          encoding='natural unsigned INT4; eight N values per int32, low nibble first',
                          dequantization_rounding='FP32 subtraction/multiply, then FP16 before dot',
                          workspace_bytes=0 if config['fused'] else k * n * 2,
                          workspace_scope='adapter-explicit; compiler scratch not inferred',
                          allocation_boundary='output and before dequant matrix preallocated; complete operation timed'))

    def validate(self, arm, actual, fixture, spec):
        return validate_actual(actual, fixture, self.inputs[arm])


def factory():
    return Adapter()
