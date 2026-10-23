"""Family12 grouped-M candidate; the existing row-major baseline is unchanged.

This module is an experiment adapter, not a claim of a generally faster GEMM.
The only changed device mechanism is output-tile order. Both arms use 64x128
output tiles, K blocks of 32, and ascending K traversal. No warmup/autotuning
or tensor inspection occurs in a forward.

Mechanism reference: pinned Catlass GemmIdentityBlockSwizzle<4,0>,
ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/block/block_swizzle.hpp:L79-L97
This independent grouped-M adaptation omits that source's odd-group N reversal.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

import gemm

GROUP_M = 4


def program_tile(pid, rows, columns, group_m=GROUP_M):
    """Map each valid program to exactly one tile, including a partial M group."""
    if any(type(x) is not int for x in (pid, rows, columns, group_m)) or (
            rows <= 0 or columns <= 0 or group_m <= 0 or not 0 <= pid < rows * columns):
        raise ValueError('program must be inside a positive integer tile grid')
    group_programs = group_m * columns
    first_m = (pid // group_programs) * group_m
    group_size = min(rows - first_m, group_m)
    inner = pid % group_programs
    return first_m + inner % group_size, inner // group_size


@lru_cache(maxsize=1)
def kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def gallery_grouped_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                             AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                             BM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                             GM: tl.constexpr):
        pid = tl.program_id(0)
        rows, columns = tl.cdiv(M, BM), tl.cdiv(N, TN)
        group_programs = GM * columns
        first_m = (pid // group_programs) * GM
        group_size = tl.minimum(rows - first_m, GM)
        inner = pid % group_programs
        tile_m = first_m + inner % group_size
        tile_n = inner // group_size
        rm = tile_m * BM + tl.arange(0, BM)
        rn = tile_n * TN + tl.arange(0, TN)
        rk = tl.arange(0, TK)
        acc = tl.full((BM, TN), 0, tl.float32)
        for stage in range(tl.cdiv(K, TK)):
            kk = stage * TK + rk
            a = tl.load(A + rm[:, None] * AK + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(B + kk[:, None] * BK + rn[None, :] * BN,
                        (kk[:, None] < K) & (rn[None, :] < N), other=0)
            acc += tl.dot(a, b)
        offsets = rm[:, None] * N + rn[None, :]
        tl.store(C + offsets, acc, (rm[:, None] < M) & (rn[None, :] < N))

    return gallery_grouped_gemm


class Adapter(gemm.Adapter):
    # Same provider bytes and spec as other family12 adapters: generate no
    # second expected output merely because the ordering implementation changes.
    case_sources = gemm.Adapter.case_sources
    make_case = staticmethod(gemm.make_case)

    def prepare(self, arm, spec, fixture, output):
        gemm.check_spec(spec)
        if spec['family'] != '12' or arm not in ('before', 'after'):
            raise ValueError('grouped-order adapter requires family12 and a paired arm')
        if arm == 'before':
            prepared = super().prepare(arm, spec, fixture, output)
            prepared.metadata['program_order'] = 'row-major, unchanged baseline'
            return prepared
        import torch_npu  # Register backend before transfer; no forward.
        import triton
        inputs = {name: value.to(device='npu:0', memory_format=torch.preserve_format)
                  for name, value in fixture['inputs'].items()}
        a, b = inputs['a'], inputs['b']
        if a.stride() != fixture['inputs']['a'].stride() or b.stride() != fixture['inputs']['b'].stride():
            raise ValueError('device transfer changed operand layout')
        self.inputs[arm] = inputs
        result = torch.empty_like(fixture['expected'], device='npu:0')
        m, n, k = (spec[key] for key in ('m', 'n', 'k'))
        config = gemm.configuration('before', spec)
        grouped = kernel()
        compiled = {}

        def run():
            compiled['grouped'] = grouped[(config['rows'] * config['columns'],)](
                a, b, result, m, n, k, a.stride(0), b.stride(0), b.stride(1),
                config['bm'], config['bn'], config['bk'], GROUP_M)
            return result

        return SimpleNamespace(run=run, expected_kernels=[grouped.fn.__name__],
            sources=[__file__, gemm.__file__, *self.case_sources],
            metadata=dict(backend='triton', torch=torch.__version__, triton=triton.__version__,
                          configuration=config, program_order='grouped-M', group_m=GROUP_M,
                          b_layout=spec.get('b_layout', 'row'),
                          input_strides={name: list(t.stride()) for name, t in inputs.items()},
                          workspace_bytes=0, workspace_scope='adapter-explicit allocation only',
                          allocation_boundary='output preallocated; compiler scratch retained by JIT'),
            compilation_state=lambda: gemm.compiled_identity(compiled))
