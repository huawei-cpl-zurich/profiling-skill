"""Paired GEMM transformations; no example is verified until its A3 receipt passes.

Independent implementation informed by the pinned sources cited in the three
recipe cards. Preparation allocates/transfers only; first launch compiles.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from gemm_contract import check_spec, make_case, validate_actual


def program_tile(pid, rows, columns, diagonal):
    """CPU mirror of row-wise column rotation; bijective on rectangular grids."""
    if rows <= 0 or columns <= 0 or not 0 <= pid < rows * columns:
        raise ValueError('program outside positive tile grid')
    row, column = divmod(pid, columns)
    if diagonal and rows >= 2 and columns >= 2:
        column = (column + row) % columns
    return row, column


def configuration(arm, spec):
    check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    family = spec['family']
    bm, bn = ((128, 256) if arm == 'before' else (256, 128)) if family == '05' else (64, 128)
    rows, columns = (spec['m'] + bm - 1) // bm, (spec['n'] + bn - 1) // bn
    return dict(bm=bm, bn=bn, bk=32, rows=rows, columns=columns,
                diagonal=family == '12' and arm == 'after' and rows >= 2 and columns >= 2,
                fused=family == '02' and arm == 'after')


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def gallery_gemm(A, B, Z, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                     BM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                     FUSED: tl.constexpr, DIAGONAL: tl.constexpr):
        pid = tl.program_id(0)
        columns = tl.cdiv(N, TN)
        tile_m = pid // columns
        tile_n = pid % columns
        if DIAGONAL:
            tile_n = (tile_n + tile_m) % columns
        rm = tile_m * BM + tl.arange(0, BM)
        rn = tile_n * TN + tl.arange(0, TN)
        rk = tl.arange(0, TK)
        acc = tl.full((BM, TN), 0, tl.float32)
        # K traversal is identical and ascending under both output mappings.
        for stage in range(tl.cdiv(K, TK)):
            kk = stage * TK + rk
            a = tl.load(A + rm[:, None] * AK + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(B + kk[:, None] * BK + rn[None, :] * BN,
                        (kk[:, None] < K) & (rn[None, :] < N), other=0)
            acc += tl.dot(a, b)
        offsets = rm[:, None] * N + rn[None, :]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        if FUSED:
            acc += tl.load(Z + offsets, mask, other=0)
        tl.store(C + offsets, acc, mask)

    @triton.jit
    def gallery_add(X, Z, C, SIZE: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + index, index < SIZE, other=0) + tl.load(Z + index, index < SIZE, other=0)
        tl.store(C + index, value, index < SIZE)

    return gallery_gemm, gallery_add


def compiled_identity(compiled):
    """Keep actual JIT binary identity; no compilation or launch in this hook."""
    result = []
    for item in compiled.values():
        binaries = {name: hashlib.sha256(value).hexdigest() for name, value in item.asm.items()
                    if isinstance(value, bytes)}
        result.append(dict(name=item.name, hash=str(item.hash), binaries=binaries))
    return result


class Adapter:
    case_sources = [Path(__file__).with_name('gemm_contract.py')]

    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu  # Register the device without running a kernel.
        import triton
        gemm, add = kernels()
        config = configuration(arm, spec)
        inputs = {name: value.to(device='npu:0', memory_format=torch.preserve_format)
                  for name, value in fixture['inputs'].items()}
        # Logical values and supported physical layouts are both audited.
        a, b = inputs['a'], inputs['b']
        if a.stride() != fixture['inputs']['a'].stride() or b.stride() != fixture['inputs']['b'].stride():
            raise ValueError('device transfer changed operand layout')
        self.inputs[arm] = inputs
        result = torch.empty_like(fixture['expected'], device='npu:0')
        separate_add = spec['family'] == '02' and arm == 'before'
        z = inputs.get('z', result)  # Unused by non-fused GEMM specializations.
        compiled = {}
        m, n, k = (spec[key] for key in ('m', 'n', 'k'))

        def run():
            # The unfused intermediate's host allocation and all its device
            # traffic belong to the before call, as required by the contract.
            intermediate = torch.empty_like(result) if separate_add else result
            compiled['gemm'] = gemm[(config['rows'] * config['columns'],)](
                a, b, z, intermediate, m, n, k, a.stride(0), b.stride(0), b.stride(1),
                config['bm'], config['bn'], config['bk'], config['fused'], config['diagonal'])
            if separate_add:
                compiled['add'] = add[((m * n + 1023) // 1024,)](intermediate, z, result, m * n, 1024)
            return result

        return SimpleNamespace(run=run, expected_kernels=[gemm.fn.__name__] + ([add.fn.__name__] if separate_add else []),
            sources=[__file__, self.case_sources[0]],
            metadata=dict(backend='triton', torch=torch.__version__, triton=triton.__version__,
                          configuration=config, b_layout=spec.get('b_layout', 'row'),
                          input_strides={name: list(tensor.stride()) for name, tensor in inputs.items()},
                          workspace_bytes=m * n * result.element_size() if separate_add else 0,
                          allocation_boundary='outputs preallocated; unfused intermediate allocated in each call'),
            compilation_state=lambda: compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        return validate_actual(actual, fixture, self.inputs[arm])
