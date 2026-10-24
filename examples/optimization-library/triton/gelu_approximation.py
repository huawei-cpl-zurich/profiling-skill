# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Formula provenance: ops-nn7ed342d28b44b6291aceaf747796028c48d7304c,
# GELU README and arch35 DAG; CANN Open Software License Agreement 2.0.
# Independent Triton adaptation; upstream arch35 device code is not compiled.
"""An exact-erf mode and an explicitly permitted tanh approximation."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from gelu_contract import CASE_SOURCES, parameters, make_case, diagnostics


def kernel():
    global triton, tl, tmath
    import triton
    import triton.language as tl
    import triton.language.math as tmath

    @triton.jit
    def wave005_gelu_approximation(X, Y, N: tl.constexpr, APPROX: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X+i, i < N, 0)
        if APPROX:
            z = 0.7978845608028654 * (x + .044715*x*x*x)
            y = (.5*x) * (1 + tmath.tanh(z))
        else:
            y = (.5*x) * (1 + tl.erf(x * 0.7071067811865475))
        tl.store(Y+i, y, i < N)
    return wave005_gelu_approximation


def identity(compiled):
    binaries = {}
    for key, data in compiled.asm.items():
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, (bytes, bytearray)):
            binaries[key] = hashlib.sha256(data).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


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
        n, _, allowed, *_ = parameters(spec)
        source = fixture['x']
        if source.device.type != 'cpu' or source.dtype != torch.float32 or source.shape != (n,) or not source.is_contiguous() or not bool(torch.isfinite(source).all()) or not bool((source.abs() <= 8).all()):
            raise ValueError('fixture must be contiguous finite FP32[elements] within [-8,8]')
        x = source.npu()
        y = torch.full((n,), float('nan'), dtype=torch.float32, device=x.device)
        program = kernel()
        grid = (triton.cdiv(n, 256),)
        approx = arm == 'after' and allowed
        args = (x, y, n, approx, 256)
        options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        compiled = program.warmup(*args, grid=grid, **options)
        artifacts = []
        for key, data in compiled.asm.items():
            if isinstance(data, str):
                data = data.encode()
            if isinstance(data, (bytes, bytearray)):
                path = output / ('kernel.'+key)
                path.write_bytes(data)
                artifacts.append(path)
        def run():
            program[grid](*args, **options)
            return y
        item = SimpleNamespace(run=run, x=x, allowed=allowed,
            expected_kernels=[program.fn.__name__], sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(compiled)],
            metadata=dict(arm=arm, language='triton', elements=n, block=256, dtype='FP32',
                dispatch='permitted-tanh-approximation' if approx else 'erf',
                torch_version=torch.__version__, triton_version=triton.__version__,
                source_kind='README and arch35 arithmetic; independent 910B formula adaptation',
                preallocation='Input/output outside API; every output written each API'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(actual.cpu(), fixture, arm, item.allowed)
        if not torch.equal(item.x.cpu(), fixture['x']) or actual.data_ptr() == item.x.data_ptr():
            result.update(passed=False, reason='input mutation or output alias')
        return result
