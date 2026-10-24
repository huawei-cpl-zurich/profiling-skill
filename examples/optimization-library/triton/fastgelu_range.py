# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Source arithmetic: ops-nn FastGELU DAG and README, commit
# 7ed342d28b44b6291aceaf747796028c48d7304c; CANN Open Software License 2.0.
# Independent Triton adaptation; upstream arch35 device code is not compiled.
"""Compare direct and sign-aware finite FP32 FastGELU expressions."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from fastgelu_contract import CASE_SOURCES, parameters, make_case, diagnostics


def kernel():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_fastgelu_range(X, Y, N: tl.constexpr, STABLE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X+i, i < N, 0)
        if STABLE:
            ax = tl.abs(x)
            y = (x / (1.0 + tl.exp(-1.702 * ax))) * tl.exp(.851 * (x - ax))
        else:
            y = x / (1.0 + tl.exp(-1.702 * x))
        tl.store(Y+i, y, i < N)
    return wave005_fastgelu_range


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
        n, *_ = parameters(spec)
        source = fixture['x']
        if source.device.type != 'cpu' or source.dtype != torch.float32 or source.shape != (n,) or not source.is_contiguous() or not bool(torch.isfinite(source).all()) or not bool((source.abs() <= 80).all()):
            raise ValueError('fixture must be contiguous finite FP32[elements] within [-80,80]')
        x = source.npu()
        y = torch.full((n,), float('nan'), dtype=torch.float32, device=x.device)
        program = kernel()
        grid = (triton.cdiv(n, 256),)
        args = (x, y, n, arm == 'after', 256)
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
        item = SimpleNamespace(run=run, x=x, expected_kernels=[program.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(compiled)],
            metadata=dict(arm=arm, language='triton', elements=n, block=256,
                dtype='FP32', torch_version=torch.__version__, triton_version=triton.__version__,
                source_kind='arch35 arithmetic body and README formula; independent 910B adaptation',
                preallocation='input and separate output allocated outside API; every output written each API',
                compile_only=True))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.cpu(), fixture)
        item = self.prepared[arm]
        if not torch.equal(item.x.cpu(), fixture['x']) or actual.data_ptr() == item.x.data_ptr():
            result.update(passed=False, reason='input mutation or output alias')
        return result
