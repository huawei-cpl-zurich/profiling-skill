# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
"""Independent SAME-Triton primitive pair; no normalization or Newton step.

Motivation: ops-rec e489c0bf ln_mul Sqrt+Div; triton-ascend-kernels
7acb9c40 norm_gated tl.rsqrt. See rsqrt-primitive.md for licenses/scope.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from rsqrt_contract import CASE_SOURCES, diagnostics, dimensions, make_case, validate_input

_KERNEL = None


def kernel():
    global triton, tl, _KERNEL
    import triton
    import triton.language as tl
    if _KERNEL is None:
        @triton.jit
        def wave006_rsqrt(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, DIRECT: tl.constexpr):
            offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(X + offset, offset < N, other=1.0)
            if DIRECT:
                y = tl.rsqrt(x)
            else:
                y = 1.0 / tl.sqrt(x)
            tl.store(Y + offset, y, offset < N)
        _KERNEL = wave006_rsqrt
    return _KERNEL


def compiled_identity(artifact):
    return [dict(hash=str(artifact.hash), binaries={name: hashlib.sha256(value).hexdigest()
                 for name, value in artifact.asm.items() if isinstance(value, bytes)})]


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        n, _ = dimensions(spec)
        x = fixture['x']
        validate_input(x, n)
        import torch
        import torch_npu
        import triton
        device_x = x.to(device='npu:0')
        storage = torch.full((n + 32,), -12345.0, dtype=torch.float32, device='npu:0')
        out = storage[16:16+n]
        fn = kernel()
        args = (device_x, out, n, 256, arm == 'after')
        grid = ((n + 255) // 256,)
        artifact = fn.warmup(*args, grid=grid)
        output = Path(output)
        for name, value in artifact.asm.items():
            if isinstance(value, str):
                (output / ('kernel.' + name)).write_text(value)
            elif isinstance(value, bytes):
                (output / ('kernel.' + name)).write_bytes(value)

        def run():
            fn[grid](*args)
            return out

        prepared = SimpleNamespace(run=run, expected_kernels=[fn.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES], inputs=(device_x,), storage=storage,
            compilation_state=lambda: compiled_identity(artifact),
            metadata=dict(language='triton', triton_version=triton.__version__, torch_version=torch.__version__,
                          arm=arm, N=n, block=256, grid=list(grid), compiler_flags='identical backend defaults',
                          num_autotune_trials=0, compilation='warmup compile only; zero forwards',
                          physical_layout='contiguous FP32; offsets in elements; 16 element output canaries'))
        self.prepared[arm] = prepared
        return prepared

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(actual.detach().cpu(), fixture)
        if not torch.equal(item.inputs[0].cpu(), fixture['x']):
            result.update(passed=False, reason='input mutation')
        canaries = item.storage.cpu()
        if not bool((canaries[:16] == -12345).all() & (canaries[-16:] == -12345).all()):
            result.update(passed=False, reason='output canary changed')
        return result


def factory():
    return Adapter()
