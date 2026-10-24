# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 LinkedIn Corporation
# Adapted independently from triton-ascend-kernels activation/swiglu.py,
# commit 7acb9c40a6ca827e885d902b7eaee5d45457b230 (MIT).
"""Save versus recompute sigmoid, with unchanged two-kernel API boundaries."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from swiglu_contract import CASE_SOURCES, dimensions, make_case, diagnostics


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_swiglu_forward(A, B, Y, S, N: tl.constexpr, SAVE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(A + i, i < N, 0)
        b = tl.load(B + i, i < N, 0)
        s = 1.0 / (1.0 + tl.exp(-a))
        tl.store(Y + i, (a * s) * b, i < N)
        if SAVE:
            tl.store(S + i, s, i < N)

    @triton.jit
    def wave005_swiglu_backward(A, B, DC, S, DA, DB, N: tl.constexpr,
                               RECOMPUTE: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(A + i, i < N, 0)
        b = tl.load(B + i, i < N, 0)
        dc = tl.load(DC + i, i < N, 0)
        if RECOMPUTE:
            s = 1.0 / (1.0 + tl.exp(-a))
        else:
            s = tl.load(S + i, i < N, 0)
        da = (dc * b) * (s + ((a * s) * (1.0 - s)))
        db = (dc * a) * s
        tl.store(DA + i, da, i < N)
        tl.store(DB + i, db, i < N)

    return wave005_swiglu_forward, wave005_swiglu_backward


def identity(compiled):
    binaries = {}
    for key, value in compiled.asm.items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            binaries[key] = hashlib.sha256(value).hexdigest()
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
        rows, width, _, _ = dimensions(spec)
        a = fixture['a'].npu()
        # The harness clones CPU fixture tensors independently: reconstruct the
        # actual input alias explicitly, rather than relying on CPU cache aliasing.
        b = a if spec['parameters'].get('share_ab_storage', False) else fixture['b'].npu()
        dc = fixture['dc'].npu()
        shape = (rows, width)
        outputs = tuple(torch.empty(shape, dtype=torch.float32, device=a.device) for _ in range(3))
        saved = torch.empty(shape if arm == 'before' else (1,), dtype=torch.float32, device=a.device)
        forward, backward = kernels()
        n, block = rows * width, 256
        args = ((*[a, b, outputs[0], saved], n, arm == 'before', block),
                (*[a, b, dc, saved, outputs[1], outputs[2]], n, arm == 'after', block))
        grid = (triton.cdiv(n, block),)
        options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        compiled = [kernel.warmup(*argv, grid=grid, **options)
                    for kernel, argv in zip((forward, backward), args)]
        artifacts = []
        for index, binary in enumerate(compiled):
            for key, data in binary.asm.items():
                if isinstance(data, str):
                    data = data.encode()
                if isinstance(data, (bytes, bytearray)):
                    path = output / f'kernel-{index}.{key}'
                    path.write_bytes(data)
                    artifacts.append(path)

        def run():
            forward[grid](*args[0], **options)
            backward[grid](*args[1], **options)
            return outputs

        item = SimpleNamespace(run=run, inputs=(a, b, dc),
            expected_kernels=[forward.fn.__name__, backward.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES, *artifacts],
            compilation_state=lambda: [identity(x) for x in compiled],
            metadata=dict(arm=arm, shape=shape, block=block, language='triton',
                torch_version=torch.__version__, triton_version=triton.__version__,
                saved_sigmoid_logical_bytes=4 * n if arm == 'before' else 0,
                actual_ab_alias=a.data_ptr() == b.data_ptr(), compile_only=True,
                preallocation='inputs, separate outputs and saved workspace allocated outside forward; '
                              'saved values written inside every before API, never retained after API'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(tuple(x.cpu() for x in actual), fixture)
        for value, key in zip(item.inputs, ('a', 'b', 'dc')):
            if not torch.equal(value.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        pointers = [x.data_ptr() for x in actual]
        if len(set(pointers)) != 3 or set(pointers) & {x.data_ptr() for x in item.inputs}:
            result.update(passed=False, reason='output alias')
        alias = item.inputs[0].data_ptr() == item.inputs[1].data_ptr()
        if alias != spec['parameters'].get('share_ab_storage', False):
            result.update(passed=False, reason='input alias contract')
        result['actual_ab_alias'] = alias
        return result
