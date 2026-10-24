"""Per-RHS factorization versus per-API shared factors; independent Thomas code.

The native host source motivates factor-once/solve-all ownership. This does not
transcribe or claim execution of the unseen native tridiagonal device helper.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from tridiagonal_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, eligible, dense_solve, diagnostics


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_tri_copy(B, X, TOTAL:tl.constexpr, BLOCK:tl.constexpr):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        tl.store(X+i, tl.load(B+i, i < TOTAL, 0.), i < TOTAL)

    @triton.jit
    def wave004_tri_factor(DL, D, DU, CP, INV, N:tl.constexpr):
        previous = tl.full((), 0., tl.float32)
        i = 0
        while i < N:
            inv = 1. / (tl.load(D+i)-tl.load(DL+i)*previous)
            cp = tl.load(DU+i)*inv
            tl.store(INV+i, inv)
            tl.store(CP+i, cp)
            previous = cp
            i += 1

    @triton.jit
    def wave004_tri_solve(DL, D, DU, B, X, CP, INV, N:tl.constexpr, LDB:tl.constexpr, SHARED:tl.constexpr):
        rhs = tl.program_id(0)
        base = rhs*LDB
        previous_cp = tl.full((), 0., tl.float32)
        previous_y = tl.full((), 0., tl.float32)
        i = 0
        while i < N:
            if SHARED:
                inv = tl.load(INV+i)
            else:
                inv = 1. / (tl.load(D+i)-tl.load(DL+i)*previous_cp)
                cp = tl.load(DU+i)*inv
                tl.store(CP+rhs*N+i, cp)
                previous_cp = cp
            y = (tl.load(B+base+i)-tl.load(DL+i)*previous_y)*inv
            tl.store(X+base+i, y)
            previous_y = y
            i += 1
        following = tl.full((), 0., tl.float32)
        i = N-1
        while i >= 0:
            if SHARED:
                cp = tl.load(CP+i)
            else:
                cp = tl.load(CP+rhs*N+i)
            value = tl.load(X+base+i)-cp*following
            tl.store(X+base+i, value)
            following = value
            i -= 1
    return wave004_tri_copy, wave004_tri_factor, wave004_tri_solve


class Thomas:
    def __init__(self, inputs, spec, arm):
        import torch
        m, r, ldb = dimensions(spec)
        self.inputs = inputs
        self.output = torch.empty_like(inputs['b'])
        cp = torch.empty((m if arm == 'after' else m*r,), device='npu', dtype=torch.float32)
        inv = torch.empty(m, device='npu', dtype=torch.float32)
        self.workspace = (cp, inv)
        copy, factor, solve = kernels()
        args = (inputs['dl'], inputs['d'], inputs['du'])
        self.calls = [(copy, (triton.cdiv(r*ldb, 256),), (inputs['b'], self.output, r*ldb, 256))] if r else []
        if arm == 'after' and r:
            self.calls.append((factor, (1,), (*args, cp, inv, m)))
        if r:
            self.calls.append((solve, (r,), (*args, inputs['b'], self.output, cp, inv, m, ldb, arm == 'after')))
        self.options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        self.compiled = [fn.warmup(*args, grid=grid, **self.options) for fn, grid, args in self.calls]

    def compilation_state(self):
        return [dict(hash=k.hash, binaries={name: hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()
                for name, v in k.asm.items() if isinstance(v, (str, bytes))}) for k in self.compiled]

    def run(self):
        for fn, grid, args in self.calls:
            fn[grid](*args, **self.options)
        return self.output


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        validate_inputs(fixture['inputs'], spec)
        fallback = not eligible(fixture['inputs']) or dimensions(spec)[1] == 0
        if fallback:
            # Execute the real dense solve at each API call; never read expected.
            item = SimpleNamespace(run=lambda: dense_solve(fixture['inputs']), compilation_state=lambda: [],
                sources=[Path(__file__), *CASE_SOURCES], expected_kernels=['host_dense_fallback'],
                metadata=dict(host_fallback=True, npu_calls_per_forward=0, arm=arm))
            self.prepared[arm] = None
            return item
        op = Thomas({k: v.npu() for k, v in fixture['inputs'].items()}, spec, arm)
        self.prepared[arm] = op
        artifacts = []
        for i, compiled in enumerate(op.compiled):
            for name, value in compiled.asm.items():
                if isinstance(value, (str, bytes)):
                    path = Path(output)/f'kernel-{i}.{name}'
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts.append(path)
        return SimpleNamespace(run=op.run, compilation_state=op.compilation_state,
            sources=[Path(__file__), *CASE_SOURCES, *artifacts], expected_kernels=[fn.fn.__name__ for fn, _, _ in op.calls],
            metadata=dict(arm=arm, host_fallback=False, npu_calls_per_forward=len(op.calls),
                factor_scope='once per full API' if arm == 'after' else 'once per RHS per full API',
                workspace_bytes=sum(x.numel()*x.element_size() for x in op.workspace),
                compiler_options=op.options, division='supported FP32 /; identical ordering both arms',
                allocation_scope='preallocated output/workspace; padding copy and factors inside every API'))

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        op = self.prepared[arm]
        if op is not None:
            for k, value in op.inputs.items():
                if not torch.equal(value.cpu(), fixture['inputs'][k]):
                    result.update(passed=False, reason='input mutated: '+k)
        return result


def factory():
    return Adapter()
