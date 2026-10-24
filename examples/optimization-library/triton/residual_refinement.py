"""Independent SOURCE-DOCUMENTED FP32 adaptation, not native extended exponent code."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from shared_tridiagonal import kernels as thomas_kernels
from refinement_contract import CASE_SOURCES, make_case, diagnostics
from tridiagonal_contract import dimensions, validate_inputs


def correction_kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_refine_residual(DL, D, DU, B, X, RES, N:tl.constexpr, LDB:tl.constexpr, BLOCK:tl.constexpr):
        rhs = tl.program_id(0)
        i = tl.arange(0, BLOCK)
        x = tl.load(X+rhs*LDB+i, i < N, 0.)
        lower = tl.load(X+rhs*LDB+i-1, (i > 0)&(i < N), 0.)
        upper = tl.load(X+rhs*LDB+i+1, i+1 < N, 0.)
        ax = (tl.load(D+i, i < N, 0.)*x+tl.load(DL+i, i < N, 0.)*lower)+tl.load(DU+i, i < N, 0.)*upper
        residual = tl.load(B+rhs*LDB+i, i < N, 0.)-ax
        tl.store(RES+rhs*N+i, residual, i < N)

    @triton.jit
    def wave004_refine_commit(X, DELTA, N:tl.constexpr, LDB:tl.constexpr, BLOCK:tl.constexpr):
        rhs = tl.program_id(0)
        i = tl.arange(0, BLOCK)
        x = tl.load(X+rhs*LDB+i, i < N, 0.)
        proposal = x+tl.load(DELTA+rhs*N+i, i < N, 0.)
        finite = (proposal == proposal)&(tl.abs(proposal) <= 3.4028234663852886e38)
        valid = tl.sum(((i < N)&~finite).to(tl.int32), 0) == 0
        tl.store(X+rhs*LDB+i, tl.where(valid, proposal, x), i < N)
    return wave004_refine_residual, wave004_refine_commit


class Refinement:
    def __init__(self, inputs, spec, arm):
        import torch
        self.inputs = inputs
        m, r, ldb = dimensions(spec)
        self.output = torch.empty_like(inputs['b'])
        copy, factor, solve = thomas_kernels()
        residual, commit = correction_kernels()
        import triton
        self.workspace = [torch.empty(m, device='npu', dtype=torch.float32) for _ in range(2)]
        cp, inv = self.workspace
        self.calls = [(copy, (triton.cdiv(r*ldb, 256),), (inputs['b'], self.output, r*ldb, 256))]
        if 'delta' in inputs:
            if arm == 'after':
                self.calls.append((commit, (r,), (self.output, inputs['delta'], m, ldb, triton.next_power_of_2(m))))
        else:
            args = (inputs['dl'], inputs['d'], inputs['du'])
            self.calls += [(factor, (1,), (*args, cp, inv, m)),
                           (solve, (r,), (*args, inputs['b'], self.output, cp, inv, m, ldb, True))]
            if arm == 'after':
                res, delta = [torch.empty((r, m), device='npu', dtype=torch.float32) for _ in range(2)]
                self.workspace += [res, delta]
                for _ in range(2):
                    self.calls += [(residual, (r,), (*args, inputs['b'], self.output, res, m, ldb, triton.next_power_of_2(m))),
                                   (solve, (r,), (*args, res, delta, cp, inv, m, m, True)),
                                   (commit, (r,), (self.output, delta, m, ldb, triton.next_power_of_2(m)))]
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
        import torch_npu
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        p = spec.get('parameters', spec)
        if p.get('component'):
            spec = dict(spec, parameters=dict(p, R=1, ldb=p['m']+8))
        validate_inputs(fixture['inputs'], spec)
        op = Refinement({k: v.npu() for k, v in fixture['inputs'].items()}, spec, arm)
        self.prepared[arm] = op
        artifacts = []
        for i, compiled in enumerate(op.compiled):
            for name, value in compiled.asm.items():
                if isinstance(value, (str, bytes)):
                    path = Path(output)/f'kernel-{i}.{name}'
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts.append(path)
        return SimpleNamespace(run=op.run, compilation_state=op.compilation_state,
            sources=[Path(__file__), Path(__file__).with_name('shared_tridiagonal.py'), *CASE_SOURCES, *artifacts],
            expected_kernels=[fn.fn.__name__ for fn, _, _ in op.calls],
            metadata=dict(arm=arm, host_fallback=False, component_guard='delta' in op.inputs,
                npu_calls_per_forward=len(op.calls), corrections=2 if arm == 'after' and 'delta' not in op.inputs else 0,
                workspace_bytes=sum(x.numel()*x.element_size() for x in op.workspace), compiler_options=op.options,
                precision='FP32 factors/initial/residual/correction/update', residual_association='(d*x+dl*lower)+du*upper',
                factor_scope='current A once per full API; retained across corrections', finite_commit='whole RHS vector',
                allocation_scope='preallocated output/workspace; copy/factor/solve/corrections inside each API'))

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        for k, value in self.prepared[arm].inputs.items():
            if not torch.equal(value.cpu(), fixture['inputs'][k]):
                result.update(passed=False, reason='input mutated: '+k)
        return result


def factory():
    return Adapter()
