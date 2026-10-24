"""W07: unblocked rank-one LU versus panel16/TRSM/IEEE FP32 Cube Schur.

Independent adaptation of the source-described blocked algorithm. All matrix
copies, permutation writes and factor work occur inside each public forward.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from blocked_lu_contract import CASE_SOURCES, dense_lu, diagnostics, dimensions, make_case, validate_inputs


COMPILER_OPTIONS = dict(num_stages=1, multibuffer=False)


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def lu_copy(A, LU, Perm, N:tl.constexpr, B:tl.constexpr):
        x = tl.program_id(0)*B + tl.arange(0, B)
        tl.store(LU+x, tl.load(A+x, x<N*N, 0), x<N*N)
        tl.store(Perm+x, x.to(tl.int64), x<N)

    @triton.jit(do_not_specialize=['K'])
    def lu_column(LU, N:tl.constexpr, K, B:tl.constexpr):
        r = tl.arange(0, B)
        pivot = tl.load(LU+K*N+K)
        column = tl.load(LU+r*N+K, (r<N)&(r>K), 0)
        tl.store(LU+r*N+K, column/pivot, (r<N)&(r>K))

    @triton.jit(do_not_specialize=['K'])
    def lu_rank_one(LU, N:tl.constexpr, K, B:tl.constexpr):
        r = tl.program_id(0)*B+tl.arange(0, B)
        c = tl.program_id(1)*B+tl.arange(0, B)
        l = tl.load(LU+r*N+K, (r<N)&(r>K), 0)
        u = tl.load(LU+K*N+c, (c<N)&(c>K), 0)
        mask = (r[:,None]<N)&(c[None,:]<N)&(r[:,None]>K)&(c[None,:]>K)
        a = tl.load(LU+r[:,None]*N+c[None,:], mask, 0)
        tl.store(LU+r[:,None]*N+c[None,:], a-l[:,None]*u[None,:], mask)

    @triton.jit(do_not_specialize=['P'])
    def lu_panel(LU, N:tl.constexpr, P, B:tl.constexpr):
        r = tl.arange(0, B)
        c = P+tl.arange(0, 16)
        mask = (r[:,None]<N)&(r[:,None]>=P)
        a = tl.load(LU+r[:,None]*N+c[None,:], mask, 0)
        for j in range(0, 16):
            k = P+j
            row = tl.sum(tl.where(r[:,None]==k, a, 0.), 0)
            pivot = tl.sum(tl.where(c==k, row, 0.), 0)
            col = tl.sum(tl.where(c[None,:]==k, a, 0.), 1)
            l = col/pivot
            a = tl.where((r[:,None]>k)&(c[None,:]>k), a-l[:,None]*row[None,:], a)
            a = tl.where((r[:,None]>k)&(c[None,:]==k), l[:,None], a)
        tl.store(LU+r[:,None]*N+c[None,:], a, mask)

    @triton.jit(do_not_specialize=['P'])
    def lu_block_row(LU, N:tl.constexpr, P):
        r = P+tl.arange(0, 16)
        c = P+16+tl.program_id(0)*16+tl.arange(0, 16)
        a = tl.load(LU+r[:,None]*N+c[None,:], c[None,:]<N, 0)
        for j in range(0, 16):
            k = P+j
            row = tl.sum(tl.where(r[:,None]==k, a, 0.), 0)
            l = tl.load(LU+r*N+k)
            a = tl.where(r[:,None]>k, a-l[:,None]*row[None,:], a)
        tl.store(LU+r[:,None]*N+c[None,:], a, c[None,:]<N)

    @triton.jit(do_not_specialize=['P'])
    def lu_schur(LU, Right, N:tl.constexpr, P):
        r = P+16+tl.program_id(0)*16+tl.arange(0, 16)
        c = P+16+tl.program_id(1)*16+tl.arange(0, 16)
        k = P+tl.arange(0, 16)
        l = tl.load(LU+r[:,None]*N+k[None,:], r[:,None]<N, 0)
        u = tl.load(Right+k[:,None]*N+c[None,:], c[None,:]<N, 0)
        product = tl.dot(l, u, input_precision='ieee')
        mask = (r[:,None]<N)&(c[None,:]<N)
        a = tl.load(LU+r[:,None]*N+c[None,:], mask, 0)
        tl.store(LU+r[:,None]*N+c[None,:], a-product, mask)

    return lu_copy, lu_column, lu_rank_one, lu_panel, lu_block_row, lu_schur


def compile_case(arm, spec):
    """Compile every launch specialization without allocating or executing tensors."""
    import torch
    from triton.runtime.jit import MockTensor
    n, _ = dimensions(spec)
    copy, column, rank, factor, solve, schur = kernels()
    f32, i64 = MockTensor(torch.float32), MockTensor(torch.int64)
    artifacts = {}
    def keep(k):
        artifacts[k.hash] = k
    keep(copy.warmup(f32, f32, i64, n, 256, grid=(triton.cdiv(n*n, 256),), **COMPILER_OPTIONS))
    if arm == "after" and n % 16 == 0:
        for start in sorted({0, max(0, n-16)}):
            keep(factor.warmup(f32, n, start, triton.next_power_of_2(n), grid=(1,), **COMPILER_OPTIONS))
            blocks = (n-start-16)//16
            if blocks:
                keep(solve.warmup(f32, n, start, grid=(blocks,), **COMPILER_OPTIONS))
                keep(schur.warmup(f32, f32, n, start, grid=(blocks, blocks), **COMPILER_OPTIONS))
    else:
        for k in sorted({0, n-2}):
            keep(column.warmup(f32, n, k, triton.next_power_of_2(n), grid=(1,), **COMPILER_OPTIONS))
            keep(rank.warmup(f32, n, k, 32, grid=(triton.cdiv(n, 32), triton.cdiv(n, 32)), **COMPILER_OPTIONS))
    return artifacts


def compilation_state(compiled):
    return [dict(hash=k.hash, binaries={key: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
                 for key, value in k.asm.items() if isinstance(value, (str, bytes))}) for k in compiled.values()]


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        if arm not in ('before', 'after'):
            raise ValueError('unknown arm')
        n, panel_size = dimensions(spec)
        no_pivot = validate_inputs(fixture, spec)
        if not no_pivot:
            # Actual elimination on every call: no cached oracle output lookup.
            def run_host():
                lu, p = dense_lu(fixture['inputs']['a'], pivot=True)
                return lu.float(), p
            item = SimpleNamespace(run=run_host, expected_kernels=['host_partial_pivot_lu'],
                sources=[Path(__file__), *CASE_SOURCES], compilation_state=lambda: [], inputs=None,
                metadata=dict(execution='host', host_forwards_per_call=1, npu_forwards_per_call=0,
                              fallback='dense CPU partial pivot; correctness-only'))
            self.prepared[arm] = item
            return item
        import torch_npu
        a = fixture['inputs']['a'].to('npu:0')
        lu = torch.full_like(a, float('nan'))
        p = torch.full((n,), -1, device='npu:0', dtype=torch.int64)
        copy, column, rank, factor, solve, schur = kernels()
        blocked = arm == 'after' and n % panel_size == 0
        compiled = {}
        def keep(key, artifact):
            compiled[artifact.hash] = artifact
        def run():
            keep('copy', copy[(triton.cdiv(n*n, 256),)](a, lu, p, n, 256, **COMPILER_OPTIONS))
            if blocked:
                for start in range(0, n, 16):
                    keep('panel', factor[(1,)](lu, n, start, triton.next_power_of_2(n), **COMPILER_OPTIONS))
                    blocks = (n-start-16)//16
                    if blocks:
                        keep('solve', solve[(blocks,)](lu, n, start, **COMPILER_OPTIONS))
                        # Separate RHS formal permits direct global Cube loads.
                        keep('schur', schur[(blocks, blocks)](lu, lu, n, start, **COMPILER_OPTIONS))
            else:
                for k in range(n-1):
                    # Launch barrier publishes the entire L column before any
                    # rank-one tile reads it; rank tiles never write column k.
                    keep('column', column[(1,)](lu, n, k, triton.next_power_of_2(n), **COMPILER_OPTIONS))
                    keep('rank', rank[(triton.cdiv(n, 32), triton.cdiv(n, 32))](lu, n, k, 32, **COMPILER_OPTIONS))
            return lu, p
        selected = [copy, factor] if blocked else [copy, column, rank]
        if blocked and n > 16:
            selected += [solve, schur]
        item = SimpleNamespace(run=run, expected_kernels=[k.fn.__name__ for k in selected],
            sources=[Path(__file__), *CASE_SOURCES], inputs=a,
            compilation_state=lambda: compilation_state(compiled), metadata=dict(
                execution='npu', arm=arm, blocked=blocked, tail_fallback=arm=='after' and not blocked,
                panel=16, shape=spec, precision='FP32; IEEE FP32 Cube; no TF32',
                launches_per_forward=1+3*(n//16)-2 if blocked else 1+2*(n-1),
                workspace_bytes=lu.numel()*lu.element_size()+p.numel()*p.element_size(),
                triton_version=triton.__version__, torch_version=torch.__version__,
                allocation='output preallocated; full input copy and permutation write in each forward'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture, spec)
        a = self.prepared[arm].inputs
        if a is not None and not torch.equal(a.detach().cpu(), fixture['inputs']['a']):
            result.update(passed=False, reason='input changed')
        return result


def factory():
    return Adapter()
