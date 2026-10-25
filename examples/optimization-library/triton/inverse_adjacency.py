"""Float atomic scatter versus stable counting CSR and exclusive gather.

Independent adaptation of the NSA inverse selected-query ownership mechanism:
ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L455-L725
Sorting is replaced with stable counting assembly. Every call rebuilds metadata
from its current edge IDs; only storage allocation is outside the forward.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from inverse_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, diagnostics


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_inverse_zero(Out, N:tl.constexpr, D:tl.constexpr, B:tl.constexpr):
        i = tl.program_id(0)*B + tl.arange(0, B)
        tl.store(Out+i, 0., i < N*D)

    @triton.jit
    def wave004_inverse_atomic(Dest, Values, Out, E:tl.constexpr, D:tl.constexpr, B:tl.constexpr):
        e = tl.program_id(0)*B + tl.arange(0, B)
        d = tl.arange(0, D)
        dest = tl.load(Dest+e, e < E, 0)
        value = tl.load(Values+e[:, None]*D+d[None, :], (e < E)[:, None], 0.)
        tl.atomic_add(Out+dest[:, None]*D+d[None, :], value, (e < E)[:, None], sem='relaxed')

    @triton.jit
    def wave004_inverse_count(Dest, Counts, E:tl.constexpr, B:tl.constexpr):
        owner = tl.program_id(0)
        lane = tl.arange(0, B)
        count = tl.full((), 0, tl.int32)
        for start in range(0, E, B):
            e = start + lane
            dest = tl.load(Dest+e, e < E, -1)
            count += tl.sum((dest == owner).to(tl.int32), 0)
        # Each slot is overwritten, so uninitialized allocation needs no zero.
        tl.store(Counts+owner, count)

    @triton.jit
    def wave004_inverse_prefix(Counts, Offsets, N:tl.constexpr, B:tl.constexpr):
        i = tl.arange(0, B)
        count = tl.load(Counts+i, i < N, 0)
        end = tl.cumsum(count, 0)
        tl.store(Offsets+i, end-count, i < N)
        tl.store(Offsets+N, tl.sum(count, 0))

    @triton.jit
    def wave004_inverse_assemble(Dest, Offsets, Edges, E:tl.constexpr, B:tl.constexpr):
        owner = tl.program_id(0)
        base = tl.load(Offsets+owner)
        lane = tl.arange(0, B)
        for start in range(0, E, B):
            e = start + lane
            dest = tl.load(Dest+e, e < E, -1)
            selected = (dest == owner) & (e < E)
            rank = tl.cumsum(selected.to(tl.int32), 0)
            tl.store(Edges+base+rank-1, e, selected)
            base += tl.sum(selected.to(tl.int32), 0)

    @triton.jit
    def wave004_inverse_gather(Values, Offsets, Edges, Out, D:tl.constexpr, B:tl.constexpr):
        owner = tl.program_id(0)
        start = tl.load(Offsets+owner)
        end = tl.load(Offsets+owner+1)
        lane = tl.arange(0, B)
        d = tl.arange(0, D)
        total = tl.zeros((D,), tl.float32)
        for block in range(start, end, B):
            i = block + lane
            edge = tl.load(Edges+i, i < end, 0)
            value = tl.load(Values+edge[:, None]*D+d[None, :], (i < end)[:, None], 0.)
            total += tl.sum(value, 0)
        tl.store(Out+owner*D+d, total)

    return (wave004_inverse_zero, wave004_inverse_atomic, wave004_inverse_count,
            wave004_inverse_prefix, wave004_inverse_assemble, wave004_inverse_gather)


class InverseReduction:
    def __init__(self, inputs, spec, arm):
        import torch
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        e, n, d, _ = dimensions(spec)
        self.inputs = inputs
        self.output = torch.full((n, d), float('nan'), dtype=torch.float32, device='npu')
        zero, atomic, count, prefix, assemble, gather = kernels()
        self.workspace = ()
        if arm == 'before':
            self.calls = [(zero, (triton.cdiv(n*d, 256),), (self.output, n, d, 256)),
                          (atomic, (triton.cdiv(e, 64),), (inputs['dest'], inputs['values'], self.output, e, d, 64))]
        else:
            counts = torch.empty(n, dtype=torch.int32, device='npu')
            offsets = torch.empty(n+1, dtype=torch.int32, device='npu')
            edges = torch.empty(e, dtype=torch.int32, device='npu')
            self.workspace = (counts, offsets, edges)
            self.calls = [(count, (n,), (inputs['dest'], counts, e, 256)),
                          (prefix, (1,), (counts, offsets, n, triton.next_power_of_2(n))),
                          (assemble, (n,), (inputs['dest'], offsets, edges, e, 256)),
                          (gather, (n,), (inputs['values'], offsets, edges, self.output, d, 64))]
        # Serial loops use the supported safe Ascend lowering for both arms.
        self.options = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        self.compiled = []

    def compile(self):
        self.compiled = [fn.warmup(*args, grid=grid, **self.options) for fn, grid, args in self.calls]

    def compilation_state(self):
        return [dict(hash=getattr(k, 'hash', None), binaries={name: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
                for name, value in k.asm.items() if isinstance(value, (str, bytes))}) for k in self.compiled]

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
        validate_inputs(fixture['inputs'], spec)
        op = InverseReduction({key: value.npu() for key, value in fixture['inputs'].items()}, spec, arm)
        op.compile()
        artifacts = []
        for i, compiled in enumerate(op.compiled):
            for name, value in compiled.asm.items():
                if isinstance(value, (str, bytes)):
                    path = Path(output) / f'kernel-{i}.{name}'
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts.append(path)
        self.prepared[arm] = op
        return SimpleNamespace(run=op.run, compilation_state=op.compilation_state,
            sources=[Path(__file__), *CASE_SOURCES, *artifacts], expected_kernels=[fn.fn.__name__ for fn, _, _ in op.calls],
            metadata=dict(language='triton', arm=arm, shape=spec, triton_version=triton.__version__, torch_version=torch.__version__,
                          workspace_bytes=sum(x.numel()*x.element_size() for x in op.workspace),
                          allocation_scope='preallocated output and workspace; all input-dependent metadata rebuilt per forward',
                          inverse_order='stable original edge order; blockwise FP32 reduction', compiler_options=op.options))

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        for key, value in self.prepared[arm].inputs.items():
            if not torch.equal(value.cpu(), fixture['inputs'][key]):
                result.update(passed=False, reason='input mutated: ' + key)
        return result


def factory():
    return Adapter()
