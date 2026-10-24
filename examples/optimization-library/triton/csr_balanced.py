"""F05: row ownership versus balanced nnz segmented scan (vector adaptation)."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from csr_balanced_contract import CASE_SOURCES, FAST_MAX_ROW, diagnostics, dimensions, make_case, validate_inputs


def kernels():
    global triton, tl, segment_combine
    import triton
    import triton.language as tl

    @triton.jit
    def wave003_csr_zero(Out, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Out + row, 0., row < ROWS)

    @triton.jit
    def wave003_csr_rows(Ptr, Col, Val, X, Out, WIDTH: tl.constexpr):
        row = tl.program_id(0)
        start = tl.load(Ptr + row)
        end = tl.load(Ptr + row + 1)
        lane = tl.arange(0, WIDTH)
        index = start + lane
        col = tl.load(Col + index, index < end, 0)
        value = tl.load(Val + index, index < end, 0.)
        x = tl.load(X + col, index < end, 0.)
        tl.store(Out + row, tl.sum(value * x, 0))

    @triton.jit
    def segment_combine(left_row, left_value, right_row, right_value):
        return right_row, tl.where(left_row == right_row, left_value + right_value, right_value)

    @triton.jit
    def wave003_csr_nnz(Ptr, Col, Val, X, Out, ROWS: tl.constexpr, NNZ: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        index = tl.program_id(0) * BLOCK + lane
        valid = index < NNZ
        # Upper bound handles duplicate offsets from empty rows correctly.
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), ROWS + 1, tl.int32)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            offset = tl.load(Ptr + mid, mid <= ROWS, NNZ + 1)
            move = (lo < hi) & (offset <= index)
            hi = tl.where((lo < hi) & ~move, mid, hi)
            lo = tl.where(move, mid + 1, lo)
        row = lo - 1
        col = tl.load(Col + index, valid, 0)
        value = tl.load(Val + index, valid, 0.) * tl.load(X + col, valid, 0.)
        _, prefix = tl.associative_scan((row, value), 0, segment_combine)
        row_end = tl.load(Ptr + row + 1, valid, NNZ)
        last = (index + 1 == row_end) | (lane == BLOCK - 1) | (index == NNZ - 1)
        tl.atomic_add(Out + row, prefix, valid & last, sem='relaxed')

    return wave003_csr_zero, wave003_csr_rows, wave003_csr_nnz


def compiled_identity(item):
    artifacts = {}
    for key, value in getattr(item, 'asm', {}).items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            artifacts[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=getattr(item, 'hash', None), artifacts=artifacts)


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
            raise ValueError('arm must be before/after')
        rows, _, nnz, _, _ = dimensions(spec)
        maximum = validate_inputs(fixture, spec)
        zero, by_row, by_nnz = kernels()
        keys = ('indptr', 'columns', 'values', 'x')
        inputs = tuple(fixture[key].npu() for key in keys)
        out = fixture['initial_y'].npu()
        # The only CPU dispatch fact is certified from the public CSR offsets;
        # no per-nnz row map, gather, scan or output initialization is cached.
        balanced = arm == 'after' and maximum <= FAST_MAX_ROW and nnz > 0
        selected = [zero, by_nnz] if balanced else ([by_row] if nnz else [zero])
        compiled = {}

        def launch(kernel, grid, *args):
            artifact = kernel[grid](*args)
            compiled[id(artifact)] = artifact

        def run():
            if balanced or nnz == 0:
                launch(zero, (triton.cdiv(rows, 256),), out, rows, 256)
            if balanced:
                launch(by_nnz, (triton.cdiv(nnz, 256),), *inputs, out, rows, nnz, 256)
            elif nnz:
                launch(by_row, (rows,), *inputs, out, max(32, triton.next_power_of_2(maximum)))
            return out

        item = SimpleNamespace(run=run, expected_kernels=[k.fn.__name__ for k in selected],
            sources=[Path(__file__), *CASE_SOURCES], inputs=inputs,
            compilation_state=lambda: [compiled_identity(v) for v in compiled.values()],
            metadata=dict(language='triton', arm=arm, rows=rows, nnz=nnz, actual_max_row=maximum,
                dispatch='balanced-nnz-scan' if balanced else 'row-vector-fallback' if nnz else 'zero-empty',
                fast_max_row=FAST_MAX_ROW, nnz_block=256, triton_version=triton.__version__,
                torch_version=torch.__version__, num_autotune_trials=0,
                adaptation='vector segmented associative scan; no Cube claim',
                included_work='output overwrite, device binary searches, products, segmented scan, atomic fragment merge'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.detach().cpu(), fixture)
        for value, key in zip(self.prepared[arm].inputs, ('indptr', 'columns', 'values', 'x')):
            if not torch.equal(value.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        if result.get('empty_rows_zero') is False:
            result.update(passed=False, reason='empty rows must be exactly zero')
        return result


def factory():
    return Adapter()
