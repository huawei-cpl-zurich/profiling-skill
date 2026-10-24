"""Two controlled normalization mechanisms; A3 qualification is still required.

Sources: pinned Triton-Ascend kernels7acb9c40a6ca, norm/l2_norm_fla.py:L7-L86,
norm/rms_norm.py:L82-L94,L145-L155. These independent pairs preserve a fixed
FP32 graph; they do not adopt the source's alternate casting modes.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
import gemm
from norm_contract import CASE_SOURCES, check_spec, make_case, diagnostics


def configuration(arm, spec):
    rows, width, _, eps = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    batch = 4 if spec['family'] == 'E01' and arm == 'after' else 1
    return dict(rows=rows, width=width, eps=eps, batch=batch,
                block=1 << (width - 1).bit_length(),
                rms=spec['family'] == 'E07',
                saved=spec['family'] == 'E01' or arm == 'after')


def program_rows(pid, rows, batch):
    if any(type(v) is not int for v in (pid, rows, batch)) or (
            rows <= 0 or batch <= 0 or not 0 <= pid < (rows + batch - 1) // batch):
        raise ValueError('program outside positive tile grid')
    return range(pid * batch, min((pid + 1) * batch, rows))


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def norm_forward(X, Y, S, R: tl.constexpr, D: tl.constexpr,
                     EPS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
                     RMS: tl.constexpr, SAVE: tl.constexpr):
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        columns = tl.arange(0, BD)
        mask = (rows[:, None] < R) & (columns[None, :] < D)
        x = tl.load(X + rows[:, None] * D + columns[None, :], mask, 0)
        square_sum = tl.sum(x * x, axis=1)
        if RMS:
            square_sum = square_sum / D
        inverse = tl.rsqrt(square_sum + EPS)
        tl.store(Y + rows[:, None] * D + columns[None, :], x * inverse[:, None], mask)
        if SAVE:
            tl.store(S + rows, inverse, rows < R)

    @triton.jit
    def rms_backward(X, DY, S, DX, R: tl.constexpr, D: tl.constexpr,
                     EPS: tl.constexpr, BD: tl.constexpr, SAVED: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BD)
        mask = columns < D
        x = tl.load(X + row * D + columns, mask, 0)
        dy = tl.load(DY + row * D + columns, mask, 0)
        if SAVED:
            inverse = tl.load(S + row)
        else:
            inverse = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
        projection = tl.sum(dy * x, axis=0) / D
        dx = inverse * (dy - x * inverse * inverse * projection)
        tl.store(DX + row * D + columns, dx, mask)

    return norm_forward, rms_backward


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm, spec)
        inputs = {name: value.npu() for name, value in fixture['inputs'].items()}
        for value in inputs.values():
            if value.stride() != (c['width'], 1) or torch_npu.get_npu_format(value) != 2:
                raise ValueError('normalization requires dense row-major ND device inputs')
        self.inputs[arm] = inputs
        y = torch.empty((c['rows'], c['width']), dtype=torch.float32, device='npu:0')
        need_stat = spec['family'] == 'E01' or c['saved']
        stat = torch.empty((c['rows'],), dtype=torch.float32, device='npu:0') if need_stat else y
        dx = torch.empty_like(y) if c['rms'] else None
        forward, backward = kernels()
        compiled = {}
        result = (y, dx) if c['rms'] else (y, stat)

        def run():
            compiled['forward'] = forward[((c['rows'] + c['batch'] - 1) // c['batch'],)](
                inputs['x'], y, stat, c['rows'], c['width'], c['eps'],
                c['block'], c['batch'], c['rms'], c['saved'])
            if c['rms']:
                compiled['backward'] = backward[(c['rows'],)](
                    inputs['x'], inputs['dy'], stat, dx, c['rows'], c['width'],
                    c['eps'], c['block'], c['saved'])
            return result

        expected = [forward.fn.__name__] + ([backward.fn.__name__] if c['rms'] else [])
        workspace = c['rows'] * 4 if c['rms'] and c['saved'] else 0
        return SimpleNamespace(run=run, expected_kernels=expected,
            sources=[__file__, gemm.__file__, *CASE_SOURCES],
            metadata=dict(configuration=c, dtype='float32', workspace_bytes=workspace,
                          workspace_scope='extra saved inverse-RMS state; all outputs preallocated',
                          operation='RMS forward plus input VJP' if c['rms'] else 'L2 forward and inverse norm',
                          input_strides={name: list(t.stride()) for name, t in inputs.items()}),
            compilation_state=lambda: gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name, tensor in self.inputs[arm].items():
            if not torch.equal(tensor.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: ' + name)
        return diagnostics(tuple(t.detach().cpu() for t in actual), fixture)


def factory():
    return Adapter()
