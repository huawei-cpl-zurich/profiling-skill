"""E08: contended FP32 atomics versus private FP32 gradients and final reduction.

Independent implementation inspired by pinned Triton-Ascend rms_norm.py
135-191,295-359,440-488. A3 atomics, resource limits and speed are unqualified
until the controlled pair harness has actually executed both complete arms.
"""
from functools import lru_cache
from types import SimpleNamespace
import torch
import gemm
from gradient_contract import CASE_SOURCES, check_spec, diagnostics, make_case


def configuration(arm, spec):
    rows, width, _, eps = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    return dict(rows=rows, width=width, eps=eps, private=arm == 'after',
                partials=32, block=1 << (width - 1).bit_length(), reduction_tile=64)


def program_rows(pid, rows, programs):
    if any(type(v) is not int for v in (pid, rows, programs)) or not (
            rows > 0 and programs > 0 and 0 <= pid < programs):
        raise ValueError('positive dimensions and a valid program required')
    return range(pid, rows, programs)


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def gradient_zero(DW, D: tl.constexpr, BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(DW + col, 0., col < D)

    @triton.jit
    def weighted_vjp(X, W, DY, DX, DW, R: tl.constexpr, D: tl.constexpr,
                     EPS: tl.constexpr, BD: tl.constexpr, PROGRAMS: tl.constexpr,
                     PRIVATE: tl.constexpr):
        pid = tl.program_id(0)
        col = tl.arange(0, BD)
        weight = tl.load(W + col, col < D, 0)
        partial = tl.zeros((BD,), tl.float32)
        for block in range((R + PROGRAMS - 1) // PROGRAMS):
            row = pid + block * PROGRAMS
            mask = (row < R) & (col < D)
            x = tl.load(X + row * D + col, mask, 0)
            dy = tl.load(DY + row * D + col, mask, 0)
            inverse = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            m = dy * weight
            projection = tl.sum(m * x, 0) / D
            dx = inverse * (m - x * inverse * inverse * projection)
            tl.store(DX + row * D + col, dx, mask)
            contribution = dy * (x * inverse)
            if PRIVATE:
                partial += contribution
            else:
                tl.atomic_add(DW + col, contribution, mask, sem='relaxed')
        if PRIVATE:
            # Even programs with no rows write zeros, never uninitialized memory.
            tl.store(DW + pid * D + col, partial, col < D)

    @triton.jit
    def gradient_reduce(Partial, DW, D: tl.constexpr, P: tl.constexpr,
                        BD: tl.constexpr):
        col = tl.program_id(0) * BD + tl.arange(0, BD)
        part = tl.arange(0, P)
        values = tl.load(Partial + part[:, None] * D + col[None, :],
                         col[None, :] < D, 0)
        tl.store(DW + col, tl.sum(values, 0), col < D)

    return gradient_zero, weighted_vjp, gradient_reduce


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm, spec)
        inputs = {name: value.npu() for name, value in fixture['inputs'].items()}
        shapes = dict(x=(c['rows'], c['width']), dy=(c['rows'], c['width']), w=(c['width'],))
        for name, value in inputs.items():
            if tuple(value.shape) != shapes[name] or value.dtype != torch.float32 or (
                    not value.is_contiguous() or torch_npu.get_npu_format(value) != 2):
                raise ValueError('weighted RMS VJP requires dense FP32 ND inputs')
        self.inputs[arm] = inputs
        dx = torch.empty((c['rows'], c['width']), dtype=torch.float32, device='npu:0')
        dw = torch.empty((c['width'],), dtype=torch.float32, device='npu:0')
        partial = torch.empty((c['partials'], c['width']), dtype=torch.float32,
                              device='npu:0') if c['private'] else dw
        zero, backward, reduce = kernels()
        compiled = {}

        def run():
            if not c['private']:
                compiled['zero'] = zero[((c['width'] + 255) // 256,)](dw, c['width'], 256)
            programs = c['partials'] if c['private'] else c['rows']
            compiled['backward'] = backward[(programs,)](inputs['x'], inputs['w'], inputs['dy'],
                dx, partial, c['rows'], c['width'], c['eps'], c['block'], programs, c['private'])
            if c['private']:
                compiled['reduce'] = reduce[((c['width'] + c['reduction_tile'] - 1) // c['reduction_tile'],)](
                    partial, dw, c['width'], c['partials'], c['reduction_tile'])
            return dx, dw

        expected = ([backward.fn.__name__, reduce.fn.__name__] if c['private'] else
                    [zero.fn.__name__, backward.fn.__name__])
        return SimpleNamespace(run=run, expected_kernels=expected,
            sources=[__file__, gemm.__file__, *CASE_SOURCES],
            metadata=dict(configuration=c, dtype='float32', operation='weighted RMS input and weight VJP',
                          workspace_bytes=c['partials'] * c['width'] * 4 if c['private'] else 0,
                          workspace_scope='private dW partials; initialization and final sum timed',
                          input_strides={name: list(value.stride()) for name, value in inputs.items()}),
            compilation_state=lambda: gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name, value in self.inputs[arm].items():
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: ' + name)
        return diagnostics(tuple(value.detach().cpu() for value in actual), fixture)


def factory():
    return Adapter()
