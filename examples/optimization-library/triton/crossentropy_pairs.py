"""E10: full log-probabilities versus row sufficient statistics, loss+dlogits only.

Independent implementation of stable cross-entropy math; exact pinned source
context and A3 performance qualification are retained separately by the study.
"""
from functools import lru_cache
from types import SimpleNamespace
import torch
import gemm
from crossentropy_contract import CASE_SOURCES, check_spec, diagnostics, make_case


def configuration(arm, spec):
    rows, width, _ = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    direct = arm == 'after'
    return dict(rows=rows, width=width, direct=direct,
                block=1 << (width - 1).bit_length(),
                workspace_bytes=4 * rows * (1 if direct else width))


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def ce_statistics(X, Targets, Scratch, Loss, D: tl.constexpr,
                      BD: tl.constexpr, DIRECT: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BD)
        x = tl.load(X + row * D + col, col < D, -float('inf'))
        maximum = tl.max(x, 0)
        lse = maximum + tl.log(tl.sum(tl.exp(x - maximum), 0))
        if DIRECT:
            # Public labels stay INT64; checked vocabulary<=8192 makes the
            # local index exact in INT32 and keeps comparison chains in INT32.
            target = tl.load(Targets + row).to(tl.int32)
            selected = tl.load(X + row * D + target)
            tl.store(Scratch + row, lse)
            tl.store(Loss + row, lse - selected)
        else:
            tl.store(Scratch + row * D + col, x - lse, col < D)

    @triton.jit
    def ce_gradient(X, Targets, Scratch, Loss, Gradient, D: tl.constexpr,
                    BD: tl.constexpr, DIRECT: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BD)
        target = tl.load(Targets + row).to(tl.int32)
        if DIRECT:
            x = tl.load(X + row * D + col, col < D, -float('inf'))
            log_probability = x - tl.load(Scratch + row)
        else:
            log_probability = tl.load(Scratch + row * D + col, col < D, -float('inf'))
            selected = tl.load(Scratch + row * D + target)
            tl.store(Loss + row, -selected)
        gradient = tl.exp(log_probability) - (col == target).to(tl.float32)
        tl.store(Gradient + row * D + col, gradient, col < D)

    return ce_statistics, ce_gradient


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm, spec)
        inputs = {key: value.npu() for key, value in fixture['inputs'].items()}
        for name, dtype, shape in (('logits', torch.float32, (c['rows'], c['width'])),
                                   ('targets', torch.int64, (c['rows'],))):
            value = inputs[name]
            if tuple(value.shape) != shape or value.dtype != dtype or (
                    not value.is_contiguous() or torch_npu.get_npu_format(value) != 2):
                raise ValueError('dense ND FP32 logits and INT64 targets required')
        self.inputs[arm] = inputs
        statistics, gradient = kernels()
        compiled = {}

        def run():
            # Allocation is inside the complete caller; no initialization is free.
            loss = torch.empty((c['rows'],), dtype=torch.float32, device='npu:0')
            dx = torch.empty((c['rows'], c['width']), dtype=torch.float32, device='npu:0')
            scratch = torch.empty((c['rows'],) if c['direct'] else dx.shape,
                                  dtype=torch.float32, device='npu:0')
            compiled['statistics'] = statistics[(c['rows'],)](
                inputs['logits'], inputs['targets'], scratch, loss,
                c['width'], c['block'], c['direct'])
            compiled['gradient'] = gradient[(c['rows'],)](
                inputs['logits'], inputs['targets'], scratch, loss, dx,
                c['width'], c['block'], c['direct'])
            return loss, dx

        return SimpleNamespace(run=run,
            expected_kernels=[statistics.fn.__name__, gradient.fn.__name__],
            sources=[__file__, gemm.__file__, *CASE_SOURCES],
            metadata=dict(configuration=c, operation='per-row cross-entropy and sum-VJP',
                          workspace_bytes=c['workspace_bytes'], dtype='float32',
                          outputs='loss[rows] and dlogits[rows,width]; log-probabilities not public',
                          input_strides={key: list(value.stride()) for key, value in inputs.items()}),
            compilation_state=lambda: gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name, value in self.inputs[arm].items():
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: ' + name)
        return diagnostics(tuple(value.detach().cpu() for value in actual), fixture)


def factory():
    return Adapter()
