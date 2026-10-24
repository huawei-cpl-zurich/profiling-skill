"""One-query vector attention versus query-tiled Cube attention with K/V reuse.

Preparation allocates and transfers; only the sole correctness call compiles.
Forwards contain no tensor inspection, warmup, autotuning or output checking.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from attention_contract import CASE_SOURCES, diagnostics, dimensions, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave001_attention_row(Q, K, V, O, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                              SCALE: tl.constexpr, BN: tl.constexpr):
        row, head = tl.program_id(0), tl.program_id(1)
        channels = tl.arange(0, D)
        positions = tl.arange(0, BN)
        query = tl.load(Q + (head * NQ + row) * D + channels).to(tl.float32)
        maximum = tl.full((), -float('inf'), tl.float32)
        denominator = tl.full((), 0, tl.float32)
        numerator = tl.full((D,), 0, tl.float32)
        for start in range(0, NK, BN):
            indices = start + positions
            key = tl.load(K + (head * NK + indices[:, None]) * D + channels[None, :],
                          indices[:, None] < NK, 0).to(tl.float32)
            value = tl.load(V + (head * NK + indices[:, None]) * D + channels[None, :],
                            indices[:, None] < NK, 0).to(tl.float32)
            score = tl.sum(key * query[None, :], 1) * SCALE
            score = tl.where(indices < NK, score, -float('inf'))
            next_maximum = tl.maximum(maximum, tl.max(score, 0))
            rescale = tl.exp(maximum - next_maximum)
            probability = tl.exp(score - next_maximum)
            numerator = numerator * rescale + tl.sum(probability.to(tl.float16).to(tl.float32)[:, None] * value, 0)
            denominator = denominator * rescale + tl.sum(probability, 0)
            maximum = next_maximum
        tl.store(O + (head * NQ + row) * D + channels, numerator / denominator)

    @triton.jit
    def wave001_attention_query_tile(Q, K, V, O, NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                                     SCALE: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
        query_rows = tl.program_id(0) * BM + tl.arange(0, BM)
        head = tl.program_id(1)
        channels = tl.arange(0, D)
        positions = tl.arange(0, BN)
        query = tl.load(Q + (head * NQ + query_rows[:, None]) * D + channels[None, :],
                        query_rows[:, None] < NQ, 0)
        maximum = tl.full((BM,), -float('inf'), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        numerator = tl.full((BM, D), 0, tl.float32)
        for start in range(0, NK, BN):
            indices = start + positions
            key = tl.load(K + (head * NK + indices[None, :]) * D + channels[:, None],
                          indices[None, :] < NK, 0)
            value = tl.load(V + (head * NK + indices[:, None]) * D + channels[None, :],
                            indices[:, None] < NK, 0)
            score = tl.dot(query, key) * SCALE
            score = tl.where(indices[None, :] < NK, score, -float('inf'))
            next_maximum = tl.maximum(maximum, tl.max(score, 1))
            rescale = tl.exp(maximum - next_maximum)
            probability = tl.exp(score - next_maximum[:, None])
            numerator = numerator * rescale[:, None] + tl.dot(probability.to(tl.float16), value)
            denominator = denominator * rescale + tl.sum(probability, 1)
            maximum = next_maximum
        tl.store(O + (head * NQ + query_rows[:, None]) * D + channels[None, :],
                 numerator / denominator[:, None], query_rows[:, None] < NQ)

    return wave001_attention_row, wave001_attention_query_tile


def compiled_identity(artifact):
    result = {}
    for key, value in getattr(artifact, 'asm', {}).items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            result[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=getattr(artifact, 'hash', None), artifacts=result)


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
        b, h, nq, nk, d, _ = dimensions(spec)
        inputs = [fixture[key] for key in ('q', 'k', 'v')]
        shapes = [(b, h, nq, d), (b, h, nk, d), (b, h, nk, d)]
        for value, shape in zip(inputs, shapes):
            if value.dtype != torch.float16 or value.shape != shape or not value.is_contiguous():
                raise ValueError('contiguous FP16 tensors with declared shapes required')
            if not torch.isfinite(value).all():
                raise ValueError('finite inputs required')
        device_inputs = tuple(value.npu() for value in inputs)
        out = torch.empty(shapes[0], dtype=torch.float32, device=device_inputs[0].device)
        kernel = kernels()[0 if arm == 'before' else 1]
        compiled = {}
        bm, bn = 16, 64
        grid = (nq if arm == 'before' else triton.cdiv(nq, bm), b * h)

        def run():
            args = (*device_inputs, out, nq, nk, d, d ** -0.5, bn)
            artifact = kernel[grid](*args, *(() if arm == 'before' else (bm,)))
            compiled[id(artifact)] = artifact
            return out

        prepared = SimpleNamespace(run=run, expected_kernels=[kernel.fn.__name__],
            sources=[Path(__file__), *CASE_SOURCES], inputs=device_inputs,
            metadata=dict(language='triton', arm=arm, torch_version=torch.__version__,
                          triton_version=triton.__version__, query_tile=1 if arm == 'before' else bm,
                          key_tile=bn, grid=list(grid), num_autotune_trials=0,
                          probability_dtype='fp16 for weighted values; fp32 denominator'),
            compilation_state=lambda: [compiled_identity(item) for item in compiled.values()])
        self.prepared[arm] = prepared
        return prepared

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.detach().cpu(), fixture)
        for value, key in zip(self.prepared[arm].inputs, ('q', 'k', 'v')):
            if not torch.equal(value.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        return result


def factory():
    return Adapter()
