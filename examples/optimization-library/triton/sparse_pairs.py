"""E05: identical 16x32 tiles, dense masked visits versus selected-block visits.

Sparse traversal is motivated by pinned NSA code, independently implemented:
ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L85-L93
ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py:L131-L166
Unlike the source's real_topk prefix assumption, -1 holes never terminate visits.
No speedup is claimed before the recorded A3 paired qualification.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

from sparse_contract import BM, BN, CASE_SOURCES, check_spec, diagnostics, make_case, validate_inputs


@lru_cache(maxsize=1)
def kernels():
    global tl, _online_update
    import triton
    import triton.language as tl

    @triton.jit
    def _online_update(QV, K, V, MAX, DEN, ACC, HEAD, ROWS, BLOCK, ACTIVE,
               NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
        positions = BLOCK * BN + tl.arange(0, BN)
        channels = tl.arange(0, D)
        key = tl.load(K + (HEAD*NK+positions[None, :])*D + channels[:, None],
                      (positions[None, :] >= 0) & (positions[None, :] < NK), other=0)
        value = tl.load(V + (HEAD*NK+positions[:, None])*D + channels[None, :],
                        (positions[:, None] >= 0) & (positions[:, None] < NK), other=0)
        valid = ACTIVE & (positions[None, :] < NK) & (ROWS[:, None] < NQ)
        scores = tl.where(valid, tl.dot(QV, key) * (D**-0.5), -float('inf'))
        next_maximum = tl.maximum(MAX, tl.max(scores, 1))
        # All-masked tiles keep neutral state rather than exp(-inf - -inf).
        origin = tl.where(next_maximum == -float('inf'), 0.0, next_maximum)
        rescale = tl.exp(MAX-origin)
        probability = tl.where(valid, tl.exp(scores-origin[:, None]), 0.0)
        ACC = ACC * rescale[:, None] + tl.dot(probability.to(tl.float16), value)
        DEN = DEN * rescale + tl.sum(probability, 1)
        return next_maximum, DEN, ACC

    @triton.jit
    def selected_attention(Q, K, V, SELECTED, O, NQ: tl.constexpr, NK: tl.constexpr,
                           D: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
                           SLOTS: tl.constexpr, SI: tl.constexpr, SPARSE: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr):
        tile, head = tl.program_id(0), tl.program_id(1)
        rows = tile*BM + tl.arange(0, BM)
        channels = tl.arange(0, D)
        q = tl.load(Q+(head*NQ+rows[:, None])*D+channels[None, :], rows[:, None] < NQ, 0)
        selection = SELECTED + (head*tl.cdiv(NQ, BM)+tile)*SLOTS
        maximum = tl.full((BM,), -float('inf'), tl.float32)
        denominator = tl.full((BM,), 0, tl.float32)
        numerator = tl.full((BM, D), 0, tl.float32)
        if SPARSE:
            for slot in range(SLOTS):
                block = tl.load(selection+slot)
                if (block >= 0) & (block < tl.cdiv(NK, BN)):
                    maximum, denominator, numerator = _online_update(q, K, V, maximum, denominator,
                        numerator, head//(HQ//HK), rows, block, True, NQ, NK, D, BM, BN)
        else:
            slots = tl.arange(0, SI)
            selected = tl.load(selection+slots, slots < SLOTS, other=-1)
            for block in range(tl.cdiv(NK, BN)):
                active = tl.sum((selected == block).to(tl.int32), 0) > 0
                maximum, denominator, numerator = _online_update(q, K, V, maximum, denominator,
                    numerator, head//(HQ//HK), rows, block, active, NQ, NK, D, BM, BN)
        output = numerator/tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(O+(head*NQ+rows[:, None])*D+channels[None, :], output, rows[:, None] < NQ)

    return selected_attention


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        if arm not in ('before', 'after'):
            raise ValueError('paired arm required')
        hq, hk, nq, nk, d, slots = check_spec(spec)
        validate_inputs(spec, fixture['inputs'])
        import torch_npu
        import triton
        inputs = {key: value.to('npu:0') for key, value in fixture['inputs'].items()}
        if any(value.stride()!=fixture['inputs'][key].stride()
               or torch_npu.get_npu_format(value)!=2 for key,value in inputs.items()):
            raise ValueError('device inputs must preserve dense ND physical layout')
        self.inputs[arm] = inputs
        result = torch.empty_like(fixture['expected'], device='npu:0')
        if torch_npu.get_npu_format(result)!=2:
            raise ValueError('output requires dense ND physical layout')
        kernel, compiled = kernels(), {}
        def run():
            artifact = kernel[((nq+BM-1)//BM, hq)](*[inputs[k] for k in ('q','k','v','selected')],
                result, nq, nk, d, hq, hk, slots, triton.next_power_of_2(slots), arm=='after', BM, BN)
            compiled[str(artifact.hash)] = dict(name=artifact.name, hash=str(artifact.hash),
                binaries={k:hashlib.sha256(v).hexdigest() for k,v in artifact.asm.items() if isinstance(v,bytes)})
            return result
        return SimpleNamespace(run=run, expected_kernels=[kernel.fn.__name__], inputs=inputs,
            sources=[Path(__file__), *CASE_SOURCES], compilation_state=lambda:list(compiled.values()),
            metadata=dict(family='E05', query_tile=BM, key_tile=BN, sparse=arm=='after',
                          workspace_bytes=0, selection_origin='supplied immutable list, both arms',
                          workspace_scope='adapter explicit allocation; compiler scratch not inferred',
                          nominal_fp32_output_accumulator_bytes=BM*d*4,
                          selection_semantics='unique valid block IDs; arbitrary -1 holes/order',
                          torch=torch.__version__, triton=triton.__version__))

    def validate(self, arm, actual, fixture, spec):
        import torch
        torch.npu.synchronize()
        return diagnostics(actual.detach().cpu(), fixture, self.inputs[arm])


def factory():
    return Adapter()
