"""F08: centered two-pass statistics versus streaming lane-local Welford.

Project-owned Triton formulation of the statistics algorithm described in
BatchNormV3Welford; this is not a BatchNorm implementation. All preparation is
host-side allocation. Each forward initializes its local state and writes both
outputs; no warmup, inspection, autotuning or additional timing launches.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from welford_contract import CASE_SOURCES, diagnostics, dimensions, make_case, validate_inputs


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave003_centered_statistics(X, Mean, Variance, C:tl.constexpr, HW:tl.constexpr,
                                   R:tl.constexpr, B:tl.constexpr):
        channel = tl.program_id(0)
        lane = tl.arange(0, B)
        if R <= B:
            offset = (lane//HW*C+channel)*HW+lane%HW
            x = tl.load(X+offset, lane<R, 0)
            mean = tl.sum(x, 0)/R
            residual = tl.where(lane<R, x-mean, 0.)
            variance = tl.sum(residual*residual, 0)/R
        else:
            partial = tl.full((B,), 0., tl.float32)
            for start in range(tl.cdiv(R, B)):
                index = start*B+lane
                offset = (index//HW*C+channel)*HW+index%HW
                x = tl.load(X+offset, index<R, 0)
                partial += x
            mean = tl.sum(partial, 0)/R
            squared = tl.full((B,), 0., tl.float32)
            for start in range(tl.cdiv(R, B)):
                index = start*B+lane
                offset = (index//HW*C+channel)*HW+index%HW
                x = tl.load(X+offset, index<R, 0)
                residual = tl.where(index<R, x-mean, 0.)
                squared += residual*residual
            variance = tl.sum(squared, 0)/R
        tl.store(Mean+channel, mean)
        tl.store(Variance+channel, variance)

    @triton.jit
    def wave003_welford_statistics(X, Mean, Variance, C:tl.constexpr, HW:tl.constexpr,
                                  R:tl.constexpr, B:tl.constexpr):
        channel = tl.program_id(0)
        lane = tl.arange(0, B)
        count = tl.full((B,), 0., tl.float32)
        mean = tl.full((B,), 0., tl.float32)
        m2 = tl.full((B,), 0., tl.float32)
        for start in range(tl.cdiv(R, B)):
            index = start*B+lane
            valid = index<R
            offset = (index//HW*C+channel)*HW+index%HW
            x = tl.load(X+offset, valid, 0)
            next_count = count+valid.to(tl.float32)
            denominator = tl.maximum(next_count, 1.)
            delta = tl.where(valid, x-mean, 0.)
            mean += delta/denominator
            m2 += delta*delta*count/denominator
            count = next_count
        merged = tl.sum(mean*count, 0)/R
        correction = mean-merged
        variance = tl.sum(m2+count*correction*correction, 0)/R
        tl.store(Mean+channel, merged)
        tl.store(Variance+channel, variance)

    return wave003_centered_statistics, wave003_welford_statistics


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        n, c, h, w = dimensions(spec)
        validate_inputs(fixture['x'], spec)
        import torch
        import torch_npu
        x = fixture['x'].npu()
        mean = torch.empty(c, dtype=torch.float32, device='npu')
        variance = torch.empty_like(mean)
        before, after = kernels()
        kernel = before if arm == 'before' else after
        compiled = {}

        def run():
            artifact = kernel[(c,)](x, mean, variance, c, h*w, n*h*w, 256)
            compiled[id(artifact)] = artifact
            return mean, variance

        def compilation_state():
            rows = []
            for artifact in compiled.values():
                asm = {k:hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()
                       for k,v in getattr(artifact, 'asm', {}).items() if isinstance(v, (str, bytes))}
                rows.append(dict(hash=getattr(artifact, 'hash', None), artifacts=asm))
            return rows

        population = n*h*w
        passes = 2 if arm == 'before' and population > 256 else 1
        item = SimpleNamespace(run=run, inputs=x, compilation_state=compilation_state,
            expected_kernels=[kernel.fn.__name__], sources=[Path(__file__), *CASE_SOURCES],
            metadata=dict(language='triton', torch_version=torch.__version__, triton_version=triton.__version__,
                shape=spec, block=256, arm=arm, arithmetic='FP32', workspace_bytes=0,
                stages_per_forward=1, logical_input_passes=passes,
                logical_input_bytes=passes*x.numel()*x.element_size(),
                logical_output_bytes=2*c*4,
                traffic_claim='logical algorithmic bytes only; physical traffic unavailable'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        if not torch.equal(self.prepared[arm].inputs.detach().cpu(), fixture['x']):
            result.update(passed=False, reason='input mutation')
        return result


def factory():
    return Adapter()
