"""E09 reuse sin/cos over Q/K heads; source-inspired, not yet A3-qualified.

Pinned source: Triton-Ascend 7acb9c40a6ca, index/fused_rotary_emb.py:L51-L74.
Both arms use the same token/frequency tile and FP16 coefficient rounding.
After trades head parallelism for reuse and a longer coefficient live range.
"""
from functools import lru_cache
from types import SimpleNamespace
import torch
import gemm
from rotary_contract import CASE_SOURCES, check_spec, make_case, diagnostics


def configuration(arm, spec):
    b, t, hq, hk, d, _, scale = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    return dict(batch=b, tokens=t, q_heads=hq, k_heads=hk, width=d,
                token_tile=8, frequency_tile=d//2, scale=scale,
                head_parallel=arm == 'before')


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def rotary_head_parallel(Q, K, Pos, Freq, OQ, OK, T: tl.constexpr,
                             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                             SCALE: tl.constexpr, BT: tl.constexpr, BF: tl.constexpr):
        b, tile, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        token = tile*BT+tl.arange(0, BT)[:, None]
        f = tl.arange(0, BF)[None, :]
        position = tl.load(Pos+b*T+token, token < T, 0).to(tl.float32)
        angle = position / SCALE * tl.load(Freq+f)
        cos = tl.cos(angle).to(tl.float16).to(tl.float32)
        sin = tl.sin(angle).to(tl.float16).to(tl.float32)
        if h < HQ:
            off = ((b*T+token)*HQ+h)*D+f
            lo = tl.load(Q+off, token < T, 0).to(tl.float32)
            hi = tl.load(Q+off+BF, token < T, 0).to(tl.float32)
            tl.store(OQ+off, lo*cos-hi*sin, token < T)
            tl.store(OQ+off+BF, hi*cos+lo*sin, token < T)
        else:
            off = ((b*T+token)*HK+h-HQ)*D+f
            lo = tl.load(K+off, token < T, 0).to(tl.float32)
            hi = tl.load(K+off+BF, token < T, 0).to(tl.float32)
            tl.store(OK+off, lo*cos-hi*sin, token < T)
            tl.store(OK+off+BF, hi*cos+lo*sin, token < T)

    @triton.jit
    def rotary_shared_coefficients(Q, K, Pos, Freq, OQ, OK, T: tl.constexpr,
                                    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                                    SCALE: tl.constexpr, BT: tl.constexpr, BF: tl.constexpr):
        b, tile = tl.program_id(0), tl.program_id(1)
        token = tile*BT+tl.arange(0, BT)[:, None]
        f = tl.arange(0, BF)[None, :]
        position = tl.load(Pos+b*T+token, token < T, 0).to(tl.float32)
        angle = position / SCALE * tl.load(Freq+f)
        cos = tl.cos(angle).to(tl.float16).to(tl.float32)
        sin = tl.sin(angle).to(tl.float16).to(tl.float32)
        for h in range(HQ):
            off = ((b*T+token)*HQ+h)*D+f
            lo = tl.load(Q+off, token < T, 0).to(tl.float32)
            hi = tl.load(Q+off+BF, token < T, 0).to(tl.float32)
            tl.store(OQ+off, lo*cos-hi*sin, token < T)
            tl.store(OQ+off+BF, hi*cos+lo*sin, token < T)
        for h in range(HK):
            off = ((b*T+token)*HK+h)*D+f
            lo = tl.load(K+off, token < T, 0).to(tl.float32)
            hi = tl.load(K+off+BF, token < T, 0).to(tl.float32)
            tl.store(OK+off, lo*cos-hi*sin, token < T)
            tl.store(OK+off+BF, hi*cos+lo*sin, token < T)

    return rotary_head_parallel, rotary_shared_coefficients


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm, spec)
        inputs = {name: value.npu() for name, value in fixture['inputs'].items()}
        expected = dict(q=((c['batch'], c['tokens'], c['q_heads'], c['width']), torch.float16),
            k=((c['batch'], c['tokens'], c['k_heads'], c['width']), torch.float16),
            positions=((c['batch'], c['tokens']), torch.int32), freq=((c['width']//2,), torch.float32))
        converted = []
        for name, value in inputs.items():
            if torch_npu.get_npu_format(value) != 2:
                normalized = torch_npu.npu_format_cast(value, 2)
                if (normalized.shape != value.shape or normalized.dtype != value.dtype
                        or normalized.stride() != value.stride()
                        or torch_npu.get_npu_format(normalized) != 2
                        or not torch.equal(normalized.detach().cpu(), fixture['inputs'][name])):
                    raise ValueError('trusted ND preparation changed input ABI or values: '+name)
                inputs[name] = value = normalized
                converted.append(name)
            shape, dtype = expected[name]
            if (value.shape != shape or value.dtype != dtype or not value.is_contiguous()
                    or value.stride() != fixture['inputs'][name].stride()
                    or torch_npu.get_npu_format(value) != 2):
                raise ValueError('rotary requires dense contiguous ND inputs')
        self.inputs[arm] = inputs
        q, k = inputs['q'], inputs['k']
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        kernel = kernels()[0 if arm == 'before' else 1]
        grid = (c['batch'], (c['tokens']+7)//8)
        if c['head_parallel']:
            grid += (c['q_heads']+c['k_heads'],)
        compiled = {}

        def run():
            compiled['rotary'] = kernel[grid](q, k, inputs['positions'], inputs['freq'], oq, ok,
                c['tokens'], c['q_heads'], c['k_heads'], c['width'], c['scale'],
                c['token_tile'], c['frequency_tile'], enable_fp_fusion=False)
            return oq, ok

        return SimpleNamespace(run=run, expected_kernels=[kernel.fn.__name__],
            sources=[__file__, gemm.__file__, *CASE_SOURCES],
            metadata=dict(configuration=c, workspace_bytes=0,
                trusted_nd_preparation=converted,
                coefficient_sets_per_valid_token=(c['q_heads']+c['k_heads'])
                    if c['head_parallel'] else 1,
                operation='complete split-half Q/K rotary', arithmetic='FP16 coefficients; FP32 products; FP16 outputs',
                input_strides={name:list(v.stride()) for name,v in inputs.items()}),
            compilation_state=lambda: gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name, value in self.inputs[arm].items():
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: '+name)
        return diagnostics(tuple(v.detach().cpu() for v in actual), fixture)


def factory():
    return Adapter()
