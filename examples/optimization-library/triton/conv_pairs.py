"""E06 eliminate sequential state traffic with immutable virtual token windows.

Source: Triton-Ascend 7acb9c40a6ca, conv/causal_conv1d_update.py:L53-L164.
This scoped pair has no accepted-token indices, padding slots, bias or activation.
The after path rereads immutable windows in parallel; it does not claim all
windows/weights are loaded once. A3 qualification is required before teaching.
"""
from functools import lru_cache
from types import SimpleNamespace
import torch
import gemm
from conv_contract import CASE_SOURCES, check_spec, make_case, diagnostics


def configuration(arm, spec):
    b, t, d, k, _ = check_spec(spec)
    if arm not in ('before', 'after'):
        raise ValueError('arm must be before or after')
    return dict(batch=b, tokens=t, width=d, kernel=k, block=256,
                taps=1 << (k-1).bit_length(), launches=t+1 if arm == 'before' else 1)


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def conv_state_init(I, S, N: tl.constexpr, BD: tl.constexpr):
        i = tl.program_id(0) * BD + tl.arange(0, BD)
        tl.store(S+i, tl.load(I+i, i < N, 0), i < N)

    @triton.jit
    def conv_one_token(X, W, S, Y, T: tl.constexpr, D: tl.constexpr,
                       K: tl.constexpr, TOKEN, BD: tl.constexpr,
                       BK: tl.constexpr):
        b = tl.program_id(0)
        d = tl.program_id(1) * BD + tl.arange(0, BD)
        lag = tl.arange(0, BK)
        cur = tl.load(X + (b*T+TOKEN)*D+d, d < D, 0)
        state = tl.load(S + (b*D+d[None, :])*(K-1)+lag[:, None],
                        (d[None, :] < D) & (lag[:, None] < K-1), 0)
        sample = tl.where(lag[:, None] == K-1, cur[None, :], state).to(tl.float32)
        weight = tl.load(W+d[None, :]*K+lag[:, None],
                         (d[None, :] < D) & (lag[:, None] < K), 0).to(tl.float32)
        # All prior samples needed by both output and shift are read before stores.
        shifted = tl.load(S + (b*D+d[None, :])*(K-1)+lag[:, None]+1,
                          (d[None, :] < D) & (lag[:, None] < K-2), 0)
        next_state = tl.where(lag[:, None] == K-2, cur[None, :], shifted)
        y = tl.sum(sample * weight, axis=0)
        tl.store(Y+(b*T+TOKEN)*D+d, y, d < D)
        tl.store(S+(b*D+d[None, :])*(K-1)+lag[:, None], next_state,
                 (d[None, :] < D) & (lag[:, None] < K-1))

    @triton.jit
    def conv_token_windows(X, W, I, Y, S, T: tl.constexpr, D: tl.constexpr,
                           K: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
        b = tl.program_id(0)
        token = tl.program_id(1)
        d = tl.program_id(2) * BD + tl.arange(0, BD)
        lag = tl.arange(0, BK)
        pos = token + lag[:, None]
        mask = (d[None, :] < D) & (lag[:, None] < K)
        prefix = tl.load(I+(b*D+d[None, :])*(K-1)+pos,
                         mask & (pos < K-1), 0)
        x = tl.load(X+(b*T+pos-(K-1))*D+d[None, :],
                    mask & (pos >= K-1), 0)
        sample = tl.where(pos < K-1, prefix, x).to(tl.float32)
        weight = tl.load(W+d[None, :]*K+lag[:, None], mask, 0).to(tl.float32)
        tl.store(Y+(b*T+token)*D+d, tl.sum(sample*weight, axis=0), d < D)
        # Unique writer for each final-state element; no mutable input dependency.
        if token == 0:
            pos = T + lag[:, None]
            mask = (d[None, :] < D) & (lag[:, None] < K-1)
            prefix = tl.load(I+(b*D+d[None, :])*(K-1)+pos,
                             mask & (pos < K-1), 0)
            x = tl.load(X+(b*T+pos-(K-1))*D+d[None, :],
                        mask & (pos >= K-1), 0)
            tl.store(S+(b*D+d[None, :])*(K-1)+lag[:, None],
                     tl.where(pos < K-1, prefix, x), mask)

    return conv_state_init, conv_one_token, conv_token_windows


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        c = configuration(arm, spec)
        b, t, d, k = (c[n] for n in ('batch', 'tokens', 'width', 'kernel'))
        inputs = {name: value.npu() for name, value in fixture['inputs'].items()}
        for name, value in inputs.items():
            if value.dtype != torch.float16 or not value.is_contiguous() or (
                    value.stride() != fixture['inputs'][name].stride()
                    or torch_npu.get_npu_format(value) != 2):
                raise ValueError('convolution requires dense contiguous FP16 ND inputs')
        self.inputs[arm] = inputs
        y = torch.empty((b, t, d), dtype=torch.float16, device='npu:0')
        state = torch.empty((b, d, k-1), dtype=torch.float16, device='npu:0')
        init, one, windows = kernels()
        compiled = {}
        grid = (b, (d+c['block']-1)//c['block'])

        def run():
            if arm == 'before':
                compiled['init'] = init[((b*d*(k-1)+255)//256,)](
                    inputs['initial'], state, b*d*(k-1), 256)
                for token in range(t):
                    compiled[f'token-{token}'] = one[grid](inputs['x'], inputs['w'], state, y,
                        t, d, k, token, c['block'], c['taps'], enable_fp_fusion=False)
            else:
                compiled['windows'] = windows[(b, t, grid[1])](
                    inputs['x'], inputs['w'], inputs['initial'], y, state,
                    t, d, k, c['block'], c['taps'], enable_fp_fusion=False)
            return y, state

        expected = [init.fn.__name__, one.fn.__name__] if arm == 'before' else [windows.fn.__name__]
        return SimpleNamespace(run=run, expected_kernels=expected,
            sources=[__file__, gemm.__file__, *CASE_SOURCES],
            metadata=dict(configuration=c, dtype='float16', workspace_bytes=0,
                state_output_bytes=2*b*d*(k-1),
                state_workspace_scope='before mutates preallocated output state after timed initialization',
                sequential_state_write_bytes=2*b*d*(k-1)*(t+1) if arm == 'before' else 2*b*d*(k-1),
                operation='complete causal depthwise convolution and final state',
                input_strides={name: list(v.stride()) for name, v in inputs.items()}),
            compilation_state=lambda: gemm.compiled_identity(compiled))

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        for name, value in self.inputs[arm].items():
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: '+name)
        return diagnostics(tuple(v.detach().cpu() for v in actual), fixture)


def factory():
    return Adapter()
