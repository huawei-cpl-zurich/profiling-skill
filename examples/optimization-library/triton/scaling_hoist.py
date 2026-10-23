"""Wave001 family01: change beta scaling location around a BF16 dot.

No autotuning, warmup, output inspection or allocation occurs in a forward.
Use optimization_pair.py with this directory on PYTHONPATH.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from scaling_contract import CASE_SOURCES, diagnostics, dimensions, make_case


def kernels():
    # Older Ascend JIT frontends resolve AST names through fn.__globals__.
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave001_kkt_scale_before(K, KRight, Beta, Out, BT: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr):
        group = tl.program_id(0)
        row = tl.arange(0, BT)
        beta = tl.load(Beta + group * BT + row)
        product = tl.zeros((BT, BT), tl.float32)
        for block in range(tl.cdiv(DK, BK)):
            channel = block * BK + tl.arange(0, BK)
            k = tl.load(K + group * BT * DK + row[:, None] * DK + channel[None, :], channel[None, :] < DK, 0)
            rhs = tl.load(KRight + group * BT * DK + row[None, :] * DK + channel[:, None], channel[:, None] < DK, 0)
            scaled = (k.to(tl.float32) * beta[:, None]).to(k.dtype)
            product = tl.dot(scaled, rhs, product)
        value = tl.where(row[:, None] > row[None, :], product, 0.0)
        tl.store(Out + group * BT * BT + row[:, None] * BT + row[None, :], value)

    @triton.jit
    def wave001_kkt_scale_after(K, KRight, Beta, Out, BT: tl.constexpr, DK: tl.constexpr, BK: tl.constexpr):
        group = tl.program_id(0)
        row = tl.arange(0, BT)
        beta = tl.load(Beta + group * BT + row)
        product = tl.zeros((BT, BT), tl.float32)
        for block in range(tl.cdiv(DK, BK)):
            channel = block * BK + tl.arange(0, BK)
            k = tl.load(K + group * BT * DK + row[:, None] * DK + channel[None, :], channel[None, :] < DK, 0)
            rhs = tl.load(KRight + group * BT * DK + row[None, :] * DK + channel[:, None], channel[:, None] < DK, 0)
            product = tl.dot(k, rhs, product)
        value = tl.where(row[:, None] > row[None, :], product * beta[:, None], 0.0)
        tl.store(Out + group * BT * BT + row[:, None] * BT + row[None, :], value)

    return wave001_kkt_scale_before, wave001_kkt_scale_after


def compiled_identity(compiled):
    """Hash emitted compiler artifacts without executing or guessing names."""
    result = {}
    for key, value in getattr(compiled, 'asm', {}).items():
        if isinstance(value, str):
            value = value.encode()
        if isinstance(value, (bytes, bytearray)):
            result[key] = hashlib.sha256(value).hexdigest()
    return dict(hash=getattr(compiled, 'hash', None), artifacts=result)


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        import torch_npu  # Registers NPU backend before moving inputs.
        import triton

        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        groups, bt, dk, _ = dimensions(spec)
        k, beta = fixture['k'], fixture['beta']
        if k.dtype != torch.bfloat16 or beta.dtype != torch.float32 or not k.is_contiguous() or not beta.is_contiguous():
            raise ValueError('contiguous BF16 K and FP32 beta required')
        if tuple(k.shape) != (groups, bt, dk) or tuple(beta.shape) != (groups, bt):
            raise ValueError('fixture shape mismatch')
        if not torch.isfinite(k).all() or not torch.isfinite(beta).all() or not ((beta >= 0) & (beta <= 1)).all():
            raise ValueError('finite inputs and beta in [0,1] required')
        device_k, device_beta = k.npu(), beta.npu()
        out = torch.empty((groups, bt, bt), dtype=torch.float32, device=device_k.device)
        kernel = kernels()[0 if arm == 'before' else 1]
        compiled = {}

        def run():
            # The two read-only formals alias the same input, without a copy.
            # Keep RHS's Cube load distinct from the vector-scaled left load.
            artifact = kernel[(groups,)](device_k, device_k, device_beta, out, bt, dk, max(32, min(64, triton.next_power_of_2(dk))))
            # Only compiler object identities are recorded; tensors are untouched.
            compiled[id(artifact)] = artifact
            return out

        def compilation_state():
            return [compiled_identity(item) for item in compiled.values()]

        prepared = SimpleNamespace(run=run, expected_kernels=[kernel.fn.__name__],
                                   sources=[Path(__file__), *CASE_SOURCES],
                                   metadata=dict(language='triton', triton_version=triton.__version__,
                                                 torch_version=torch.__version__, arm=arm,
                                                 groups=groups, bt=bt, dk=dk, num_autotune_trials=0,
                                                 rhs_input_alias='same K pointer through separate read-only formal',
                                                 compiler_artifacts='compilation_state snapshots'),
                                   compilation_state=compilation_state,
                                   inputs=(device_k, device_beta))
        self.prepared[arm] = prepared
        return prepared

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual.detach().cpu(), fixture)
        for device_input, key in zip(self.prepared[arm].inputs, ('k', 'beta')):
            if not torch.equal(device_input.cpu(), fixture[key]):
                result.update(passed=False, reason='input mutation: ' + key)
        return result


def factory():
    return Adapter()
