"""F09: separate launches versus deterministic resident grouped tile ownership.

Independent implementation of the cited grouped-M scheduling mechanism, using
an identical FP16/FP32 Cube tile and ascending K traversal in both arms.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from resident_gemm_contract import check_spec, check_prefix, make_case, validate_actual


@lru_cache(maxsize=1)
def kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def resident_descriptors(MS, D, G: tl.constexpr, BLOCK: tl.constexpr):
        g = tl.arange(0, BLOCK)
        m = tl.load(MS + g, g < G, other=0)
        start = tl.cumsum(m) - m
        tl.store(D + g * 2, start, g < G)
        tl.store(D + g * 2 + 1, m, g < G)

    @triton.jit
    def resident_group_gemm(A, B, C, D, N: tl.constexpr, K: tl.constexpr,
                            FIRST: tl.constexpr, GROUPS: tl.constexpr,
                            STRIDE: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr, BK: tl.constexpr):
        task = tl.program_id(0)
        base = 0
        cols = tl.cdiv(N, BN)
        for g in range(FIRST, FIRST + GROUPS):
            row = tl.load(D + g * 2)
            m = tl.load(D + g * 2 + 1)
            count = tl.cdiv(m, BM) * cols
            while task < base + count:
                tile = task - base
                rm = (tile // cols) * BM + tl.arange(0, BM)
                rn = (tile % cols) * BN + tl.arange(0, BN)
                rk = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for stage in range(tl.cdiv(K, BK)):
                    kk = stage * BK + rk
                    a = tl.load(A + (row + rm[:, None]) * K + kk[None, :],
                                (rm[:, None] < m) & (kk[None, :] < K), other=0)
                    b = tl.load(B + g * K * N + kk[:, None] * N + rn[None, :],
                                (kk[:, None] < K) & (rn[None, :] < N), other=0)
                    acc = tl.dot(a, b, acc)
                tl.store(C + (row + rm[:, None]) * N + rn[None, :], acc,
                         (rm[:, None] < m) & (rn[None, :] < N))
                task += STRIDE
            # Zero-M consumes no tiles but g still advances B's descriptor.
            base += count
    return resident_descriptors, resident_group_gemm


def identity(compiled):
    return [dict(name=x.name, hash=str(x.hash), binaries={k: hashlib.sha256(v).hexdigest()
            for k, v in x.asm.items() if isinstance(v, bytes)}) for x in compiled.values()]


class Adapter:
    case_sources = [Path(__file__).with_name('resident_gemm_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.inputs = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        import triton
        if arm not in ('before', 'after'):
            raise ValueError('unknown arm')
        p = check_spec(spec)
        ms, n, k = p['Ms'], p['N'], p['K']
        reject = spec.get('invalid_prefix') is not None
        if reject and spec.get('role') != 'guard':
            raise ValueError('invalid descriptor is correctness-only')
        desc, gemm = kernels()
        inputs = {} if reject else {key: value.to('npu:0') for key, value in fixture['inputs'].items()}
        self.inputs[arm] = inputs
        result = None if reject else torch.full_like(fixture['expected'], float('nan'), device='npu:0')
        descriptors = None if reject else torch.empty((len(ms), 2), device='npu:0', dtype=torch.int32)
        compiled = {}
        launches = ([(0, len(ms), 20)] if arm == 'after' else
                    [(g, 1, ((m + 15) // 16) * ((n + 63) // 64)) for g, m in enumerate(ms) if m])

        def execute(compile_only=False):
            if reject:
                try:
                    check_prefix(spec['invalid_prefix'], len(ms), sum(ms))
                except ValueError:
                    return True
                raise ValueError('invalid prefix unexpectedly accepted')
            desc_args = [inputs['ms'], descriptors, len(ms), triton.next_power_of_2(len(ms))]
            def invoke(fn, grid, args, key):
                compiled[key] = fn.warmup(*args, grid=grid) if compile_only else fn[grid](*args)
            invoke(desc, (1,), desc_args, 'descriptor')
            for group, groups, grid in launches:
                args = [inputs['a'], inputs['b'], result, descriptors, n, k, group, groups, grid, 16, 64, 64]
                invoke(gemm, (grid,), args, 'group-' + str(group))
            return result

        def compile_only():
            execute(True)
            output.mkdir(parents=True, exist_ok=True)
            for key, item in compiled.items():
                for name, value in item.asm.items():
                    if isinstance(value, (str, bytes)):
                        (output / (key + '.' + name)).write_bytes(value.encode() if isinstance(value, str) else value)
            return identity(compiled)

        return SimpleNamespace(run=execute, compile_only=compile_only,
            expected_kernels=[desc.fn.__name__, gemm.fn.__name__],
            sources=[__file__, self.case_sources[0]], compilation_state=lambda: identity(compiled),
            metadata=dict(backend='triton', tile=[16, 64, 64], accumulator='fp32', input_bound=.125,
                resident_cores=20 if arm == 'after' else None, descriptor_bytes=len(ms)*8,
                device_launches_per_forward=0 if reject else 1 + len(launches),
                gemm_launches_per_forward=0 if reject else len(launches),
                dispatch='reject-prefix' if reject else 'resident' if arm == 'after' else 'per-group',
                allocation_boundary='raw FP16 A/B, raw group Ms uploaded; output and descriptor buffer preallocated; '
                'prefix descriptor construction and every group tile inside API forward',
                torch=torch.__version__, triton=triton.__version__))

    def validate(self, arm, actual, fixture, spec):
        if spec.get('invalid_prefix') is not None and actual is not True:
            raise ValueError('invalid prefix was not rejected')
        return validate_actual(actual, fixture, self.inputs[arm])
