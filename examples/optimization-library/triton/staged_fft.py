"""Device-resident pedagogical Cooley-Tukey complex FP32 transform.

Algorithm lead: ops-fft 1878ff9c940d7bbdb592acfff008a1cc98494345 FFT-B,
fft1_d_b.cpp and fft1_d_b_kernel.h (CANN Open Software License 2.0).
This independently written Triton example generalizes small sizes outside the
native FFT-B minimum 256; it does not execute that implementation or wrapper.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from fft_contract import CASE_SOURCES, coefficients, diagnostics, dimensions, factors, make_case, validate_inputs


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_full_dft(R, I, C, S, OR, OI, N:tl.constexpr, K:tl.constexpr):
        batch = tl.program_id(0)
        k = tl.program_id(1) * 8 + tl.arange(0, 8)
        j = tl.arange(0, K)
        r = tl.load(R + batch*N + j, j < N, 0)
        i = tl.load(I + batch*N + j, j < N, 0)
        index = k[:, None]*j[None, :] % N
        c = tl.load(C + index)
        s = tl.load(S + index)
        real = tl.sum(r[None, :]*c - i[None, :]*s, 1)
        imag = tl.sum(r[None, :]*s + i[None, :]*c, 1)
        tl.store(OR + batch*N + k, real, k < N)
        tl.store(OI + batch*N + k, imag, k < N)

    @triton.jit
    def wave004_fft_first(R, I, C, S, TR, TI, A:tl.constexpr, B:tl.constexpr):
        batch = tl.program_id(0)
        n1 = tl.program_id(1)
        k2 = tl.arange(0, B)
        n2 = tl.arange(0, B)
        r = tl.load(R + batch*A*B + n1 + A*n2)
        i = tl.load(I + batch*A*B + n1 + A*n2)
        index = (k2[:, None]*n2[None, :]*A) % (A*B)
        c = tl.load(C + index); s = tl.load(S + index)
        real = tl.sum(r[None, :]*c - i[None, :]*s, 1)
        imag = tl.sum(r[None, :]*s + i[None, :]*c, 1)
        tc = tl.load(C + n1*k2); ts = tl.load(S + n1*k2)
        # Every scratch element is overwritten on every forward.
        tl.store(TR + batch*A*B + n1*B + k2, real*tc - imag*ts)
        tl.store(TI + batch*A*B + n1*B + k2, real*ts + imag*tc)

    @triton.jit
    def wave004_fft_second(TR, TI, C, S, OR, OI, A:tl.constexpr, B:tl.constexpr):
        batch = tl.program_id(0)
        k2 = tl.program_id(1)
        k1 = tl.arange(0, A)
        n1 = tl.arange(0, A)
        r = tl.load(TR + batch*A*B + n1*B + k2)
        i = tl.load(TI + batch*A*B + n1*B + k2)
        index = k1[:, None]*n1[None, :]*B % (A*B)
        c = tl.load(C + index); s = tl.load(S + index)
        real = tl.sum(r[None, :]*c - i[None, :]*s, 1)
        imag = tl.sum(r[None, :]*s + i[None, :]*c, 1)
        # Natural frequency ordering is written inside the API, not cached.
        tl.store(OR + batch*A*B + k2+B*k1, real)
        tl.store(OI + batch*A*B + k2+B*k1, imag)

    return wave004_full_dft, wave004_fft_first, wave004_fft_second


class ComplexFFT:
    """Reusable split-component forward transform; prepare once, run repeatedly.

    Only mathematical tables and allocations occur during preparation. No
    input-dependent transform or permutation occurs outside run().
    """
    def __init__(self, real, imag, staged):
        import torch
        if (real.dtype != torch.float32 or imag.dtype != torch.float32 or real.shape != imag.shape
                or real.ndim != 2 or not real.is_contiguous() or not imag.is_contiguous()
                or real.device != imag.device or real.device.type != 'npu'):
            raise ValueError('matching contiguous NPU FP32[B,N] components required')
        self.batch, self.n = dimensions(dict(batch=real.shape[0], N=real.shape[1]))
        self.inputs = (real, imag)
        self.output = (torch.empty_like(real), torch.empty_like(imag))
        self.tables = tuple(x.to(real.device) for x in coefficients(self.n))
        self.factorization = factors(self.n) if staged else None
        self.workspace = tuple(torch.empty_like(real) for _ in range(2)) if self.factorization else ()
        direct, first, second = kernels()
        b, n = self.batch, self.n
        if self.factorization:
            a, radix = self.factorization
            self.calls = [(first, (b, a), (*self.inputs, *self.tables, *self.workspace, a, radix)),
                          (second, (b, radix), (*self.workspace, *self.tables, *self.output, a, radix))]
        else:
            self.calls = [(direct, (b, triton.cdiv(n, 8)), (*self.inputs, *self.tables, *self.output, n, triton.next_power_of_2(n)))]
        self.compiled = []

    def compile(self):
        # Triton warmup compiles only: never launches a kernel.
        self.compiled = [fn.warmup(*args, grid=grid, enable_fp_fusion=False) for fn, grid, args in self.calls]
        return self.compilation_state()

    def compilation_state(self):
        return [dict(hash=getattr(k, 'hash', None), binaries={name: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
                for name, value in k.asm.items() if isinstance(value, (str, bytes))}) for k in self.compiled]

    def run(self):
        for fn, grid, args in self.calls:
            fn[grid](*args, enable_fp_fusion=False)
        return self.output


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        validate_inputs(fixture['inputs'], spec)
        op = ComplexFFT(*(fixture['inputs'][key].npu() for key in ('real', 'imag')), staged=arm == 'after')
        op.compile()
        artifacts = []
        for index, compiled in enumerate(op.compiled):
            for kind, value in compiled.asm.items():
                if isinstance(value, (str, bytes)):
                    path = Path(output) / f'kernel-{index}.{kind}'
                    path.write_bytes(value.encode() if isinstance(value, str) else value)
                    artifacts.append(path)
        self.prepared[arm] = op
        return SimpleNamespace(run=op.run, compilation_state=op.compilation_state,
            sources=[Path(__file__), *CASE_SOURCES, *artifacts], expected_kernels=[fn.fn.__name__ for fn, _, _ in op.calls],
            metadata=dict(language='triton', arm=arm, shape=spec, precision='IEEE FP32, no dot/TF32, fusion disabled',
                          workspace_bytes=sum(x.numel()*x.element_size() for x in op.workspace),
                          factorization=op.factorization, triton_version=triton.__version__, torch_version=torch.__version__))

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual, fixture)
        for value, key in zip(self.prepared[arm].inputs, ('real', 'imag')):
            if not torch.equal(value.cpu(), fixture['inputs'][key]):
                result.update(passed=False, reason='input mutated')
        return result


def factory():
    return Adapter()
