"""W03 independently derived half-length real FFT reconstruction.

The native host N/2 plan and A/B coefficient formula motivate this example.
No unseen native device helper is copied or claimed executed. Complex kernels
are reused from staged_fft; N=1024 is qualified only through this RFFT API.
"""
from pathlib import Path
from types import SimpleNamespace
import staged_fft
from fft_contract import coefficients
from rfft_contract import CASE_SOURCES, diagnostics, dimensions, make_case, validate_inputs


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_real_pack(X, R, I, N:tl.constexpr, K:tl.constexpr):
        row = tl.program_id(0)
        j = tl.program_id(1)*128 + tl.arange(0, 128)
        tl.store(R + row*K+j, tl.load(X+row*N+2*j, j<K, 0), j<K)
        tl.store(I + row*K+j, tl.load(X+row*N+2*j+1, j<K, 0), j<K)

    @triton.jit
    def wave004_real_extract(R, I, OR, OI, N:tl.constexpr, H:tl.constexpr):
        row = tl.program_id(0)
        k = tl.program_id(1)*128 + tl.arange(0,128)
        r = tl.load(R+row*N+k, k<H, 0)
        i = tl.load(I+row*N+k, k<H, 0)
        i = tl.where((k==0) | ((N%2==0) & (k==N//2)), 0., i)
        tl.store(OR+row*H+k, r, k<H)
        tl.store(OI+row*H+k, i, k<H)

    @triton.jit
    def wave004_real_reconstruct(R, I, C, S, OR, OI, M:tl.constexpr):
        row = tl.program_id(0)
        k = tl.program_id(1)*128 + tl.arange(0,128)
        p = k % M
        q = (M-p) % M
        ar = tl.load(R+row*M+p, k<=M, 0)
        ai = tl.load(I+row*M+p, k<=M, 0)
        br = tl.load(R+row*M+q, k<=M, 0)
        bi = -tl.load(I+row*M+q, k<=M, 0)
        c = tl.load(C+k, k<=M, 0)
        s = tl.load(S+k, k<=M, 0)
        # E=(Z[k]+conj(Z[-k]))/2; O=(Z[k]-conj(Z[-k]))/(2i).
        # X[k]=E+exp(-2*pi*i*k/N)*O, with natural ordered output.
        er = .5*(ar+br); ei = .5*(ai+bi)
        dr = .5*(ar-br); di = .5*(ai-bi)
        real = er+c*di+s*dr
        imag = ei+s*di-c*dr
        imag = tl.where((k==0) | (k==M), 0., imag)
        tl.store(OR+row*(M+1)+k, real, k<=M)
        tl.store(OI+row*(M+1)+k, imag, k<=M)

    return wave004_real_pack, wave004_real_extract, wave004_real_reconstruct


class RealFFT(staged_fft.ComplexFFT):
    """Prepare tables/storage only; run every input-dependent step each call."""
    def __init__(self, real, packed):
        import torch
        if (real.dtype != torch.float32 or real.ndim != 2 or not real.is_contiguous() or real.device.type != 'npu'):
            raise ValueError('contiguous NPU FP32[B,N] required')
        b, n = dimensions(dict(batch=real.shape[0], N=real.shape[1]))
        self.inputs = (real,)
        self.packed = packed and n >= 32 and n & (n-1) == 0
        m = n//2 if self.packed else n
        h = n//2+1
        self.output = tuple(torch.empty((b,h), device=real.device, dtype=real.dtype) for _ in range(2))
        self.tables = tuple(x.to(real.device) for x in coefficients(m))
        r = torch.empty((b,m), device=real.device, dtype=real.dtype) if self.packed else real
        i = torch.empty_like(r) if self.packed else torch.zeros_like(r)
        self.complex_output = tuple(torch.empty_like(r) for _ in range(2))
        direct, first, second = staged_fft.kernels()
        pack, extract, reconstruct = kernels()
        self.factorization = None
        self.workspace = ()
        self.calls = []
        if self.packed:
            self.calls.append((pack,(b,triton.cdiv(m,128)),(real,r,i,n,m)))
        if n >= 64 and n & (n-1) == 0:
            a = 1 << ((m.bit_length()-1)//2)
            radix = m//a
            self.factorization = (a,radix)
            self.workspace = tuple(torch.empty_like(r) for _ in range(2))
            self.calls += [(first,(b,a),(r,i,*self.tables,*self.workspace,a,radix)),
                           (second,(b,radix),(*self.workspace,*self.tables,*self.complex_output,a,radix))]
        else:
            self.calls.append((direct,(b,triton.cdiv(m,8)),(r,i,*self.tables,*self.complex_output,m,triton.next_power_of_2(m))))
        if self.packed:
            self.reconstruction_tables = tuple(x.to(real.device) for x in coefficients(n))
            self.calls.append((reconstruct,(b,triton.cdiv(h,128)),(*self.complex_output,*self.reconstruction_tables,*self.output,m)))
        else:
            self.calls.append((extract,(b,triton.cdiv(h,128)),(*self.complex_output,*self.output,n,h)))
        self.compiled = []


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        validate_inputs(fixture['inputs'], spec)
        op = RealFFT(fixture['inputs']['real'].npu(), packed=arm=='after')
        op.compile()
        artifacts = []
        for index, compiled in enumerate(op.compiled):
            for kind, value in compiled.asm.items():
                if isinstance(value, (str,bytes)):
                    path = Path(output)/f'kernel-{index}.{kind}'
                    path.write_bytes(value.encode() if isinstance(value,str) else value)
                    artifacts.append(path)
        self.prepared[arm] = op
        return SimpleNamespace(run=op.run, compilation_state=op.compilation_state,
            sources=[Path(__file__),Path(staged_fft.__file__),Path(staged_fft.coefficients.__code__.co_filename),*CASE_SOURCES,*artifacts],
            expected_kernels=[fn.fn.__name__ for fn,_,_ in op.calls],
            metadata=dict(language='triton',arm=arm,shape=spec,packed=op.packed,factorization=op.factorization,
                          precision='IEEE FP32, fusion disabled',triton_version=triton.__version__,torch_version=torch.__version__))

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual,fixture)
        if not torch.equal(self.prepared[arm].inputs[0].cpu(),fixture['inputs']['real']):
            result.update(passed=False,reason='input mutated')
        return result


def factory():
    return Adapter()
