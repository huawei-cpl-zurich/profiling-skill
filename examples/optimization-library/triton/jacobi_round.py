"""W004 W09: serial pair updates versus one conflict-free staged round."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from jacobi_contract import CASE_SOURCES, diagnostics, make_case, parameters, validate_input


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def jacobi_copy(A, Out, N:tl.constexpr, B:tl.constexpr):
        x = tl.program_id(0)*B + tl.arange(0,B)
        tl.store(Out+x, tl.load(A+x,x<N*N,0), x<N*N)

    @triton.jit
    def jacobi_angles(A, Mate, Sign, C, S, N:tl.constexpr, B:tl.constexpr):
        x = tl.program_id(0)*B + tl.arange(0,B)
        q = tl.load(Mate+x,x<N,0)
        sign = tl.load(Sign+x,x<N,0)
        p = tl.minimum(x,q)
        q = tl.maximum(x,q)
        app = tl.load(A+p*N+p,x<N,0)
        aqq = tl.load(A+q*N+q,x<N,0)
        apq = tl.load(A+p*N+q,x<N,0)
        tau = (aqq-app)/(2*tl.where(apq==0,1.,apq))
        t = tl.where(tau>=0,1.,-1.)/(tl.abs(tau)+tl.sqrt(1+tau*tau))
        t = tl.where((apq==0)|(sign==0),0.,t)
        c = 1./tl.sqrt(1+t*t)
        tl.store(C+x,c,x<N)
        tl.store(S+x,t*c*sign,x<N)

    @triton.jit(do_not_specialize=['pairIndex'])
    def jacobi_serial(A, Pairs, C, S, pairIndex, N:tl.constexpr, B:tl.constexpr, LEFT:tl.constexpr):
        p = tl.load(Pairs+pairIndex*2)
        q = tl.load(Pairs+pairIndex*2+1)
        x = tl.program_id(0)*B+tl.arange(0,B)
        c = tl.load(C+p)
        s = tl.load(S+q)
        if LEFT:
            po, qo = p*N+x, q*N+x
        else:
            po, qo = x*N+p, x*N+q
        a, b = tl.load(A+po,x<N,0), tl.load(A+qo,x<N,0)
        tl.store(A+po,c*a-s*b,x<N)
        tl.store(A+qo,s*a+c*b,x<N)

    @triton.jit
    def jacobi_stage(A, Out, Mate, C, S, N:tl.constexpr, B:tl.constexpr, LEFT:tl.constexpr):
        x = tl.program_id(0)*B+tl.arange(0,B)
        row, col = x//N, x%N
        axis = row if LEFT else col
        mate = tl.load(Mate+axis,x<N*N,0)
        c, s = tl.load(C+axis,x<N*N,0), tl.load(S+axis,x<N*N,0)
        other = mate*N+col if LEFT else row*N+mate
        a, b = tl.load(A+x,x<N*N,0), tl.load(A+other,x<N*N,0)
        tl.store(Out+x,c*a+s*b,x<N*N)

    return jacobi_copy, jacobi_angles, jacobi_serial, jacobi_stage


class Stage:
    """Reusable single-stage primitive; preparation never executes a kernel."""
    def __init__(self, arm, spec, fixture):
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        pairs, mate, sign = validate_input(spec, fixture)
        import torch
        import torch_npu
        self.n, _ = parameters(spec)
        self.arm, self.pairs = arm, pairs
        self.a = fixture['a'].npu()
        self.out = torch.full_like(self.a, float('nan'))
        self.scratch = torch.empty_like(self.a)
        self.mate = torch.tensor(mate,dtype=torch.int32,device='npu')
        self.sign = torch.tensor(sign,dtype=torch.int32,device='npu')
        self.table = torch.tensor(pairs,dtype=torch.int32,device='npu')
        self.c = torch.empty(self.n,dtype=torch.float32,device='npu')
        self.s = torch.empty_like(self.c)
        self.k = kernels()
        self.artifacts = {}
        self.options = dict(num_warps=4,num_stages=1,multibuffer=False,enable_fp_fusion=False)

    def calls(self):
        n, b = self.n, 128
        copy, angles, serial, stage = self.k
        yield angles, ((n+b-1)//b,), (self.a,self.mate,self.sign,self.c,self.s,n,b)
        if self.arm == 'before':
            yield copy, ((n*n+b-1)//b,), (self.a,self.out,n,b)
            for index in range(len(self.pairs)):
                for left in (True,False):
                    yield serial, ((n+b-1)//b,), (self.out,self.table,self.c,self.s,index,n,b,left)
        else:
            yield stage, ((n*n+b-1)//b,), (self.a,self.scratch,self.mate,self.c,self.s,n,b,True)
            yield stage, ((n*n+b-1)//b,), (self.scratch,self.out,self.mate,self.c,self.s,n,b,False)

    def keep(self, artifact):
        self.artifacts[artifact.hash] = artifact

    def compile_only(self):
        # Compile every shape/side and representative runtime indices 0 and 1.
        seen = set()
        serial_hashes = {}
        for kernel, grid, args in self.calls():
            if kernel is self.k[2]:
                if args[4] not in (0,1):
                    continue
            key = (kernel.fn.__name__, tuple(x for x in args if type(x) in (int,bool)))
            if key in seen:
                continue
            seen.add(key)
            artifact = kernel.warmup(*args,grid=grid,**self.options)
            self.keep(artifact)
            if kernel is self.k[2]:
                serial_hashes.setdefault(args[-1], []).append(artifact.hash)
        if any(len(set(values)) != 1 for values in serial_hashes.values()):
            raise RuntimeError('pairIndex unexpectedly specialized the kernel')
        return self.compilation_state()

    def run(self):
        for kernel, grid, args in self.calls():
            self.keep(kernel[grid](*args,**self.options))
        return self.out

    def compilation_state(self):
        return [dict(hash=getattr(a,'hash',None),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in getattr(a,'asm',{}).items() if isinstance(v,(str,bytes))}) for a in self.artifacts.values()]


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        stage = Stage(arm,spec,fixture)
        item = SimpleNamespace(run=stage.run,compile_only=stage.compile_only,stage=stage,
                compilation_state=stage.compilation_state,sources=[Path(__file__),*CASE_SOURCES],
                expected_kernels=list(dict.fromkeys(k.fn.__name__ for k,_,_ in stage.calls())),
                metadata=dict(language='triton',arm=arm,operation='one-prescribed-round',block_size=1,
                              num_stages=1,multibuffer=False,precision='FP32 angles and updates'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual,fixture)
        if not torch.equal(self.prepared[arm].stage.a.cpu(),fixture['a']):
            result.update(passed=False,reason='input mutation')
        return result


def factory():
    return Adapter()
