"""W08: retain the identical staged A path, omit unrequested V work."""
import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from eigenvalues_contract import CASE_SOURCES, diagnostics, make_case, parameters, validate_input
from jacobi_contract import round_pairs
import jacobi_round


@lru_cache(maxsize=1)
def kernels():
    global triton, tl, parent_copy, parent_angles, parent_stage
    import triton
    import triton.language as tl
    parent_copy, parent_angles, _, parent_stage = jacobi_round.kernels()

    @triton.jit
    def eigen_copy(A, Out, N:tl.constexpr, B:tl.constexpr):
        batch = tl.program_id(1)
        parent_copy(A+batch*N*N,Out+batch*N*N,N,B)

    @triton.jit
    def eigen_angles(A, Mate, Sign, C, S, N:tl.constexpr, B:tl.constexpr):
        batch = tl.program_id(1)
        parent_angles(A+batch*N*N,Mate,Sign,C+batch*N,S+batch*N,N,B)

    @triton.jit
    def eigen_stage(A, Out, Mate, C, S, N:tl.constexpr, B:tl.constexpr, LEFT:tl.constexpr):
        batch = tl.program_id(1)
        parent_stage(A+batch*N*N,Out+batch*N*N,Mate,C+batch*N,S+batch*N,N,B,LEFT)

    @triton.jit
    def eigen_identity(V, N:tl.constexpr, B:tl.constexpr):
        x = tl.program_id(0)*B+tl.arange(0,B)
        tl.store(V+tl.program_id(1)*N*N+x,tl.cast(x//N==x%N,tl.float32),x<N*N)

    @triton.jit
    def eigen_sort(A, V, Values, SortedV, N:tl.constexpr, K:tl.constexpr, VECTORS:tl.constexpr):
        # One diagonal index per program bounds UB while preserving stable ranks.
        index, batch = tl.program_id(0), tl.program_id(1)
        x = tl.arange(0,K)
        d = tl.load(A+batch*N*N+index*N+index)
        other = tl.load(A+batch*N*N+x*N+x,x<N,float('inf'))
        rank = tl.sum(tl.cast((x<N)&((other<d)|((other==d)&(x<index))),tl.int32),0)
        tl.store(Values+batch*N+rank,d)
        if VECTORS:
            v = tl.load(V+batch*N*N+x*N+index,x<N,0)
            tl.store(SortedV+batch*N*N+x*N+rank,v,x<N)

    return eigen_copy,eigen_angles,eigen_stage,eigen_identity,eigen_sort


class Solver:
    """Five full round-robin sweeps. Metadata is cached; numerical work is not."""
    validate_input = staticmethod(validate_input)

    def __init__(self, arm, spec, fixture):
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        self.batch,self.n,self.vectors = self.validate_input(spec,fixture)
        import torch
        import torch_npu
        self.track = arm=='before' or self.vectors
        self.a = fixture['a'].npu()
        self.work = torch.empty_like(self.a)
        self.scratch = torch.empty_like(self.a)
        self.values = torch.full((self.batch,self.n),float('nan'),dtype=torch.float32,device='npu')
        self.v = torch.empty_like(self.a) if self.track else None
        self.vtmp = torch.empty_like(self.a) if self.track else None
        self.sorted_v = torch.full_like(self.a,float('nan')) if self.vectors else None
        self.c = torch.empty((self.batch,self.n),dtype=torch.float32,device='npu')
        self.s = torch.empty_like(self.c)
        self.maps = []
        for r in range(self.n-1):
            mate,sign = list(range(self.n)),[0]*self.n
            for p,q in round_pairs(self.n,r):
                mate[p],mate[q],sign[p],sign[q] = q,p,-1,1
            self.maps.append((torch.tensor(mate,dtype=torch.int32,device='npu'),
                              torch.tensor(sign,dtype=torch.int32,device='npu')))
        self.k = kernels()
        self.artifacts = {}
        self.options = dict(num_warps=4,num_stages=1,multibuffer=False,enable_fp_fusion=False)

    def calls(self, representative=False):
        n,b,batch = self.n,128,self.batch
        copy,angles,stage,identity,sort = self.k
        matrix_grid = ((n*n+b-1)//b,batch)
        yield copy,matrix_grid,(self.a,self.work,n,b)
        if self.track:
            yield identity,matrix_grid,(self.v,n,b)
        for _ in range(1 if representative else 5):
            for mate,sign in self.maps[:2] if representative else self.maps:
                yield angles,(1,batch),(self.work,mate,sign,self.c,self.s,n,b)
                yield stage,matrix_grid,(self.work,self.scratch,mate,self.c,self.s,n,b,True)
                yield stage,matrix_grid,(self.scratch,self.work,mate,self.c,self.s,n,b,False)
                if self.track:
                    yield stage,matrix_grid,(self.v,self.vtmp,mate,self.c,self.s,n,b,False)
                    yield copy,matrix_grid,(self.vtmp,self.v,n,b)
        yield sort,(n,batch),(self.work,self.v if self.track else self.work,
            self.values,self.sorted_v if self.vectors else self.work,n,1<<(n-1).bit_length(),self.vectors)

    def keep(self, artifact):
        self.artifacts[artifact.hash] = artifact

    def compile_only(self):
        hashes = {}
        for kernel,grid,args in self.calls(representative=True):
            artifact = kernel.warmup(*args,grid=grid,**self.options)
            self.keep(artifact)
            key = (kernel.fn.__name__,tuple(x for x in args if type(x) in (int,bool)))
            hashes.setdefault(key,set()).add(artifact.hash)
        if any(len(v)!=1 for v in hashes.values()):
            raise RuntimeError('round metadata unexpectedly specialized compiled binary')
        return self.compilation_state()

    def run(self):
        for kernel,grid,args in self.calls():
            self.keep(kernel[grid](*args,**self.options))
        return (self.values,self.sorted_v) if self.vectors else self.values

    def compilation_state(self):
        return [dict(hash=a.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
            for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in self.artifacts.values()]


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        solver = Solver(arm,spec,fixture)
        item = SimpleNamespace(run=solver.run,compile_only=solver.compile_only,solver=solver,
            compilation_state=solver.compilation_state,sources=[Path(__file__),Path(jacobi_round.__file__),*CASE_SOURCES],
            expected_kernels=list(dict.fromkeys(k.fn.__name__ for k,_,_ in solver.calls(representative=True))),
            metadata=dict(language='triton',arm=arm,operation='five-full-round-robin-sweeps',
                batch=solver.batch,N=solver.n,sweeps=5,rounds_per_sweep=solver.n-1,
                want_vectors=solver.vectors,accumulate_vectors=solver.track,num_stages=1,
                multibuffer=False,precision='FP32 angles and updates'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = diagnostics(actual,fixture,parameters(spec)[2])
        if not torch.equal(self.prepared[arm].solver.a.cpu(),fixture['a']):
            result.update(passed=False,reason='input mutation')
        return result


def factory():
    return Adapter()
