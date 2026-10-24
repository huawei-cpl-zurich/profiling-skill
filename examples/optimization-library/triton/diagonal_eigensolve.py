"""W10: full device certificate and real one-bit dispatch on every owned API."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import eigenvalues_only
import jacobi_round
from diagonal_contract import CASE_SOURCES, diagnostics, make_case, parameters, validate_input


@lru_cache(maxsize=1)
def certificate_kernel():
    global triton,tl
    import triton
    import triton.language as tl

    @triton.jit
    def diagonal_certificate(A, Flag, N:tl.constexpr, TILE:tl.constexpr):
        # One program owns the reduction, scanning bounded tiles without atomics
        # or a stale flag initialization. All entries are loaded on every call.
        lane=tl.arange(0,TILE)
        valid=tl.full((TILE,),True,tl.int1)
        for start in range(tl.cdiv(N*N,TILE)):
            x=start*TILE+lane
            a=tl.load(A+x,x<N*N,0.)
            finite=(a==a)&(tl.abs(a)!=float('inf'))
            valid=valid&((x>=N*N)|(finite&((x//N==x%N)|(a==0.))))
        tl.store(Flag,tl.min(valid.to(tl.int32),0))
    return diagonal_certificate


class Solver(eigenvalues_only.Solver):
    validate_input=staticmethod(validate_input)

    def __init__(self,arm,spec,fixture):
        # Parent after arm is exactly the five-sweep values-only solver.
        super().__init__('after',spec,fixture)
        import torch
        self.arm=arm
        self.flag=torch.empty((),dtype=torch.int32,device='npu')
        self.certificate=certificate_kernel()
        self.last_dispatch=None

    def compile_only(self):
        result=super().compile_only()
        if self.arm=='after':
            self.keep(self.certificate.warmup(self.a,self.flag,self.n,256,grid=(1,),**self.options))
        return self.compilation_state()

    def run(self):
        if self.arm=='after':
            self.keep(self.certificate[(1,)](self.a,self.flag,self.n,256,**self.options))
            # Mandatory algorithmic D2H and caller gap belong inside this API.
            certified=bool(self.flag.item())
            self.last_dispatch='diagonal-sort' if certified else 'five-sweep-fallback'
            if certified:
                sort=self.k[-1]
                self.keep(sort[(self.n,self.batch)](self.a,self.a,self.values,self.work,
                    self.n,1<<(self.n-1).bit_length(),False,**self.options))
                return self.values
        else:
            self.last_dispatch='five-sweep-baseline'
        return super().run()


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('invalid arm')
        solver=Solver(arm,spec,fixture)
        # Expected selector only. Numerical dispatch always reads the device flag.
        expected_diagonal=spec['parameters']['structure'] in ('diagonal','unsorted-repeated-diagonal')
        baseline=list(dict.fromkeys(k.fn.__name__ for k,_,_ in solver.calls(representative=True)))
        expected=(['diagonal_certificate','eigen_sort'] if expected_diagonal else
                  ['diagonal_certificate',*baseline]) if arm=='after' else baseline
        item=SimpleNamespace(run=solver.run,compile_only=solver.compile_only,solver=solver,
            compilation_state=solver.compilation_state,expected_kernels=expected,
            sources=[Path(__file__),Path(eigenvalues_only.__file__),Path(jacobi_round.__file__),*CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,operation='exact-diagonal-certificate-five-sweeps',
                batch=1,N=solver.n,sweeps=5,want_vectors=False,num_stages=1,multibuffer=False,
                dispatch='actual device finite/exact-zero flag.item() inside API',
                timing_scope='instrumented whole device operation span including dispatch gaps'))
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        solver=self.prepared[arm].solver
        result=diagnostics(actual,fixture,False)
        expected='diagonal-sort' if spec['parameters']['structure'] in ('diagonal','unsorted-repeated-diagonal') else 'five-sweep-fallback'
        result.update(actual_dispatch=solver.last_dispatch)
        if arm=='after' and solver.last_dispatch!=expected:
            result.update(passed=False,reason='wrong actual device certificate dispatch')
        if not torch.equal(solver.a.cpu(),fixture['a']):
            result.update(passed=False,reason='input mutation')
        if spec['parameters']['structure']=='unsorted-repeated-diagonal' and not torch.equal(actual.cpu(),fixture['expected'].float()):
            result.update(passed=False,reason='repeated diagonal sort not exact')
        return result


def factory():return Adapter()
