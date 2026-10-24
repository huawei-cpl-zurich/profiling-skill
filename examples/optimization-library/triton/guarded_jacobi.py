"""SOURCE-DOCUMENTED independent Jacobi; upstream device body was unseen."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from guarded_jacobi_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, diagnostics


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave004_jacobi_certificate(DL,D,DU,B,C,N:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.arange(0,BLOCK); mask=i<N
        dl=tl.load(DL+i,mask,0.); d=tl.load(D+i,mask,1.); du=tl.load(DU+i,mask,0.); b=tl.load(B+i,mask,0.)
        ad=tl.abs(d); off=tl.abs(dl)+tl.abs(du); ab=tl.abs(b)
        # Comparisons reject NaN and infinity as well as the strict boundary.
        valid=(ad>2.*off)&(ad>=2.**-24)&(ad<=2.**24)&(ab<=2.**24)&(off<=2.**23)
        mn=tl.min(tl.where(mask,ad,float('inf')),0); mx=tl.max(ab,0)
        ok=(tl.min(tl.where(mask,valid,True).to(tl.int32),0)==1)&(mx/mn<=4.)
        tl.store(C,ok & (N>=128) & (N<=6144))

    @triton.jit
    def wave004_jacobi_reset(B,X,Y,N:tl.constexpr,LDB:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        b=tl.load(B+i,i<LDB,0.)
        tl.store(X+i,tl.where(i<N,0.,b),i<LDB)
        tl.store(Y+i,tl.where(i<N,0.,b),i<LDB)

    @triton.jit
    def wave004_jacobi_step(DL,D,DU,B,X,Y,C,N:tl.constexpr,BLOCK:tl.constexpr):
        if tl.load(C)!=0:
            i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); mask=i<N
            left=tl.load(X+i-1,(i>0)&mask,0.); right=tl.load(X+i+1,i+1<N,0.)
            value=((tl.load(B+i,mask,0.)-tl.load(DL+i,mask,0.)*left)-tl.load(DU+i,mask,0.)*right)/tl.load(D+i,mask,1.)
            tl.store(Y+i,value,mask)

    @triton.jit
    def wave004_jacobi_thomas(DL,D,DU,B,X,CP,C,N:tl.constexpr,LDB:tl.constexpr,GUARDED:tl.constexpr):
        execute=True
        if GUARDED: execute=tl.load(C)==0
        if execute:
            previous_cp=tl.full((),0.,tl.float32); previous_y=tl.full((),0.,tl.float32)
            i=0
            while i<N:
                inv=1./(tl.load(D+i)-tl.load(DL+i)*previous_cp)
                cp=tl.load(DU+i)*inv;tl.store(CP+i,cp);previous_cp=cp
                y=(tl.load(B+i)-tl.load(DL+i)*previous_y)*inv
                tl.store(X+i,y);previous_y=y;i+=1
            following=tl.full((),0.,tl.float32);i=N-1
            while i>=0:
                value=tl.load(X+i)-tl.load(CP+i)*following
                tl.store(X+i,value);following=value;i-=1
    return wave004_jacobi_certificate,wave004_jacobi_reset,wave004_jacobi_step,wave004_jacobi_thomas


class Operation:
    def __init__(self, inputs, spec, arm):
        import torch
        m,r,ldb=dimensions(spec);self.inputs=inputs
        self.output=torch.empty_like(inputs['b'])
        y=torch.empty_like(self.output);cp=torch.empty(m,device='npu',dtype=torch.float32)
        cert=torch.empty((),device='npu',dtype=torch.int32)
        self.workspace=(y,cp,cert)
        certificate,reset,step,solve=kernels();args=tuple(inputs[k] for k in ('dl','d','du','b'))
        self.calls=[]
        if arm=='after':self.calls.append((certificate,(1,),(*args,cert,m,triton.next_power_of_2(m))))
        self.calls.append((reset,(triton.cdiv(ldb,256),),(inputs['b'],self.output,y,m,ldb,256)))
        if arm=='after':
            old,new=self.output,y
            for _ in range(40):
                self.calls.append((step,(triton.cdiv(m,256),),(*args,old,new,cert,m,256)))
                old,new=new,old
        self.calls.append((solve,(1,),(*args,self.output,cp,cert,m,ldb,arm=='after')))
        self.options=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        self.compiled=[fn.warmup(*a,grid=g,**self.options) for fn,g,a in self.calls]

    def compilation_state(self):
        return [dict(hash=k.hash,binaries={n:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
            for n,v in k.asm.items() if isinstance(v,(str,bytes))}) for k in self.compiled]

    def run(self):
        for fn,g,a in self.calls:fn[g](*a,**self.options)
        return self.output


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)
    def __init__(self):self.prepared={}
    def prepare(self,arm,spec,fixture,output):
        import torch_npu
        if arm not in ('before','after'):raise ValueError('unknown arm')
        validate_inputs(fixture['inputs'],spec)
        op=Operation({k:v.npu() for k,v in fixture['inputs'].items()},spec,arm);self.prepared[arm]=op
        artifacts=[]
        for i,k in enumerate(op.compiled):
            for name,value in k.asm.items():
                if isinstance(value,(str,bytes)):
                    p=Path(output)/f'kernel-{i}.{name}';p.write_bytes(value.encode() if isinstance(value,str) else value);artifacts.append(p)
        return SimpleNamespace(run=op.run,compilation_state=op.compilation_state,
            sources=[Path(__file__),*CASE_SOURCES,*artifacts],expected_kernels=sorted(set(fn.fn.__name__ for fn,_,_ in op.calls)),
            metadata=dict(arm=arm,host_fallback=False,npu_calls_per_forward=len(op.calls),
                certificate='GPU current A/b every API; device conditional, no host synchronization',
                iterations=40 if arm=='after' else 0,compiler_options=op.options,
                workspace_bytes=sum(x.numel()*x.element_size() for x in op.workspace)))
    def validate(self,arm,actual,fixture,spec):
        import torch
        result=diagnostics(actual,fixture)
        for k,v in self.prepared[arm].inputs.items():
            if not torch.equal(v.cpu(),fixture['inputs'][k]):result.update(passed=False,reason='input mutated: '+k)
        return result


def factory():return Adapter()
