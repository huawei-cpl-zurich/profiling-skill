"""Compiler-noop scaled-MM screen; independent numeric execution is disabled."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from scaled_index_contract import CASE_SOURCES,dimensions,strides,dispatch_audit,make_case,check_fixture


def identity(artifact):
    return dict(hash=artifact.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in artifact.asm.items() if isinstance(v,(str,bytes))})


UNQUALIFIED = ('T10 compiler-noop screen: no independent numeric gate ran; '
               'public execution and capture are disabled. See experiments/optimization-wave008/T10-result.json.')


class StudyUnqualified(ValueError):
    """Compiled equality is not an independent numerical correctness gate."""


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self, *, compile_inspection=False):
        self.prepared={};self.compile_inspection=compile_inspection

    def capture(self,*args,**kwargs):raise StudyUnqualified(UNQUALIFIED)

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        if not self.compile_inspection:raise StudyUnqualified(UNQUALIFIED)
        m,n,k,_,p=dimensions(spec)
        import torch,torch_npu
        ast,bst,_=strides(spec)
        data=dict(a=torch.empty_strided((m,k),ast,dtype=torch.int8,device='npu:0'),
                  b=torch.empty_strided((k,n),bst,dtype=torch.int8,device='npu:0'),
                  sa=torch.empty((1,),device='npu:0'),sb=torch.empty((1,),device='npu:0'),
                  bias=torch.empty((n,),device='npu:0') if p.get('bias',True) else None)
        return self._prepare(arm,spec,data,probe=True)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        if not self.compile_inspection:raise StudyUnqualified(UNQUALIFIED)
        check_fixture(spec,fixture)
        import torch,torch_npu
        data={k:(None if v is None else v.to(device='npu:0',memory_format=torch.preserve_format)) for k,v in fixture.items() if k!='expected'}
        return self._prepare(arm,spec,data)

    def _prepare(self,arm,spec,data,probe=False):
        import torch,triton
        import triton.language as tl
        from scaled_index_kernel import scaled_mm_kernel
        m,n,k,_,_=dimensions(spec);ast,bst,cst=strides(spec)
        if tuple(data['a'].stride())!=ast or tuple(data['b'].stride())!=bst:raise ValueError('device transfer/prototype changed physical strides')
        audit=dispatch_audit(spec);bits=64 if arm=='before' else audit['candidate_bits']
        store=torch.empty((m*n+2,),dtype=torch.float32,device=data['a'].device) if probe else torch.full((m*n+2,),-76543.,dtype=torch.float32,device=data['a'].device)
        result=store[1:-1].view(m,n)
        args=(data['a'],data['b'],data['sa'],data['sb'],result,data['bias'],m,n,k,*ast,*bst,*cst)
        kw=dict(INDEX_DTYPE=tl.int32 if bits==32 else tl.int64,ACCUMULATOR_DTYPE=tl.int32,
                BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=128,BLOCK_SIZE_SCALE_A=1,BLOCK_SIZE_SCALE_B=1,
                num_stages=1,multibuffer=False,enable_fp_fusion=False)
        grid=(triton.cdiv(m,16)*triton.cdiv(n,64),);compiled={};qualified=[False]
        def compile_only():
            if not qualified[0]:
                # Commit qualification only after all compiler/artifact checks succeed.
                artifact=scaled_mm_kernel.warmup(*args,grid=grid,**kw)
                if 'npubin' not in artifact.asm:raise RuntimeError('linked NPU binary required')
                staged={artifact.hash:artifact}
                compiled.update(staged);qualified[0]=True
            return list(compiled.values())
        def run():
            # Even explicit compilation inspection never authorizes execution.
            raise StudyUnqualified(UNQUALIFIED)
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[identity(x) for x in compiled.values()],
            expected_kernels=['scaled_mm_kernel'],sources=[Path(__file__),Path(__file__).with_name('scaled_index_kernel.py'),*CASE_SOURCES],
            metadata=dict(execution_enabled=False,arm=arm,role=spec.get('role'),parameters=spec.get('parameters',spec),index_bits=bits,
                index_audit=audit,physical_strides=dict(a=list(ast),b=list(bst),out=list(cst)),scope='direct prepared scaled-MM component',
                prepared_output_reuse=True,allocator_excluded=True,prototype=probe,
                fixture_defaults_sha256=hashlib.sha256(CASE_SOURCES[-1].read_bytes()).hexdigest()),
            inputs=data,store=store,result=result,spec=spec)
        if not probe:self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]
        if item.spec!=spec:raise ValueError('stale prepared case')
        host=actual.detach().cpu();expected=fixture['expected']
        numerical=host.dtype==torch.float32 and host.shape==expected.shape and torch.equal(host.view(torch.int32),expected.view(torch.int32))
        immutable=all(v is None or torch.equal(v.cpu(),fixture[k]) for k,v in item.inputs.items())
        layouts=tuple(item.inputs['a'].stride())==tuple(fixture['a'].stride()) and tuple(item.inputs['b'].stride())==tuple(fixture['b'].stride())
        canary=bool((item.store[[0,-1]].cpu()==-76543).all())
        nonalias=actual.untyped_storage().data_ptr() not in [x.untyped_storage().data_ptr() for x in item.inputs.values() if x is not None]
        return dict(passed=bool(numerical and immutable and layouts and canary and nonalias),bitwise_all_outputs=bool(numerical),
                    immutable_inputs=immutable,physical_layout_preserved=layouts,canaries=canary,nonalias=nonalias,
                    selected_index_bits=item.metadata['index_bits'],ga_manual_scope=item.metadata['index_audit']['wide_guard_scope'])


def factory():raise StudyUnqualified(UNQUALIFIED)


def main():
    """Disabled compiler-noop CLI; independent numeric qualification is unavailable."""
    import argparse
    parser=argparse.ArgumentParser(description=main.__doc__)
    for name in ('case','output','cache','provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--correctness-only',action='store_true');args=parser.parse_args()
    factory()  # Before fixture/provenance loading, cache creation or NPU imports.



if __name__=='__main__':raise SystemExit(main())
