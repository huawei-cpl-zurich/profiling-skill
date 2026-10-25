"""Prepared zero-limit clipped forward; public CLI is exploratory, not frozen collection."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import torch
from clipped_zero_contract import check_spec,check_input,make_case,validate_actual
from clipped_zero_tiles import _pick_block_sizes
from clipped_zero_screen import elf_state


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
            for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('clipped_zero_contract.py')]
    make_case=staticmethod(make_case)

    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        """Input-only compile API: no expected payload, no oracle and no run."""
        p=check_spec(spec)
        item=self._prepare(arm,spec,dict(gup=torch.zeros(p['M'],2*p['K'],dtype=torch.float32)))
        def reject_run():
            raise RuntimeError('compile-only probe cannot execute numerical work')
        item.run=reject_run
        item.metadata.update(compile_only_probe=True,device_kernel_calls=0)
        return item

    def prepare(self,arm,spec,fixture,output):
        p=check_spec(spec);e=fixture.get('expected')
        if not isinstance(e,torch.Tensor) or e.shape!=(p['M'],p['K']) or e.dtype!=torch.float32:
            raise ValueError('FP32 expected output required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu_inputs):
        p=check_spec(spec)
        if arm not in ('before','after'):raise ValueError('paired arm required')
        if set(cpu_inputs)!={'gup'}:raise ValueError('exactly one GUP input required')
        check_input(cpu_inputs['gup'],p,'cpu')
        import torch_npu
        from clipped_zero_kernels import _clipped_swiglu_kernel,_zero_limit_swiglu_kernel
        inputs={n:v.to('npu:0') for n,v in cpu_inputs.items()};check_input(inputs['gup'],p,'npu')
        if arm in self.state:self.state.clear();self.actual.clear()
        m,k=p['M'],p['K'];bm,bk=_pick_block_sizes(m,k,torch.float32)
        buffer=torch.full((m*k+64,),12345.,dtype=torch.float32).to('npu:0');out=buffer[32:-32].view(m,k)
        self.state[arm]=(inputs['gup'],buffer)
        kernel=_clipped_swiglu_kernel if arm=='before' else _zero_limit_swiglu_kernel
        args=(inputs['gup'],out,m,k,1.702,p.get('limit',0))
        kwargs=dict(BLOCK_M=bm,BLOCK_K=bk,LARGE_TENSOR=False,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        grid=((m+bm-1)//bm,(k+bk-1)//bk);compiled=[];qualification={}
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,**kwargs,grid=grid)
                elf=elf_state(c.asm.get('npubin'),c.name)
                metadata=getattr(c.metadata,'_asdict',lambda:None)()
                signature=getattr(getattr(c,'src',None),'signature',None)
                abi=dict(known=metadata is not None and signature is not None)
                if abi['known']:
                    abi.update(signature={str(k):str(v) for k,v in signature.items()},metadata={k:repr(v) for k,v in metadata.items() if k not in ('hash','name','kernel_name')})
                qualification.update(elf=elf,abi=abi)
                compiled.append(c)  # Commit only after all qualification succeeds; failed attempts leave no cached success.
            return compiled
        def run():
            compile_only();kernel[grid](*args,**kwargs);return out
        sources=[__file__,self.case_sources[0],*[Path(__file__).with_name(n) for n in ('clipped_zero_kernels.py','clipped_zero_tiles.py','clipped_zero_screen.py')]]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),qualification=lambda:dict(qualification),sources=sources,
            expected_kernels=[kernel.fn.__name__],metadata=dict(component='clipped forward zero-limit only',
            configuration=dict(BLOCK_M=bm,BLOCK_K=bk,limit=p.get('limit',0)),grid=grid,device_kernel_calls=1,reused_output=True,
            workspace_scope='prepared output; compiler-managed scratch retained by JIT',
            allocator_limit='no allocation or complete model latency claim',input_strides={n:list(v.stride()) for n,v in inputs.items()}))

    def validate(self,arm,actual,fixture,spec):
        inputs,buffer=self.state[arm];result=validate_actual(actual,fixture,inputs,buffer)
        self.actual[arm]=actual.detach().cpu()
        other='after' if arm=='before' else 'before'
        if other in self.actual:
            result['paired_exact_bits']=bool(torch.equal(self.actual[arm].view(torch.int32),self.actual[other].view(torch.int32)))
            result['passed'] &= result['paired_exact_bits']
            result['paired_max_absolute_error']=float((self.actual[arm]-self.actual[other]).abs().max())
        return result


def main():
    import argparse,json,sys
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case','output','cache','provenance'):p.add_argument('--'+key,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args()
    spec=json.loads(Path(a.case).read_text());check_spec(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    provenance=json.loads(Path(a.provenance).read_text());provenance['qualification']='exploratory CLI, not frozen all-case collector'
    report=run_pair(Adapter(),spec,a.output,a.cache,provenance,correctness_only=a.correctness_only)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
