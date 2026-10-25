"""Prepared given-saved-state L2 backward; public CLI is exploratory, not frozen collection."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import struct
import torch
from l2_backward_contract import check_spec,check_inputs,make_case,validate_actual


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
            for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('l2_backward_contract.py')]
    make_case=staticmethod(make_case)

    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        """Input-only compile API: no expected payload, no oracle and no run."""
        p=check_spec(spec);rows,d=p['rows'],p['D']
        item=self._prepare(arm,spec,dict(y=torch.zeros(rows,d),rstd=torch.ones(rows),dy=torch.zeros(rows,d)))
        def reject_run():
            raise RuntimeError('compile-only probe cannot execute numerical work')
        item.run=reject_run
        item.metadata.update(compile_only_probe=True,device_kernel_calls=0)
        return item

    def prepare(self,arm,spec,fixture,output):
        p=check_spec(spec);e=fixture.get('expected')
        if not isinstance(e,torch.Tensor) or e.shape!=(p['rows'],p['D']) or e.dtype!=torch.float32:
            raise ValueError('FP32 expected output required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu_inputs):
        p=check_spec(spec)
        if arm not in ('before','after'):raise ValueError('paired arm required')
        check_inputs(cpu_inputs,p,'cpu')
        import torch_npu
        from l2_backward_kernels import l2norm_bwd_kernel1,l2norm_bwd_factored
        inputs={n:v.to('npu:0') for n,v in cpu_inputs.items()};check_inputs(inputs,p,'npu')
        if arm in self.state:self.state.clear();self.actual.clear()
        rows,d=p['rows'],p['D'];bd=1<<(d-1).bit_length()
        buffer=torch.full((rows*d+64,),12345.,dtype=torch.float32,device='npu:0');out=buffer[32:-32].view(rows,d)
        self.state[arm]=(inputs,buffer)
        kernel=l2norm_bwd_kernel1 if arm=='before' else l2norm_bwd_factored
        args=(inputs['y'],inputs['rstd'],inputs['dy'],out,1e-6,d)
        kwargs=dict(BD=bd,num_warps=4,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        grid=(rows,);compiled=[]
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,**kwargs,grid=grid);data=c.asm.get('npubin')
                if not isinstance(data,bytes) or not data.startswith(b'\x7fELF'):raise ValueError('missing actual npubin')
                offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
                sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
                table=sections[names];strings=data[table[4]:table[4]+table[5]]
                if not any(strings[t[0]:].split(b'\0',1)[0]==b'.text' and t[5]>0 for t in sections):raise ValueError('missing executable text')
                compiled.append(c)
            return compiled
        def run():
            compile_only();kernel[grid](*args,**kwargs);return out
        sources=[__file__,self.case_sources[0],Path(__file__).with_name('l2_backward_kernels.py')]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),sources=sources,
            expected_kernels=[kernel.fn.__name__],metadata=dict(component='provided-saved-state L2 backward only',
            configuration=dict(BD=bd,num_warps=4),grid=grid,device_kernel_calls=1,reused_output=True,
            workspace_scope='prepared output; compiler-managed scratch retained by JIT',
            allocator_limit='no allocation or complete model latency claim',input_strides={n:list(v.stride()) for n,v in inputs.items()}))

    def validate(self,arm,actual,fixture,spec):
        inputs,buffer=self.state[arm];result=validate_actual(actual,fixture,inputs,buffer)
        self.actual[arm]=actual.detach().cpu()
        other='after' if arm=='before' else 'before'
        if other in self.actual:
            result['paired_exact_bits']=bool(torch.equal(self.actual[arm].view(torch.int32),self.actual[other].view(torch.int32)))
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
