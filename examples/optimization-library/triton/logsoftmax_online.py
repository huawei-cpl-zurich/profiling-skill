"""Exploratory prepared FP32 log-softmax; controlled all-case qualification is separate."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from logsoftmax_online_contract import dimensions,make_case,check_fixture,backing,diagnostics


def identity(compiled):
    return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('logsoftmax_online_contract.py')]
    make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('arm')
        r,c,pad=dimensions(spec)
        import torch_npu
        x=torch.empty((r,c+pad),dtype=torch.float32,device='npu:0')[:,:c]
        return self._prepare(arm,spec,x,True)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('arm')
        check_fixture(spec,fixture) # Domain before backend/cache or device movement.
        import torch_npu
        cpu=fixture['inputs']['x'];r,c,pad=dimensions(spec)
        x=backing(cpu).to('npu:0').view(r,c+pad)[:,:c]
        return self._prepare(arm,spec,x,False)

    def _prepare(self,arm,spec,x,probe):
        import triton
        from logsoftmax_online_kernels import logsoftmax_source,logsoftmax_online
        r,c,pad=dimensions(spec)
        assert x.stride()==(c+pad,1)
        cores=triton.runtime.driver.active.utils.get_device_properties(0)['num_vectorcore']
        kernel=logsoftmax_source if arm=='before' else logsoftmax_online
        store=(torch.empty if probe else torch.full)((r*c+64,),**({} if probe else {'fill_value':12345.}),dtype=torch.float32,device=x.device)
        out=store[32:-32].view(r,c);compiled=[]
        args=(x,out,x.stride(0),out.stride(0),r,c)
        kw=dict(BLOCK_SIZE=2048,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        def compile_only():
            if not compiled:
                result=kernel.warmup(*args,grid=(cores,),**kw);data=result.asm.get('npubin')
                if not isinstance(data,bytes) or not data.startswith(b'\x7fELF'):raise ValueError('linked NPU ELF required')
                compiled.append(result) # Commit qualification only after all checks pass.
            return compiled
        def run():
            if probe:raise RuntimeError('compile prototype cannot execute')
            compile_only();kernel[(cores,)](*args,**kw)
            return out
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),
            expected_kernels=[kernel.fn.__name__],sources=[Path(__file__),*self.case_sources,Path(__file__).with_name('logsoftmax_online_kernels.py')],
            metadata=dict(arm=arm,prototype=probe,grid=[cores],BLOCK_SIZE=2048,input_stride=list(x.stride()),output_stride=list(out.stride()),
                allocator_cost_excluded=True,prepared_output_reused=True,component='FP32 log-softmax only',api_kernels=1),
            x=x,out=out,store=store,spec=spec)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.items={};self.numeric={};self.current=key
            self.items[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];check_fixture(spec,fixture)
        if item.spec!=spec:raise ValueError('stale case')
        if (not isinstance(actual,torch.Tensor) or actual.shape!=fixture['expected'].shape or actual.dtype!=torch.float32
                or actual.device!=item.x.device or not actual.is_contiguous() or actual.data_ptr()!=item.out.data_ptr()):raise ValueError('output shape/dtype/device/layout')
        b=item.store.detach().cpu() # Sole output-storage transfer includes both halos.
        host=b[32:-32].view_as(fixture['expected'])
        result=diagnostics(host,fixture['expected'])
        immutable=torch.equal(backing(item.x).cpu(),backing(fixture['inputs']['x']))
        canary=bool((b[:32]==12345).all() and (b[-32:]==12345).all())
        nonalias=actual.untyped_storage().data_ptr()!=item.x.untyped_storage().data_ptr()
        original=fixture['inputs']['x'];duplicate=(original.shape[0]>=2 and torch.equal(original[0],original[1]))
        consistent=not duplicate or torch.equal(host[0].view(torch.int32),host[1].view(torch.int32))
        other=self.numeric.get('after' if arm=='before' else 'before')
        result.update(inputs_immutable=immutable,canaries=canary,nonalias=nonalias,duplicate_rows_consistent=consistent,
            paired_bitwise_equal=None if other is None else bool(torch.equal(host.view(torch.int32),other.view(torch.int32))),
            paired_max_absolute_error=None if other is None else float((host-other).abs().max()))
        self.numeric[arm]=host.clone();result['passed'] &= immutable and canary and nonalias and consistent
        return result


def factory():return Adapter()


def main():
    import argparse,sys
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):p.add_argument('--'+name,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args()
    spec=json.loads(Path(a.case).read_text());dimensions(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    report=run_pair(Adapter(),spec,a.output,a.cache,json.loads(Path(a.provenance).read_text()),correctness_only=a.correctness_only)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
