"""Prepared rotary Q/K: head-parallel pair versus sequence programs sharing trig."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rotary_trig_contract import CASE_SOURCES,dimensions,make_case,validate_inputs,check_output


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)
    def __init__(self):self.prepared={};self.bits={};self.pair_key=None

    def _device_item(self,arm,spec,v,probe=False):
        b,s,hq,hk,d=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,triton
        from rotary_trig_kernel import _fused_rotary_emb_kernel,rotary_head_kernel
        buffers=[torch.empty(v[n].numel()+2,dtype=torch.float32,device=v[n].device) for n in ('Q','K')]
        out=tuple(buf[1:-1].view_as(v[n]) for buf,n in zip(buffers,('Q','K')))
        if not probe:
            for buf in buffers:buf[0]=-701.;buf[-1]=-701.
        flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False,num_warps=4)
        scale=spec.get('parameters',spec)['scaling'];p,i=v['P'],v['I'];calls=[]
        if arm=='before':
            for n,o,h in zip(('Q','K'),out,(hq,hk)):
                x=v[n];args=(x,p,i,scale,o,*x.stride(),*p.stride(),s)
                calls.append((rotary_head_kernel,(b,(s+31)//32,h),args,dict(BLOCK=32,F=d//2,**flags)))
        else:
            args=(v['Q'],v['K'],p,i,scale,*out,*v['Q'].stride(),*v['K'].stride(),*p.stride(),s)
            calls.append((_fused_rotary_emb_kernel,(b,(s+31)//32),args,dict(BLOCK=32,BLOCK_HQ=hq,BLOCK_HK=hk,BLOCK_F=d//2,**flags)))
        compiled=[]
        def compile_only():
            if not compiled:
                pending=[kernel.warmup(*args,grid=grid,**kw) for kernel,grid,args,kw in calls]
                compiled.extend(pending)
            return list(compiled)
        def run():
            if probe:raise RuntimeError('input-only compile probe cannot execute')
            compile_only()
            for kernel,grid,args,kw in calls:kernel[grid](*args,**kw)
            return out
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for key,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,inputs=v,buffers=buffers,outputs=out,
            sources=[Path(__file__),Path(__file__).with_name('rotary_trig_kernel.py'),*CASE_SOURCES],expected_kernels=list(dict.fromkeys(x[0].fn.__name__ for x in calls)),
            metadata=dict(arm=arm,grids=[x[1] for x in calls],api_component_count=len(calls),position_strides=p.stride(),flags=flags,block=32,
                scope='prepared Q/K forward; both private outputs reused; allocation excluded',baseline='educational head-parallel, not vendor baseline',
                source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',torch_version=torch.__version__,triton_version=triton.__version__))

    def probe(self,arm,spec):
        b,s,hq,hk,d=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,torch_npu
        ps=spec.get('parameters',spec).get('position_stride',1)
        v={n:torch.empty(shape,dtype=torch.float32,device='npu') for n,shape in [('Q',(b,s,hq,d)),('K',(b,s,hk,d)),('I',(d//2,))]}
        v['P']=torch.empty((b,s*ps),dtype=torch.int64,device='npu')[:,::ps]
        return self._device_item(arm,spec,v,probe=True)

    def prepare(self,arm,spec,fixture,output):
        dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        validate_inputs(fixture['inputs'],spec)
        import torch_npu
        key=json.dumps(spec,sort_keys=True)
        if key!=self.pair_key or arm in self.prepared:self.prepared={};self.bits={};self.pair_key=key
        v={n:x.npu() for n,x in fixture['inputs'].items() if n!='P'}
        p=fixture['inputs']['P'];ps=p.stride(1)
        v['P']=p.as_strided((p.untyped_storage().nbytes()//8,),(1,),storage_offset=0).npu().view(p.shape[0],p.shape[1]*ps)[:,::ps]
        item=self._device_item(arm,spec,v);self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];result=check_output(actual,fixture['expected'])
        halos=all(bool((x[[0,-1]].cpu()==-701).all()) for x in item.buffers)
        def raw(x):return x.as_strided((x.untyped_storage().nbytes()//x.element_size(),),(1,),storage_offset=0).cpu().view(torch.uint8)
        immutable=all(torch.equal(raw(x),raw(fixture['inputs'][n])) for n,x in item.inputs.items())
        private=isinstance(actual,tuple) and len(actual)==2 and all(isinstance(x,torch.Tensor) for x in actual)
        if private:private=len({x.untyped_storage().data_ptr() for x in (*actual,*item.inputs.values())})==6
        paired=None;pair_error=None
        if result['passed']:
            self.bits[arm]=tuple(x.detach().cpu().clone() for x in actual)
            if len(self.bits)==2:
                paired=all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(self.bits['before'],self.bits['after']))
                pair_error=[float((a-b).abs().max()) for a,b in zip(self.bits['before'],self.bits['after'])]
        result.update(canaries=halos,immutable=immutable,nonalias=private,paired_output_bits=paired,max_pair_errors=pair_error)
        result['passed']=result['passed'] and halos and immutable and private
        return result


def factory():return Adapter()
