"""Prepared source-controlled mean CE: delayed versus unconditional eager gradient."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ce_eager_contract import CASE_SOURCES,dimensions,make_case,validate_inputs,check_output


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)
    def __init__(self):self.prepared={};self.host_outputs={};self.pair_key=None

    def _device_item(self,arm,spec,v,n,probe=False):
        r,c=dimensions(spec);p=spec.get('parameters',spec);need=p['needs_grad']
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,triton
        import ce_eager_kernel as k
        b=min(2048,1<<(c-1).bit_length());private=arm=='after' or need
        def guarded(size):
            buf=torch.empty(size+2,dtype=torch.float32,device=v['X'].device)
            if not probe:buf[0]=-701.;buf[-1]=-701.
            return buf,buf[1:-1]
        lossbuf,loss=guarded(r);scalarbuf,scalar=guarded(1);scalar=scalar.view(())
        workbuf,work=guarded(r*c) if private else (None,v['X']);work=work.view(r,c)
        stats=torch.empty((2,r),dtype=torch.float32,device=v['X'].device) if arm=='before' and need else None
        upstream=torch.empty((),dtype=torch.float32,device=v['X'].device)
        if not probe and need:upstream.fill_(p['upstream'])
        flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False);calls=[]
        if private:calls.append((k.ce_reset_copy,(r,(c+b-1)//b),(v['X'],work,loss),dict(V=c,B=b)))
        else:calls.append((k.ce_loss_zero,(r,),(loss,),dict(ROWS=r)))
        args=(work,c,v['Y'],1,None,loss,None,1,c,n,n,0.,-100)
        kw=dict(lse_square_scale=0.,label_smoothing=p['smoothing'],reduction='mean',softcap=None,RETURN_Z_LOSS=False,BLOCK_SIZE=b,HAS_WEIGHT=False,HAS_SOFTCAPPING=False)
        if arm=='after':calls.append((k.cross_entropy_kernel,(r,),args,kw))
        else:
            args=(work,stats[0] if stats is not None else None,stats[1] if stats is not None else None,need,*args[1:])
            calls.append((k.ce_stats_forward,(r,),args,kw))
            if need:calls.append((k.ce_delayed_gradient,(r,),(work,v['Y'],stats[0],stats[1],c,n),dict(label_smoothing=p['smoothing'],BLOCK_SIZE=b)))
        scale=need and p['upstream']!=1
        if scale:calls.append((k.element_mul_kernel,(r,),(work,c,upstream,c),dict(BLOCK_SIZE=b,num_warps=32)))
        compiled=[]
        def compile_only():
            if not compiled:
                pending=[kernel.warmup(*args,grid=grid,**dict(flags,**kw)) for kernel,grid,args,kw in calls]
                compiled.extend(pending)
            return list(compiled)
        def run():
            if probe:raise RuntimeError('input-only compile probe cannot execute')
            compile_only()
            for index,(kernel,grid,args,kw) in enumerate(calls):
                kernel[grid](*args,**dict(flags,**kw))
                if index==1:torch.sum(loss,dim=0,out=scalar)
            return scalar,work if need else None
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest() for key,value in a.asm.items() if isinstance(value,(str,bytes))}) for a in compiled]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,inputs=v,
            buffers=[x for x in (lossbuf,scalarbuf,workbuf) if x is not None],outputs=(scalar,work if need else None),
            sources=[Path(__file__),Path(k.__file__),*CASE_SOURCES],expected_kernels=[x[0].fn.__name__ for x in calls],
            metadata=dict(arm=arm,block=b,grids=[x[1] for x in calls],triton_component_count=len(calls),torch_sum_count=1,
                api_component_count=len(calls)+1,private_reset_copy=private,saved_stats=stats is not None,upstream=p['upstream'],needs_grad=need,
                scope='prepared source-controlled CE; charged reset helper and Torch SUM; reused private buffers; excludes fresh allocator/autograd',
                flags=flags,source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',torch_version=torch.__version__,triton_version=triton.__version__))

    def probe(self,arm,spec):
        r,c=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,torch_npu
        v=dict(X=torch.empty((r,c),dtype=torch.float32,device='npu'),Y=torch.empty((r,),dtype=torch.int64,device='npu'))
        p=spec.get('parameters',spec);n=0 if p.get('all_ignored') else (sum(y!=-100 for y in p['targets']) if 'targets' in p else r-(r+6)//7)
        return self._device_item(arm,spec,v,n,probe=True)

    def prepare(self,arm,spec,fixture,output):
        dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        n=validate_inputs(fixture['inputs'],spec)
        import torch_npu
        key=json.dumps(spec,sort_keys=True)
        if key!=self.pair_key or arm in self.prepared:self.prepared={};self.host_outputs={};self.pair_key=key
        item=self._device_item(arm,spec,{name:x.npu() for name,x in fixture['inputs'].items()},n);self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]
        host=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in actual) if isinstance(actual,tuple) else actual
        result=check_output(host,fixture['expected'])
        halos=all(bool((buf[[0,-1]].cpu()==-701).all()) for buf in item.buffers)
        immutable=all(torch.equal(x.cpu(),fixture['inputs'][n]) for n,x in item.inputs.items())
        tensors=[x for x in actual if isinstance(x,torch.Tensor)] if isinstance(actual,tuple) else []
        private=len({x.untyped_storage().data_ptr() for x in (*tensors,*item.inputs.values())})==len(tensors)+2
        pair=None
        if result['passed']:
            self.host_outputs[arm]=host
            if len(self.host_outputs)==2:pair=all((a is None and b is None) or (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a.view(torch.int32),b.view(torch.int32))) for a,b in zip(self.host_outputs['before'],self.host_outputs['after']))
        result.update(canaries=halos,immutable=immutable,nonalias=private,paired_output_bits_secondary=pair)
        result['passed']=result['passed'] and halos and immutable and private
        return result


def factory():return Adapter()
