"""Prepared full Compressor pair; only proven output zeroing is omitted."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
from compressor_zero_contract import CASE_SOURCES,EPS,ATOL,RTOL,dimensions,complete_write,output_shape,make_case,check_fixture


def identity(artifact):
    return dict(hash=artifact.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in artifact.asm.items() if isinstance(v,(str,bytes))})


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):self.prepared={};self.paired={};self.current=None

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        check_fixture(spec,fixture)
        import torch_npu
        data={k:v.npu() for k,v in fixture.items() if k!='expected'}
        return self._build(arm,spec,data)

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        b,s,_,_,packed,_,_,_=dimensions(spec)
        import torch,torch_npu
        data={k:torch.empty(shape,dtype=dtype,device='npu:0') for k,shape,dtype in [('x',(b*s,5120) if packed else (b,s,5120),torch.bfloat16),('wkv',(512,5120),torch.bfloat16),('wgate',(512,5120),torch.bfloat16),('norm_w',(512,),torch.bfloat16),('cache',(b,16,1024),torch.float32)]}
        return self._build(arm,spec,data,probe=True)

    def _build(self,arm,spec,data,probe=False):
        import torch,triton
        import compressor_zero_vendor as vendor
        b,s,r,_,packed,start,used,table=dimensions(spec);m=b*s
        device=data['x'].device;shape=output_shape(spec);n=__import__('math').prod(shape)
        alloc=lambda n,dtype,canary:torch.empty(n+2,device=device,dtype=dtype) if probe else torch.full((n+2,),canary,device=device,dtype=dtype)
        out_store=alloc(n,torch.bfloat16,-120.);out=out_store[1:-1].view(shape)
        cache_store=alloc(b*16*1024,torch.float32,-1234.);cache=cache_store[1:-1].view(b,16,1024)
        if not probe:cache.copy_(data['cache'])
        data={**data,'cache':cache};kv=torch.empty((m,512),device=device,dtype=torch.float32);score=torch.empty_like(kv,dtype=torch.float32,device=device)
        groups=(start+used)//r;rows=(s+r-1)//r
        meta=torch.tensor([start]*b+[used]*b+[i*(groups if packed else rows) for i in range(b)]+[i*s for i in range(b)]+table,dtype=torch.int32,device=device)
        mapping=meta[4*b:];cores=triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())['num_aicore']
        cfg=dict(BM=32,BN=64,BK=128,GROUP=8);common=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False);launches=[]
        def add(kernel,grid,args,kw):launches.append((kernel,grid,args,{**kw,**common}))
        if m<=64:
            add(vendor._proj2_kernel,(min(cores,triton.cdiv(m,32)*16),),(kv,score,data['x'].view(m,5120),data['wkv'],data['wgate'],m),dict(IN_DIM=5120,OUT_DIM=512,**cfg))
        else:
            for dest,weight in ((kv,data['wkv']),(score,data['wgate'])):
                add(vendor._proj_kernel,(min(cores,triton.cdiv(m,32)*8),),(dest,data['x'].view(m,5120),weight,m),dict(IN_DIM=5120,OUT_DIM=512,**cfg))
        pad=triton.next_power_of_2(r);gt=max(1,16//pad);tasks=1+triton.cdiv(triton.cdiv(used,r),gt)
        add(vendor._pool_kernel,(min(cores,b*tasks),),(out.view(-1,512),kv,score,cache,data['norm_w'],meta,mapping,b*tasks,tasks,b),dict(CACHE_SIZE=16,HEAD_DIM=512,RATIO=r,RATIO_PAD=pad,CHUNK_ROWS=min(pad,8),GROUPS_TILE=gt,eps=EPS,SINGLE_BLOCK=True))
        tail=min(used,16)
        add(vendor._cache_update_kernel,(min(cores,b*tail),),(kv,score,cache,meta,mapping,b*tail,tail,b),dict(CACHE_SIZE=16,HEAD_DIM=512))
        artifacts={};qualified=[False];omit=arm=='after' and complete_write(spec)
        def compile_only():
            if not qualified[0]:
                pending={}
                for kernel,grid,args,kw in launches:
                    a=kernel.warmup(*args,grid=grid,**kw)
                    if 'npubin' not in a.asm:raise RuntimeError('linked NPU binary required')
                    pending[a.hash]=a
                artifacts.update(pending);qualified[0]=True
            return list(artifacts.values())
        def run():
            if probe:raise ValueError('input-only compile prototype cannot run')
            compile_only()  # Qualification is atomic; a failure publishes no partial state.
            if not omit:out.zero_()
            for kernel,grid,args,kw in launches:kernel[grid](*args,**kw)
            return dict(output=out,cache=cache)
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[identity(a) for a in artifacts.values()],
            expected_kernels=list(dict.fromkeys(k.fn.__name__ for k,_,_,_ in launches)),
            sources=[Path(__file__),Path(__file__).with_name('compressor_zero_vendor.py'),*CASE_SOURCES],
            metadata=dict(arm=arm,role=spec.get('role'),parameters=spec.get('parameters',spec),omit_output_zero=omit,complete_write_proof=complete_write(spec),common_projection_config=cfg,projection_strategy='dual' if m<=64 else 'separate',eps=EPS,triton_launches_per_api=len(launches),zero_api_calls_per_api=int(not omit),zero_backend='Torch Tensor.zero_; actual correlated device operations must be included',allocator_excluded=True,metadata_uploaded_once=True,mutable_cache_initialized_once=True,output_poisoned_once=not probe,prototype=probe),
            inputs=data,output=out,cache=cache,kv=kv,score=score,out_store=out_store,cache_store=cache_store,spec=spec)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.prepared:self.prepared={};self.paired={};self.current=key
            self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]
        if item.spec!=spec:raise ValueError('stale prepared case')
        if not isinstance(actual,dict) or set(actual)!={'output','cache'}:return dict(passed=False,reason='full return arity')
        expected=fixture['expected'];host={k:v.detach().cpu() for k,v in actual.items()}
        if any(host[k].dtype!=expected[k].dtype or host[k].shape!=expected[k].shape for k in host):return dict(passed=False,reason='full shape/dtype')
        err=(host['output'].float()-expected['output'].float()).abs();relative=err/expected['output'].float().abs().clamp_min(1e-12)
        numerical=bool(torch.isfinite(host['output']).all() and (err<=ATOL+RTOL*expected['output'].float().abs()).all())
        b,s,r,_,packed,start,used,_=dimensions(spec);groups=(start+used)//r
        zero_rows=bool((host['output'][groups:] if packed else host['output'][:,groups:]).eq(0).all())
        bits=lambda a,b:torch.equal(a.view(torch.int32),b.view(torch.int32))
        exact=bits(host['cache'],expected['cache']) and bits(item.kv.cpu(),expected['kv']) and bits(item.score.cpu(),expected['score'])
        immutable=all(k=='cache' or torch.equal(v.cpu(),fixture[k]) for k,v in item.inputs.items())
        canaries=bool((item.out_store[[0,-1]].cpu()==-120).all() and (item.cache_store[[0,-1]].cpu()==-1234).all())
        self.paired[arm]=host['output'].clone();other='after' if arm=='before' else 'before'
        paired=torch.equal(host['output'].view(torch.int16),self.paired[other].view(torch.int16)) if other in self.paired else None
        ptrs=[v.untyped_storage().data_ptr() for v in item.inputs.values()]+[item.output.untyped_storage().data_ptr(),item.kv.untyped_storage().data_ptr(),item.score.untyped_storage().data_ptr()]
        nonalias=len(set(ptrs))==len(ptrs)
        quantiles=torch.tensor([0.,.5,.9,.99,1.],dtype=err.dtype,device=err.device)
        return dict(passed=numerical and exact and zero_rows and immutable and canaries and nonalias and paired is not False,unwritten_rows_exact_zero=zero_rows,exact_projection_cache_bits=exact,paired_output_bits=paired,canaries=canaries,inputs_immutable=immutable,nonalias=nonalias,max_absolute_error=float(err.max()),max_relative_error=float(relative.max()),absolute_error_quantiles=torch.quantile(err.flatten(),quantiles).tolist(),relative_error_quantiles=torch.quantile(relative.flatten(),quantiles).tolist())


def factory():return Adapter()


def main():
    """Exploratory single-case CLI; controlled qualification requires all-case screens."""
    import argparse,sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    p=argparse.ArgumentParser(description=main.__doc__)
    for name in ('case','output','cache','provenance'):p.add_argument('--'+name,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args()
    spec=json.loads(Path(a.case).read_text());dimensions(spec)
    result=run_pair(factory(),spec,Path(a.output),Path(a.cache),json.loads(Path(a.provenance).read_text()),correctness_only=a.correctness_only)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
