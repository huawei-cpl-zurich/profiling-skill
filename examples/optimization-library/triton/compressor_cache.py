"""Compiler-failed full prepared Compressor; execution is disabled."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
from compressor_cache_contract import CASE_SOURCES,EPS,ATOL,RTOL,dimensions,make_case,check_fixture


def identity(artifact):
    return dict(hash=artifact.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in artifact.asm.items() if isinstance(v,(str,bytes))})


FAILURE = ('T07 failed study: candidate backend buffer-shape compilation failed before correctness; '
           'public execution and capture are disabled. See experiments/optimization-wave008/T07-result.json.')


class StudyUnsupported(ValueError):
    """This retained candidate has no passing kernel correctness gate."""


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self, *, compile_inspection=False):
        self.prepared={};self.paired={};self.current=None;self.compile_inspection=compile_inspection

    def capture(self,*args,**kwargs):raise StudyUnsupported(FAILURE)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        if not self.compile_inspection:raise StudyUnsupported(FAILURE)
        check_fixture(spec,fixture)
        import torch_npu
        data={k:v.npu() for k,v in fixture.items() if k!='expected'}
        return self._build(arm,spec,data)

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        if not self.compile_inspection:raise StudyUnsupported(FAILURE)
        m,c,_,_,_=dimensions(spec)
        import torch,torch_npu
        data={k:torch.empty(shape,dtype=dtype,device='npu:0') for k,shape,dtype in
              [('x',(1,m,5120),torch.bfloat16),('wkv',(512,5120),torch.bfloat16),
               ('wgate',(512,5120),torch.bfloat16),('norm_w',(512,),torch.bfloat16),('cache',(1,c,1024),torch.float32)]}
        return self._build(arm,spec,data,probe=True)

    def _build(self,arm,spec,data,probe=False):
        import torch,triton
        import compressor_cache_vendor as vendor
        from compressor_cache_projection import cache_proj2_kernel,cache_proj_kernel
        m,c,r,start,_=dimensions(spec);fused=arm=='after' and start%r==0
        device=data['x'].device;rows=(m+r-1)//r
        alloc=lambda n,dtype,canary:(torch.empty((n+2,),device=device,dtype=dtype) if probe else torch.full((n+2,),canary,device=device,dtype=dtype))
        out_store=alloc(rows*512,torch.bfloat16,-120.);out=out_store[1:-1].view(1,rows,512)
        cache_store=alloc(c*1024,torch.float32,-1234.);cache=cache_store[1:-1].view(1,c,1024)
        if not probe:cache.copy_(data['cache'])
        data={**data,'cache':cache}
        kv=torch.empty((m,512),device=device);score=torch.empty_like(kv)
        # Prepared metadata is immutable and uploaded once, outside forwards.
        meta=torch.tensor([start,m,0,0,0],dtype=torch.int32,device=device);mapping=meta[4:5]
        cores=triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())['num_aicore']
        cfg=dict(BM=32,BN=64,BK=128,GROUP=8);common=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        launches=[]
        def add(kernel,grid,args,kw):launches.append((kernel,grid,args,{**kw,**common}))
        if m<=64:
            kernel=cache_proj2_kernel if fused else vendor._proj2_kernel
            args=(kv,score,data['x'].view(m,5120),data['wkv'],data['wgate'],m)
            add(kernel,(min(cores,triton.cdiv(m,32)*16),),args+((cache,start,c) if fused else ()),dict(IN_DIM=5120,OUT_DIM=512,**cfg))
        else:
            for half,(dest,weight) in enumerate(((kv,data['wkv']),(score,data['wgate']))):
                kernel=cache_proj_kernel if fused else vendor._proj_kernel
                args=(dest,data['x'].view(m,5120),weight,m)
                add(kernel,(min(cores,triton.cdiv(m,32)*8),),args+((cache,start,c,half) if fused else ()),dict(IN_DIM=5120,OUT_DIM=512,**cfg))
        pad=triton.next_power_of_2(r);gt=max(1,16//pad);tasks=1+triton.cdiv(triton.cdiv(m,r),gt)
        add(vendor._pool_kernel,(min(cores,tasks),),(out.view(-1,512),kv,score,cache,data['norm_w'],meta,mapping,tasks,tasks,1),
            dict(CACHE_SIZE=c,HEAD_DIM=512,RATIO=r,RATIO_PAD=pad,CHUNK_ROWS=min(pad,8),GROUPS_TILE=gt,eps=EPS,SINGLE_BLOCK=True))
        if not fused:
            tail=min(m,c)
            add(vendor._cache_update_kernel,(min(cores,tail),),(kv,score,cache,meta,mapping,tail,tail,1),dict(CACHE_SIZE=c,HEAD_DIM=512))
        artifacts={};qualified=[False]
        def compile_only():
            if not qualified[0]:
                for kernel,grid,args,kw in launches:
                    a=kernel.warmup(*args,grid=grid,**kw)
                    if 'npubin' not in a.asm:raise RuntimeError('linked NPU binary required')
                    artifacts[a.hash]=a
                qualified[0]=True
            return list(artifacts.values())
        def run():
            # Explicit compile inspection never authorizes a numeric launch.
            raise StudyUnsupported(FAILURE)
        paths=[Path(__file__),Path(__file__).with_name('compressor_cache_vendor.py'),Path(__file__).with_name('compressor_cache_projection.py'),*CASE_SOURCES]
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[identity(a) for a in artifacts.values()],
            expected_kernels=list(dict.fromkeys(k.fn.__name__ for k,_,_,_ in launches)),sources=paths,
            metadata=dict(execution_enabled=False,arm=arm,role=spec.get('role'),parameters=spec.get('parameters',spec),fused_cache=fused,
                common_projection_config=cfg,projection_strategy='dual' if m<=64 else 'separate',eps=EPS,
                logical_components=1+len(launches),zero_each_api=True,zero_backend='torch Tensor.zero_ common; actual tasks must be accounted',
                scope='full prepared B1 BSND Compressor',allocator_excluded=True,metadata_uploaded_once=True,
                mutable_cache_initialized_once=True,prototype=probe),
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
        valid=all(host[k].dtype==expected[k].dtype and host[k].shape==expected[k].shape for k in host)
        if not valid:return dict(passed=False,reason='full return shape/dtype')
        err=(host['output'].float()-expected['output'].float()).abs();relative=err/expected['output'].float().abs().clamp_min(1e-12)
        numerical=bool(torch.isfinite(host['output']).all() and (err<=ATOL+RTOL*expected['output'].float().abs()).all())
        m,_,r,start,_=dimensions(spec)
        zero_rows=bool((host['output'][:,(start+m)//r:]==0).all())
        bits=lambda a,b:torch.equal(a.view(torch.int32),b.view(torch.int32))
        exact=bits(host['cache'],expected['cache']) and bits(item.kv.cpu(),expected['kv']) and bits(item.score.cpu(),expected['score'])
        immutable=all(k=='cache' or torch.equal(v.cpu(),fixture[k]) for k,v in item.inputs.items())
        canaries=bool((item.out_store[[0,-1]].cpu()==-120).all() and (item.cache_store[[0,-1]].cpu()==-1234).all())
        self.paired[arm]=host['output'].clone();other='after' if arm=='before' else 'before'
        paired=torch.equal(host['output'].view(torch.int16),self.paired[other].view(torch.int16)) if other in self.paired else None
        pointers=[v.untyped_storage().data_ptr() for v in item.inputs.values()]+[item.output.untyped_storage().data_ptr(),item.kv.untyped_storage().data_ptr(),item.score.untyped_storage().data_ptr()]
        nonalias=len(set(pointers))==len(pointers)
        return dict(passed=numerical and exact and zero_rows and immutable and canaries and nonalias and paired is not False,
            unwritten_rows_exact_zero=zero_rows,
            exact_projection_cache_bits=exact,paired_output_bits=paired,canaries=canaries,inputs_immutable=immutable,nonalias=nonalias,
            max_absolute_error=float(err.max()),max_relative_error=float(relative.max()),
            absolute_error_quantiles=torch.quantile(err.flatten(),torch.tensor([0.,.5,.9,.99,1.])).tolist(),
            relative_error_quantiles=torch.quantile(relative.flatten(),torch.tensor([0.,.5,.9,.99,1.])).tolist())


def factory():raise StudyUnsupported(FAILURE)


def main():
    """Disabled failed-study CLI; rejects before fixtures or backends."""
    import argparse
    parser=argparse.ArgumentParser(description=main.__doc__)
    for name in ('case','output','cache','provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--correctness-only',action='store_true')
    args=parser.parse_args()
    factory()  # Before case/provenance loading, fixture generation, or backend imports.



if __name__=='__main__':raise SystemExit(main())
