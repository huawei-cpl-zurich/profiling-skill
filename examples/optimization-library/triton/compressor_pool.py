"""Exploratory prepared pooling component, not full-compressor throughput.

Controlled acquisition additionally screens every compiled case before scientific
CPU oracle generation and checks every pair before any profile. No autotuning.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from compressor_pool_contract import CASE_SOURCES, ATOL, RTOL, dimensions, make_case, check_fixture


def identity(artifact):
    binaries={}
    for key,value in artifact.asm.items():
        if isinstance(value,str):value=value.encode()
        if isinstance(value,bytes):binaries[key]=hashlib.sha256(value).hexdigest()
    return dict(hash=artifact.hash,binaries=binaries)


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):self.prepared={};self.numeric={};self.current=None

    def compile_probe(self,arm,spec,output):
        """Allocate uninitialized prototype storage, never a scientific fixture."""
        if arm not in ('before','after'):raise ValueError('unknown arm')
        shape=dimensions(spec)
        import torch,torch_npu
        _,_,_,used,res,_=shape
        data={name:torch.empty(size,dtype=dtype,device='npu:0') for name,size,dtype in
              [('kv',(used,512),torch.float32),('score',(used,512),torch.float32),('weight',(512,),torch.bfloat16)]}
        data['cache']=torch.empty((1,8,1024),device='npu:0',dtype=torch.float32) if res else None
        return self._prepare(arm,spec,data,probe=True)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        check_fixture(spec,fixture)  # Reject before backend imports/cache access.
        import torch_npu
        data={k:(None if v is None else v.npu()) for k,v in fixture.items() if k!='expected'}
        return self._prepare(arm,spec,data)

    def _prepare(self,arm,spec,data,probe=False):
        if arm not in ('before','after'):raise ValueError('unknown arm')
        import torch,triton
        from compressor_pool_kernel import pool_general,pool_ratio2
        g,r,start,used,res,_=dimensions(spec)
        optimized=arm=='after' and r==2 and not res
        kernel=pool_ratio2 if optimized else pool_general
        store=torch.empty((g*512+2,),dtype=torch.bfloat16,device=data['kv'].device) if probe else torch.full((g*512+2,),-120.,dtype=torch.bfloat16,device=data['kv'].device)
        result=store[1:-1].view(g,512)
        args=(data['kv'],data['score'],data['cache'],data['weight'],result)
        kw=dict(RATIO=r,RATIO_PAD=triton.next_power_of_2(r),START=start,HAS_RES=res,
                num_stages=1,multibuffer=False,enable_fp_fusion=False)
        artifacts={};qualified=[False]
        def compile_only():
            if not qualified[0]:
                a=kernel.warmup(*args,grid=(g,),**kw)
                if 'npubin' not in a.asm:raise RuntimeError('compiled NPU binary required')
                artifacts[a.hash]=a;qualified[0]=True
            return list(artifacts.values())
        def run():
            if probe:raise RuntimeError('compile prototype cannot execute')
            compile_only()
            a=kernel[(g,)](*args,**kw);artifacts[a.hash]=a
            return result
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[identity(a) for a in artifacts.values()],
            expected_kernels=[kernel.fn.__name__],sources=[Path(__file__),Path(__file__).with_name('compressor_pool_kernel.py'),*CASE_SOURCES],
            metadata=dict(arm=arm,parameters=spec.get('parameters',spec),role=spec.get('role'),groups=g,ratio=r,
                start_pos=start,used_len=used,residual=res,route='optimized' if optimized else 'reference',
                groups_per_program=1,api_kernels=1,scope='pooling component only',allocator_cost_excluded=True,
                prepared_output_reuse=True,bf16_pool_boundary=True,prototype=probe),
            inputs=data,store=store,result=result,spec=spec)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.prepared:self.prepared={};self.numeric={};self.current=key
            self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]
        if item.spec!=spec:raise ValueError('stale prepared case')
        host=actual.detach().cpu();expected=fixture['expected']
        if host.dtype!=torch.bfloat16 or host.shape!=expected.shape:return dict(passed=False,reason='output shape/dtype')
        error=(host.float()-expected.float()).abs();allowed=ATOL+RTOL*expected.float().abs()
        valid=bool(torch.isfinite(host).all() and (error<=allowed).all())
        immutable=all(x is None or torch.equal(x.cpu(),fixture[name]) for name,x in item.inputs.items())
        canary=bool((item.store[[0,-1]].cpu()==-120).all())
        nonalias=actual.untyped_storage().data_ptr() not in [x.untyped_storage().data_ptr() for x in item.inputs.values() if x is not None]
        self.numeric[arm]=host.clone()
        fallback=dimensions(spec)[1]!=2 or dimensions(spec)[4]
        other='after' if arm=='before' else 'before'
        paired=(torch.equal(host.view(torch.int16),self.numeric[other].view(torch.int16)) if other in self.numeric else None) if fallback else None
        quantiles=torch.tensor([0.,.5,.9,.99,1.],dtype=error.dtype,device=error.device)
        return dict(passed=valid and immutable and canary and nonalias and paired is not False,
            atol=ATOL,rtol=RTOL,max_absolute_error=float(error.max()),
            mean_absolute_error=float(error.mean()),max_relative_error=float((error/expected.float().abs().clamp_min(1e-12)).max()),
            absolute_error_quantiles=torch.quantile(error.flatten(),quantiles).tolist(),
            fallback_paired_bits=paired,canaries=canary,inputs_immutable=immutable,nonalias=nonalias,
            selected_kernel=item.expected_kernels,rounded_pool_diagnostics='unavailable; no extra diagnostic buffer or forward')


def factory():return Adapter()


def main():
    """Exploratory CLI; the controlled all-case collector is separate."""
    import argparse,sys
    parser=argparse.ArgumentParser(description=main.__doc__)
    for name in ('case','output','cache','provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--correctness-only',action='store_true')
    args=parser.parse_args()
    spec=json.loads(Path(args.case).read_text())
    dimensions(spec)  # Unsupported host cases fail before cache or NPU imports.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    report=run_pair(Adapter(),spec,args.output,args.cache,json.loads(Path(args.provenance).read_text()),correctness_only=args.correctness_only)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
