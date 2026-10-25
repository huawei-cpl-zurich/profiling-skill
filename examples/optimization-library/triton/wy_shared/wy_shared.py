"""Prepared IEEE WY W/U component, not full GDN/public-wrapper latency."""
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
import torch
from wy_shared_contract import dimensions,layout,make_case,check_fixture,backing,halo,CANARY,ATOL,RTOL
HERE=Path(__file__).resolve().parent


def identity(compiled):return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def ieee_policy(c):
    ir=c.asm.get('ttir','');ir=ir.decode() if isinstance(ir,bytes) else ir
    dots=[line for line in ir.splitlines() if re.search(r'\btt\.dot\b',line)]
    default=getattr(getattr(c,'metadata',None),'default_dot_input_precision',None);origins=[]
    for line in dots:
        policy=re.search(r'inputPrecision\s*=\s*([A-Za-z0-9_]+)',line)
        if (('inputPrecision' in line and policy is None) or line.count('xf32')<3 or 'xf16' in line or 'xbf16' in line
                or (policy is not None and policy[1]!='ieee') or (policy is None and default!='ieee')):
            raise ValueError('compiled FP32 IEEE dot policy required before numerical references')
        origins.append('explicitIR' if policy else 'actualCompilerMetadataDefaultIEEE')
    if not dots:raise ValueError('compiled FP32 IEEE dot operation required')
    return dict(status='ieee',dot_operations=len(dots),compiler_default_dot_input_precision=default,policy_origins=origins,raw_TTIR_sha256=hashlib.sha256(ir.encode()).hexdigest(),dot_lines=dots)


def load_json(path):
    def bad(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(Path(path).read_text(),parse_constant=bad)


class Adapter:
    case_sources=[HERE/'wy_shared_contract.py']
    make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('arm')
        t,_=dimensions(spec)
        import torch_npu
        inputs={}
        for k,(shape,dtype) in layout(t).items():
            n=__import__('math').prod(shape);h=128//torch.empty((),dtype=dtype,device='cpu').element_size()
            inputs[k]=torch.empty((n+2*h,),dtype=dtype,device='npu:0')[h:-h].view(shape)
        return self._prepare(arm,spec,inputs,True)

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('arm')
        check_fixture(spec,fixture)
        import torch_npu
        inputs={k:backing(x).to('npu:0')[halo(x):-halo(x)].view(x.shape) for k,x in fixture['inputs'].items()}
        return self._prepare(arm,spec,inputs,False)

    def _prepare(self,arm,spec,z,probe):
        from wy_shared_generated import wy_w_kernel,wy_u_kernel,wy_fused_kernel
        import torch_npu
        t,nt=dimensions(spec);shape=(1,t,8,128);device=z['k'].device;stores=[];outputs=[]
        for _ in range(2):
            store=(torch.empty if probe else torch.full)((t*8*128+128,),**({} if probe else {'fill_value':CANARY}),dtype=torch.float16,device=device);stores.append(store);outputs.append(store[64:-64].view(shape))
        physical=[]
        for name,x in list(z.items())+list(zip(('w','u'),outputs)):
            h=halo(x);fmt=int(torch_npu.get_npu_format(x));size=int(torch_npu.get_storage_size(x))
            if (x.device!=device or not x.is_contiguous() or x.data_ptr()%128 or fmt!=2 or size<x.numel()+2*h
                    or x.storage_offset()!=h or backing(x).numel()!=x.numel()+2*h):raise ValueError('actual dense aligned ND owned physical storage required')
            physical.append(dict(name=name,shape=list(x.shape),strides=list(x.stride()),storage_offset=h,storage_bytes=x.untyped_storage().nbytes(),format=fmt,physical_storage_numel=size))
        if len({x.untyped_storage().data_ptr() for x in [*z.values(),*outputs]})!=9:raise ValueError('device aliases')
        w,u=outputs;args=(z['k'],z['v'],z['beta'],w,u,z['A'],z['g'],z['cu'],z['indices'],t)
        kw=dict(H=8,Hg=8,K=128,V=128,BT=64,BK=64,BV=64,IS_VARLEN=True,num_warps=4,num_stages=3,enable_fp_fusion=False)
        kernels=[wy_w_kernel,wy_u_kernel] if arm=='before' else [wy_fused_kernel];compiled=[];raw=[];policies=[]
        def compile_only():
            if not compiled:
                pending=[];proof=[]
                with torch.npu.device(device):
                    for kernel in kernels:
                        c=kernel.warmup(*args,grid=(nt,1),**kw);raw.append(c)
                        if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin'].startswith(b'\x7fELF'):raise ValueError('linked NPU ELF required')
                        proof.append(ieee_policy(c));pending.append(c)
                policies.extend(proof);compiled.extend(pending)
            return compiled
        def run():
            if probe:raise RuntimeError('compile prototype cannot execute')
            with torch.npu.device(device):
                compile_only()
                for kernel in kernels:kernel[(nt,1)](*args,**kw)
            return tuple(outputs)
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),raw_artifacts=raw,IEEE_policies=policies,
            expected_kernels=[k.fn.__name__ for k in kernels],sources=[Path(__file__),*self.case_sources,HERE/'wy_shared_generated.py',HERE/'wy_shared_codegen.py',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())],
            metadata=dict(arm=arm,prototype=probe,grid=[nt,1],parameters=kw,physical_layouts=physical,api_kernels=len(kernels),B=1,T=t,component='prepared packed-single-sequence WY W+U',prepared_outputs_reused=True,allocator_cost_excluded=True,common_A_beta_g_indices_prepared=True,prepared_device=str(device),caller_device_restored=True,IEEE_policy='bothsourceU/Wallow_tf32False'),inputs=z,outputs=tuple(outputs),stores=stores,spec=spec)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.items={};self.numeric={};self.current=key
            self.items[arm]=item
        return item

    def capture(self,run,item,output):
        # Events/profiler setup and synchronous export share the prepared device.
        with torch.npu.device(item.outputs[0].device):
            import sys
            helpers=HERE.parents[3]/'skills/ascend-model-benchmarking/scripts'
            if str(helpers) not in sys.path:sys.path.insert(0,str(helpers))
            from profile_ten import capture_ten
            return capture_ten(run,output,expected_kernel=item.expected_kernels)

    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];check_fixture(spec,fixture);shape=(1,dimensions(spec)[0],8,128)
        if item.spec!=spec or not isinstance(actual,tuple) or len(actual)!=2:raise ValueError('BOTH W/U outputs required')
        details={};numeric={};canary=True
        for name,x,want,store in zip(('w','u'),actual,item.outputs,item.stores):
            if not isinstance(x,torch.Tensor) or x.dtype!=torch.float16 or x.shape!=shape or x.device!=want.device or not x.is_contiguous() or x.stride()!=want.stride() or x.data_ptr()!=want.data_ptr():raise ValueError('W/U output ABI')
            raw=store.detach().cpu();h=raw[64:-64].view(shape);e=fixture['expected'][name];hd=h.double();ed=e.double();diff=(hd-ed).abs();finite=torch.isfinite(hd);bad=(~finite)|(diff>ATOL+RTOL*ed.abs());coords=bad.nonzero();count=int(bad.sum())
            def scalar(v):return float(v) if bool(torch.isfinite(v)) else str(float(v))
            details[name]=dict(passed=count==0,mismatch_count=count,first_mismatches=[dict(coordinates=c.tolist(),expected=scalar(e[tuple(c)]),actual=scalar(h[tuple(c)])) for c in coords[:8]],max_absolute_error=float(diff.max()) if bool(finite.all()) else None,nonfinite_count=int((~finite).sum()),zero_sign_differences=int(((h==0)&(e==0)&(h.view(torch.int16)!=e.view(torch.int16))).sum()))
            numeric[name]=h.clone();canary &= bool((raw[:64]==CANARY).all() and (raw[-64:]==CANARY).all())
        immutable=all(torch.equal(backing(x).cpu(),backing(fixture['inputs'][name])) for name,x in item.inputs.items());nonalias=len({x.untyped_storage().data_ptr() for x in [*item.inputs.values(),*item.outputs]})==9
        other=self.numeric.get('after' if arm=='before' else 'before');self.numeric[arm]=numeric
        return dict(passed=all(x['passed'] for x in details.values()) and immutable and canary and nonalias,outputs=details,atol=ATOL,rtol=RTOL,inputs_immutable=immutable,canaries=canary,nonalias=nonalias,paired_bits_secondary=None if other is None else {name:bool(torch.equal(h.view(torch.int16),other[name].view(torch.int16))) for name,h in numeric.items()},output_storage_CPU_transfers=2)


def factory():return Adapter()


def main():
    import argparse,sys
    p=argparse.ArgumentParser(description='Exploratory prepared WY pair, not controlled all-case/fullGDN qualification')
    for name in ('case','output','cache','provenance'):p.add_argument('--'+name,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args();spec=load_json(a.case);dimensions(spec)
    sys.path.insert(0,str(HERE.parents[3]/'scripts'))
    from optimization_pair import run_pair
    result=run_pair(Adapter(),spec,a.output,a.cache,load_json(a.provenance),correctness_only=a.correctness_only)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
