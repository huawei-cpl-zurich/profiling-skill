"""Exploratory prepared DSA score pair; frozen all-case collector supplies qualification."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import hashlib,torch
from dsa_indexer_contract import parameters,shapes,check_inputs,make_case,bits


def backend():
    import torch_npu
    import dsa_indexer_kernels as k
    from dsa_indexer_source import dsa_indexer_fused
    return 'npu:0',k,dsa_indexer_fused


def guard(device):return torch.npu.device(device) if str(device).split(':')[0]=='npu' else nullcontext()


def identity(compiled):
    return [dict(hash=str(c.hash),sha256=hashlib.sha256(c.asm['npubin']).hexdigest(),format='Triton-Ascend-npubin',identity_source='compiled executable npubin',name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def layout(t):
    high=t.storage_offset()+sum((n-1)*s for n,s in zip(t.shape,t.stride()));capacity=t.untyped_storage().nbytes()//t.element_size()
    if not t.is_contiguous() or high>=capacity:raise ValueError('physical contiguous storage span required')
    fmt=None
    if t.device.type=='npu':
        import torch_npu
        fmt=torch_npu.get_npu_format(t)
        if fmt!=2:raise ValueError('physical ND format2 required')
    return dict(shape=list(t.shape),strides=list(t.stride()),storage_offset=t.storage_offset(),maximum_offset=high,storage_elements=capacity,format=fmt)


class Adapter:
    case_sources=[Path(__file__).with_name('dsa_indexer_contract.py')];make_case=staticmethod(make_case)
    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        item=self._prepare(arm,spec,None)
        def reject():raise RuntimeError('input-only compile probe cannot run')
        item.run=reject;item.metadata.update(compile_only_probe=True,device_kernel_calls=0)
        return item

    def prepare(self,arm,spec,fixture,output):
        p=parameters(spec);check_inputs(fixture['inputs'],p)
        for key,dtype in [('expected',torch.float32),('expected_fp64',torch.float64)]:
            e=fixture.get(key)
            if not isinstance(e,torch.Tensor) or e.device.type!='cpu' or e.dtype!=dtype or e.shape!=(p['B'],p['M'],p['N']):raise ValueError('independent CPU FP64/FP32 expected score matrices required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu):
        p=parameters(spec)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,k,fused=backend();buffers=[];inputs={};scratch=[];compiled=[];closed=False
        def allocate(shape,dtype,value=None):
            size=1
            for n in shape:size*=n
            host=torch.full((size+64,),1024.,dtype=dtype)
            if value is not None:host[32:-32].view(shape).copy_(value)
            with guard(device):buffer=host.to(device)
            buffers.append(buffer);return buffer[32:-32].view(shape)
        for name,shape in shapes(p).items():inputs[name]=allocate(shape,torch.float32 if name=='weights' else torch.float16,None if cpu is None else cpu[name])
        b,m,h,d,n=(p[x] for x in ('B','M','H','D','N'));out=allocate((b,m,n),torch.float32)
        if arm=='before':
            scratch=[allocate((b,m,h,n),torch.float32) for _ in range(2)]
            calls=[(k.dsa_indexer_qk,(inputs['q'],inputs['k'],scratch[0],b,h,m,n,d,128)),(k.dsa_indexer_activate,(scratch[0],inputs['weights'],scratch[1],h,n,128)),(k.dsa_indexer_headsum,(scratch[1],out,h,n,128))]
        else:calls=[(fused,(inputs['q'],inputs['k'],out,inputs['weights'],b,h,m,n,d,128))]
        flags=dict(num_warps=4,num_stages=1,multibuffer=False);grid=(b*m,)
        with guard(device):physical={name:layout(t) for name,t in inputs.items()};physical['out']=layout(out)
        def compile_only():
            if closed:raise RuntimeError('prepared item is closed')
            if not compiled:
                with guard(device):pending=[kernel.warmup(*args,grid=grid,**flags) for kernel,args in calls]
                if any(not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin'] for c in pending):raise RuntimeError('executable npubin required')
                compiled.extend(pending)
            return list(compiled)
        def run():
            compile_only()
            with guard(device):
                for kernel,args in calls:kernel[grid](*args,**flags)
            return out
        if arm in self.state:self.actual.clear()
        state=dict(inputs=inputs,buffers=buffers,out=out);self.state[arm]=state
        def close():
            nonlocal closed,out
            if closed:return
            closed=True;calls.clear();compiled.clear();inputs.clear();buffers.clear();scratch.clear();out=None;state.clear()
            if self.state.get(arm) is state:self.state.pop(arm);self.actual.pop(arm,None)
        return SimpleNamespace(run=run,close=close,compile_only=compile_only,compiled=lambda:list(compiled),compilation_state=lambda:identity(compiled),sources=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('dsa_indexer_source.py','dsa_indexer_kernels.py')]],expected_kernels=[kernel.fn.__name__ for kernel,_ in calls],metadata=dict(arm=arm,route='optimized' if arm=='after' else 'reference',component='prepared complete DSA score',prepared_device=str(device),grid=grid,flags=flags,BLOCK_SIZE_N=128,device_kernel_calls=len(calls),physical=physical,reused_outputs=True,scope='all QK/ReLU/weight/headsum stages charged; private allocations/initial copies outside both APIs; output/scratch reused; no per-profile IO checks'))

    def validate(self,arm,actual,fixture,spec):
        s=self.state[arm];e=fixture['expected'];p=parameters(spec)
        if not isinstance(actual,torch.Tensor) or actual is not s['out'] or actual.dtype!=torch.float32 or actual.shape!=e.shape or not actual.is_contiguous():return dict(passed=False,reason='one private contiguous FP32 score required')
        value=actual.detach().cpu();bad=(value!=e)|~torch.isfinite(value);coords=bad.nonzero().tolist();first=[dict(coordinate=c,actual=repr(float(value[tuple(c)])),expected=float(e[tuple(c)]),actual_bits=int(value.view(torch.int32)[tuple(c)]),expected_bits=int(e.view(torch.int32)[tuple(c)])) for c in coords[:8]]
        immutable=all(bits(x,fixture['inputs'][name]) for name,x in s['inputs'].items());canary=all(bool((x[:32].cpu()==1024).all() and (x[-32:].cpu()==1024).all()) for x in s['buffers']);private=all(actual.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in s['inputs'].values())
        self.actual[arm]=value.clone();other='after' if arm=='before' else 'before';paired=other not in self.actual or torch.equal(value,self.actual[other]);paired_bits=None if other not in self.actual else bits(value,self.actual[other])
        return dict(passed=bool(not coords and immutable and canary and private and paired),mismatch_count=len(coords),first_mismatches=first,nonfinite_count=int((~torch.isfinite(value)).sum()),max_FP64_error=float((value.double()-fixture['expected_fp64']).abs().max()),immutable_inputs=immutable,canaries=canary,private_output=private,paired_nonzero_bits_numeric_zeros=paired if other in self.actual else None,paired_all_bits_secondary=paired_bits,CPU_zero_sign_secondary=bits(value,e))


def factory():return Adapter()


def main():
    import argparse,json,sys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--correctness-only',action='store_true');a=parser.parse_args();spec=json.loads(Path(a.case).read_text());parameters(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    provenance=json.loads(Path(a.provenance).read_text());provenance['qualification']='exploratory CLI, not frozen all-case collector'
    return 0 if run_pair(Adapter(),spec,a.output,a.cache,provenance,correctness_only=a.correctness_only)['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
