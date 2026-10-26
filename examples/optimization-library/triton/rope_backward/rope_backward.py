"""Exploratory bounded prepared RoPE backward; frozen collector qualifies all cases."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import hashlib,torch
from rope_backward_contract import parameters,shapes,check_inputs,make_case as _make_case,bits


def backend():
    import torch_npu
    import rope_backward_kernels as k
    return 'npu:0',k


def guard(device):return torch.npu.device(device) if str(device).split(':')[0]=='npu' else nullcontext()


def physical(t):
    if not t.is_contiguous() or t.storage_offset()+t.numel()>t.untyped_storage().nbytes()//t.element_size():raise ValueError('contiguous physical storage span required')
    fmt=None
    if t.device.type=='npu':
        import torch_npu
        fmt=torch_npu.get_npu_format(t)
        if fmt!=2 or t.data_ptr()%128:raise ValueError('128B aligned physical ND format2 required')
    return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),format=fmt)


def identity(compiled):
    return [dict(hash=str(c.hash),sha256=hashlib.sha256(c.asm['npubin']).hexdigest(),format='Triton-Ascend-npubin',identity_source='compiled executable npubin',name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class _MeasuredAdapter:
    case_sources=[Path(__file__).with_name('rope_backward_contract.py')];make_case=staticmethod(_make_case)
    def __init__(self):self.state={};self.actual={}
    def compile_probe(self,arm,spec,output):
        item=self._prepare(arm,spec,None)
        def reject():raise RuntimeError('input-only compile probe cannot run')
        item.run=reject;item.metadata.update(compile_only_probe=True,device_kernel_calls=0);return item
    def prepare(self,arm,spec,fixture,output):
        p=parameters(spec);check_inputs(fixture['inputs'],p);e=fixture.get('expected');f=fixture.get('expected_fp64');expected_shapes=[shapes(p)['dy'],shapes(p)['cos'],shapes(p)['sin']]
        for value,dtypes in [(e,(torch.float16,torch.float32,torch.float32)),(f,(torch.float64,)*3)]:
            if not isinstance(value,(tuple,list)) or len(value)!=3 or any(not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.shape!=s or x.dtype!=d for x,s,d in zip(value,expected_shapes,dtypes)):raise ValueError('three independent CPU expected gradient tensors required')
        return self._prepare(arm,spec,fixture['inputs'])
    def _prepare(self,arm,spec,cpu):
        p=parameters(spec)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,k=backend();buffers=[];inputs={};compiled=[];closed=False
        def allocate(shape,dtype,value=None):
            size=1
            for n in shape:size*=n
            halo=128//torch.empty((),dtype=dtype).element_size();host=torch.full((size+2*halo,),1024.,dtype=dtype)
            if value is not None:host[halo:-halo].view(shape).copy_(value)
            with guard(device):buffer=host.to(device)
            buffers.append((buffer,halo));return buffer[halo:-halo].view(shape)
        for name,shape in shapes(p).items():inputs[name]=allocate(shape,torch.float16 if name in ('dy','x_rope') else torch.float32,None if cpu is None else cpu[name])
        outputs=(allocate(shapes(p)['dy'],torch.float16),allocate(shapes(p)['cos'],torch.float32),allocate(shapes(p)['sin'],torch.float32));b,s,h,r,n=(p[x] for x in ('B','S','H','half','nope_dim'));br=min(h,64) if arm=='before' else h;total=b*s*h;flags=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)
        kernel=k.rope_backward_atomic if arm=='before' else k.rope_backward_owned;args=(*[inputs[x] for x in ('dy','x_rope','cos','sin')],*outputs,total,s,h,n,2*r,r,br,64)
        with guard(device):layouts={name:physical(t) for name,t in inputs.items()};layouts.update({name:physical(t) for name,t in zip(('dx','dcos','dsin'),outputs)})
        calls=[(kernel,args,(total//br,))]
        if arm=='before':calls.insert(0,(k.rope_backward_zero,(outputs[1],outputs[2],b*s*r,256),((b*s*r+255)//256,)))
        def compile_only():
            if closed:raise RuntimeError('prepared item is closed')
            if not compiled:
                with guard(device):pending=[fn.warmup(*args,grid=grid,**flags) for fn,args,grid in calls]
                if any(not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin'] for c in pending):raise RuntimeError('actual executable npubin required')
                compiled.extend(pending)
            return list(compiled)
        def run():
            compile_only()
            with guard(device):
                for fn,args,grid in calls:fn[grid](*args,**flags)
            return outputs
        if arm in self.state:self.actual.clear()
        state=dict(inputs=inputs,outputs=outputs,buffers=buffers);self.state[arm]=state
        item=SimpleNamespace(run=run,outputs=outputs,compile_only=compile_only,compiled=lambda:list(compiled),compilation_state=lambda:identity(compiled),expected_kernels=[fn.fn.__name__ for fn,_,_ in calls],sources=[Path(__file__),*self.case_sources,Path(__file__).with_name('rope_backward_kernels.py')],metadata=dict(arm=arm,prepared_device=device,route='optimized' if arm=='after' else 'reference',component='complete saved-pre-rotation backward dx/dcos/dsin',flags=flags,BLOCK_ROWS=br,BLOCK_NOPE=64,grids=[list(grid) for _,_,grid in calls],device_kernel_calls=len(calls),physical=layouts,reused_outputs=True,scope='before charged coefficientzero+atomicbackward, afterownedoverwrite; all dx/prefix work; setup/allocations outside APIs; no profileIO'))
        def close():
            nonlocal closed,outputs
            if closed:return
            closed=True;calls.clear();compiled.clear();inputs.clear();buffers.clear();outputs=();item.outputs=();state.clear()
            if self.state.get(arm) is state:self.state.pop(arm);self.actual.pop(arm,None)
        item.close=close;return item
    def capture(self,run,item,output):
        import sys
        helpers=Path(__file__).resolve().parents[4]/'skills/ascend-model-benchmarking/scripts'
        if str(helpers) not in sys.path:sys.path.insert(0,str(helpers))
        from profile_ten import capture_ten
        with guard(item.metadata['prepared_device']):return capture_ten(run,output,expected_kernel=item.expected_kernels,metric_group='PipeUtilization')
    def validate(self,arm,actual,fixture,spec):
        state=self.state[arm];e=fixture['expected']
        with guard(str(state['outputs'][0].device)):
            if not isinstance(actual,(tuple,list)) or len(actual)!=3 or any(not isinstance(x,torch.Tensor) or x is not out or x.dtype!=expected.dtype or x.shape!=expected.shape for x,out,expected in zip(actual,state['outputs'],e)):return dict(passed=False,reason='three private full gradient tensors required')
            values=tuple(x.detach().cpu() for x in actual);checks=[]
            for name,x,expected,exact in zip(('dx','dcos','dsin'),values,e,fixture['expected_fp64']):
                bad=(x!=expected)|~torch.isfinite(x);coords=bad.nonzero().tolist();checks.append(dict(name=name,mismatch_count=len(coords),first_mismatches=[dict(coordinate=c,actual=repr(float(x[tuple(c)])),expected=float(expected[tuple(c)])) for c in coords[:8]],max_FP64_error=float((x.double()-exact).abs().max()),zero_sign_bits_secondary=bits(x,expected)))
            immutable=all(bits(x,fixture['inputs'][name]) for name,x in state['inputs'].items());canary=all(bool((buf[:halo].cpu()==1024).all() and (buf[-halo:].cpu()==1024).all()) for buf,halo in state['buffers']);private=len({x.untyped_storage().data_ptr() for x in (*actual,*state['inputs'].values())})==7
            other=self.actual.get('after' if arm=='before' else 'before');paired=other is None or all(torch.equal(x,y) for x,y in zip(values,other));self.actual[arm]=tuple(x.clone() for x in values)
            return dict(passed=bool(not any(c['mismatch_count'] for c in checks) and immutable and canary and private and paired),outputs=checks,immutable_inputs=immutable,canaries=canary,private_outputs=private,paired_numeric=paired if other is not None else None,paired_bits_secondary=None if other is None else all(bits(x,y) for x,y in zip(values,other)))


def _quarantine(*args,**kwargs):raise RuntimeError('RoPE backward study is unqualified: full-head H256 compile exceeds UB; numerical/profile dispatch quarantined')


make_case=_quarantine


class Adapter:
    case_sources=_MeasuredAdapter.case_sources;make_case=staticmethod(_quarantine)
    prepare=validate=capture=staticmethod(_quarantine)
    def compile_probe(self,arm,spec,output,*,inspect_compile=False):
        if not inspect_compile:return _quarantine()
        return _MeasuredAdapter().compile_probe(arm,spec,output)


def factory():return Adapter()
