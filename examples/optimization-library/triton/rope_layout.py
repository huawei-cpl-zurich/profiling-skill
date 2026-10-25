"""Exploratory prepared RoPE layout study; forward only, with source alias semantics."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,torch
from rope_layout_contract import parameters,check_inputs,input_case,make_case,equal


def backend():
    import torch_npu
    from rope_layout_source import _triton_rope
    from rope_layout_kernels import _strided_rope
    return 'npu:0',_triton_rope,_strided_rope


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('rope_layout_contract.py')]
    make_case=staticmethod(make_case)

    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        item=self._prepare(arm,spec,input_case(spec))
        def reject():raise RuntimeError('input-only probe cannot run')
        item.run=reject;item.metadata.update(compile_only_probe=True,device_kernel_calls=0)
        return item

    def prepare(self,arm,spec,fixture,output):
        p=parameters(spec);check_inputs(fixture['inputs'],p);expected=fixture.get('expected')
        if not isinstance(expected,(tuple,list)) or len(expected)!=2 or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32 or v.shape!=fixture['inputs'][n].shape for v,n in zip(expected,('q','k'))):
            raise ValueError('two CPU FP32 expected rotations required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu):
        p=parameters(spec);check_inputs(cpu,p)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,source,strided=backend();inputs={};buffers=[];copies=[];physical=[];aliases=[]
        def allocate(shape,initial=None):
            size=1
            for d in shape:size*=d
            host=torch.full((size+64,),12345.,dtype=torch.float32)
            if initial is not None:host[32:-32].view(shape).copy_(initial)
            buffer=host.to(device);buffers.append(buffer)
            return buffer[32:-32].view(shape)
        for name in ('q','k'):
            b,h,s,d=cpu[name].shape;shape=(b,h,s,d) if p['producer']=='BHSD' else (b,s,h,d)
            initial=cpu[name] if p['producer']=='BHSD' else cpu[name].transpose(1,2)
            backing=allocate(shape,initial);value=backing if p['producer']=='BHSD' else backing.transpose(1,2)
            inputs[name]=value;t=value.transpose(1,2);aliases.append(t.is_contiguous())
            out=t if aliases[-1] else allocate((b,s,h,d));physical.append(out)
            if not aliases[-1]:copies.append((out,t))
        for name in ('cos','sin'):inputs[name]=cpu[name].clone().to(device)
        b,s,hq,hk,d=(p[n] for n in ('B','S','Hq','Hk','D'));power=lambda x:1<<(x-1).bit_length()
        tail=(inputs['cos'],d,inputs['sin'],d,s,b,p['cos_B'],hq,hk,d,power(hq),power(hk),power(d),max(power(hq),power(hk)))
        source_route=arm=='before' or p['producer']=='BSHD-backing';kernel=source if source_route else strided
        args=(physical[0],physical[0].stride(1),physical[1],physical[1].stride(1),*tail) if source_route else (inputs['q'],physical[0].stride(1),inputs['k'],physical[1].stride(1),*physical,*inputs['q'].stride(),*inputs['k'].stride(),*tail)
        flags=dict(BACKWARD_PASS=False,num_stages=1,multibuffer=False,enable_fp_fusion=False);grid=(b*s,);compiled=[]
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,grid=grid,**flags)
                if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin']:raise RuntimeError('missing executable binary')
                compiled.append(c)
            return compiled
        outputs=tuple(v.transpose(1,2) for v in physical)+(inputs['cos'],inputs['sin'])
        def run():
            compile_only()
            if arm=='before':
                for destination,original in copies:destination.copy_(original)
            kernel[grid](*args,**flags);return outputs
        if arm in self.state:self.state.clear();self.actual.clear()
        self.state[arm]=dict(inputs=inputs,buffers=buffers,outputs=outputs,aliases=aliases)
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),compiled=lambda:list(compiled),
            sources=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('rope_layout_source.py','rope_layout_kernels.py')]],
            expected_kernels=[kernel.fn.__name__],metadata=dict(component='prepared source rope_forward',arm=arm,route='reference' if source_route else 'optimized',
            source_aliases=dict(zip(('q','k'),aliases)),charged_torch_copies=len(copies) if arm=='before' else 0,device_kernel_calls=1,
            grid=grid,flags=flags,reused_outputs=True,input_strides={n:list(t.stride()) for n,t in inputs.items()},output_strides=[list(v.stride()) for v in outputs],
            allocator_scope='preallocated inputs/private BSHD outputs outside BOTH APIs; actual Torch copies every before call; no allocator/ordinary latency claim'))

    def validate(self,arm,actual,fixture,spec):
        state=self.state[arm];p=parameters(spec);ok=isinstance(actual,(tuple,list)) and len(actual)==4 and all(isinstance(v,torch.Tensor) for v in actual)
        if not ok:return dict(passed=False,reason='four source outputs required')
        alias_checks={}
        for i,name in enumerate(('q','k')):
            value,owned,original=actual[i],state['outputs'][i],state['inputs'][name];alias=state['aliases'][i]
            h=p['Hq'] if name=='q' else p['Hk'];d=p['D']
            expected_stride=fixture['inputs'][name].stride() if alias else (p['S']*h*d,d,h*d,1)
            ok &= value is owned and value.stride()==expected_stride and equal(value,fixture['expected'][i])
            alias_checks[name]=(value.data_ptr()==original.data_ptr())==alias;ok &= alias_checks[name]
            ok &= equal(original,fixture['expected'][i] if alias else fixture['inputs'][name],bits=not alias)
        for i,name in enumerate(('cos','sin'),2):ok &= actual[i] is state['inputs'][name] and equal(actual[i],fixture['inputs'][name],bits=True)
        for buffer in state['buffers']:ok &= bool((buffer[:32]==12345).all() and (buffer[-32:]==12345).all())
        private=[v.data_ptr() for i,v in enumerate(actual[:2]) if not state['aliases'][i]]
        ok &= len(set(private))==len(private) and all(v not in [t.data_ptr() for t in state['inputs'].values()] for v in private)
        self.actual[arm]=tuple(v.detach().cpu().clone() for v in actual[:2]);other='before' if arm=='after' else 'after';result={}
        if other in self.actual:result['paired_exact_bits']=all(equal(a,b,bits=True) for a,b in zip(self.actual[arm],self.actual[other]))
        result.update(passed=bool(ok),exact_numerical=True,source_alias_checks=alias_checks,source_aliases=state['aliases'],zero_sign_oracle_gate=False)
        return result


def main():
    import argparse,json,sys
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):p.add_argument('--'+name,required=True)
    p.add_argument('--correctness-only',action='store_true');a=p.parse_args();spec=json.loads(Path(a.case).read_text());parameters(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    provenance=json.loads(Path(a.provenance).read_text());provenance['qualification']='exploratory CLI, not frozen all-case collector'
    return 0 if run_pair(Adapter(),spec,a.output,a.cache,provenance,correctness_only=a.correctness_only)['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
