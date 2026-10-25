"""Exploratory prepared pack+unpack API; frozen collection requires the all-case controller."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import torch
from pack_prefix_contract import parameters,check_inputs,input_case,make_case,bits_equal


def backend():
    import torch_npu
    import pack_prefix_kernels as k
    return 'npu:0',k


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('pack_prefix_contract.py')]
    make_case=staticmethod(make_case)

    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        inputs=input_case(spec)  # Deterministic inputs only: identical N specialization, no expected payload.
        item=self._prepare(arm,spec,inputs)
        def reject():raise RuntimeError('compile-only probe cannot launch')
        item.run=reject;item.metadata.update(compile_only_probe=True,device_kernel_calls=0)
        return item

    def prepare(self,arm,spec,fixture,output):
        p=parameters(spec);n=check_inputs(fixture['inputs'],p)
        expected=fixture.get('expected');prefix=fixture.get('expected_prefix')
        shapes=((p['B'],p['Lmax'],p['D']),(n,p['D']))
        if not isinstance(expected,(tuple,list)) or len(expected)!=2 or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32 or tuple(v.shape)!=s for v,s in zip(expected,shapes)):
            raise ValueError('both CPU FP32 expected outputs required')
        if not isinstance(prefix,torch.Tensor) or prefix.device.type!='cpu' or prefix.dtype!=torch.int32 or prefix.shape!=(p['B']+1,):
            raise ValueError('cached CPU prefix required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu):
        p=parameters(spec);n=check_inputs(cpu,p)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,k=backend();inputs={name:t.to(device).clone() for name,t in cpu.items()}
        b,l,d=p['B'],p['Lmax'],p['D'];buffers=[];outputs=[]
        for shape in ((b,l,d),(n,d)):
            size=1
            for dim in shape:size*=dim
            buffer=torch.full((size+64,),12345.).to(device);buffers.append(buffer);outputs.append(buffer[32:-32].view(shape))
        prefix_buffer=torch.full((b+65,),-12345,dtype=torch.int32).to(device);starts=prefix_buffer[32:-32]
        grid=(b,(l+63)//64,(d+63)//64);flags=dict(num_warps=4,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        if arm=='before':
            calls=[(k._pack_seq_kernel,grid,(inputs['x'],outputs[0],inputs['lengths'],n,d,l,-9.5,64,64)),
                   (k._unpack_seq_triton_kernel,grid,(outputs[0],outputs[1],inputs['lengths'],b,l,d,64,64))]
        else:
            calls=[(k._exclusive_prefix,(1,),(inputs['lengths'],starts,b,1<<(b-1).bit_length())),
                   (k._pack_seq_kernel_prefix,grid,(inputs['x'],outputs[0],starts,inputs['lengths'],n,d,l,-9.5,64,64)),
                   (k._unpack_seq_triton_kernel_prefix,grid,(outputs[0],outputs[1],starts,inputs['lengths'],b,l,d,64,64))]
        compiled=[]
        def compile_only():
            if not compiled:
                pending=[kernel.warmup(*args,grid=g,**flags) for kernel,g,args in calls]
                for c in pending:
                    if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin']:raise RuntimeError('missing executable binary')
                compiled.extend(pending)  # Atomic: no partial successful qualification survives failure.
            return compiled
        def run():
            compile_only()
            for kernel,g,args in calls:kernel[g](*args,**flags)
            return tuple(outputs)
        if arm in self.state:self.state.clear();self.actual.clear()
        self.state[arm]=dict(inputs=inputs,buffers=buffers,prefix_buffer=prefix_buffer,starts=starts,outputs=outputs)
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),
            compiled=lambda:list(compiled),sources=[Path(__file__),*self.case_sources,Path(__file__).with_name('pack_prefix_kernels.py')],
            expected_kernels=[kernel.fn.__name__ for kernel,_,_ in calls],metadata=dict(arm=arm,component='prepared pack+unpack',
            grid=grid,tiles=[64,64],flags=flags,device_kernel_calls=len(calls),reused_outputs=True,
            prefix_scope='after GPU recomputed every API; B+1 int32 workspace',host_scope='CPU validation/max/sum and all allocations outside both APIs',
            allocator_limit='reused prepared outputs/workspace; no ordinary allocation/API latency claim',input_strides={name:list(t.stride()) for name,t in inputs.items()}))

    def validate(self,arm,actual,fixture,spec):
        state=self.state[arm];p=parameters(spec);ok=isinstance(actual,(tuple,list)) and len(actual)==2 and all(isinstance(v,torch.Tensor) for v in actual)
        result=dict(passed=False)
        if not ok:return dict(passed=False,reason='both packed and roundtrip outputs required')
        for value,owned,expected in zip(actual,state['outputs'],fixture['expected']):
            ok &= value is owned and bits_equal(value,expected)
        pointers=[v.data_ptr() for v in [*state['inputs'].values(),*actual,state['starts']]]
        ok &= len(set(pointers))==len(pointers)
        for name,value in state['inputs'].items():ok &= bits_equal(value,fixture['inputs'][name])
        for buffer in state['buffers']:ok &= bool((buffer[:32]==12345).all() and (buffer[-32:]==12345).all())
        pb=state['prefix_buffer'];ok &= bool((pb[:32]==-12345).all() and (pb[-32:]==-12345).all())
        if arm=='after':
            result['prefix_exact']=bits_equal(state['starts'],fixture['expected_prefix']);ok &= result['prefix_exact']
        self.actual[arm]=tuple(v.detach().cpu().clone() for v in actual)
        other='before' if arm=='after' else 'after'
        if other in self.actual:
            result['paired_exact_bits']=all(bits_equal(a,b) for a,b in zip(self.actual[arm],self.actual[other]));ok &= result['paired_exact_bits']
        result.update(passed=bool(ok),exact_bits=True,arity=2)
        return result


def main():
    import argparse,json,sys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--correctness-only',action='store_true');a=parser.parse_args()
    spec=json.loads(Path(a.case).read_text());parameters(spec)
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    provenance=json.loads(Path(a.provenance).read_text());provenance['qualification']='exploratory CLI'
    return 0 if run_pair(Adapter(),spec,a.output,a.cache,provenance,correctness_only=a.correctness_only)['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
