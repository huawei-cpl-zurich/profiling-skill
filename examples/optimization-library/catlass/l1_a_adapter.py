"""Exploratory prepared native FullLoadA pair; compile/resource screen before math."""
from pathlib import Path
from types import SimpleNamespace
import torch,hashlib
from l1_a_contract import parameters,check_inputs,make_case,bits
import l1_a_native as native_backend
from l1_a_native import backend,Native,inspect_build
from l1_a_build import BASE,sources,sha


class Adapter:
    case_sources=[BASE/'l1_a_contract.py'];make_case=staticmethod(make_case)
    def __init__(self,directory=None):self.directory=Path(directory or Path.cwd()/'.catlass-r12-build');self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        p=parameters(spec);compiled=[]
        def compile_only():
            if not compiled:compiled.append(inspect_build(arm,p,self.directory/arm))
            return compiled[0]
        def reject():raise RuntimeError('input-only native compile probe cannot run')
        return SimpleNamespace(run=reject,compile_only=compile_only,compilation_state=lambda:compiled[:],sources=[*sources(),Path(__file__),Path(native_backend.__file__),*self.case_sources],metadata=dict(arm=arm,compile_only_probe=True,device_kernel_calls=0,parameters=p))

    def prepare(self,arm,spec,fixture,output):
        p=parameters(spec);check_inputs(fixture['inputs'],p);e=fixture.get('expected')
        if not isinstance(e,(tuple,list)) or len(e)!=2 or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.shape!=(p['M'],p['N']) or v.dtype!=d for v,d in zip(e,(torch.float64,torch.float16))):raise ValueError('independent CPU FP64 and rounded FP16 expected matrices required')
        return self._prepare(arm,spec,fixture['inputs'])

    def _prepare(self,arm,spec,cpu,inspection=False):
        p=parameters(spec)
        if not inspection:check_inputs(cpu,p)
        if arm not in ('before','after'):raise ValueError('before/after required')
        inspection_result=inspect_build(arm,p,self.directory/arm)
        expected_names=[inspection_result['entrypoints']['host_registered_selector']]
        device,stream=backend();buffers=[];inputs={}
        def allocate(shape,value=None):
            size=1
            for n in shape:size*=n
            host=torch.full((size+64,),1024.,dtype=torch.float16)
            if value is not None:host[32:-32].view(shape).copy_(value)
            with native_backend.device_guard(device):tensor=host.to(device)
            buffers.append(tensor);return tensor[32:-32].view(shape)
        for key in ('a','b'):inputs[key]=allocate(cpu[key].shape,None if inspection else cpu[key])
        out=allocate((p['M'],p['N']));native=[];closed=False
        def compile_only():
            if closed:raise RuntimeError('prepared item is closed')
            if not native:native.append(Native(arm,p,[inputs['a'],inputs['b']],out,self.directory/arm,stream))
            return native[0].manifest
        def run():compile_only();return native[0].run()
        if arm in self.state:self.actual.clear()
        state=dict(inputs=inputs,out=out,buffers=buffers,native=native);self.state[arm]=state
        def close():
            nonlocal out,closed
            if closed:return
            closed=True
            try:
                for context in native:context.close()
            finally:
                native.clear();inputs.clear();buffers.clear();out=None;state.clear()
                if self.state.get(arm) is state:self.state.pop(arm);self.actual.pop(arm,None)
        return SimpleNamespace(run=run,close=close,compile_only=compile_only,compilation_state=lambda:native_backend.binary_identity(native[0].manifest) if native else [],
            sources=[*sources(),Path(__file__),Path(native_backend.__file__),*self.case_sources],expected_kernels=expected_names,
            metadata=dict(native_compile_abi_resources=inspection_result,arm=arm,component='native source DeviceGemm FullLoadA',prepared_device=str(device),grid=24,L1=[128,256,256],L0=[128,256,64],stages=2,unit_flag=True,device_kernel_calls=1,reused_outputs=True,workspace_bytes=0,
            scope='opaque source DeviceGemm Params initializedonce; currentACLstream eachAPI; noH2D/sync/reset/allocator insidepreparedAPI; allactualAIC/AIVtasktypescharged'))

    def validate(self,arm,actual,fixture,spec):
        s=self.state[arm];exact,expected=fixture['expected'];p=parameters(spec)
        if not isinstance(actual,torch.Tensor) or actual is not s['out'] or actual.shape!=expected.shape or actual.dtype!=torch.float16 or actual.stride()!=(p['N'],1):return dict(passed=False,reason='private contiguous FP16 matrix required')
        value=actual.detach().cpu();bad=(value!=expected)|~torch.isfinite(value);coords=bad.nonzero().tolist();first=[dict(coordinate=c,actual=repr(float(value[tuple(c)])),expected=float(expected[tuple(c)]),actual_bits=int(value.view(torch.int16)[tuple(c)]),expected_bits=int(expected.view(torch.int16)[tuple(c)])) for c in coords[:8]]
        immutable=all(bits(t,fixture['inputs'][k]) for k,t in s['inputs'].items());canary=all(bool((b[:32].cpu()==1024).all() and (b[-32:].cpu()==1024).all()) for b in s['buffers'])
        private=all(actual.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for t in s['inputs'].values());self.actual[arm]=value.clone();other='before' if arm=='after' else 'after';paired=other not in self.actual or torch.equal(value,self.actual[other])
        return dict(passed=bool(not coords and immutable and canary and private and paired),mismatch_count=len(coords),first_mismatches=first,nonfinite_count=int((~torch.isfinite(value)).sum()),max_fp64_forward_error=float((value.double()-exact).abs().max()),immutable_inputs=immutable,canaries=canary,private_output=private,paired_nonzero_bits=paired if other in self.actual else None,zero_sign_bits_secondary=bits(value,expected))


def factory():return Adapter()


def main():
    import argparse,json,sys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output'):parser.add_argument('--'+name,required=True)
    for name in ('cache','provenance','build-directory'):parser.add_argument('--'+name)
    parser.add_argument('--inspect-compile',action='store_true');parser.add_argument('--arm',choices=('before','after'),default='before');parser.add_argument('--correctness-only',action='store_true');a=parser.parse_args()
    spec=json.loads(Path(a.case).read_text());parameters(spec);adapter=Adapter(a.build_directory or Path(a.output)/'native-build')
    if a.inspect_compile:adapter.compile_probe(a.arm,spec,a.output).compile_only();return 0
    if not a.cache or not a.provenance:parser.error('exploratory numeric CLI requires cache and provenance')
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    provenance=json.loads(Path(a.provenance).read_text());provenance['qualification']='exploratory CLI; not frozen all-case native collector'
    return 0 if run_pair(adapter,spec,a.output,a.cache,provenance,correctness_only=a.correctness_only)['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
