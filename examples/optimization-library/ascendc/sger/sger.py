"""Prepared column-major SGER; private A and product dispatcher/C ABI."""
import ctypes,hashlib,json,sys,uuid,weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch
from sger_contract import dimensions,make_case,check_inputs,check_output,tiling
from sger_compile import HERE,PIN,build
sys.path.insert(0,str(HERE.parents[3]/'scripts'))
from optimization_native_bridge import audit_tensor,digest
_LAUNCHES=weakref.WeakValueDictionary();_REG=[]


class Native:
    def __init__(self,b,inputs,outputs,raw,stream,device=None):
        self.closed=False;self.device=device
        if len(raw)!=56:raise ValueError('56B host tiling required')
        if digest(b['binary'])!=b['manifest']['binary_sha256']:raise ValueError('binary identity mismatch')
        self.inputs=inputs;self.outputs=outputs;self.stream=stream;self.tile=ctypes.create_string_buffer(raw);self.library=ctypes.CDLL(str(b['binary']))
        if ctypes.addressof(self.tile)%8:raise ValueError('host tiling alignment')
        self.library.catlass_bridge_abi_version.restype=ctypes.c_uint32
        if self.library.catlass_bridge_abi_version()!=1:raise ValueError('native ABI version')
        self.function=self.library.catlass_bridge_launch;self.function.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint64,ctypes.c_void_p];self.function.restype=ctypes.c_int
        self.layouts=[audit_tensor(t) for t in (*inputs,*outputs)]
        if any(t.device!=outputs[0].device for t in (*inputs,*outputs)):raise ValueError('device mismatch')
        self.pointers=(ctypes.c_void_p*6)(*[t.data_ptr() for t in (*inputs,*outputs)],ctypes.addressof(self.tile))
        if not _REG:
            lib=torch.library.Library('optimization_sger','DEF');lib.define('launch(Tensor[] originals, Tensor(a!) a, Tensor(b!) product, str token) -> ()')
            def launch(originals,x,y,token):_LAUNCHES[token]._launch()
            for backend in ('CPU','PrivateUse1'):lib.impl('launch',launch,backend)
            _REG.append(lib)
        self.token=uuid.uuid4().hex;_LAUNCHES[self.token]=self
    def context(self):return torch.npu.device(self.device) if self.device is not None else nullcontext()
    def _launch(self):
        if self.closed:raise RuntimeError('prepared native item is closed')
        with self.context():code=self.function(self.pointers,6,self.stream())
        if code:raise RuntimeError('native SGER ABI launch status '+str(code))
    def run(self):
        if self.closed:raise RuntimeError('prepared native item is closed')
        torch.ops.optimization_sger.launch(list(self.inputs),*self.outputs,self.token)
        return self.outputs[0]
    def close(self):
        if self.closed:return
        self.closed=True;_LAUNCHES.pop(self.token,None)
        with self.context():self.inputs=();self.outputs=();self.pointers=None;self.tile=None


class Prepared(SimpleNamespace):
    def close(self):
        if getattr(self,'closed',False):return
        self.closed=True
        with self.native.context():
            self.native.close();self.stores=();self.originals=();self.outputs=()
        owner=self.owner()
        if owner is not None and owner.items.get(self.arm) is self:owner.items.pop(self.arm);owner.host.pop(self.arm,None)


def runtime_cores(device):
    import triton.runtime.driver as driver
    actual=driver.active.utils.get_device_properties(device)
    if actual.get('num_vectorcore')!=48:raise RuntimeError('fixed prepared policy requires actual48 vector cores')
    return actual


class Adapter:
    make_case=staticmethod(make_case);case_sources=[HERE/'sger_contract.py']
    def __init__(self,directory=None):self.directory=Path(directory or '.');self.builds={};self.items={};self.host={};self.current=None
    def compile_probe(self,arm,spec,output=None):
        m,n,lda,use,cols=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('arm')
        compiled=[]
        def compile_only():
            if not compiled:
                b=self.builds.get(arm)
                if b is None:
                    b=build(arm,self.directory);self.builds[arm]=b
                pending=[SimpleNamespace(hash=digest(obj),asm={kind:obj.read_bytes()},metadata=b['resources']) for kind,obj in b.get('objects',[('device_vec_o',b['object'])])]
                pending[0].asm['host_shared_so']=b['binary'].read_bytes();compiled.extend(pending)
            return list(compiled)
        def run():raise RuntimeError('input-only native probe cannot execute')
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:[dict(hash=a.hash,binaries={k:hashlib.sha256(v).hexdigest() for k,v in a.asm.items()}) for a in compiled],
            sources=[Path(__file__),*self.case_sources,HERE/'sger_compile.py',HERE/'native.cpp',*sorted((HERE/'vendor').glob('*'))],expected_kernels=[],
            metadata=dict(arm=arm,m=m,n=n,lda=lda,grid=[use],fixed_prepared_policy_AIV=48,source_host_helper='UNKNOWN',tiling_bytes=56,prototype_tiling_sha256=hashlib.sha256(tiling(spec)).hexdigest(),source_UB_bytes=196608,compiler_managed_scratch='UNKNOWN if unreported',api_component_count=3 if arm=='before' else 2,route='optimized'))
    def capture(self,run,item,output):
        sys.path.insert(0,str(HERE.parents[3]/'skills/ascend-model-benchmarking/scripts'))
        from profile_ten import capture_ten
        with item.native.context():return capture_ten(run,output,expected_kernel=item.expected_kernels,metric_group='PipeUtilization')

    def prepare(self,arm,spec,fixture,output):
        check_inputs(spec,fixture['inputs'])
        if arm not in ('before','after'):raise ValueError('arm')
        import torch_npu
        device=torch.npu.current_device()
        with torch.npu.device(device):
            actual=runtime_cores(device)
            probe=self.compile_probe(arm,spec,output);probe.compile_only();b=self.builds[arm]
            originals=tuple(fixture['inputs'][name].to('npu:'+str(device)) for name in ('A','X','Y'));size=originals[0].numel()
            stores=tuple(torch.full((size+64,),12345.,dtype=torch.float32,device='npu:'+str(device)) for _ in range(2));outputs=tuple(x[32:-32] for x in stores)
            if any(t.data_ptr()%128 for t in outputs):raise ValueError('128B working alignment required')
            raw=tiling(spec,outputs[0].data_ptr(),originals[1].data_ptr(),originals[2].data_ptr());native=Native(b,originals,outputs,raw,lambda:torch.npu.current_stream().npu_stream,device=device)
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.items={};self.host={};self.current=key
            item=Prepared(owner=weakref.ref(self),arm=arm,run=native.run,compile_only=probe.compile_only,compilation_state=probe.compilation_state,sources=probe.sources+b['sources'],expected_kernels=b['selectors'],
                metadata=dict(probe.metadata,prototype=False,prepared_device=device,actual_driver_properties=actual,prepared_private_outputs_reused=True,allocator_excluded=True,tiling_sha256=hashlib.sha256(raw).hexdigest(),ABI_layouts=native.layouts,source_revision=PIN),stores=stores,originals=originals,outputs=outputs,native=native)
            self.items[arm]=item;return item
    def validate(self,arm,actual,fixture,spec):
        with self.items[arm].native.context():return self._validate(arm,actual,fixture,spec)
    def _validate(self,arm,actual,fixture,spec):
        item=self.items[arm]
        if not isinstance(actual,torch.Tensor) or actual.data_ptr()!=item.outputs[0].data_ptr() or actual.dtype!=torch.float32 or actual.shape!=item.outputs[0].shape or actual.device!=item.outputs[0].device or not actual.is_contiguous():return dict(passed=False,reason='private full A output ABI')
        full=tuple(s.detach().cpu() for s in item.stores);host=full[0][32:-32];result=check_output(host,fixture['expected'])
        halo=all(bool((s[:32]==12345).all() and (s[-32:]==12345).all()) for s in full)
        immutable=all(torch.equal(a.cpu().view(torch.int32),fixture['inputs'][name].view(torch.int32)) for a,name in zip(item.originals,('A','X','Y')))
        private=len({x.untyped_storage().data_ptr() for x in (*item.originals,*item.outputs)})==5
        other=self.host.get('after' if arm=='before' else 'before');self.host[arm]=host.clone()
        result.update(canaries=halo,inputs_immutable=immutable,nonalias=private,paired_bits_secondary=None if other is None else torch.equal(host.view(torch.int32),other.view(torch.int32)))
        result['passed']=result['passed'] and halo and immutable and private;return result



def factory():return Adapter()


def main():
    import argparse
    p=argparse.ArgumentParser(description='Bounded prepared SGER component, not vendor host API')
    for n in ('case','output','cache','provenance'):p.add_argument('--'+n,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();spec=json.loads(Path(a.case).read_text());dimensions(spec)
    if a.validate_only:print(json.dumps(dict(status='admitted',role=spec.get('role'),qualification='exploratory')));return 0
    from optimization_pair import run_pair
    report=run_pair(Adapter(Path(a.output).parent/'native'),spec,a.output,a.cache,json.loads(Path(a.provenance).read_text()))
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
