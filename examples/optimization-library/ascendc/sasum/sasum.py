"""Source-derived native SASUM pair; controlled qualification is separate."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys,weakref
import torch
from sasum_contract import dimensions,make_case as _provider_make_case,check_fixture,tiling,full
from sasum_compile import build,HERE
sys.path.insert(0,str(HERE.parents[3]/'scripts'))
from optimization_native_bridge import NativeArm as BaseNativeArm,digest


def device_objects(b):return b.get('objects',[('device_aiv_elf',b['object'])])


def binary_state(b):return [dict(hash=digest(p),binaries={role:digest(p)}) for role,p in device_objects(b)]


class NativeArm(BaseNativeArm):
    def __init__(self,*args,device=None,**kwargs):
        self.device=device;self.closed=False;super().__init__(*args,**kwargs)
    def context(self):return torch.npu.device(self.device)
    def _launch(self):
        if self.closed:raise RuntimeError('prepared native item is closed')
        with self.context():return super()._launch()
    def run(self):
        if self.closed:raise RuntimeError('prepared native item is closed')
        with self.context():return super().run()
    def close(self):
        if self.closed:return
        self.closed=True;super().close()
        with self.context():self.inputs=[];self.output=None;self._pointers=None


class Prepared(SimpleNamespace):
    def close(self):
        if getattr(self,'closed',False):return
        self.closed=True
        with self.native.context():
            self.native.close();self.stores=[];self.x=None;self.result=None
        owner=self.owner()
        if owner is not None and owner.items.get(self.arm) is self:owner.items.pop(self.arm);owner.numeric.pop(self.arm,None)


class _NumericalAdapter:
    make_case=staticmethod(_provider_make_case)
    case_sources=[HERE/'sasum_contract.py']
    def __init__(self,directory=None,*,compile_inspection=False):self.directory=Path(directory or '.');self.builds={};self.items={};self.current=None;self.numeric={}

    def _build(self,arm):
        if arm not in self.builds:
            candidate=build(arm,self.directory);self.builds[arm]=candidate
        return self.builds[arm]

    def compile_probe(self,arm,spec,output):
        if arm not in ('before','after'):raise ValueError('arm')
        n,inc=dimensions(spec);raw=tiling(n,inc);compiled=[]
        def compile_only():
            if not compiled:
                b=self._build(arm);pending=[SimpleNamespace(hash=digest(p),asm={role:p.read_bytes()},metadata=b['resources']) for role,p in device_objects(b)];compiled.extend(pending)
            return compiled
        def run():raise RuntimeError('input-only native prototype cannot execute')
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:binary_state(self.builds[arm]) if compiled else [],
            expected_kernels=[],sources=[__file__,*self.case_sources,HERE/'sasum_compile.py',HERE/'native.cpp',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())],
            metadata=dict(arm=arm,prototype=True,n=n,incx=inc,grid=[8],tiling_bytes=len(raw),tiling_sha256=hashlib.sha256(raw).hexdigest(),source_declared_UB_bytes=165216+(32 if arm=='after' else 0),automatic_scratch='unknown',ABI=dict(pointer_count=4,input_span_floats=(n-1)*inc+1,workspace_bytes=1024,logical_result_floats=1,alignment=128)))

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'):raise ValueError('arm')
        check_fixture(spec,fixture);n,inc=dimensions(spec);probe=self.compile_probe(arm,spec,output);probe.compile_only();b=self.builds[arm]
        import torch_npu
        device=torch.npu.current_device()
        with torch.npu.device(device):
            # Transfer full backing, preserving real stride2 holes and 128B halos.
            backing=full(fixture['x']).to('npu:'+str(device));x=backing[32:-32]
            def guarded(size):
                store=torch.full((size+64,),12345.,dtype=torch.float32,device='npu:'+str(device));return store,store[32:-32]
            result_store,result=guarded(1);workspace_store,workspace=guarded(256)
            tile=torch.tensor(list(tiling(n,inc)),dtype=torch.uint8,device='cpu').to('npu:'+str(device))
            for t in (x,result,workspace):
                if t.data_ptr()%128:raise ValueError('128B pointer alignment required')
            native=NativeArm(b['binary'],[x,workspace,tile],result,sources=probe.sources+b['sources'],source_revision='c2332551586763f8c2d27621f8bc119bd1c3a301',build_manifest=b['manifest'],stream_provider=lambda:torch.npu.current_stream().npu_stream,device=device)
            # Runtime reports actual SDK registration, which can carry a mix_aiv suffix.
            native.expected_kernels=b['selectors']
            item=Prepared(owner=weakref.ref(self),arm=arm,run=native.run,compile_only=probe.compile_only,compilation_state=lambda:binary_state(b),sources=native.sources,expected_kernels=b['selectors'],
                metadata=dict(probe.metadata,prototype=False,prepared_device=device,prepared_output_reused=True,allocator_cost_excluded=True,selected_device_object_sha256=digest(b['object']),native_resources=b['resources'],native_binding=native.metadata),
                result=result,stores=[result_store,workspace_store],x=x,native=native,spec=spec)
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.items={};self.numeric={};self.current=key
            self.items[arm]=item;return item

    def validate(self,arm,actual,fixture,spec):
        with self.items[arm].native.context():return self._validate(arm,actual,fixture,spec)
    def _validate(self,arm,actual,fixture,spec):
        item=self.items[arm]
        if item.spec!=spec or not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32 or actual.shape!=(1,) or actual.device!=item.result.device or actual.data_ptr()!=item.result.data_ptr():raise ValueError('scalar output ABI')
        result=item.stores[0].detach().cpu();host=result[32:-32];expected=fixture['expected'];exact=bool(torch.equal(host.view(torch.int32),expected.view(torch.int32))) if bool(expected!=0) else bool(host==0)
        immutable=bool(torch.equal(full(item.x).cpu().view(torch.int32),full(fixture['x']).view(torch.int32)))
        canaries=True
        for i,store in enumerate(item.stores):
            h=result if i==0 else store.detach().cpu();canaries &= bool((h[:32]==12345).all() and (h[-32:]==12345).all())
        nonalias=len({t.untyped_storage().data_ptr() for t in (item.x,*item.stores)})==3
        other=self.numeric.get('after' if arm=='before' else 'before');self.numeric[arm]=host.clone()
        return dict(passed=exact and immutable and canaries and nonalias,mismatch_count=0 if exact else 1,first_mismatches=[] if exact else [dict(coordinates=[0],expected=float(expected[0]),actual=float(host[0]))],max_absolute_error=float((host-expected).abs().max()),nonfinite_count=int((~torch.isfinite(host)).sum()),paired_bitwise_secondary=None if other is None else bool(torch.equal(host.view(torch.int32),other.view(torch.int32))),zero_sign_secondary=int(host.view(torch.int32)[0]) if bool(expected==0) else None,inputs_immutable=immutable,canaries=canaries,nonalias=nonalias)


FAILURE=('P04 is numerically qualified on all twelve ASC arm/case gates, but profile accounting '
         'is unavailable: MIX_AIV compute tasks lack explicit CPU/NPU correlation. '
         'Full performance qualification is incomplete; public numeric execution is disabled. '
         'See experiments/optimization-wave010/P04-result.json.')


class StudyUnqualified(ValueError):pass


class Adapter(_NumericalAdapter):
    def __init__(self,directory=None,*,compile_inspection=False):
        super().__init__(directory);self.compile_inspection=compile_inspection
    @staticmethod
    def make_case(spec):raise StudyUnqualified(FAILURE)
    def prepare(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def validate(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def capture(self,*args,**kwargs):raise StudyUnqualified(FAILURE)
    def compile_probe(self,arm,spec,output):
        if not self.compile_inspection:raise StudyUnqualified(FAILURE)
        return super().compile_probe(arm,spec,output)


def make_case(spec):raise StudyUnqualified(FAILURE)


def factory():raise StudyUnqualified(FAILURE)


def main():
    import argparse
    p=argparse.ArgumentParser(description=FAILURE)
    for k in ('case','output','cache','provenance'):p.add_argument('--'+k,required=True)
    p.add_argument('--correctness-only',action='store_true');p.parse_args()
    factory() # Reject before files, cache, expected generation, backend imports or input movement.


if __name__=='__main__':raise SystemExit(main())
