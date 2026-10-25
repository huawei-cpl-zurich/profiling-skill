"""AWQ packed expansion versus educational output lanes; prepared component."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from awq_expansion_contract import CASE_SOURCES,dimensions,make_case,validate_inputs,check_output


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):self.prepared={};self.bits={};self.pair_key=None

    def _device_item(self,arm,spec,inputs,probe=False):
        if arm not in ('before','after'):raise ValueError('invalid arm')
        k,n,g=dimensions(spec)
        import torch
        import triton
        from awq_expansion_kernel import awq_lane_kernel,awq_dequantize_kernel
        storage=torch.empty(k*n+2,dtype=torch.float16,device=inputs['Q'].device)
        out=storage[1:-1].view(k,n)
        if not probe:storage[0]=-701.;storage[-1]=-701.
        kernel=awq_lane_kernel if arm=='before' else awq_dequantize_kernel
        grid=((n//8+15)//16,(k+15)//16)
        flags=dict(BLOCK_SIZE_X=16,BLOCK_SIZE_Y=16,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        args=(inputs['Q'],inputs['S'],inputs['Z'],g,out,n//8,k);compiled={}
        def compile_only():
            if not compiled:
                pending=kernel.warmup(*args,grid=grid,**flags)
                compiled[0]=pending
            return list(compiled.values())
        def run():
            if probe:raise RuntimeError('input-only compile probe cannot execute')
            compile_only()
            kernel[grid](*args,**flags)
            return out
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for key,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            sources=[Path(__file__),Path(__file__).with_name('awq_expansion_kernel.py'),*CASE_SOURCES],
            expected_kernels=[kernel.fn.__name__],inputs=inputs,output_storage=storage,output=out,
            metadata=dict(arm=arm,grid=grid,flags=flags,api_component_count=1,
                output_tile=[16,128],packed_tile=[16,16],scope='prepared dequantization component; output reused; allocation excluded',
                baseline='educational per-output lanes, not vendor baseline',source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',
                torch_version=torch.__version__,triton_version=triton.__version__))

    def probe(self,arm,spec):
        k,n,g=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch
        import torch_npu
        inputs={name:torch.empty(shape,dtype=dtype,device='npu') for name,shape,dtype in
            [('Q',(k,n//8),torch.int32),('Z',(k//g,n//8),torch.int32),('S',(k//g,n),torch.float16)]}
        return self._device_item(arm,spec,inputs,probe=True)

    def prepare(self,arm,spec,fixture,output):
        dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        validate_inputs(fixture['inputs'],spec)
        import torch_npu
        pair_key=json.dumps(spec,sort_keys=True)
        if self.pair_key!=pair_key or arm in self.prepared:self.prepared={};self.bits={};self.pair_key=pair_key
        item=self._device_item(arm,spec,{name:v.npu() for name,v in fixture['inputs'].items()})
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];result=check_output(actual,fixture['expected'])
        canaries=bool((item.output_storage[[0,-1]].cpu()==-701).all())
        immutable=all(torch.equal(v.cpu().view(torch.uint8),fixture['inputs'][name].view(torch.uint8)) for name,v in item.inputs.items())
        private=isinstance(actual,torch.Tensor) and all(actual.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in item.inputs.values())
        pair=None
        if result['passed']:
            self.bits[arm]=actual.detach().cpu().view(torch.int16).clone()
            if len(self.bits)==2:pair=torch.equal(self.bits['before'],self.bits['after'])
        result.update(canaries=canaries,immutable=immutable,nonalias=private,paired_output_bits=pair)
        result['passed']=result['passed'] and canaries and immutable and private and pair is not False
        return result


def factory():return Adapter()
