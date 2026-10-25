"""Request metadata row reuse versus educational row/tile grid; prepared component."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from request_metadata_contract import CASE_SOURCES,dimensions,validate_inputs


class Adapter:
    case_sources=CASE_SOURCES
    def make_case(self,*args):raise RuntimeError('R05 qualification failed; scientific payload generation disabled')

    def __init__(self,*,compile_inspection=False):
        self.compile_inspection=compile_inspection;self.prepared={};self.bits={};self.pair_key=None

    def _device_item(self,arm,spec,inputs,probe=False):
        if arm not in ('before','after'):raise ValueError('invalid arm')
        t,w,m,r=dimensions(spec)
        import torch
        import triton
        from request_metadata_kernel import request_tile_kernel,_convert_req_index_to_global_index_kernel
        storage=torch.empty(t*w+2,dtype=torch.int32,device=inputs['R'].device)
        out=storage[1:-1].view(t,w)
        if not probe:storage[0]=-701.;storage[-1]=-701.
        kernel=request_tile_kernel if arm=='before' else _convert_req_index_to_global_index_kernel
        cores=vector_cores()
        grid=(t,w//128) if arm=='before' else (cores,)
        flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        args=(inputs['R'],inputs['B'],inputs['I'],out,inputs.get('P'),inputs.get('W'),m,(t+cores-1)//cores,w//128,t,64,128,spec.get('parameters',spec)['HAS_PREFILL'],m,1,w,1,w,1);compiled={}
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
            sources=[Path(__file__),Path(__file__).with_name('request_metadata_kernel.py'),*CASE_SOURCES],
            expected_kernels=[kernel.fn.__name__],inputs=inputs,output_storage=storage,output=out,
            metadata=dict(arm=arm,grid=grid,flags=flags,api_component_count=1,
                num_vectorcore=cores,rows_per_prog=(t+cores-1)//cores,column_tile=128,block_size=64,scope='prepared exact request mapping; private output reused; allocation excluded',
                baseline='educational row/column tile, not vendor baseline',source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',
                torch_version=torch.__version__,triton_version=triton.__version__))

    def probe(self,arm,spec):
        if not self.compile_inspection:raise RuntimeError('qualification failed; explicit compile_inspection=True required')
        t,w,m,r=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch
        import torch_npu
        pre=spec.get('parameters',spec)['HAS_PREFILL']
        shapes={'R':(t,),'B':(r,m),'I':(t,w)}
        if pre:shapes.update(P=(t,),W=(3,))
        inputs={name:torch.empty(shape,dtype=torch.int32,device='npu') for name,shape in shapes.items()}
        return self._device_item(arm,spec,inputs,probe=True)

    def prepare(self,arm,spec,fixture,output):
        raise RuntimeError('R05 qualification failed on GB before; numeric execution is disabled')

    def _prepare_historical(self,arm,spec,fixture,output):
        # Retained implementation for NONFROZEN mocked regression only; not a public numeric route.
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
            self.bits[arm]=actual.detach().cpu().view(torch.int32).clone()
            if len(self.bits)==2:pair=torch.equal(self.bits['before'],self.bits['after'])
        result.update(canaries=canaries,immutable=immutable,nonalias=private,paired_output_bits=pair)
        result['passed']=result['passed'] and canaries and immutable and private and pair is not False
        return result


def check_output(actual,expected):
    import torch
    if not isinstance(actual,torch.Tensor) or actual.dtype!=torch.int32 or actual.shape!=expected.shape or not actual.is_contiguous():
        return dict(passed=False,reason='one contiguous int32 output required')
    got=actual.detach().cpu()
    mismatch=got!=expected;count=int(mismatch.sum());first=None
    if count:
        index=tuple(int(i) for i in mismatch.nonzero()[0])
        first=dict(index=list(index),actual=int(got[index]),expected=int(expected[index]))
    return dict(passed=count==0,exact_integer_bits=count==0,mismatch_count=count,first_mismatch=first)


def vector_cores():
    from triton.runtime import driver
    count=driver.active.utils.get_device_properties(0)['num_vectorcore']
    if type(count) is not int or count<=0:raise ValueError('invalid queried VEC count')
    return count


def factory():return Adapter()
