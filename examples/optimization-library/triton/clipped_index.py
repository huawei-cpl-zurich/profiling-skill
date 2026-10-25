"""Flat divmod versus exact source 2D clipped forward, prepared output reuse."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from clipped_index_contract import CASE_SOURCES,dimensions,plan,make_case,validate_input,check_output


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):
        self.prepared={};self.bits={};self.pair_key=None

    def _device_item(self,arm,spec,gup,probe=False):
        if arm not in ('before','after'):raise ValueError('invalid arm')
        _,m,k=dimensions(spec);layout=plan(m,k)
        import torch
        import triton
        from clipped_index_kernel import _clipped_swiglu_kernel,_clipped_swiglu_flat_kernel
        storage=torch.empty(m*k+2,dtype=torch.float32,device=gup.device)
        out=storage[1:-1].view(m,k)
        if not probe:storage[0]=-701.;storage[-1]=-701.
        kernel=_clipped_swiglu_flat_kernel if arm=='before' else _clipped_swiglu_kernel
        grid=tuple(layout['flat_grid'] if arm=='before' else layout['source_grid'])
        flags=dict(LARGE_TENSOR=layout['int64'],num_stages=1,multibuffer=False,enable_fp_fusion=False)
        if arm=='before':flags['TILE']=layout['tile']
        else:flags.update(BLOCK_M=layout['block_m'],BLOCK_K=layout['block_k'])
        args=(gup,out,m,k,1.702,7.0);compiled={}
        def compile_only():
            if not compiled:compiled[0]=kernel.warmup(*args,grid=grid,**flags)
            return list(compiled.values())
        def run():
            compile_only()
            compiled[0]=kernel[grid](*args,**flags)
            return out
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for key,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            sources=[Path(__file__),Path(__file__).with_name('clipped_index_kernel.py'),*CASE_SOURCES],
            expected_kernels=[kernel.fn.__name__],inputs=gup,output_storage=storage,
            metadata=dict(arm=arm,grid=grid,layout=layout,flags=flags,api_component_count=1,
                          scope='prepared clipped forward; allocation excluded',source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',
                          torch_version=torch.__version__,triton_version=triton.__version__))

    def probe(self,arm,spec):
        """Input-only compile probe: no fixture/oracle, initialization or execution."""
        import torch
        import torch_npu
        _,m,k=dimensions(spec)
        return self._device_item(arm,spec,torch.empty((m,2*k),dtype=torch.float32,device='npu'),probe=True)

    def prepare(self,arm,spec,fixture,output):
        dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        if set(fixture['inputs'])!={'GUP'}:raise ValueError('GUP input required')
        validate_input(fixture['inputs']['GUP'],spec)
        import torch_npu
        pair_key=json.dumps(spec,sort_keys=True)
        if self.pair_key!=pair_key or arm in self.prepared:
            self.prepared={};self.bits={};self.pair_key=pair_key
        item=self._device_item(arm,spec,fixture['inputs']['GUP'].npu())
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];result=check_output(actual,fixture['expected'])
        canaries=bool((item.output_storage[[0,-1]].cpu()==-701).all())
        immutable=torch.equal(item.inputs.cpu().view(torch.uint8),fixture['inputs']['GUP'].view(torch.uint8))
        nonalias=isinstance(actual,torch.Tensor) and actual.untyped_storage().data_ptr()!=item.inputs.untyped_storage().data_ptr()
        pair=None
        if result['passed']:
            self.bits[arm]=actual.detach().cpu().view(torch.int32).clone()
            if len(self.bits)==2:pair=torch.equal(self.bits['before'],self.bits['after'])
        result.update(canaries=canaries,immutable=immutable,nonalias=nonalias,paired_output_bits=pair)
        result['passed']=result['passed'] and canaries and immutable and nonalias and pair is not False
        return result


def factory():return Adapter()
