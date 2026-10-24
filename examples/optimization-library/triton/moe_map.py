"""Bounded supplied-metadata map component; prefix preparation and allocation excluded.

One prepared API call launches one map kernel. Metadata are supplied inputs;
this does not measure the whole make_chunk_sort_map pipeline. MindSpeedOps
4efe2b675c2d51517c3fdd14c721020b8d41e241, Apache-2.0.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from moe_map_contract import CASE_SOURCES,dimensions,make_case,validate_inputs


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)
    def __init__(self): self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        if arm not in ('before','after'): raise ValueError('invalid arm')
        _,n,s=dimensions(spec);validate_inputs(fixture['inputs'],spec)
        import torch
        import torch_npu
        import triton
        from moe_map_kernel import moe_map_before,moe_map_after
        cores=torch.npu.get_device_properties(0).vector_core_num
        if type(cores) is not int or not 0<cores<=64: raise ValueError('common float range requires queried AIV cores <=64')
        blocks=min(cores,n);bs=triton.cdiv(n,blocks)
        inputs={k:v.npu() for k,v in fixture['inputs'].items()}
        storage=torch.full((n+2,),-701,device='npu',dtype=torch.int32);out=storage[1:-1]
        kernel=moe_map_before if arm=='before' else moe_map_after
        args=tuple(inputs[k] for k in ('split_sizes','input_cumsum','inverse_sorted_indices','output_cumsum'))+(out,s,n,bs,64,triton.next_power_of_2(s))
        flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        compiled={};qualified=None
        def remember(a): compiled[id(a)]=a;return a
        def compile_only():
            nonlocal qualified
            if qualified is None: qualified=remember(kernel.warmup(*args,grid=(blocks,1,1),**flags))
            return qualified
        def run():
            compile_only()
            remember(kernel[(blocks,1,1)](*args,**flags))
            return out
        def compilation_state():
            return [dict(hash=a.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            expected_kernels=[kernel.fn.__name__],inputs=inputs,storage=storage,outputs=out,
            sources=[Path(__file__),Path(__file__).with_name('moe_map_kernel.py'),*CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',
                queried_aiv_cores=cores,grid=[blocks,1,1],block_size=bs,sub_block_size=64,flags=flags,
                api_component_count=1,scope='supplied-metadata map component; prefix/inverse generation and fresh allocator excluded'))
        self.prepared[arm]=item;return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];_,n,_=dimensions(spec)
        valid=(isinstance(actual,torch.Tensor) and actual.dtype==torch.int32 and actual.device.type=='npu'
               and tuple(actual.shape)==(n,) and actual.is_contiguous())
        if valid:
            valid=torch.equal(actual.cpu(),fixture['expected']) and all(actual.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in item.inputs.values())
        mutated=[k for k,v in item.inputs.items() if not torch.equal(v.cpu(),fixture['inputs'][k])]
        canary=bool((item.storage[[0,-1]].cpu()==-701).all())
        return dict(passed=bool(valid and canary and not mutated),exact_int32=bool(valid),canaries=canary,mutated_inputs=mutated)


def factory(): return Adapter()
