"""Provided-saved-state RMS backward: BR1 versus BR4, common zero and sum.

Source-style portable helpers replace vendor torch zeros/sum on both arms.
The generic CLI is exploratory; controlled qualification uses the frozen collector.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rms_backward_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_outputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        p, rows, d = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        import torch
        import torch_npu
        import triton
        import triton.runtime.driver as driver
        from rms_backward_kernel import _block_rms_norm_backward_kernel as kernel, zero_dx, sum_dw
        cores = driver.active.utils.get_device_properties(torch.npu.current_device())['num_vectorcore']
        inputs = {k:v.npu() for k,v in fixture['inputs'].items()}
        # Valid RSTD is contiguous, with a NaN halo to expose unmasked tail reads.
        halo = torch.full((rows+4,), float('nan'), dtype=torch.float32, device='npu')
        halo[:rows].copy_(inputs['RSTD']); inputs['RSTD'] = halo[:rows]
        storages, outputs = [], []
        for shape in ((rows,d),(d,)):
            storage = torch.empty(__import__('math').prod(shape)+2, device='npu', dtype=torch.float32)
            storage[0] = -701.; storage[-1] = -701.
            storages.append(storage); outputs.append(storage[1:-1].view(shape))
        dx, dw = outputs
        partial = torch.empty((cores,d), device='npu', dtype=torch.float32)
        blockrow = 4 if arm == 'after' and d <= 1024 else 1
        flags = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        args = (inputs['dY'],d,dx,d,inputs['X'],d,triton.language.float32,inputs['W'],1,
                inputs['RSTD'],1,partial,d,rows,d,p['offset'],(rows+cores-1)//cores,1)
        launches = [(zero_dx,(dx,rows*d),((rows*d+255)//256,),dict(BLOCK=256)),
                    (kernel,args,(cores,),dict(BLOCK_SIZE=triton.next_power_of_2(d),BLOCK_ROW=blockrow)),
                    (sum_dw,(partial,dw),(triton.cdiv(d,128),),
                     dict(D=d,CORES=cores,ROWS=triton.next_power_of_2(cores),BLOCK=128))]
        compiled = {}
        def compile_only():
            if not compiled:
                for i,(fn,a,grid,kw) in enumerate(launches):
                    compiled[i] = fn.warmup(*a,grid=grid,**kw,**flags)
            return list(compiled.values())
        def run():
            compile_only()  # Public first launch qualifies all three kernels before math.
            for i,(fn,a,grid,kw) in enumerate(launches):
                compiled[i] = fn[grid](*a,**kw,**flags)
            return dx,dw
        def compilation_state():
            return [dict(hash=k.hash,binaries={n:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for n,v in k.asm.items() if isinstance(v,(str,bytes))}) for k in compiled.values()]
        item = SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            sources=[Path(__file__),Path(__file__).with_name('rms_backward_kernel.py'),*CASE_SOURCES],
            inputs=inputs,output_storages=storages,expected_kernels=[fn.fn.__name__ for fn,*_ in launches],
            metadata=dict(arm=arm,block_row=blockrow,grid=[cores],num_vectorcore=cores,flags=flags,
                scope='provided-saved-stat backward component',api_component_count=3,
                source_style='common portable Triton zero DX + blocked backward + final DW sum',
                route='optimized' if blockrow==4 else 'reference',
                source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',
                partial_dw_shape=[cores,d],torch_version=torch.__version__,triton_version=triton.__version__))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = check_outputs(actual,fixture['expected'])
        canaries = all(bool((s[[0,-1]].cpu()==-701).all()) for s in item.output_storages)
        mutated = [k for k,v in item.inputs.items() if not torch.equal(v.cpu().view(torch.uint8),fixture['inputs'][k].view(torch.uint8))]
        nonalias = len({v.untyped_storage().data_ptr() for v in [*item.inputs.values(),*actual]})==6 if isinstance(actual,tuple) and len(actual)==2 else False
        result.update(canaries=canaries,mutated_inputs=mutated,nonalias=nonalias)
        result['passed'] = result['passed'] and canaries and not mutated and nonalias
        return result


def factory():
    return Adapter()
