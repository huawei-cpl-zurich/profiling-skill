"""Full prepared RMS forward/backward: cached stats versus recomputation."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rms_training_contract import CASE_SOURCES,dimensions,make_case,validate_inputs,check_outputs


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):self.prepared={};self.y_bits={};self.pair_key=None

    def _device_item(self,arm,spec,inputs,probe=False):
        p,rows,d=dimensions(spec)
        import torch
        import triton
        import triton.runtime.driver as driver
        from rms_training_kernel import _rms_norm_forward_kernel as forward, _block_rms_norm_backward_kernel as backward, zero_dx, sum_dw
        cores=driver.active.utils.get_device_properties(torch.npu.current_device())['num_vectorcore']
        recompute=arm=='after' and d<=1024
        storages=[];outputs=[]
        for shape in ((rows,d),(rows,d),(d,)):
            storage=torch.empty(__import__('math').prod(shape)+2,dtype=torch.float32,device=inputs['X'].device)
            if not probe:storage[0]=-701.;storage[-1]=-701.
            storages.append(storage);outputs.append(storage[1:-1].view(shape))
        y,dx,dw=outputs
        stats=torch.empty(rows,dtype=torch.float32,device=inputs['X'].device)
        partial=torch.empty((cores,d),dtype=torch.float32,device=inputs['X'].device)
        common=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        block=triton.next_power_of_2(d)
        fargs=(y,d,inputs['X'],d,inputs['W'],1,stats,1,d,1e-6,p['offset'],1)
        bargs=(inputs['DY'],d,dx,d,inputs['X'],d,triton.language.float32,inputs['W'],1,
               stats,1,partial,d,rows,d,p['offset'],1e-6,(rows+cores-1)//cores,1)
        launches=[(forward,fargs,(rows,),dict(BLOCK_SIZE=block,SAVE_STATS=not recompute)),
                  (zero_dx,(dx,rows*d),((rows*d+255)//256,),dict(BLOCK=256)),
                  (backward,bargs,(cores,),dict(BLOCK_SIZE=block,BLOCK_ROW=1,RECOMPUTE=recompute)),
                  (sum_dw,(partial,dw),(triton.cdiv(d,128),),dict(D=d,CORES=cores,ROWS=triton.next_power_of_2(cores),BLOCK=128))]
        compiled={}
        def compile_only():
            if not compiled:
                pending={i:fn.warmup(*a,grid=grid,**kw,**common) for i,(fn,a,grid,kw) in enumerate(launches)}
                compiled.update(pending)
            return list(compiled.values())
        def run():
            if probe:raise RuntimeError('compile-only probe cannot execute')
            compile_only()
            for i,(fn,a,grid,kw) in enumerate(launches):compiled[i]=fn[grid](*a,**kw,**common)
            return tuple(outputs)
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for key,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            expected_kernels=[fn.fn.__name__ for fn,*_ in launches],inputs=inputs,output_storages=storages,
            sources=[Path(__file__),Path(__file__).with_name('rms_training_kernel.py'),*CASE_SOURCES],
            metadata=dict(arm=arm,recompute=recompute,route='optimized' if recompute else 'cached',
                forward_grid=[rows],backward_grid=[cores],num_vectorcore=cores,BLOCK_ROW=1,forward_row_mode=True,
                flags=common,api_component_count=4,scope='full prepared source-style forward+backward, allocator excluded',
                source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',internal_rstd_observed=False,
                common_helpers='portable zero DX and final partial DW sum every API',
                fallback_reason='D>1024 exact cached-stat route' if arm=='after' and not recompute else None))

    def probe(self,arm,spec):
        """Input-only compile protocol; no fixture/oracle, initialization or execution."""
        _,rows,d=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch
        import torch_npu
        inputs={k:torch.empty(shape,dtype=torch.float32,device='npu')
                for k,shape in dict(X=(rows,d),DY=(rows,d),W=(d,)).items()}
        return self._device_item(arm,spec,inputs,probe=True)

    def prepare(self,arm,spec,fixture,output):
        dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        validate_inputs(fixture['inputs'],spec)
        import torch_npu
        pair_key=json.dumps(spec,sort_keys=True)
        if pair_key!=self.pair_key or arm in self.prepared:
            self.prepared={};self.y_bits={};self.pair_key=pair_key
        item=self._device_item(arm,spec,{k:v.npu() for k,v in fixture['inputs'].items()})
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];result=check_outputs(actual,fixture['expected'])
        canaries=all(bool((s[[0,-1]].cpu()==-701).all()) for s in item.output_storages)
        immutable=all(torch.equal(v.cpu().view(torch.uint8),fixture['inputs'][k].view(torch.uint8)) for k,v in item.inputs.items())
        nonalias=isinstance(actual,tuple) and len(actual)==3 and len({v.untyped_storage().data_ptr() for v in [*item.inputs.values(),*actual]})==6
        pair=None
        if result['passed']:
            self.y_bits[arm]=actual[0].detach().cpu().view(torch.int32).clone()
            if len(self.y_bits)==2:pair=torch.equal(self.y_bits['before'],self.y_bits['after'])
        result.update(canaries=canaries,immutable=immutable,nonalias=nonalias,paired_y_bits=pair)
        result['passed']=result['passed'] and canaries and immutable and nonalias and pair is not False
        return result


def factory():return Adapter()
