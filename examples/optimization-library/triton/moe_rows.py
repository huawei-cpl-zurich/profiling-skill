"""Supplied-map row movement, not end-to-end MoE/chunk-map construction.

Source: MindSpeedOps 4efe2b675c2d51517c3fdd14c721020b8d41e241 (Apache-2.0).
One API is one movement kernel; empty N performs no kernel launch.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_rows_contract import CASE_SOURCES, dimensions, make_case, validate_inputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)
    correctness_only = staticmethod(lambda spec: dimensions(spec)[1] == 0)

    def __init__(self): self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before','after'): raise ValueError('invalid arm')
        p,n,d = dimensions(spec)
        validate_inputs(fixture['inputs'],spec)
        import torch
        import torch_npu
        import triton
        from moe_rows_kernel import moe_rows_before,moe_rows_after,GET_ELEMENT_SOURCE,get_element
        cores = torch.npu.get_device_properties(0).vector_core_num
        if type(cores) is not int or cores <= 0: raise ValueError('queried AIV cores unavailable')
        blocks = min(cores,n) if n else 0
        bs = triton.cdiv(n,blocks) if n else 0
        bt = triton.next_power_of_2(bs) if n else 1
        inputs = {k:v.npu() for k,v in fixture['inputs'].items()}
        storage = torch.full((n*d+2,),-701.,device='npu',dtype=torch.float32)
        out = storage[1:-1].view(n,d)
        ps = torch.full((n+2,),-701.,device='npu',dtype=torch.float32) if p['with_probs'] else None
        probs = ps[1:-1] if ps is not None else None
        kernel = moe_rows_before if arm == 'before' else moe_rows_after
        args = (inputs['x'],out,inputs['row_map'],inputs.get('probs'),probs,n,d,bs,bt)
        flags = dict(PROBS=p['with_probs'],FORWARD=p['forward'],USE_ELEMENT=get_element is not None,
                     num_stages=1,multibuffer=False,enable_fp_fusion=False)
        compiled = {}
        qualified = False
        qualified_artifact = None
        def remember(artifact): compiled[id(artifact)]=artifact; return artifact
        def compile_only():
            nonlocal qualified, qualified_artifact
            if qualified: return qualified_artifact
            if not n:
                qualified = True
                return None
            try:
                qualified_artifact = remember(kernel.warmup(*args,grid=(blocks,1,1),**flags))
            except Exception as error:
                if arm != 'before' or not flags['USE_ELEMENT']: raise
                # Probe the real extension lowering before selecting a portable baseline.
                item.metadata['get_element_compile_failure'] = dict(type=type(error).__name__,message=str(error))
                item.metadata['baseline'] = 'portable-source-style scalar map load; changed map traffic'
                flags['USE_ELEMENT'] = False
                qualified_artifact = remember(kernel.warmup(*args,grid=(blocks,1,1),**flags))
            qualified = True
            return qualified_artifact
        def run():
            compile_only()  # Select a supported specialization before the first launch.
            if n: remember(kernel[(blocks,1,1)](*args,**flags))
            return out,probs
        def compilation_state():
            return [dict(hash=a.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item=SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            expected_kernels=[kernel.fn.__name__],inputs=inputs,storages=(storage,ps),outputs=(out,probs),
            sources=[Path(__file__),Path(__file__).with_name('moe_rows_kernel.py'),*CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',
                queried_aiv_cores=cores,grid=[blocks,1,1],block_size=bs,block_tokens=bt,row_tile=4,
                get_element_source=GET_ELEMENT_SOURCE,baseline='source-style get_element' if get_element is not None else 'portable-source-style scalar map load; changed map traffic',
                flags=flags,api_component_count=int(n>0),map_construction='excluded; supplied API input'))
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]; p,n,d=dimensions(spec)
        valid = isinstance(actual,tuple) and len(actual)==2
        reasons=[]
        if valid:
            for value,expected,shape in zip(actual,fixture['expected'],((n,d),(n,))):
                if expected is None:
                    valid = valid and value is None
                elif (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32
                      or tuple(value.shape)!=shape or value.device.type!='npu' or not value.is_contiguous()):
                    valid=False
                else:
                    valid = valid and torch.equal(value.cpu().view(torch.int32),expected.view(torch.int32))
            for v in actual:
                if v is not None and v.numel():
                    valid = valid and all(v.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in item.inputs.values())
        for key,value in item.inputs.items():
            if not torch.equal(value.cpu().view(torch.uint8),fixture['inputs'][key].view(torch.uint8)): reasons.append(key)
        for storage in item.storages:
            if storage is not None: valid = valid and bool((storage[[0,-1]].cpu()==-701.).all())
        return dict(passed=bool(valid and not reasons),exact_fp32_bits=bool(valid),mutated_inputs=reasons)


def factory(): return Adapter()
