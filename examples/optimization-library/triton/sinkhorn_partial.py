"""Private prepared Sinkhorn reducer buffers; no producer/backward claim."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sinkhorn_partial_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_outputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before','after'):
            raise ValueError('unknown arm')
        _, n = dimensions(spec)
        validate_inputs(fixture['inputs'],spec)
        import torch
        import torch_npu
        import triton
        from sinkhorn_partial_kernel import deterministic_reduce_kernel as kernel
        inputs = {k:v.npu() for k,v in fixture['inputs'].items()}
        storages, outputs = [], []
        for name in ('scale','base'):
            initial = fixture['inputs'][name]
            storage = torch.empty(initial.numel()+2,dtype=torch.float32,device='npu')
            storage[0] = -701.; storage[-1] = -701.
            value = storage[1:-1]
            value.copy_(inputs[name])
            storages.append(storage); outputs.append(value)
        pointers = [v.untyped_storage().data_ptr() for v in [*inputs.values(),*outputs]]
        if len(set(pointers)) != 6:
            raise ValueError('private nonalias buffers required')
        vector = arm == 'after' and n <= 1024
        flags = dict(N=n,VECTOR=vector,BLOCK=(1 << (n-1).bit_length()) if vector else 1,
                     num_stages=1,multibuffer=False,enable_fp_fusion=False)
        args = (inputs['partials_scale'],inputs['partials_base'],*outputs)
        compiled = {}
        def compile_only():
            if not compiled:
                artifact = kernel.warmup(*args,grid=(1,),**flags)
                if not isinstance(artifact.asm.get('npubin'),bytes):
                    raise ValueError('compiled NPU binary qualification required')
                compiled['kernel'] = artifact
            return list(compiled.values())
        def run():
            compile_only()  # Public first launch is qualified, never a warmup forward.
            artifact = kernel[(1,)](*args,**flags)
            compiled['kernel'] = artifact
            return tuple(outputs)
        def compilation_state():
            return [dict(hash=a.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                    for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item = SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            inputs=inputs,output_storages=storages,expected_kernels=[kernel.fn.__name__],
            sources=[Path(__file__),Path(__file__).with_name('sinkhorn_partial_kernel.py'),*CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,flags=flags,grid=(1,),vector=vector,
                          api_component_count=1,source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',
                          torch_version=torch.__version__,triton_version=triton.__version__,
                          prepared_output_reuse=True,allocation_excluded=True))
        self.prepared[arm] = item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item = self.prepared[arm]
        result = check_outputs(actual,fixture)
        canaries = all(bool((s[[0,-1]].cpu() == -701).all()) for s in item.output_storages)
        mutated = [k for k,v in item.inputs.items() if not torch.equal(v.cpu().view(torch.int32),fixture['inputs'][k].view(torch.int32))]
        result.update(canaries=canaries,mutated_inputs=mutated)
        result['passed'] = result['passed'] and canaries and not mutated
        return result


def factory():
    return Adapter()
