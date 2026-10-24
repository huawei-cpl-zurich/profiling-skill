"""Source-style arch32 MHC post: four API components versus three."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhc_post_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_output


class Adapter:
    validate_case = staticmethod(dimensions)
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}
        self.pair_key = None
        self.bits = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('invalid arm')
        _, bs, d = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        key = (str(Path(output).parent.resolve()), json.dumps(spec, sort_keys=True))
        if key != self.pair_key:
            self.pair_key, self.bits, self.prepared = key, {}, {}
        import torch
        import torch_npu
        import triton
        import mhc_post_kernel as k
        inputs = {name: value.npu() for name, value in fixture['inputs'].items()}
        storages = []
        def allocate(shape):
            size = __import__('math').prod(shape)
            storage = torch.empty(size+2, dtype=torch.float32, device='npu')
            storage[0] = -701.; storage[-1] = -701.
            storages.append(storage)
            return storage[1:-1].view(shape)
        ht = allocate((bs,4,4)); a = allocate((bs,4,d)); b = allocate((bs,4,d)); y = allocate((bs,4,d))
        buffers = [*inputs.values(), ht,a,b,y]
        if len({v.untyped_storage().data_ptr() for v in buffers}) != len(buffers):
            raise ValueError('private nonalias buffers required')
        flags = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        common = dict(GROUP=1, BLOCK_C=d)
        components = [
            (k.transpose_user_h, (inputs['H'],ht), ((bs*16+255)//256,), dict(BS=bs,BLOCK=256)),
            (k.hc_post_bmm1_fwd_kernel, (inputs['hp'],inputs['x'],a,bs,d,4,1,d,1,4*d,d,1), (bs,1), dict(**common,DIVISIBLE_C=True)),
            (k.hc_post_bmm2_fwd_kernel, (ht,inputs['R'],y if arm=='after' else b,a,arm=='after',k.USE_SLICE,16,4,1,4*d,d,1,4*d,d,1), (bs,1), dict(**common,C=d))]
        if arm == 'before':
            components.append((k.add_outputs,(a,b,y),((bs*4*d+255)//256,),dict(N=bs*4*d,BLOCK=256)))
        compiled = {}
        def execute(compile_only):
            artifacts = []
            for kernel,args,grid,kw in components:
                artifact = (kernel.warmup(*args,grid=grid,**kw,**flags) if compile_only
                            else kernel[grid](*args,**kw,**flags))
                compiled[id(artifact)] = artifact
                artifacts.append(artifact)
            return artifacts if compile_only else y
        def compilation_state():
            return [dict(hash=v.hash,binaries={name:hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()
                    for name,data in v.asm.items() if isinstance(data,(str,bytes))}) for v in compiled.values()]
        item = SimpleNamespace(run=lambda:execute(False),compile_only=lambda:execute(True),
            compilation_state=compilation_state,expected_kernels=[x[0].fn.__name__ for x in components],
            sources=[Path(__file__),Path(k.__file__),*CASE_SOURCES],inputs=inputs,storages=storages,
            metadata=dict(language='triton',arm=arm,shape=spec,flags=flags,api_component_count=len(components),
                source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',source_style=True,
                slice_probe=k.SLICE_PROBE,coefficient_loads='source extract_slice' if k.USE_SLICE else 'common sixteen scalar loads',
                transpose='common one-pass GPU materialization inside every API',torch_version=torch.__version__,triton_version=triton.__version__))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        result = check_output(actual, fixture['expected'])
        item = self.prepared[arm]
        canaries = all(bool((s[[0,-1]].cpu() == -701).all()) for s in item.storages)
        mutated = [key for key,v in item.inputs.items() if not torch.equal(v.cpu().view(torch.uint8),fixture['inputs'][key].view(torch.uint8))]
        bits = actual.detach().cpu().view(torch.int32).clone() if isinstance(actual,torch.Tensor) else None
        self.bits[arm] = bits
        paired = None
        if set(self.bits) == {'before','after'}:
            paired = self.bits['before'] is not None and self.bits['after'] is not None and torch.equal(self.bits['before'],self.bits['after'])
        result.update(canaries=canaries,mutated_inputs=mutated,paired_exact_bits=paired)
        result['passed'] = result['passed'] and canaries and not mutated and paired is not False
        return result


def factory():
    return Adapter()
