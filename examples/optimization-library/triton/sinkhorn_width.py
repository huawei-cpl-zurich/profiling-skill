"""Width-only Sinkhorn pair: two prepared kernels per full API call."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinkhorn_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_outputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}
        self.prepost_bits = None

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        _, bs, iters, eps = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        import torch
        import torch_npu
        import triton
        from sinkhorn_kernel import hc_split_sinkhorn_forward_prepost_kernel as prepost, hc_split_sinkhorn_forward_comb_kernel as comb
        inputs = {k: v.npu() for k, v in fixture['inputs'].items()}
        storages, outputs = [], []
        for shape in ((bs,4), (bs,4), (bs,4,8)):
            count = __import__('math').prod(shape)
            storage = torch.empty(count+2, dtype=torch.float32, device='npu')
            storage[0] = -701.; storage[-1] = -701.
            storages.append(storage); outputs.append(storage[1:-1].view(shape))
        pre, post, backing = outputs
        result = (pre, post, backing[:, :, :4])
        pointers = [v.untyped_storage().data_ptr() for v in [*inputs.values(), *outputs]]
        if len(set(pointers)) != 6:
            raise ValueError('private nonalias buffers required')
        common = dict(BLOCK_HC=4, GROUP=32, num_stages=1, multibuffer=False, enable_fp_fusion=False)
        comb_flags = dict(common, BLOCK_ALIGN=8 if arm == 'before' else 4, SINKHORN_ITERS=iters)
        pre_args = (inputs['mixes'], inputs['scale'], inputs['base'], pre, post, bs, 4, eps, 24)
        comb_args = (inputs['mixes'], inputs['scale'], inputs['base'], backing, bs, 4, eps, 24)
        grid = ((bs+31)//32,)
        compiled = {}
        def remember(artifact):
            compiled[id(artifact)] = artifact
            return artifact
        def compile_only():
            return [remember(prepost.warmup(*pre_args, grid=grid, **common)),
                    remember(comb.warmup(*comb_args, grid=grid, **comb_flags))]
        def run():
            remember(prepost[grid](*pre_args, **common))
            remember(comb[grid](*comb_args, **comb_flags))
            return result
        def compilation_state():
            return [dict(hash=a.hash, binaries={k: hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item = SimpleNamespace(run=run, compile_only=compile_only, compilation_state=compilation_state,
            expected_kernels=[prepost.fn.__name__, comb.fn.__name__], inputs=inputs,
            output_storages=storages, backing=backing,
            sources=[Path(__file__), Path(__file__).with_name('sinkhorn_kernel.py'), *CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,parameters=spec,api_component_count=2,grid=grid,
                flags=comb_flags,group_adaptation='source default48 unsupported; common GROUP32',
                sigmoid='common faithful 1/(1+exp(-x)) expansion',output_strides=[32,8,1],
                allocator_cost_excluded=True,output_buffers_reused=True,torch_version=torch.__version__,
                triton_version=triton.__version__,source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = check_outputs(actual, fixture['expected'])
        pads = item.backing[:,:,4:].detach().cpu()
        padding = bool((pads == 0).all() and not torch.signbit(pads).any())
        canaries = all(bool((s[[0,-1]].cpu() == -701).all()) for s in item.output_storages)
        immutable = all(torch.equal(v.cpu().view(torch.uint8), fixture['inputs'][k].view(torch.uint8)) for k,v in item.inputs.items())
        layout = isinstance(actual,tuple) and len(actual)==3 and tuple(actual[2].stride())==(32,8,1)
        bits = tuple(v.detach().cpu().view(torch.int32) for v in actual[:2]) if layout else None
        identical = bits is not None and (self.prepost_bits is None or all(torch.equal(a,b) for a,b in zip(bits,self.prepost_bits)))
        if arm == 'before':
            self.prepost_bits = bits
        result['common_prepost_bitwise_equal'] = identical
        result.update(padding_positive_zero=padding,canaries=canaries,input_immutable=immutable,layout=layout)
        result['passed'] = result['passed'] and padding and canaries and immutable and layout and identical
        return result


def factory():
    return Adapter()
