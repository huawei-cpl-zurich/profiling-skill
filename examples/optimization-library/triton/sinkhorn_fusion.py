"""Sinkhorn launch fusion: unchanged 8-lane math, two launches versus one."""
import hashlib
import json
from pathlib import Path
import sys
import struct
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinkhorn_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_outputs


def screen_binary(data):
    """Reject absent/truncated code before executing a compiled artifact."""
    if not isinstance(data, bytes) or data[:6] != b'\x7fELF\x02\x01' or len(data) < 64:
        raise ValueError('expected ELF64 little-endian npubin')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, names = struct.unpack_from('<HHH', data, 58)
    if size != 64 or names >= count or offset + size*count > len(data):
        raise ValueError('invalid npubin section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
    strings = sections[names]
    table = data[strings[4]:strings[4]+strings[5]]
    text = []
    for section in sections:
        if section[1] != 8 and section[4]+section[5] > len(data):
            raise ValueError('truncated npubin section')
        name = table[section[0]:].split(b'\0', 1)[0]
        if name == b'.text':
            text.append(section)
    if len(text) != 1 or not text[0][5]:
        raise ValueError('missing executable .text')
    return hashlib.sha256(data).hexdigest()


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}
        self.pair_bits = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        _, bs, iters, eps = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        import torch
        import torch_npu
        import triton
        from sinkhorn_kernel import hc_split_sinkhorn_forward_prepost_kernel as prepost, hc_split_sinkhorn_forward_comb_kernel as comb
        from sinkhorn_fusion_kernel import hc_sinkhorn_forward_fused_kernel as fused
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
        comb_flags = dict(common, BLOCK_ALIGN=8, SINKHORN_ITERS=iters)
        pre_args = (inputs['mixes'], inputs['scale'], inputs['base'], pre, post, bs, 4, eps, 24)
        comb_args = (inputs['mixes'], inputs['scale'], inputs['base'], backing, bs, 4, eps, 24)
        grid = ((bs+31)//32,)
        fused_args = (*pre_args[:5], backing, *pre_args[5:])
        kernels = [(prepost, pre_args, common), (comb, comb_args, comb_flags)] if arm == 'before' else [(fused, fused_args, comb_flags)]
        compiled = {}
        screened = False
        def remember(artifact):
            compiled[id(artifact)] = artifact
            return artifact
        def compile_only():
            nonlocal screened
            artifacts = [remember(k.warmup(*args, grid=grid, **flags)) for k,args,flags in kernels]
            for artifact in artifacts:
                screen_binary(artifact.asm['npubin'])
            screened = True
            return artifacts
        def run():
            if not screened:
                compile_only()  # Public callers receive the same prelaunch screen.
            for kernel, args, flags in kernels:
                remember(kernel[grid](*args, **flags))
            return result
        def compilation_state():
            return [dict(hash=a.hash, binaries={k: hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item = SimpleNamespace(run=run, compile_only=compile_only, compilation_state=compilation_state,
            expected_kernels=[k.fn.__name__ for k,_,_ in kernels], inputs=inputs,
            output_storages=storages, backing=backing,
            pair_key=(str(Path(output).resolve().parent), json.dumps(spec, sort_keys=True)),
            sources=[Path(__file__), Path(__file__).with_name('sinkhorn_kernel.py'), Path(__file__).with_name('sinkhorn_fusion_kernel.py'), *CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,parameters=spec,api_component_count=len(kernels),grid=grid,
                flags=comb_flags,group_adaptation='source default48 unsupported; common GROUP32',
                sigmoid='common faithful 1/(1+exp(-x)) expansion',output_strides=[32,8,1],
                allocator_cost_excluded=True,output_buffers_reused=True,torch_version=torch.__version__,
                triton_version=triton.__version__,source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241'))
        if arm in self.pair_bits.get(item.pair_key, {}):
            del self.pair_bits[item.pair_key]
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
        bits = tuple(v.detach().cpu().view(torch.int32).clone() for v in actual) if layout else None
        identical = None
        if bits is not None:
            pair = self.pair_bits.setdefault(item.pair_key, {})
            pair[arm] = bits
            if set(pair) == {'before', 'after'}:
                identical = all(torch.equal(a, b) for a, b in zip(pair['before'], pair['after']))
                del self.pair_bits[item.pair_key]
        result['paired_output_bitwise_equal'] = identical
        result['paired_output_comparison'] = 'pending' if identical is None else 'complete'
        result.update(padding_positive_zero=padding,canaries=canaries,input_immutable=immutable,layout=layout)
        result['passed'] = result['passed'] and padding and canaries and immutable and layout and bits is not None and identical is not False
        return result


def factory():
    return Adapter()
