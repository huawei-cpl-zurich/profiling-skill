"""Static versus runtime iteration binding, using two kernels per complete API."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinkhorn_runtime_contract import CASE_SOURCES, dimensions, make_case
from sinkhorn_contract import validate_inputs, check_outputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}
        self.pair_bits = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        p, bs, iters, eps = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        import torch
        import torch_npu
        import triton
        from sinkhorn_kernel import hc_split_sinkhorn_forward_prepost_kernel as prepost
        from sinkhorn_kernel import hc_split_sinkhorn_forward_comb_kernel as static_comb
        from sinkhorn_runtime_kernel import hc_runtime_sinkhorn_forward_comb_kernel as runtime_comb
        comb = static_comb if arm == 'before' else runtime_comb
        inputs = {k:v.npu() for k,v in fixture['inputs'].items()}
        counts = p.get('counts_sequence',[iters])
        input_args = tuple(inputs[k] for k in ('mixes','scale','base'))
        components = []
        for count in counts:
            storages, outputs = [], []
            for shape in ((bs,4),(bs,4),(bs,4,8)):
                storage = torch.empty(__import__('math').prod(shape)+2,dtype=torch.float32,device='npu')
                storage[0] = -701.; storage[-1] = -701.
                storages.append(storage); outputs.append(storage[1:-1].view(shape))
            pre, post, backing = outputs
            components.append(SimpleNamespace(count=count,storages=storages,backing=backing,
                result=(pre,post,backing[:,:,:4]),
                pre_args=(*input_args,pre,post,bs,4,eps,24),
                comb_args=(*input_args,backing,bs,4,eps,24)))
        pointers = [v.untyped_storage().data_ptr() for v in [*inputs.values(),
                    *(s for component in components for s in component.storages)]]
        if len(set(pointers)) != len(pointers):
            raise ValueError('private nonalias buffers required')
        common = dict(BLOCK_HC=4,GROUP=32,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        grid = ((bs+31)//32,)
        compiled, compile_records, run_records = {}, [], []
        def remember(artifact):
            compiled[id(artifact)] = artifact
            return artifact
        def flags(count):
            return dict(common,BLOCK_ALIGN=8,SINKHORN_ITERS=count)
        def compile_only():
            artifacts = []
            for c in components:
                for kind,kernel,args,options in (('prepost',prepost,c.pre_args,common),
                                                ('comb',comb,c.comb_args,flags(c.count))):
                    start = time.perf_counter()
                    artifact = remember(kernel.warmup(*args,grid=grid,**options))
                    compile_records.append(dict(kind=kind,count=c.count,hash=artifact.hash,
                        wall_seconds=time.perf_counter()-start,cache_hit=getattr(artifact,'cache_hit',None)))
                    artifacts.append(artifact)
            return artifacts
        result = tuple(c.result for c in components) if len(components)>1 else components[0].result
        def run():
            for c in components:
                callback = getattr(item,'on_component',None)
                if callback is not None: callback(c.count)
                remember(prepost[grid](*c.pre_args,**common))
                artifact = remember(comb[grid](*c.comb_args,**flags(c.count)))
                run_records.append(dict(count=c.count,comb_hash=artifact.hash))
            return result
        def compilation_state():
            return [dict(hash=a.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
                for k,v in a.asm.items() if isinstance(v,(str,bytes))}) for a in compiled.values()]
        item = SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,
            compile_records=compile_records,run_records=run_records,components=components,inputs=inputs,
            expected_kernels=[prepost.fn.__name__,comb.fn.__name__],
            pair_key=(str(Path(output).resolve().parent),json.dumps(spec,sort_keys=True)),
            sources=[Path(__file__),Path(__file__).with_name('sinkhorn_kernel.py'),
                     Path(__file__).with_name('sinkhorn_runtime_kernel.py'),*CASE_SOURCES],
            metadata=dict(language='triton',arm=arm,parameters=spec,full_api_components=len(components),
                kernels_per_full_api=2,counts=counts,grid=grid,flags=dict(common,BLOCK_ALIGN=8),
                iteration_binding='constexpr-static_range' if arm=='before' else 'runtime-do_not_specialize-range',
                allocator_cost_excluded=True,output_buffers_reused=True,output_strides=[32,8,1],
                torch_version=torch.__version__,triton_version=triton.__version__,
                source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        compound = len(item.components)>1
        values = actual if compound else (actual,)
        expected = fixture['expected'] if compound else (fixture['expected'],)
        if not isinstance(values,tuple) or len(values)!=len(item.components):
            return dict(passed=False,reason='compound output must contain two complete output tuples')
        rows, bits = [], []
        for value,wanted,c in zip(values,expected,item.components):
            row = check_outputs(value,wanted)
            layout = (isinstance(value,tuple) and len(value)==3 and all(isinstance(v,torch.Tensor) for v in value)
                      and tuple(value[2].stride())==(32,8,1))
            pads = c.backing[:,:,4:].detach().cpu()
            padding = bool((pads==0).all() and not torch.signbit(pads).any())
            canaries = all(bool((s[[0,-1]].cpu()==-701).all()) for s in c.storages)
            row.update(layout=layout,padding_positive_zero=padding,canaries=canaries)
            row['passed'] = row['passed'] and layout and padding and canaries
            rows.append(row)
            if layout:
                bits.extend(v.detach().cpu().view(torch.int32).clone() for v in value)
        immutable = all(torch.equal(v.cpu().view(torch.uint8),fixture['inputs'][k].view(torch.uint8))
                        for k,v in item.inputs.items())
        pair = self.pair_bits.setdefault(item.pair_key,{})
        pair[arm] = bits
        bit_equal = None
        if set(pair)=={'before','after'}:
            bit_equal = len(pair['before'])==len(pair['after']) and all(torch.equal(a,b) for a,b in zip(pair['before'],pair['after']))
            del self.pair_bits[item.pair_key]
        hashes = {r['comb_hash'] for r in item.run_records}
        unified = len(hashes)==1 if compound and arm=='after' else None
        return dict(passed=all(r['passed'] for r in rows) and immutable,
            components=rows,input_immutable=immutable,crosspath_bitwise_equal_read_only=bit_equal,
            runtime_variant_unified=unified,run_bindings=list(item.run_records))


def factory():
    return Adapter()
