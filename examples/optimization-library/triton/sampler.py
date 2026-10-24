"""Wave007 recovered-token draft zeroing: global store/restore vs registers.

Source pin 7acb9c40a6ca827e885d902b7eaee5d45457b230, Huawei/vLLM;
see sampler_kernel.py attribution and SAMPLER_LICENSE. One API is one kernel.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

# Also supports importlib loading from an unrelated working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampler_contract import CASE_SOURCES, dimensions, make_case, qualify_ranking, validate_inputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        p, lengths, vocab = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        qualify_ranking(fixture['inputs'], fixture['expected'])
        import torch
        import torch_npu
        import triton
        from sampler_kernel import sample_recovered_tokens_kernel as kernel
        inputs = {key: value.npu() for key, value in fixture['inputs'].items()}
        n = sum(lengths)
        storage = torch.full((n + 2,), -701, dtype=torch.int32, device='npu')
        out = storage[1:-1]
        args = (out, inputs['ends'], inputs['ids'], inputs.get('draft'), inputs['target'], inputs['q'],
                vocab, triton.next_power_of_2(vocab))
        flags = dict(NO_DRAFT_PROBS=not p['draft_probs_present'], REGISTER_EXCLUSION=arm == 'after',
                     SUB_BLOCK=4096, num_stages=1, multibuffer=False, enable_fp_fusion=False)
        compiled = {}
        def remember(artifact):
            compiled[id(artifact)] = artifact
            return artifact
        def compile_only():
            return remember(kernel.warmup(*args, grid=(len(lengths), max(lengths)), **flags))
        def run():
            remember(kernel[(len(lengths), max(lengths))](*args, **flags))
            return out
        def compilation_state():
            return [dict(hash=getattr(item, 'hash', None), binaries={key: hashlib.sha256(
                value.encode() if isinstance(value, str) else value).hexdigest()
                for key, value in item.asm.items() if isinstance(value, (str, bytes))})
                for item in compiled.values()]
        item = SimpleNamespace(run=run, compile_only=compile_only, compiled=compiled,
            expected_kernels=[kernel.fn.__name__], compilation_state=compilation_state,
            sources=[Path(__file__), Path(__file__).with_name('sampler_kernel.py'), *CASE_SOURCES],
            inputs=inputs, output_storage=storage,
            metadata=dict(language='triton', arm=arm, shape=spec, flags=flags,
                          source_pin='7acb9c40a6ca827e885d902b7eaee5d45457b230',
                          triton_version=triton.__version__, torch_version=torch.__version__,
                          api_component_count=1, target_boundary='bitwise restored; exclusive nonalias rows'))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        passed = torch.equal(actual.detach().cpu(), fixture['expected'])
        canaries = item.output_storage[[0, -1]].cpu()
        passed = passed and bool((canaries == -701).all())
        mutated = []
        for name, value in item.inputs.items():
            if not torch.equal(value.cpu().view(torch.uint8), fixture['inputs'][name].view(torch.uint8)):
                mutated.append(name)
        return dict(passed=passed and not mutated, integer_exact=passed, mutated_inputs=mutated)


def factory():
    return Adapter()
