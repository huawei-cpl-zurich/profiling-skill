"""MHC coefficient-gradient atomics versus proved exclusive stores.

A source-backed application, not a new mechanism or whole-vendor-pipeline claim.
"""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mhc_pre_contract import CASE_SOURCES, dimensions, make_case, validate_inputs, check_outputs


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        _, bs, d, tile = dimensions(spec)
        validate_inputs(fixture['inputs'], spec)
        import torch
        import torch_npu
        import triton
        from mhc_pre_kernel import hc_pre_bmm_bwd_fused_kernel as kernel, zero_gradient_kernel as zero
        inputs = {key: value.npu() for key, value in fixture['inputs'].items()}
        storages, outputs = [], []
        for shape in ((bs, 4), (bs, 4, d)):
            count = __import__('math').prod(shape)
            storage = torch.empty(count + 2, dtype=torch.float32, device='npu')
            storage[0] = -701.; storage[-1] = -701.
            storages.append(storage)
            outputs.append(storage[1:-1].view(shape))
        dh, dx = outputs
        pointers = [x.untyped_storage().data_ptr() for x in [*inputs.values(), dh, dx]]
        if len(set(pointers)) != 5 or not all(x.is_contiguous() for x in [*inputs.values(), dh, dx]):
            raise ValueError('private nonalias contiguous buffers required')
        grid = (bs, (d + tile - 1) // tile)
        # GROUP=1 and only these two program axes; each row owns offsets 4*b+0..3.
        exclusive = (arm == 'after' and grid[1] == 1 and tile == d
                     and tuple(dh.stride()) == (4, 1) and dh.shape == (bs, 4))
        flags = dict(GROUP=1, BLOCK_D=tile, DIVISIBLE_D=True, EXCLUSIVE=exclusive,
                     num_stages=1, multibuffer=False, enable_fp_fusion=False)
        args = (inputs['H'], inputs['X'], inputs['dY'], dx, dh, bs, d,
                4, 1, 4*d, d, 1, d, 1, 4*d, d, 1, 4, 1)
        zero_flags = dict(BLOCK=256, num_stages=1, multibuffer=False, enable_fp_fusion=False)
        compiled = {}
        def remember(artifact):
            compiled[id(artifact)] = artifact
            return artifact
        def compile_only():
            artifacts = []
            if not exclusive:
                artifacts.append(remember(zero.warmup(dh, bs*4, grid=((bs*4+255)//256,), **zero_flags)))
            artifacts.append(remember(kernel.warmup(*args, grid=grid, **flags)))
            return artifacts
        def run():
            if not exclusive:
                remember(zero[((bs*4+255)//256,)](dh, bs*4, **zero_flags))
            remember(kernel[grid](*args, **flags))
            return dh, dx
        def compilation_state():
            return [dict(hash=artifact.hash, binaries={key: hashlib.sha256(
                value.encode() if isinstance(value, str) else value).hexdigest()
                for key, value in artifact.asm.items() if isinstance(value, (str, bytes))})
                for artifact in compiled.values()]
        expected = [kernel.fn.__name__] if exclusive else [zero.fn.__name__, kernel.fn.__name__]
        item = SimpleNamespace(run=run, compile_only=compile_only, compilation_state=compilation_state,
            expected_kernels=expected, sources=[Path(__file__), Path(__file__).with_name('mhc_pre_kernel.py'), *CASE_SOURCES],
            inputs=inputs, output_storages=storages,
            metadata=dict(language='triton', arm=arm, shape=spec, flags=flags, exclusive=exclusive,
                api_component_count=len(expected), grid=grid, source_pin='4efe2b675c2d51517c3fdd14c721020b8d41e241',
                torch_version=torch.__version__, triton_version=triton.__version__,
                zero_init='one-pass portable Triton adaptation of torch.zeros, inside every baseline/fallback API',
                signed_zero_bitwise_guarantee=False))
        self.prepared[arm] = item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        item = self.prepared[arm]
        result = check_outputs(actual, fixture['expected'])
        canaries = all(bool((s[[0, -1]].cpu() == -701).all()) for s in item.output_storages)
        mutated = [k for k, v in item.inputs.items()
                   if not torch.equal(v.cpu().view(torch.uint8), fixture['inputs'][k].view(torch.uint8))]
        if spec.get('parameters', spec).get('zero_cancelled_and_signedzero') and isinstance(actual, tuple) and len(actual) == 2:
            result['signed_zero_bits'] = [v.detach().cpu().view(torch.int32)[v.detach().cpu() == 0].tolist() for v in actual]
        result.update(canaries=canaries, mutated_inputs=mutated)
        result['passed'] = result['passed'] and canaries and not mutated
        return result


def factory():
    return Adapter()
