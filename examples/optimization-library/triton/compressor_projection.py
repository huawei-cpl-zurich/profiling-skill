"""Prepared two-projection component; common controlled tiles, no X reuse.

Public CLI is exploratory. The frozen collector additionally screens all cases
before math and gates every pair before any diagnostic profile capture.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import struct
import torch
from compressor_projection_contract import check_spec, check_inputs, make_case, validate_actual, CONFIG, K, N


def identity(compiled):
    return [dict(hash=str(item.hash), name=item.name,
                 binaries={k: hashlib.sha256(v.encode() if isinstance(v, str) else v).hexdigest()
                           for k, v in item.asm.items() if isinstance(v, (str, bytes))})
            for item in compiled]


class Adapter:
    case_sources = [Path(__file__).with_name('compressor_projection_contract.py')]
    make_case = staticmethod(make_case)

    def __init__(self):
        self.state = {}

    def prepare(self, arm, spec, fixture, output):
        m = check_spec(spec)
        if arm not in ('before', 'after'):
            raise ValueError('paired arm required')
        check_inputs(fixture['inputs'], m, 'cpu')
        if (not isinstance(fixture.get('expected'), (list, tuple)) or len(fixture['expected']) != 2
                or any(v.shape != (m, N) or v.dtype != torch.float32 for v in fixture['expected'])):
            raise ValueError('two FP32 expected outputs required')
        import torch_npu
        from compressor_projection_kernels import _proj_kernel, _proj2_kernel
        from compressor_projection_utils import get_npu_aicore_num
        inputs = {k: v.to('npu:0') for k, v in fixture['inputs'].items()}
        check_inputs(inputs, m, 'npu')
        buffers = [torch.full((m*N+64,), 12345., device='npu:0') for _ in range(2)]
        outputs = tuple(b[32:32+m*N].view(m, N) for b in buffers)
        cores = int(get_npu_aicore_num())
        if cores <= 0:
            raise ValueError('invalid Cube count')
        tiles = ((m + 31) // 32) * (N // 64)
        kernel = _proj_kernel if arm == 'before' else _proj2_kernel
        args = ([(outputs[i], inputs['x'], inputs['w'+str(i)], m, K, N) for i in range(2)]
                if arm == 'before' else [(*outputs, inputs['x'], inputs['w0'], inputs['w1'], m, K, N)])
        grid = (min(cores, tiles * (1 if arm == 'before' else 2)),)
        flags = dict(num_stages=1, multibuffer=False, enable_fp_fusion=False)
        compiled = []

        def compile_only():
            if m and not compiled:
                for a in args:
                    c = kernel.warmup(*a, **CONFIG, **flags, grid=grid)
                    if not isinstance(c.asm.get('npubin'), bytes) or not c.asm['npubin'].startswith(b'\x7fELF'):
                        raise ValueError('missing actual ELF npubin')
                    data = c.asm['npubin']
                    offset = struct.unpack_from('<Q', data, 40)[0]
                    size, count, names = struct.unpack_from('<HHH', data, 58)
                    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
                    table = sections[names]; strings = data[table[4]:table[4]+table[5]]
                    if not any(strings[t[0]:].split(b'\0',1)[0] == b'.text' and t[5] > 0 for t in sections):
                        raise ValueError('missing executable text')
                    compiled.append(c)
            return compiled

        def run():
            compile_only()  # Cached first-launch qualification; warmup is compile only.
            if m:
                for a in args:
                    kernel[grid](*a, **CONFIG, **flags)
            return outputs

        self.state[arm] = (inputs, buffers)
        sources = [__file__, self.case_sources[0], Path(__file__).with_name('compressor_projection_kernels.py'),
                   Path(__file__).with_name('compressor_projection_utils.py')]
        return SimpleNamespace(run=run, compile_only=compile_only, compilation_state=lambda: identity(compiled),
            sources=sources, expected_kernels=[kernel.fn.__name__],
            metadata=dict(component='two BF16 projections', cube_cores=cores, configuration=CONFIG,
                          grid=grid, device_kernel_calls=(2 if arm == 'before' else 1) if m else 0,
                          reused_outputs=True, x_reuse=False, allocation_boundary='outputs prepared; JIT scratch excluded',
                          allocator_limit='no allocator or full Compressor latency claim',
                          address_bound_elements=max(256*K, N*K), input_strides={k:list(v.stride()) for k,v in inputs.items()}))

    def validate(self, arm, actual, fixture, spec):
        inputs, buffers = self.state[arm]
        return validate_actual(actual, fixture, inputs, buffers)


def main():
    import argparse, json, sys
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('case', 'output', 'cache', 'provenance'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--correctness-only', action='store_true')
    a = p.parse_args()
    spec = json.loads(Path(a.case).read_text()); m = check_spec(spec)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
    from optimization_pair import run_pair
    provenance = json.loads(Path(a.provenance).read_text())
    provenance['qualification'] = 'exploratory public CLI; not frozen all-case scientific collector'
    r = run_pair(Adapter(), spec, a.output, a.cache, provenance, correctness_only=a.correctness_only or m == 0)
    return 0 if r['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
