"""Exact signed byte widening; same staging, grid and scratch on both sides."""
import ctypes
import json
import os
import re
from pathlib import Path
import shutil
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, build_shared, digest


def dimensions(spec):
    p = spec.get('parameters', spec)
    n = p.get('N')
    if type(n) is not int or not 0 <= n <= 1048576:
        raise ValueError('N must be an integer in 0..1048576')
    if p.get('dtype', 'int8') != 'int8' or p.get('layout', 'contiguous') != 'contiguous':
        raise ValueError('contiguous signed int8 input required')
    return n


def check_input(x, n):
    if x.dtype != torch.int8 or x.ndim != 1 or x.numel() != n or not x.is_contiguous():
        raise ValueError('contiguous signed int8 input of shape [N] required')


def make_case(spec):
    n = dimensions(spec)  # Reject unsupported input declarations before cache creation.
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    if n >= 256:
        permutation = torch.randperm(256, generator=generator, device='cpu') - 128
        x = permutation.repeat((n + 255) // 256)[:n].to(dtype=torch.int8)
    else:
        x = torch.randint(-128, 128, (n,), generator=generator, device='cpu', dtype=torch.int32).to(torch.int8)
        prefix = torch.tensor([-128, -1, 0, 1, 127], dtype=torch.int8, device='cpu')
        x[:min(n, 5)] = prefix[:min(n, 5)]
    # Independent Python integer -> FP32 conversion; no FP16 or candidate code.
    expected = torch.tensor([float(int(v)) for v in x.tolist()], dtype=torch.float32, device='cpu')
    if spec.get('parameters', {}).get('also_empty'):
        expected = (expected, torch.empty((0,), dtype=torch.float32, device='cpu'))
    return dict(x=x, expected=expected)


def compile_all(specs, output):
    """SDK-only build, retaining the actual SDK declaration and compiler artifacts."""
    output = Path(output).resolve()
    compiler = Path(shutil.which('bisheng')).resolve()
    roots = [Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME') if os.environ.get(k)]
    cann = next(p for p in roots + list(compiler.parents) if (p/'include/acl/acl.h').is_file())
    operators = list((cann/'include').rglob('kernel_operator.h'))
    if len(operators) != 1:
        raise RuntimeError('exactly one SDK kernel_operator.h required')
    api = operators[0].resolve().parent
    source = Path(__file__).with_suffix('.cpp')
    includes = [cann/'include', api.parent, api, api/'interface', api/'impl']
    output.mkdir(parents=True, exist_ok=False)
    # Preserve exact installed declarations; compiler/runtime qualify the selected fields.
    declarations = []
    for path in api.parent.rglob('*.h'):
        if path.stat().st_size > 1000000:
            continue
        text = path.read_text(errors='replace')
        match = re.search(r'struct DataCopyExtParams\s*\{|enum (?:class )?KernelType|Cast\(', text)
        if match:
            declarations.append(dict(path=str(path), sha256=digest(path), declaration=text[match.start():match.start()+2000]))
    (output/'sdk-declarations.json').write_text(json.dumps(declarations, indent=2))
    if not declarations:
        raise RuntimeError('selected SDK DataCopyExtParams declaration unavailable')
    for spec in specs:
        n = dimensions(spec)
        for arm in ('before', 'after'):
            name = f'{n}-{arm}'
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DSTUDY_N={n}', f'-DAFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-N01')



def verify_binary(binary, n, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('n', n), ('after', int(arm == 'after'))):
        function = getattr(library, 'int8_widening_' + key)
        function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')


class CompositeArm:
    def __init__(self, native, empty):
        self.native, self.empty = native, empty
        self.sources, self.expected_kernels, self.metadata = native.sources, native.expected_kernels, native.metadata
        self.metadata.update(component_operations=dict(widen=1, empty=1, native_launches=1, empty_native_launches=0))

    def compilation_state(self):
        return self.native.compilation_state()

    def run(self):
        return self.native.run(), self.empty

    def close(self):
        self.native.close()


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('INT8_WIDENING_DIRECTORY', '.')).resolve()
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        n = dimensions(spec)
        check_input(fixture['x'], n)  # Actual dtype/layout rejection before binary/device access.
        if arm not in ('before', 'after') or n == 0:
            raise ValueError('nonempty native preparation and before/after arm required')
        name = f'{n}-{arm}'
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, n, arm)
        import torch_npu
        input_storage = torch.full((n + 64,), -93, dtype=torch.int8, device='npu:0')
        x = input_storage[32:-32]; x.copy_(fixture['x'])
        storage = torch.full((n + 64,), -8191.0, dtype=torch.float32, device='npu:0')
        result = storage[32:-32]
        item = NativeArm(binary, [x], result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                         source_revision='wave006-N01', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        item.metadata.update(backend='ascendc-cabi-v1', N=n, tile=4096, ub_bytes=28672,
                             half_scratch_bytes=8192, grid=(n+4095)//4096,
                             component_operations=dict(widen=1, empty=0, native_launches=1))
        empty_input = torch.empty((0,), dtype=torch.int8, device='npu:0')
        if spec.get('parameters', {}).get('also_empty'):
            item = CompositeArm(item, empty_result(empty_input))
        self.prepared[arm] = (item, input_storage, storage, empty_input)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, input_storage, storage, empty_input = self.prepared[arm]
        composite = isinstance(actual, tuple)
        result = actual[0] if composite else actual
        expected = fixture['expected'][0] if composite else fixture['expected']
        exact = torch.equal(result.cpu().view(torch.int32), expected.view(torch.int32))
        immutable = torch.equal(input_storage[32:-32].cpu(), fixture['x'])
        canaries = bool((storage[:32].cpu() == -8191).all() and (storage[-32:].cpu() == -8191).all()
                        and (input_storage[:32].cpu() == -93).all() and (input_storage[-32:].cpu() == -93).all())
        distinct_empty = not composite or (actual[1] is not empty_input and actual[1].shape == (0,)
                                           and actual[1].dtype == torch.float32 and actual[1].device == empty_input.device)
        return dict(passed=exact and immutable and canaries and distinct_empty, exact_bitwise=exact,
                    immutable_input=immutable, gm_canaries=canaries, distinct_empty=distinct_empty,
                    native_launches=1, empty_native_launches=0)


def empty_result(x):
    check_input(x, 0)
    return torch.empty((0,), dtype=torch.float32, device=x.device)


def factory():
    return Adapter()
