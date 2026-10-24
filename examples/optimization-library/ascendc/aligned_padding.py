"""Aligned required-padding fill; source-backed standalone native adaptation."""
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
    rows, width, pitch = (p.get(k) for k in ('rows', 'width', 'pitch'))
    if any(type(n) is not int for n in (rows, width, pitch)) or rows < 0 or not 1 <= width <= pitch <= 1024 or pitch % 8:
        raise ValueError('integer rows>=0, 1<=width<=pitch<=1024 and pitch%8==0 required')
    if rows * pitch * 4 > 65536:
        raise ValueError('local-storage software scope exceeds 65536 bytes')
    return rows, width, pitch


def vector_admitted(rows, width, pitch):
    return rows > 0 and pitch > width and width % 8 == 0 and pitch % 8 == 0


def empty_result(x, width):
    if x.ndim != 2 or x.shape[0] != 0 or x.dtype != torch.float32 or not x.is_contiguous() or not 1 <= width <= x.shape[1]:
        raise ValueError('contiguous empty FP32 rows required')
    return torch.empty_like(x)


def make_case(spec):
    rows, width, pitch = dimensions(spec)
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.rand((rows, pitch), generator=generator, dtype=torch.float32, device='cpu') * 2 - 1
    x[:, width:] = 73.25
    if rows:
        x[:, 0] = -0.0
        if width > 1:
            x[:, 1] = 0.0
    expected = x.clone()
    expected[:, width:] = 0.0
    result = dict(x=x, expected=expected)
    if spec['parameters'].get('also_empty_rows'):
        result['empty'] = torch.empty((0, pitch), dtype=torch.float32, device='cpu')
    return result


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
        match = re.search(r'(?:struct DataCopyExtParams\s*\{|.*void Duplicate\()', text)
        if match:
            declarations.append(dict(path=str(path), sha256=digest(path), declaration=text[match.start():match.start()+2000]))
    (output/'sdk-declarations.json').write_text(json.dumps(declarations, indent=2))
    if not declarations:
        raise RuntimeError('selected SDK DataCopyExtParams declaration unavailable')
    for spec in specs:
        rows, width, pitch = dimensions(spec)
        for arm in ('before', 'after'):
            name = f'{rows}-{width}-{pitch}-{arm}'
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DW006_N05_ROWS={rows}', f'-DW006_N05_WIDTH={width}', f'-DW006_N05_PITCH={pitch}', f'-DW006_N05_AFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-N05')



def verify_binary(binary, rows, width, pitch, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('rows', rows), ('width', width), ('pitch', pitch), ('after', int(arm == 'after'))):
        function = getattr(library, 'aligned_padding_' + key)
        function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('ALIGNED_PADDING_DIRECTORY', '.')).resolve()
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        rows, width, pitch = dimensions(spec)
        if arm not in ('before', 'after') or rows == 0:
            raise ValueError('native preparation requires positive rows and a valid arm')
        name = f'{rows}-{width}-{pitch}-{arm}'
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, rows, width, pitch, arm)
        import torch_npu
        storages = [torch.full((rows*pitch+64,), -8191.0, dtype=torch.float32, device='npu:0') for _ in range(2)]
        x, result = [s[32:-32].view(rows, pitch) for s in storages]
        x.copy_(fixture['x'])
        item = NativeArm(binary, [x], result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                         source_revision='wave006-N05', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        assert x.data_ptr() % 32 == result.data_ptr() % 32 == 0
        item.metadata.update(backend='ascendc-cabi-v1', rows=rows, width=width, pitch=pitch,
                             ub_bytes=rows*pitch*4, optimized=arm=='after' and vector_admitted(rows,width,pitch),
                             grid=1, full_row_copy_commands=rows*2, full_row_gm_bytes=rows*pitch*8)
        if 'empty' in fixture:
            empty = fixture['empty'].to(device='npu:0', dtype=torch.float32)
            original = item.run
            def composite():
                item.empty_output = empty_result(empty, width)
                return original()
            item.run = composite
            item.metadata.update(api_components=2, native_components=1, empty_native_calls=0)
        self.prepared[arm] = (item, storages)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, storages = self.prepared[arm]
        immutable = torch.equal(item.inputs[0].cpu().view(torch.int32), fixture['x'].view(torch.int32))
        exact = torch.equal(actual.cpu().view(torch.int32), fixture['expected'].view(torch.int32))
        canaries = all(bool((s[:32].cpu() == -8191).all() and (s[-32:].cpu() == -8191).all()) for s in storages)
        nonalias = item.inputs[0].untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
        empty = 'empty' not in fixture or (item.empty_output.shape == fixture['empty'].shape and item.empty_output.numel() == 0)
        return dict(passed=exact and immutable and canaries and nonalias and empty, exact_bitwise=exact,
                    immutable_input=immutable, gm_canaries=canaries, nonalias_output=nonalias, empty_component=empty)


def factory():
    return Adapter()
