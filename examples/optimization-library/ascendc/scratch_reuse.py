"""Independent four-stage FP32 storage-lifetime study; software reservation bounds."""
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
    rows, width = p.get('rows'), p.get('width')
    if type(rows) is not int or type(width) is not int or min(rows, width) <= 0:
        raise ValueError('positive integer rows and width required')
    tile = p.get('tile_elements', rows * width)
    late = p.get('needs_first_late', False)
    if type(late) is not bool or type(tile) is not int or tile <= 0 or tile % 8:
        raise ValueError('boolean late mode and positive eight-aligned tile required')
    if tile * 16 > 192 * 1024:
        raise ValueError('four-buffer software reservation exceeds 192KiB')
    return rows, width, tile, late


def make_case(spec):
    rows, width, _, late = dimensions(spec)
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randint(-32, 33, (rows, width), generator=generator, device='cpu', dtype=torch.int32).float() / 16
    a = (x.double() * x.double()).float()
    y = (a.double() + x.double()).float()
    b = (0.5 * y.double()).float()
    expected = (b.double() + a.double() if late else b.double() - 0.25).float()
    return dict(x=x, expected=expected)


def name_for(spec, arm):
    rows, width, tile, late = dimensions(spec)
    return f'{rows}-{width}-{tile}-{int(late)}-{arm}'


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
        match = re.search(r'struct DataCopyExtParams\s*\{', text)
        if match:
            declarations.append(dict(path=str(path), sha256=digest(path), declaration=text[match.start():match.start()+2000]))
    (output/'sdk-declarations.json').write_text(json.dumps(declarations, indent=2))
    if not declarations:
        raise RuntimeError('selected SDK DataCopyExtParams declaration unavailable')
    for spec in specs:
        rows, width, tile, late = dimensions(spec)
        for arm in ('before', 'after'):
            name = name_for(spec, arm)
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DROWS={rows}', f'-DWIDTH={width}', f'-DTILE={tile}', f'-DLATE={int(late)}', f'-DAFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-L06')


def verify_binary(binary, spec, arm):
    rows, width, tile, late = dimensions(spec)
    library = ctypes.CDLL(str(binary))
    for key, expected in (('rows', rows), ('width', width), ('tile', tile), ('late', int(late)), ('after', int(arm == 'after'))):
        function = getattr(library, 'scratch_reuse_' + key)
        function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm/lifetime contract mismatch')


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('SCRATCH_REUSE_DIRECTORY', '.')).resolve()
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        rows, width, tile, late = dimensions(spec)
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        name = name_for(spec, arm)
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, spec, arm)
        import torch_npu
        input_storage = torch.full((rows * width + 64,), -8191.0, dtype=torch.float32, device='npu:0')
        x = input_storage[32:-32].view(rows, width)
        x.copy_(fixture['x'])
        storage = torch.full((rows * width + 64,), -8191.0, dtype=torch.float32, device='npu:0')
        result = storage[32:-32].view(rows, width)
        item = NativeArm(binary, [x], result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                         source_revision='wave006-L06', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        item.metadata.update(backend='ascendc-cabi-v1', rows=rows, width=width, tile_elements=tile,
                             logical_reserved_bytes=tile*4*(3 if arm=='after' and not late else 4),
                             optimized=arm=='after' and not late, needs_first_late=late, grid=1)
        self.prepared[arm] = (item, storage, input_storage)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, storage, input_storage = self.prepared[arm]
        output = actual.cpu()
        immutable = torch.equal(item.inputs[0].cpu().view(torch.int32), fixture['x'].view(torch.int32))
        exact = torch.equal(output.view(torch.int32), fixture['expected'].view(torch.int32))
        canaries = bool((storage[:32].cpu() == -8191).all() and (storage[-32:].cpu() == -8191).all())
        canaries = canaries and bool((input_storage[:32].cpu() == -8191).all() and (input_storage[-32:].cpu() == -8191).all())
        nonalias = item.inputs[0].untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
        return dict(passed=exact and immutable and canaries and nonalias, exact_bitwise=exact,
                    immutable_input=immutable, gm_canaries=canaries, nonalias_output=nonalias)


def factory():
    return Adapter()
