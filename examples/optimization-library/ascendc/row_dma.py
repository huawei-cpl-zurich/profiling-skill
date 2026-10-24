"""Regular row-command aggregation; all software bounds are study policy."""
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
    rows, columns = p.get('rows'), p.get('columns')
    if type(rows) is not int or type(columns) is not int or rows <= 0 or not 1 <= columns <= 128:
        raise ValueError('positive integer rows and columns 1..128 required')
    pitch = ((columns + 7) // 8) * 8
    if p.get('destination_pitch_elements', pitch) != pitch:
        raise ValueError('UB pitch must equal columns rounded to eight FP32 elements')
    if rows * pitch * 4 > 65536:
        raise ValueError('conservative local-storage software scope exceeds 65536 bytes')
    return rows, columns, pitch


def make_case(spec):
    rows, columns, _ = dimensions(spec)  # Reject before fixture cache/device activity.
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.rand((rows, columns), generator=generator, dtype=torch.float32, device='cpu') * 2 - 1
    x.flatten()[0] = -1
    return dict(x=x, expected=(x.to(torch.float64) + 1).to(torch.float32))


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
        rows, columns, _ = dimensions(spec)
        for arm in ('before', 'after'):
            name = f'{rows}-{columns}-{arm}'
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DROWS={rows}', f'-DCOLUMNS={columns}', f'-DAFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-L04')


def verify_binary(binary, rows, columns, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('rows', rows), ('columns', columns), ('after', int(arm == 'after'))):
        function = getattr(library, 'row_dma_' + key)
        function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('ROW_DMA_DIRECTORY', '.')).resolve()
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        rows, columns, pitch = dimensions(spec)
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        name = f'{rows}-{columns}-{arm}'
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, rows, columns, arm)
        import torch_npu
        x = fixture['x'].to(device='npu:0', dtype=torch.float32)
        storage = torch.full((rows * columns + 64,), -8191.0, dtype=torch.float32, device='npu:0')
        result = storage[32:-32].view(rows, columns)
        item = NativeArm(binary, [x], result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                         source_revision='wave006-L04', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        item.metadata.update(backend='ascendc-cabi-v1', rows=rows, columns=columns, ub_pitch_elements=pitch,
                             ub_bytes=rows*pitch*4, source_expected_input_commands=1 if arm=='after' and rows<=128 else rows,
                             optimized=arm=='after' and rows<=128, grid=1)
        self.prepared[arm] = (item, storage)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, storage = self.prepared[arm]
        output = actual.cpu()
        immutable = torch.equal(item.inputs[0].cpu().view(torch.int32), fixture['x'].view(torch.int32))
        exact = torch.equal(output.view(torch.int32), fixture['expected'].view(torch.int32))
        canaries = bool((storage[:32].cpu() == -8191).all() and (storage[-32:].cpu() == -8191).all())
        nonalias = item.inputs[0].untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
        return dict(passed=exact and immutable and canaries and nonalias, exact_bitwise=exact,
                    immutable_input=immutable, gm_canaries=canaries, nonalias_output=nonalias)


def factory():
    return Adapter()
