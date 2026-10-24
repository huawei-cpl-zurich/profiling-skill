"""Exact local indexed fetch, scalar byte reads versus native vector Gather."""
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
    n, k = p.get('N'), p.get('K')
    if type(n) is not int or type(k) is not int or not 1 <= n <= 65536 or not 1 <= k <= 4096:
        raise ValueError('integer N 1..65536 and K 1..4096 required')
    return n, k


def validate_indices(indices, k):
    if indices.device.type != 'cpu' or indices.dtype != torch.int32 or indices.ndim != 1 or not indices.is_contiguous():
        raise ValueError('contiguous CPU int32 indices required')
    if bool(((indices < 0) | (indices >= k)).any()):
        raise ValueError('indices must be in [0,K)')


def make_case(spec):
    n, k = dimensions(spec)
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    table = torch.rand((k,), generator=generator, dtype=torch.float32, device='cpu') * 8 - 4
    table[0], table[-1] = 0.0, -0.0
    pattern = spec.get('parameters', spec).get('pattern', 'uniform')
    if pattern == 'uniform':
        indices = torch.randint(k, (n,), generator=generator, dtype=torch.int32, device='cpu')
    elif pattern == 'repeat-one':
        indices = torch.full((n,), k // 2, dtype=torch.int32, device='cpu')
    elif pattern in ('alternating-two', 'endpoints-and-signedzero'):
        indices = (torch.arange(n, dtype=torch.int32, device='cpu') % 2) * (k - 1)
    else:
        raise ValueError('unsupported index pattern')
    # Explicit index vectors are a public host interface for invalid-input checks.
    if 'indices' in spec.get('parameters', spec):
        indices = torch.tensor(spec.get('parameters', spec)['indices'], dtype=torch.int32, device='cpu')
        if indices.numel() != n:
            raise ValueError('index extent differs from N')
    validate_indices(indices, k)
    return dict(table=table, indices=indices, expected=table[indices.to(torch.int64)].clone())


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
        match = re.search(r'(Gather\(|struct DataCopyExtParams\s*\{)', text)
        if match:
            declarations.append(dict(path=str(path), sha256=digest(path), declaration=text[match.start():match.start()+2000]))
    (output/'sdk-declarations.json').write_text(json.dumps(declarations, indent=2))
    if not declarations:
        raise RuntimeError('selected SDK DataCopyExtParams declaration unavailable')
    seen = set()
    for spec in specs:
        n, k = dimensions(spec)
        if (n, k) in seen:
            continue
        seen.add((n, k))
        for arm in ('before', 'after'):
            name = f'{n}-{k}-{arm}'
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DGATHER_N={n}', f'-DGATHER_K={k}', f'-DAFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-N03')


def verify_binary(binary, n, k, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('n', n), ('k', k), ('after', int(arm == 'after'))):
        function = getattr(library, 'ub_gather_' + key)
        function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('UB_GATHER_DIRECTORY', '.')).resolve()
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        n, k = dimensions(spec)
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        validate_indices(fixture['indices'], k)
        if fixture['indices'].numel() != n:
            raise ValueError('index extent differs from N')
        table = fixture['table']
        if table.device.type != 'cpu' or table.dtype != torch.float32 or table.shape != (k,) or not table.is_contiguous():
            raise ValueError('contiguous CPU FP32 table[K] required')
        if not bool(torch.isfinite(table).all()) or bool((table.abs() > 4).any()):
            raise ValueError('finite table values in [-4,4] required')
        name = f'{n}-{k}-{arm}'
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, n, k, arm)
        import torch_npu
        validate_indices(fixture['indices'], k)
        x = fixture['table'].to(device='npu:0', dtype=torch.float32)
        indices = fixture['indices'].to(device='npu:0', dtype=torch.int32)
        storage = torch.full((n + 64,), -8191.0, dtype=torch.float32, device='npu:0')
        result = storage[32:-32]
        item = NativeArm(binary, [x, indices], result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                         source_revision='wave006-N03', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        item.metadata.update(backend='ascendc-cabi-v1', N=n, K=k, tile=4096,
                             ub_bytes=(k + 8192)*4, grid=1, byte_offsets=True)
        self.prepared[arm] = (item, storage)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, storage = self.prepared[arm]
        output = actual.cpu()
        immutable = (torch.equal(item.inputs[0].cpu().view(torch.int32), fixture['table'].view(torch.int32))
                     and torch.equal(item.inputs[1].cpu(), fixture['indices']))
        exact = torch.equal(output.view(torch.int32), fixture['expected'].view(torch.int32))
        canaries = bool((storage[:32].cpu() == -8191).all() and (storage[-32:].cpu() == -8191).all())
        nonalias = item.inputs[0].untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
        return dict(passed=exact and immutable and canaries and nonalias, exact_bitwise=exact,
                    immutable_input=immutable, gm_canaries=canaries, nonalias_output=nonalias)


def factory():
    return Adapter()
