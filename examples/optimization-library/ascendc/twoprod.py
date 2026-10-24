"""Dense adaptation of pinned sparse TwoProd guard hoisting; A2/A3 only."""
import ctypes
import hashlib
import json
import os
import re
import struct
from pathlib import Path
import torch
from optimization_native_bridge import NativeArm, build_shared, digest


def dimensions(spec):
    p = spec.get('parameters', spec)
    rows, width = p.get('rows'), p.get('width')
    if type(rows) is not int or type(width) is not int or not 1 <= rows <= 64 or not 1 <= width <= 4096:
        raise ValueError('integer rows 1..64 and width 1..4096 required')
    distribution = p.get('distribution', 'uniform')
    if distribution not in ('uniform', 'splitter-unsafe-finite-products', 'mixed-safe-unsafe-cancellation'):
        raise ValueError('undeclared distribution')
    if distribution != 'uniform' and (width < 4 or (distribution.startswith('mixed') and rows != 3)):
        raise ValueError('guard requires width>=4, mixed requires three rows')
    return rows, width


def make_case(spec):
    rows, width = dimensions(spec)
    seed = spec.get('seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    a, b = [torch.rand((rows, width), generator=generator, device='cpu', dtype=torch.float32)*8-4 for _ in range(2)]
    distribution = spec['parameters'].get('distribution', 'uniform')
    if distribution != 'uniform':
        a.zero_(); b.zero_()
        unsafe = 1 if distribution.startswith('mixed') else 0
        a[unsafe, 0] = 1e35; b[unsafe, 0] = 1e-35; b[unsafe, 1] = 1e35
        if unsafe:
            a[0, :4] = torch.tensor([2**24, 1, -2**24, 2**-10], dtype=torch.float32, device='cpu')
            a[2, :4] = torch.tensor([1, -1, 2**-20, 2**-20], dtype=torch.float32, device='cpu')
            b[0, :4] = 1; b[2, :4] = 1
    expected = (a.to(torch.float64)*b.to(torch.float64)).sum(dim=1)
    return dict(a=a, b=b, expected=expected)


def expected_flags(a, b):
    """CPU validation only; never called by prepare or run."""
    ma, mb = a.abs().amax(dim=1), b.abs().amax(dim=1)
    split = torch.tensor(8e34, dtype=torch.float32, device='cpu')
    product = torch.tensor(1.6e38, dtype=torch.float32, device='cpu')
    limit = torch.minimum(split, product/mb)
    limit[mb > split] = -1
    return (ma <= limit).to(torch.float32)


def validate_fixture(fixture, spec):
    shape = dimensions(spec)
    for key in ('a', 'b'):
        value = fixture[key]
        if value.device.type != 'cpu' or value.dtype != torch.float32 or tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError('CPU contiguous FP32 inputs matching declared shape required')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('finite inputs required')
    if spec['parameters'].get('distribution', 'uniform') == 'uniform':
        if any(bool((fixture[k].abs() > 4).any()) for k in ('a', 'b')):
            raise ValueError('uniform input outside [-4,4]')


def compile_all(specs, output):
    import shutil
    output = Path(output).resolve()
    compiler = Path(shutil.which('bisheng')).resolve()
    roots = [Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME') if os.environ.get(k)]
    cann = next(p for p in roots + list(compiler.parents) if (p/'include/acl/acl.h').is_file())
    operators = list((cann/'include').rglob('kernel_operator.h'))
    if len(operators) != 1:
        raise RuntimeError('one selected SDK kernel_operator.h required')
    api = operators[0].resolve().parent
    source = Path(__file__).with_suffix('.cpp')
    helper = source.with_name('twoprod_helpers.h')
    includes = [source.parent, cann/'include', api.parent, api, api/'interface', api/'impl']
    output.mkdir(parents=True, exist_ok=False)
    for rows, width in sorted(set(dimensions(spec) for spec in specs)):
        for arm in ('before', 'after'):
            name = f'{rows}-{width}-{arm}'; binary = output/f'{name}.so'
            translation = output/f'{name}.cpp'; translation.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-ffp-contract=off', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220', f'-DROWS={rows}', f'-DWIDTH={width}',
                       f'-DAFTER={int(arm=="after")}', *[f'-I{p}' for p in includes], str(translation), '-o', str(binary)]
            build_shared(command, source_paths=[source, helper, translation, Path(__file__)], binary=binary,
                         output=output/f'build-{name}', source_revision='ops-sparse-bfc7c19d-wave006-N02')


def verify_binary(binary, rows, width, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('rows', rows), ('width', width), ('after', int(arm == 'after'))):
        function = getattr(library, 'twoprod_'+key); function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('TWOPROD_DIRECTORY', '.')).resolve()
        self.prepared = {}

    @staticmethod
    def capture(run, item, output):
        raise RuntimeError(
            "N02 profiling unavailable: paired FP32 preflight is unqualified; "
            "use --correctness-only until separate paired qualification")

    def prepare(self, arm, spec, fixture, output):
        rows, width = dimensions(spec)
        if arm not in ('before', 'after'):
            raise ValueError('arm must be before or after')
        validate_fixture(fixture, spec)
        name = f'{rows}-{width}-{arm}'; binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        verify_binary(binary, rows, width, arm)
        import torch_npu
        inputs = [fixture[k].to(device='npu:0', dtype=torch.float32) for k in ('a', 'b')]
        storage = torch.full((rows*2+64,), -8191.0, device='npu:0', dtype=torch.float32)
        result = storage[32:-32].view(rows, 2)
        item = NativeArm(binary, inputs, result, sources=[Path(__file__), Path(__file__).with_suffix('.cpp'),
                         Path(__file__).with_name('twoprod_helpers.h'), self.directory/f'build-{name}/build.json'],
                         source_revision='ops-sparse-bfc7c19d-wave006-N02', build_manifest=manifest,
                         stream_provider=lambda: torch.npu.current_stream().npu_stream)
        item.metadata.update(backend='ascendc-cabi-v1', rows=rows, width=width, grid=1,
                             reduction_workspace_floats=256, ub_bytes=(2*((width+7)//8*8)+272)*4,
                             output_columns=['dot', 'GPU_fast_flag'], fresh_certification=arm=='after')
        self.prepared[arm] = (item, storage)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        item, storage = self.prepared[arm]; cpu = actual.cpu(); y = cpu[:, 0].contiguous()
        numerical = bool(torch.isfinite(y).all() and torch.allclose(y.to(torch.float64), fixture['expected'], atol=1e-6, rtol=1e-6))
        flags = expected_flags(fixture['a'], fixture['b']) if arm == 'after' else torch.zeros_like(y)
        dispatch = torch.equal(cpu[:, 1], flags)
        immutable = all(torch.equal(t.cpu().view(torch.int32), fixture[k].view(torch.int32)) for t, k in zip(item.inputs, ('a', 'b')))
        canaries = bool((storage[:32].cpu() == -8191).all() and (storage[-32:].cpu() == -8191).all())
        nonalias = all(t.untyped_storage().data_ptr() != actual.untyped_storage().data_ptr() for t in item.inputs)
        return dict(passed=numerical and dispatch and immutable and canaries and nonalias,
                    numerical_fp64=numerical, actual_fast_flags=cpu[:, 1].tolist(), dispatch_verified=dispatch,
                    immutable_input=immutable, gm_canaries=canaries, nonalias_output=nonalias,
                    fp32_output_sha256=hashlib.sha256(y.view(torch.int32).numpy().tobytes()).hexdigest())


def factory():
    return Adapter()


def audit_placement(directory, specs):
    """Qualify generated vector entry, no Cube function, and SDK-bound ELF kind."""
    def read_elf(path):
        data = path.read_bytes()
        if data[:6] != b'\x7fELF\x02\x01':
            raise ValueError('little-endian ELF64 required')
        offset = struct.unpack_from('<Q', data, 40)[0]
        size, count, strings_index = struct.unpack_from('<HHH', data, 58)
        sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset+i*size) for i in range(count)]
        table = sections[strings_index]; names = data[table[4]:table[4]+table[5]]
        metadata, functions = {}, []
        for sec in sections:
            name = names[sec[0]:names.find(b'\0', sec[0])].decode()
            if name.startswith('.ascend.meta.'):
                metadata[name] = data[sec[4]:sec[4]+sec[5]]
            if sec[1] not in (2, 11):
                continue
            strings = sections[sec[6]]; symbols = data[strings[4]:strings[4]+strings[5]]
            for pos in range(sec[4], sec[4]+sec[5], sec[9]):
                index, info, _, ndx, _, length = struct.unpack_from('<IBBHQQ', data, pos)
                if info & 15 == 2 and ndx and length:
                    functions.append(symbols[index:symbols.find(b'\0', index)].decode())
        return metadata, functions
    directory = Path(directory); result = []
    for rows, width in sorted(set(dimensions(s) for s in specs)):
        for arm in ('before', 'after'):
            name = f'{rows}-{width}-{arm}'; entry = 'wave006_twoprod_'+arm
            vectors = list(directory.glob(name+'-*-vec.o'))
            cubes = list(directory.glob(name+'-*-cube.o'))
            contexts = list(directory.glob(name+'-*-vec.ccei'))
            if len(vectors) != 1 or len(contexts) != 1:
                raise ValueError('one generated vector ELF and SDK context required')
            context = contexts[0].read_text(errors='replace')
            enum = re.search(r'enum (?:class )?KernelType\s*\{(.*?)\}', context, re.S)
            binding = re.search(r'K_TYPE_AIV\s*=\s*(\d+)', enum[1]) if enum else None
            if binding is None:
                raise ValueError('selected SDK AIV kind declaration unavailable')
            kind = int(binding[1]); metadata, functions = read_elf(vectors[0])
            actual = metadata.get('.ascend.meta.'+entry)
            if not any(entry in f for f in functions) or actual != struct.pack('<HHI', 1, 4, kind):
                raise ValueError('generated vector entry or SDK-bound metadata mismatch')
            if any(entry in f for cube in cubes for f in read_elf(cube)[1]):
                raise ValueError('unexpected Cube entry')
            result.append(dict(variant=name, vector_sha256=digest(vectors[0]), sdk_context_sha256=digest(contexts[0]),
                               sdk_enum=enum[0], actual_metadata_hex=actual.hex(), aiv_kind=kind,
                               vector_functions=functions, cube_entry=False))
    return result
