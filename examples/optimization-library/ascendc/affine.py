"""Within-call FP32 affine coefficient residency on native AIV cores."""
import ctypes
import json
import os
import re
from pathlib import Path
import shutil
import sys
import hashlib
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, build_shared, digest


def dimensions(spec):
    p = spec.get('parameters', spec)
    rows, width = p.get('rows'), p.get('width')
    if type(rows) is not int or type(width) is not int or not 1 <= rows <= 65536 or not 1 <= width <= 1024:
        raise ValueError('integer rows 1..65536 and width 1..1024 required')
    return rows, width, (width+7)//8*8


def make_case(spec):
    rows, width, _ = dimensions(spec)
    seed = spec['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    def rand(shape, low, high):
        return torch.rand(shape, generator=generator, dtype=torch.float32, device='cpu')*(high-low)+low
    x = rand((rows,width), -2, 2)
    gamma = rand((width,), -1.5, 1.5); beta = rand((width,), -.5, .5)
    def oracle(g,b):
        return ((x.double()*g.double()).float().double()+b.double()).float()
    value = dict(x=x, gamma=gamma, beta=beta)
    if spec.get('parameters', {}).get('two_calls_same_coefficient_pointer'):
        gamma1 = rand((width,), -1.5, 1.5); beta1 = rand((width,), -.5, .5)
        value.update(gamma1=gamma1, beta1=beta1, expected=(oracle(gamma,beta), oracle(gamma1,beta1)))
    else:
        value['expected'] = oracle(gamma,beta)
    return value


def compile_all(specs, output, aiv_count, ub_bytes):
    """SDK-only build, retaining the actual SDK declaration and compiler artifacts."""
    if type(aiv_count) is not int or aiv_count <= 0 or ub_bytes < 12288:
        raise ValueError('queried positive AIV count and sufficient UB required')
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
        rows, width, pitch = dimensions(spec)
        if 3*pitch*4 > ub_bytes:
            raise ValueError("queried UB capacity insufficient")
        grid = min(rows, aiv_count)
        for arm in ('before', 'after'):
            name = f'{rows}-{width}-{arm}'
            binary = output/f'{name}.so'
            translation_unit = output/f'{name}.cpp'
            translation_unit.write_bytes(source.read_bytes())
            command = [str(compiler), '-xcce', '-std=c++17', '-O2', '-fPIC', '-shared',
                       '-save-temps=obj', '--cce-aicore-arch=dav-c220',
                       f'-DW006_N06_ROWS={rows}', f'-DW006_N06_WIDTH={width}',
                       f'-DW006_N06_GRID={grid}', f'-DW006_N06_AFTER={int(arm=="after")}',
                       *[f'-I{p}' for p in includes], str(translation_unit), '-o', str(binary)]
            build_shared(command, source_paths=[source, translation_unit, Path(__file__), output/'sdk-declarations.json'],
                         binary=binary, output=output/f'build-{name}', source_revision='wave006-N06')


def verify_binary(binary, rows, width, arm):
    library = ctypes.CDLL(str(binary))
    for key, expected in (('rows',rows), ('width',width), ('after',int(arm=='after'))):
        function = getattr(library, 'affine_'+key); function.restype = ctypes.c_uint32
        if function() != expected:
            raise ValueError('compiled native shape/arm contract mismatch')
    library.affine_grid.restype = ctypes.c_uint32
    grid = library.affine_grid()
    if not 1 <= grid <= rows:
        raise ValueError('invalid native grid')
    return grid


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = [Path(__file__)]

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('AFFINE_DIRECTORY', '.')).resolve()
        self.prepared = {}
        self.output_bits = {}
        self.first_outputs = {}

    def prepare(self, arm, spec, fixture, output):
        rows, width, pitch = dimensions(spec)
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        name = f'{rows}-{width}-{arm}'
        binary = self.directory/f'{name}.so'
        manifest = json.loads((self.directory/f'build-{name}/build.json').read_text())
        grid = verify_binary(binary, rows, width, arm)
        import torch_npu
        inputs = [fixture[key].to(device='npu:0', dtype=torch.float32).clone() for key in ('x','gamma','beta')]
        compound = isinstance(fixture['expected'], tuple)
        storages = [torch.full((rows*width+64,), -8191., dtype=torch.float32, device='npu:0') for _ in range(2 if compound else 1)]
        results = [s[32:-32].view(rows,width) for s in storages]
        items = [NativeArm(binary, inputs, result,
                 sources=[Path(__file__), Path(__file__).with_suffix('.cpp'), self.directory/f'build-{name}/build.json', self.directory/'sdk-declarations.json'],
                 source_revision='wave006-N06', build_manifest=manifest,
                 stream_provider=lambda: torch.npu.current_stream().npu_stream) for result in results]
        item = items[0]
        item.metadata.update(backend='ascendc-cabi-v1', rows=rows, width=width, ub_pitch=pitch,
            reserved_ub_bytes=3*pitch*4, grid=grid, active_cores=(rows+((rows+grid-1)//grid)-1)//((rows+grid-1)//grid),
            component_API_calls=2 if compound else 1, caller_coefficient_copies=2 if compound else 0)
        chunk = (rows+grid-1)//grid
        item.metadata['logical_coefficient_request_bytes'] = 2*width*4*((rows+chunk-1)//chunk if arm=='after' else rows)
        if compound:
            replacements = [fixture[k].to(device='npu:0',dtype=torch.float32) for k in ('gamma1','beta1')]
            original_run = item.run
            def run():
                original_run()
                torch.npu.synchronize()
                first = results[0].cpu()
                if not torch.allclose(first, fixture['expected'][0], atol=2e-6, rtol=2e-6):
                    raise RuntimeError('compound first call failed before coefficient update')
                if not all(torch.equal(inputs[i].cpu().view(torch.int32), fixture[k].view(torch.int32))
                           for i,k in ((1,'gamma'),(2,'beta'))):
                    raise RuntimeError('compound first call mutated coefficients')
                self.first_outputs[arm] = first
                # Same caller-owned pointers, intentionally updated between full calls.
                inputs[1].copy_(replacements[0]); inputs[2].copy_(replacements[1])
                items[1].run()
                return tuple(results)
            item.run = run
            original_close = item.close
            def close():
                original_close(); items[1].close()
            item.close = close
        self.prepared[arm] = (items, storages, inputs, results)
        return item

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        items, storages, inputs, results = self.prepared[arm]
        expected = fixture['expected'] if isinstance(fixture['expected'],tuple) else (fixture['expected'],)
        outputs = ((self.first_outputs[arm], results[1].cpu()) if len(results)==2
                   else (results[0].cpu(),))
        accurate = all(torch.allclose(y,e,atol=2e-6,rtol=2e-6) for y,e in zip(outputs,expected))
        exact_oracle = all(torch.equal(y.view(torch.int32),e.view(torch.int32)) for y,e in zip(outputs,expected))
        final_keys = ('x','gamma1','beta1') if len(results)==2 else ('x','gamma','beta')
        immutable = all(torch.equal(x.cpu().view(torch.int32),fixture[k].view(torch.int32)) for x,k in zip(inputs,final_keys))
        canaries = all(bool((s[:32].cpu()==-8191).all() and (s[-32:].cpu()==-8191).all()) for s in storages)
        nonalias = all(y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for y in results for x in inputs)
        self.output_bits[arm] = tuple(y.view(torch.int32) for y in outputs)
        pair_equal = None if 'before' not in self.output_bits or 'after' not in self.output_bits else all(torch.equal(a,b) for a,b in zip(self.output_bits['before'],self.output_bits['after']))
        hashes = [hashlib.sha256(y.numpy().tobytes()).hexdigest() for y in self.output_bits[arm]]
        return dict(passed=accurate and immutable and canaries and nonalias and pair_equal is not False,
                    primary_accuracy=accurate, exact_oracle=exact_oracle, paired_bitwise=pair_equal,
                    output_bits_sha256=hashes, immutable_input=immutable, gm_canaries=canaries,
                    nonalias_output=nonalias, component_API_calls=len(results), caller_coefficient_copies=2 if len(results)==2 else 0)


def factory():
    return Adapter()
