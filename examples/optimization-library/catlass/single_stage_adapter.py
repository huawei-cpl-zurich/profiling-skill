"""Controlled scheduler/outer-K specialization with identical general fallback."""
import ctypes
import json
import os
import shutil
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, digest
from compile_tile import PIN
CASE_SOURCES = [Path(__file__)]


def check_spec(spec):
    for key in ('m', 'n', 'k'):
        if type(spec.get(key)) is not int or not 0 < spec[key] <= 32768 or spec[key] % 16:
            raise ValueError('dimensions must be positive multiples of 16 at most 32768')
    if type(spec.get('seed')) is not int:
        raise ValueError('integer seed required')


def route(spec, arm, cores):
    check_spec(spec)
    if arm not in ('before', 'after') or type(cores) is not int or cores <= 0:
        raise ValueError('arm and actual positive Cube count required')
    eligible = ((spec['m'] + 127)//128)*((spec['n'] + 255)//256) <= cores and spec['k'] <= 256
    return 'direct' if arm == 'after' and eligible else 'general'


def make_case(spec):
    check_spec(spec)
    m, n, k = [spec[x] for x in ('m', 'n', 'k')]
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    a = (torch.rand((m,k), generator=generator, dtype=torch.float64, device='cpu')*.5-.25).half()
    b = (torch.rand((k,n), generator=generator, dtype=torch.float64, device='cpu')*.5-.25).half()
    return dict(inputs=dict(a=a,b=b), expected=(a.double() @ b.double()).half())


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self, directory=None, cube_cores=None):
        self.directory = Path(directory or os.environ['CATLASS_SINGLE_STAGE_DIRECTORY']).resolve()
        self.cores = cube_cores  # Queried from runtime properties during preparation.
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch_npu
        if self.cores is None:
            self.cores = int(torch.npu.get_device_properties(0).cube_core_num)
        variant = route(spec, arm, self.cores)
        # Separate loader identity per prepared shape; globals never cross cases.
        binary = Path(output)/f'{variant}.so'
        shutil.copyfile(self.directory/f'{variant}.so', binary)
        build_path = self.directory/f'build-{variant}/build.json'
        build = json.loads(build_path.read_text())
        if build['source_revision'] != PIN or build['binary_sha256'] != digest(binary):
            raise ValueError('pinned native build identity mismatch')
        inputs = {name: value.to(device='npu', dtype=torch.float16) for name,value in fixture['inputs'].items()}
        result = torch.full(fixture['expected'].shape, float('nan'), device='npu', dtype=torch.float16)
        library = ctypes.CDLL(str(binary))
        configure = library.catlass_bridge_configure
        configure.argtypes, configure.restype = [ctypes.c_uint32]*4, ctypes.c_int
        if configure(*(spec[x] for x in ('m','n','k')), self.cores):
            raise ValueError('native shape/core gate rejected')
        native = NativeArm(binary, [inputs['a'],inputs['b']], result,
            sources=[Path(__file__), Path(__file__).with_name('single_stage_block.hpp'),
                     Path(__file__).with_name('single_stage_kernel.hpp'), build_path,
                     *[entry['path'] for entry in build['sources']]],
            source_revision=PIN, build_manifest=build,
            stream_provider=lambda: torch.npu.current_stream().npu_stream)
        native.metadata.update(arm=arm, route=variant, cube_cores=self.cores, unit_flag=False,
                               shape=[spec[x] for x in ('m','n','k')])
        self.prepared[arm] = native
        return native

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        result = actual.detach().cpu()
        if result.shape != fixture['expected'].shape or result.dtype != torch.float16 or not torch.isfinite(result).all():
            raise ValueError('output shape/dtype/finiteness/sentinel failure')
        for name, value in zip(('a','b'), self.prepared[arm].inputs):
            if not torch.equal(value.cpu(), fixture['inputs'][name]):
                raise ValueError('input mutated: '+name)
        delta = (result.double()-fixture['expected'].double()).abs()
        passed = bool(torch.all(delta <= .002 + .002*fixture['expected'].double().abs()))
        return dict(passed=passed, max_absolute_error=float(delta.max()), atol=.002, rtol=.002)


def factory():
    return Adapter()
