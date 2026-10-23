"""Family05 native pilot. Prebuild before.so/after.so; no compile in forwards."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, digest

PIN = '40aced927fdaaa0f3eaa9c6f5c5d859e06392f55'


def balanced_shape(cube_cores):
    if not isinstance(cube_cores, int) or cube_cores < 3:
        raise ValueError('actual device Cube-core count is required')
    choices = [(a*b, -a*(b+1), a, b) for b in range(3, 32, 2)
               for a in range(1, cube_cores+1) if a*b <= cube_cores < a*(b+1)]
    if not choices:
        raise ValueError('no supported tile balance case for this device')
    _, _, a, b = max(choices)
    return 256*a, 128*b, 6144


class Adapter:
    def __init__(self, directory=None, cube_cores=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton'))
        import gemm_contract
        self.contract = gemm_contract
        self.case_sources = [gemm_contract.__file__]
        self.make_case = gemm_contract.make_case
        self.directory = Path(directory or os.environ['CATLASS_NATIVE_DIRECTORY']).resolve()
        self.cube_cores = int(cube_cores or os.environ['CATLASS_CUBE_CORES'])
        if self.cube_cores <= 0:
            raise ValueError('actual device Cube-core count is required')
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        if spec.get('b_layout', 'row') != 'row' or any(spec[key] % 16 for key in ('m','n','k')):
            raise ValueError('pilot requires row-major B and aligned dimensions')
        import torch_npu  # Register backend before transferring tensors.
        binary = self.directory / f'{arm}.so'
        build = json.loads((self.directory / f'build-{arm}/build.json').read_text())
        if build['binary_sha256'] != digest(binary) or build['source_revision'] != PIN:
            raise ValueError('native build identity does not match pinned binary')
        native = NativeArm(binary, [fixture['inputs']['a'].npu(), fixture['inputs']['b'].npu()],
            torch.empty_like(fixture['expected'], device='npu'),
            sources=[entry['path'] for entry in build['sources']],
            source_revision=PIN, build_manifest=build,
            stream_provider=lambda: torch.npu.current_stream().npu_stream)
        configure = native._library.catlass_bridge_configure
        configure.argtypes = [ctypes.c_uint32]*4
        configure.restype = ctypes.c_int
        shape = [spec[key] for key in ('m','n','k')]
        if configure(*shape, self.cube_cores):
            native.close()
            raise ValueError('native shape/core configuration rejected')
        native.metadata.update(shape=shape, cube_cores=self.cube_cores,
                               torch=torch.__version__, torch_npu=torch_npu.__version__)
        self.prepared[arm] = native
        return native

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        inputs = dict(zip(('a', 'b'), self.prepared[arm].inputs))
        return self.contract.validate_actual(actual, fixture, inputs)



def adapter():
    return Adapter()
