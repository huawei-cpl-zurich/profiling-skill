"""Prepared native families 06/07; every forward launches one complete kernel."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, digest
from parallelism_contract import CASE_SOURCES, check_spec, make_case, gemm_contract
from compile_tile import PIN


def configure(library, family, arm, shape, cores):
    """Read the actual compiled heuristic and workspace size through the C ABI."""
    if family not in ('06', '07') or arm not in ('before', 'after') or type(cores) is not int or cores <= 0:
        raise ValueError('an arm and actual positive Cube-core count are required')
    library.catlass_parallelism_family.restype = ctypes.c_uint32
    if library.catlass_parallelism_family() != int(family):
        raise ValueError('binary family does not match case')
    library.catlass_bridge_kernel_names.argtypes = []
    library.catlass_bridge_kernel_names.restype = ctypes.c_char_p
    mechanism = 'splitk' if family == '06' else 'buffering'
    expected_names = [f'wave1_{mechanism}_{arm}']
    if json.loads(library.catlass_bridge_kernel_names().decode()) != expected_names:
        raise ValueError('binary kernel identities do not match the requested arm')
    function = library.catlass_bridge_configure
    function.argtypes, function.restype = [ctypes.c_uint32]*4, ctypes.c_int
    if function(*shape, cores):
        raise ValueError('native shape/core/synchronization configuration rejected')
    library.catlass_parallelism_factor.restype = ctypes.c_uint32
    library.catlass_parallelism_workspace.restype = ctypes.c_uint64
    factor = library.catlass_parallelism_factor()
    workspace = library.catlass_parallelism_workspace()
    split = family == '06' and arm == 'after'
    expected = 4 * shape[0] * shape[1] * factor if split else 0
    if factor < 1 or factor > 16 or workspace != expected or (not split and factor != 1):
        raise ValueError('compiled workspace/factor violates the declared contract')
    return dict(split_factor=factor, workspace_bytes=workspace,
                operand_stages=(2 if arm == 'after' else 1) if family == '07' else 2,
                output_tiles=((shape[0]+127)//128)*((shape[1]+255)//256))


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self, directory=None, cube_cores=None):
        self.directory = Path(directory or os.environ['CATLASS_PARALLELISM_DIRECTORY']).resolve()
        self.cores = int(cube_cores or os.environ['CATLASS_CUBE_CORES'])
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        check_spec(spec)
        import torch_npu
        shape = [spec[key] for key in ('m','n','k')]
        artifact = f'{spec["family"]}-{arm}'
        binary = self.directory/f'{artifact}.so'
        build_path = self.directory/f'build-{artifact}/build.json'
        build = json.loads(build_path.read_text())
        if build['source_revision'] != PIN or build['binary_sha256'] != digest(binary):
            raise ValueError('native build does not match the pinned binary')
        # NPU activation precedes the sync-address query; no kernel is launched.
        inputs = {name: value.npu() for name,value in fixture['inputs'].items()}
        library = ctypes.CDLL(str(binary))
        resources = configure(library, spec['family'], arm, shape, self.cores)
        workspace = torch.empty(resources['workspace_bytes'], dtype=torch.uint8, device='npu')
        result = torch.empty_like(fixture['expected'], device='npu')
        native = NativeArm(binary, [inputs['a'],inputs['b'],workspace], result,
            sources=[Path(__file__), build_path, *[entry['path'] for entry in build['sources']]],
            source_revision=PIN, build_manifest=build,
            stream_provider=lambda: torch.npu.current_stream().npu_stream)
        native.metadata.update(family=spec['family'], arm=arm, shape=shape, cube_cores=self.cores,
                               **resources, torch=torch.__version__, torch_npu=torch_npu.__version__)
        self.prepared[arm] = native
        return native

    def validate(self, arm, actual, fixture, spec):
        # Input checks exclude the intentionally mutable partial-result workspace.
        torch.npu.synchronize()
        inputs = dict(zip(('a','b'), self.prepared[arm].inputs[:2]))
        return gemm_contract.validate_actual(actual, fixture, inputs)


def factory():
    return Adapter()
