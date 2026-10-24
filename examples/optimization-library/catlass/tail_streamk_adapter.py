"""Prepared tail-only StreamK/control kernels; no work is hidden outside forwards."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from optimization_native_bridge import NativeArm, digest
from compile_tile import PIN
import tail_streamk_contract as contract


def configure(library, arm, spec, cores):
    resources = contract.geometry(spec, cores)
    library.catlass_tail_after.restype = ctypes.c_uint32
    library.catlass_bridge_kernel_names.restype = ctypes.c_char_p
    if arm not in ('before', 'after') or library.catlass_tail_after() != int(arm == 'after'):
        raise ValueError('compiled scheduling arm mismatch')
    if json.loads(library.catlass_bridge_kernel_names()) != [f'w003_tail_streamk_{arm}']:
        raise ValueError('compiled kernel selector mismatch')
    library.catlass_bridge_configure.argtypes = [ctypes.c_uint32] * 4
    library.catlass_bridge_configure.restype = ctypes.c_int
    if library.catlass_bridge_configure(*(spec[k] for k in ('m', 'n', 'k')), cores):
        raise ValueError('native configuration or hardware synchronization unavailable')
    library.catlass_tail_workspace.restype = ctypes.c_uint64
    workspace = library.catlass_tail_workspace()
    if workspace != 2 * 20 * 128 * 256 * 4:
        raise ValueError('unexpected native workspace capacity')
    return dict(resources, workspace_bytes=workspace, shuffle_k=True, operand_stages=2,
                split_tail=arm == 'after', cube_cores=cores)


class Adapter:
    case_sources = contract.CASE_SOURCES
    make_case = staticmethod(contract.make_case)

    def __init__(self, directory=None, cores=None):
        self.directory = Path(directory or os.environ['CATLASS_TAIL_DIRECTORY']).resolve()
        self.cores = int(cores or os.environ['CATLASS_CUBE_CORES'])
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        contract.check_spec(spec)
        import torch_npu
        binary = self.directory / f'{arm}.so'
        build_path = self.directory / f'build-{arm}/build.json'
        build = json.loads(build_path.read_text())
        if build['source_revision'] != PIN or build['binary_sha256'] != digest(binary):
            raise ValueError('build does not match pinned binary')
        inputs = {name: tensor.npu() for name, tensor in fixture['inputs'].items()}
        library = ctypes.CDLL(str(binary))
        resources = configure(library, arm, spec, self.cores)
        # Every consumed partial is overwritten by its owning Cube task in each
        # forward. Neither this buffer nor output contents are initial inputs.
        workspace = torch.empty(resources['workspace_bytes'], dtype=torch.uint8, device='npu')
        result = torch.full(fixture['expected'].shape, float('nan'), dtype=torch.float16, device='npu')
        native = NativeArm(binary, [inputs['a'], inputs['b'], workspace], result,
            sources=[Path(__file__), contract.__file__, build_path,
                     *[entry['path'] for entry in build['sources']]],
            source_revision=PIN, build_manifest=build,
            stream_provider=lambda: torch.npu.current_stream().npu_stream)
        native.metadata.update(arm=arm, shape=[spec[k] for k in ('m', 'n', 'k')],
                               torch=torch.__version__, torch_npu=torch_npu.__version__, **resources)
        self.prepared[arm] = native
        return native

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        inputs = dict(zip(('a', 'b'), self.prepared[arm].inputs[:2]))
        return contract.validate(actual, fixture, inputs)


def factory():
    return Adapter()
