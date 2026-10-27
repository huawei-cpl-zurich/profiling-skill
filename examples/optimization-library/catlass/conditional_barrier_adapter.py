"""Native 2201 completion-event/unit-flag pair; no compile/allocation in forwards."""
import ctypes
import json
import os
import shutil
import uuid
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm, digest
from compile_tile import PIN
from conditional_barrier_contract import CASE_SOURCES, check_spec, make_case, validate_actual


def isolate_binary(binary, output):
    """Give each prepared case its own loader image for the stateful C ABI.

    The copied bytes retain the compiled binary digest. A distinct filename
    prevents dlopen from sharing mutable shape configuration across cases.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    private = output/f'conditional-barrier-{uuid.uuid4().hex}.so'
    shutil.copyfile(binary,private)
    if digest(private) != digest(binary):
        raise ValueError('private loader image differs from compiled binary')
    return private


def configure(library, arm, spec, cores):
    tasks = check_spec(spec, cores)
    if arm not in ('before','after'):
        raise ValueError('unknown arm')
    library.catlass_stock_barrier_enabled.restype = ctypes.c_uint32
    library.catlass_bridge_kernel_names.restype = ctypes.c_char_p
    if library.catlass_stock_barrier_enabled() != int(arm=='after') or json.loads(
            library.catlass_bridge_kernel_names().decode()) != [f'wave006_conditional_barrier_{arm}']:
        raise ValueError('binary does not implement the requested arm')
    function = library.catlass_bridge_configure
    function.argtypes, function.restype = [ctypes.c_uint32]*4, ctypes.c_int
    if function(*(spec[key] for key in ('m','n','k')), cores):
        raise ValueError('native configuration rejected')
    for key in ('m','n'):
        query = getattr(library,'catlass_tile_'+key)
        query.restype = ctypes.c_uint32
        if query() != spec['tile_'+key]:
            raise ValueError('binary helper dimensions mismatch')
    return dict(tasks=tasks, cube_cores=cores, enable_unit_flag=False,
                stock_conditional_barrier=arm=='after', helper_m=spec['m'],helper_n=spec['n'],
                l1_tile=[spec['m'],spec['n'],256], l0_tile=[spec['m'],spec['n'],64], stages=2)


class Composite:
    def __init__(self, components):
        self.components = components
        self.sources = sorted(set(path for item in components for path in item.sources))
        self.expected_kernels = components[0].expected_kernels
        self.metadata = dict(components=[item.metadata for item in components],
                             component_gemms_per_api=len(components))

    def run(self):
        return tuple(item.run() for item in self.components)

    def compilation_state(self):
        return [item.compilation_state() for item in self.components]

    def close(self):
        for item in self.components:
            item.close()


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self, directory=None, cube_cores=None):
        self.directory = Path(directory or os.environ['CATLASS_CONDITIONAL_BARRIER_DIRECTORY']).resolve()
        self.cores = cube_cores if cube_cores is not None else int(os.environ['CATLASS_CUBE_CORES'])
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        check_spec(spec,self.cores)
        import torch_npu
        resource = self.directory/f'{spec["tile_m"]}x{spec["tile_n"]}'
        binary = resource/f'{arm}.so'
        build_path = resource/f'build-{arm}/build.json'
        build = json.loads(build_path.read_text())
        if build['source_revision'] != PIN or build['binary_sha256'] != digest(binary):
            raise ValueError('pinned build identity mismatch')
        binary = isolate_binary(binary,output)
        resources = configure(ctypes.CDLL(str(binary)),arm,spec,self.cores)
        components = []
        for part in fixture.get('components',(fixture,)):
            inputs = {name:value.to(device='npu',dtype=torch.float16) for name,value in part['inputs'].items()}
            backing = torch.full((part['expected'].numel()+32,),123.,device='npu',dtype=torch.float16)
            result = backing[16:-16].view(part['expected'].shape)
            result.fill_(float('nan'))
            item = NativeArm(binary,[inputs['a'],inputs['b']],result,
                sources=[Path(__file__),build_path,*[entry['path'] for entry in build['sources']]],
                source_revision=PIN,build_manifest=build,
                stream_provider=lambda:torch.npu.current_stream().npu_stream)
            item.output_canary = backing
            item.metadata.update(resources,arm=arm,torch=torch.__version__,torch_npu=torch_npu.__version__,
                                 component_gemms_per_api=1,output_sentinel='NaN fully overwritten')
            components.append(item)
        native = components[0] if len(components)==1 else Composite(components)
        self.prepared[arm] = native
        return native

    def validate(self, arm, actual, fixture, spec):
        torch.npu.synchronize()
        native = self.prepared[arm]
        components = native.components if isinstance(native,Composite) else [native]
        parts = fixture.get('components',(fixture,))
        outputs = actual if isinstance(actual,tuple) else (actual,)
        for item in components:
            if not bool((item.output_canary[:16] == 123).all()) or not bool((item.output_canary[-16:] == 123).all()):
                raise ValueError('output canary changed')
        diagnostics = [validate_actual(value,part,dict(zip(('a','b'),item.inputs)))
                       for value,part,item in zip(outputs,parts,components)]
        return dict(passed=all(item['passed'] for item in diagnostics),
                    component_gemms=len(components),components=diagnostics)


def factory():
    return Adapter()
