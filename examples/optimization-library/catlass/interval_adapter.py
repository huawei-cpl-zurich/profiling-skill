"""Prepared E12 native pair. Complete GEMM, no packing or compile in forwards."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm,digest
from interval_contract import CASE_SOURCES,check_spec,copy_plan,make_case,gemm_contract
from compile_tile import PIN


def configure(library,arm,spec,cores):
    check_spec(spec)
    if arm not in ('before','after') or type(cores) is not int or cores<=0:
        raise ValueError('arm and actual positive Cube-core count required')
    library.catlass_interval_after.restype=ctypes.c_uint32
    library.catlass_bridge_kernel_names.restype=ctypes.c_char_p
    if (library.catlass_interval_after()!=int(arm=='after') or
            json.loads(library.catlass_bridge_kernel_names())!=[f'wave2_interval_{arm}']):
        raise ValueError('binary copy mode or kernel arm differs from experiment')
    function=library.catlass_bridge_configure
    function.argtypes,function.restype=[ctypes.c_uint32]*4,ctypes.c_int
    if function(*(spec[key] for key in ('m','n','k')),cores):
        raise ValueError('native shape/core configuration rejected')
    return copy_plan(spec)


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self,directory=None,cube_cores=None):
        self.directory=Path(directory or os.environ['CATLASS_INTERVAL_DIRECTORY']).resolve()
        self.cores=int(cube_cores or os.environ['CATLASS_CUBE_CORES'])
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        check_spec(spec)
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        artifact=f'E12-{arm}'
        binary=self.directory/f'{artifact}.so'
        build_path=self.directory/f'build-{artifact}/build.json'
        build=json.loads(build_path.read_text())
        if build['source_revision']!=PIN or build['binary_sha256']!=digest(binary):
            raise ValueError('native build does not match pinned binary')
        library=ctypes.CDLL(str(binary))
        resources=configure(library,arm,spec,self.cores)
        for name in ('workspace','l1_bytes'):
            getattr(library,'catlass_interval_'+name).restype=ctypes.c_uint64
        if library.catlass_interval_workspace()!=0 or library.catlass_interval_l1_bytes()!=393216:
            raise ValueError('compiled resource contract mismatch')
        import torch_npu
        inputs={name:value.npu() for name,value in fixture['inputs'].items()}
        native=NativeArm(binary,[inputs['a'],inputs['b']],
            torch.empty_like(fixture['expected'],device='npu'),
            sources=[Path(__file__),build_path,*[entry['path'] for entry in build['sources']]],
            source_revision=PIN,build_manifest=build,
            stream_provider=lambda:torch.npu.current_stream().npu_stream)
        native.metadata.update(family='E12',arm=arm,shape=[spec[key] for key in ('m','n','k')],
            cube_cores=self.cores,**resources,torch=torch.__version__,torch_npu=torch_npu.__version__)
        self.prepared[arm]=native
        return native

    def validate(self,arm,actual,fixture,spec):
        torch.npu.synchronize()
        return gemm_contract.validate_actual(actual,fixture,
            dict(zip(('a','b'),self.prepared[arm].inputs)))


def factory():
    return Adapter()
