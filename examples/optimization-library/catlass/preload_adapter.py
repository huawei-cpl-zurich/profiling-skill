"""Prepared native families 08/09/11: one complete device launch per forward."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm,digest
from preload_contract import CASE_SOURCES,check_spec,make_case,gemm_contract
from compile_tile import PIN


def configure(library,family,arm,shape,cores):
    if arm not in ('before','after') or type(cores) is not int or cores<=0:
        raise ValueError('arm and actual positive Cube-core count required')
    spec=dict(family=family,m=shape[0],n=shape[1],k=shape[2])
    check_spec(spec)
    for name in ('family','after'):
        getattr(library,'catlass_preload_'+name).restype=ctypes.c_uint32
    if library.catlass_preload_family()!=int(family) or library.catlass_preload_after()!=int(arm=='after'):
        raise ValueError('binary family/arm differs from experiment')
    function=library.catlass_bridge_configure
    function.argtypes,function.restype=[ctypes.c_uint32]*4,ctypes.c_int
    if function(*shape,cores):
        raise ValueError('native shape/core configuration rejected')
    for name in ('workspace','l1_bytes'):
        getattr(library,'catlass_preload_'+name).restype=ctypes.c_uint64
    reserved=library.catlass_preload_l1_bytes()
    expected=512*1024 if family=='09' and arm=='after' else 384*1024
    if library.catlass_preload_workspace()!=0 or reserved!=expected:
        raise ValueError('compiled resource contract mismatch')
    tasks=((shape[0]+127)//128)*((shape[1]+255)//256)
    return dict(workspace_bytes=0,l1_reserved_bytes=reserved,output_tasks=tasks,
                max_tasks_per_core=(tasks+cores-1)//cores,k_tiles=(shape[2]+255)//256,
                source_reuse_opportunity=family=='09' and tasks>cores,
                shuffle_k=family=='11' and arm=='after',
                next_task_preload=family=='11' or family=='08' and arm=='after')


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self,directory=None,cube_cores=None):
        self.directory=Path(directory or os.environ['CATLASS_PRELOAD_DIRECTORY']).resolve()
        self.cores=int(cube_cores or os.environ['CATLASS_CUBE_CORES'])
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        check_spec(spec)
        import torch_npu
        shape=[spec[key] for key in ('m','n','k')]
        artifact=f'{spec["family"]}-{arm}'
        binary=self.directory/f'{artifact}.so'
        build_path=self.directory/f'build-{artifact}/build.json'
        build=json.loads(build_path.read_text())
        if build['source_revision']!=PIN or build['binary_sha256']!=digest(binary):
            raise ValueError('native build does not match pinned binary')
        library=ctypes.CDLL(str(binary))
        resources=configure(library,spec['family'],arm,shape,self.cores)
        inputs={name:value.npu() for name,value in fixture['inputs'].items()}
        result=torch.empty_like(fixture['expected'],device='npu')
        native=NativeArm(binary,[inputs['a'],inputs['b']],result,
            sources=[Path(__file__),build_path,*[entry['path'] for entry in build['sources']]],
            source_revision=PIN,build_manifest=build,
            stream_provider=lambda:torch.npu.current_stream().npu_stream)
        native.metadata.update(family=spec['family'],arm=arm,shape=shape,cube_cores=self.cores,
                               **resources,torch=torch.__version__,torch_npu=torch_npu.__version__)
        self.prepared[arm]=native
        return native

    def validate(self,arm,actual,fixture,spec):
        torch.npu.synchronize()
        return gemm_contract.validate_actual(actual,fixture,dict(zip(('a','b'),self.prepared[arm].inputs)))


def factory():
    return Adapter()
