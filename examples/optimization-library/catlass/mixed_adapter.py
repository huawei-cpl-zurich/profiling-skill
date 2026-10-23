"""Native mixed examples; preparation never launches a correctness/warmup call."""
import ctypes
import json
import os
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm,digest
from mixed_contract import CASE_SOURCES,check_spec,make_padding_case,gemm_contract
from compile_tile import PIN


def topology(cube,vector,configured):
    """The AIV-only add launch and mixed epilogue must have equal capacity."""
    if any(type(value) is not int or value<=0 for value in (cube,vector,configured)) or cube!=configured or vector!=2*cube:
        raise ValueError('native mixed examples require observed Vector=2*Cube and matching configured Cube count')
    return dict(observed_cube_cores=cube,observed_vector_cores=vector)


def configure(library,family,arm,shape,cores):
    if family not in ('02','10') or arm not in ('before','after') or type(cores) is not int or cores<=0:
        raise ValueError('known family/arm and observed positive Cube-core count required')
    library.catlass_mixed_family.restype=ctypes.c_uint32
    if library.catlass_mixed_family()!=int(family):
        raise ValueError('binary family mismatch')
    library.catlass_bridge_kernel_names.argtypes=[]
    library.catlass_bridge_kernel_names.restype=ctypes.c_char_p
    expected_names=([f'wave1_padding_{arm}'] if family=='10' else
                    ['wave1_add_matmul','wave1_add_vector'] if arm=='before' else ['wave1_add_fused'])
    if json.loads(library.catlass_bridge_kernel_names().decode())!=expected_names:
        raise ValueError('binary kernel identities do not match the requested arm')
    library.catlass_bridge_configure.argtypes=[ctypes.c_uint32]*4
    library.catlass_bridge_configure.restype=ctypes.c_int
    if library.catlass_bridge_configure(*shape,cores):
        raise ValueError('native dimensions or hardware synchronization rejected')
    library.catlass_mixed_workspace.restype=ctypes.c_uint64
    size=library.catlass_mixed_workspace()
    m,n,k=shape
    if family=='02' and size!=4*m*n or family=='10' and ((arm=='before' and size!=0) or (arm=='after' and size<2*(m*k+k*n))):
        raise ValueError('workspace does not match the declared operation')
    return dict(workspace_bytes=size,conversion_forced=family=='10' and arm=='after',
                launch_count=2 if family=='02' and arm=='before' else 1,
                standalone_add_vector_blocks=2*cores if family=='02' and arm=='before' else 0,
                standalone_add_logical_blocks=cores if family=='02' and arm=='before' else 0,
                intermediate_gm_bytes=4*m*n if family=='02' else 0)


class Adapter:
    def __init__(self,family,directory=None,cube_cores=None):
        if family not in ('02','10'): raise ValueError('family must be 02 or 10')
        self.family=family
        self.directory=Path(directory or os.environ['CATLASS_MIXED_DIRECTORY']).resolve()
        self.cores=int(cube_cores or os.environ['CATLASS_CUBE_CORES'])
        self.case_sources=[Path(gemm_contract.__file__)] if family=='02' else CASE_SOURCES
        self.make_case=gemm_contract.make_case if family=='02' else make_padding_case
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        check_spec(spec,self.family)
        import torch_npu
        name=f'{self.family}-{arm}'
        binary=self.directory/(name+'.so')
        build_path=self.directory/f'build-{name}/build.json'
        build=json.loads(build_path.read_text())
        if build['source_revision']!=PIN or build['binary_sha256']!=digest(binary):
            raise ValueError('native build provenance mismatch')
        inputs={key:value.npu() for key,value in fixture['inputs'].items()}
        properties=torch.npu.get_device_properties(inputs['a'].device.index)
        observed=topology(getattr(properties,'cube_core_num',None),
                          getattr(properties,'vector_core_num',None),self.cores)
        library=ctypes.CDLL(str(binary))
        resources=configure(library,self.family,arm,[spec[k] for k in ('m','n','k')],self.cores)
        workspace=torch.empty(resources['workspace_bytes'],dtype=torch.uint8,device='npu')
        addend=inputs.get('z')
        if addend is None: addend=torch.empty(0,dtype=torch.uint8,device='npu')
        result=torch.empty_like(fixture['expected'],device='npu')
        native=NativeArm(binary,[inputs['a'],inputs['b'],addend,workspace],result,
            sources=[Path(__file__),build_path,*[entry['path'] for entry in build['sources']]],
            source_revision=PIN,build_manifest=build,
            stream_provider=lambda:torch.npu.current_stream().npu_stream)
        native.metadata.update(family=self.family,arm=arm,shape=spec,cube_cores=self.cores,
                               **observed,**resources,torch=torch.__version__,torch_npu=torch_npu.__version__)
        self.prepared[arm]=(native,inputs)
        return native

    def validate(self,arm,actual,fixture,spec):
        torch.npu.synchronize()
        return gemm_contract.validate_actual(actual,fixture,self.prepared[arm][1])


def epilogue_factory(): return Adapter('02')
def padding_factory(): return Adapter('10')
