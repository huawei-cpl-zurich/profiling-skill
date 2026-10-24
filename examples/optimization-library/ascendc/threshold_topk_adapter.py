"""Source-owned F03 pair: tile sort/merge with and without threshold compaction."""
import ctypes
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm,digest
from threshold_topk_contract import CASE_SOURCES,dimensions,make_case,diagnostics


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self,directory=None):
        self.directory=Path(directory or os.environ['THRESHOLD_TOPK_DIRECTORY']).resolve()
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        import torch_npu
        if arm not in ('before','after'): raise ValueError('known arm required')
        rows,vocab,k,_=dimensions(spec)
        scores=fixture['scores']
        if scores.dtype!=torch.float16 or tuple(scores.shape)!=(rows,vocab) or not scores.is_contiguous():
            raise ValueError('contiguous FP16 score matrix required')
        if not bool(torch.isfinite(scores).all()): raise ValueError('finite scores required')
        binary=self.directory/f'{arm}.so'
        build_path=self.directory/f'build-{arm}/build.json'
        build=json.loads(build_path.read_text())
        if build['binary_sha256']!=digest(binary): raise ValueError('native binary provenance mismatch')
        library=ctypes.CDLL(str(binary))
        library.threshold_configure.argtypes=[ctypes.c_uint32]*3
        library.threshold_configure.restype=ctypes.c_int
        if library.threshold_configure(rows,vocab,k): raise ValueError('native shape guard rejected')
        device=scores.npu()
        values=torch.empty((rows,k),dtype=torch.float16,device=device.device)
        indices=torch.empty((rows,k),dtype=torch.int32,device=device.device)
        native=NativeArm(binary,[device,indices],values,
            sources=[Path(__file__),*CASE_SOURCES,build_path,*[e['path'] for e in build['sources']]],
            source_revision=build['source_revision'],build_manifest=build,
            stream_provider=lambda:torch.npu.current_stream().npu_stream)
        if native.expected_kernels!=[f'wave003_topk_{arm}']: raise ValueError('native selector mismatch')
        native.metadata.update(arm=arm,shape=spec,torch=torch.__version__,torch_npu=torch_npu.__version__,
            tile=8192,k_limit=1024,tie_policy='arbitrary valid unique tied indices',
            mechanism='running strict-GT compaction' if arm=='after' else 'all tiles fully sorted, no pruning',
            output_contract='descending exact K values and matching unique original indices',
            launches_per_forward=1,workspace_gm_bytes=0,local_buffer_bytes=196608)
        def run():
            native.run()
            return values,indices
        item=SimpleNamespace(run=run,expected_kernels=native.expected_kernels,sources=native.sources,
            metadata=native.metadata,compilation_state=native.compilation_state,native=native,inputs=device,close=native.close)
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        torch.npu.synchronize()
        result=diagnostics(actual,fixture)
        if not result['passed']:
            values,indices=(item.detach().cpu() for item in actual)
            result['index_bounds']=[int(indices.min()),int(indices.max())]
            result['first_indices']=indices[:,:16].tolist()
            result['first_values']=values[:,:16].tolist()
        if not torch.equal(self.prepared[arm].inputs.cpu(),fixture['scores']):
            result.update(passed=False,reason='input mutation')
        return result


def factory(): return Adapter()
