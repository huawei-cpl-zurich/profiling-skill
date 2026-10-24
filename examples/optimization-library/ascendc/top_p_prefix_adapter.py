"""Prepared native F04; all normalization, selection and fallback stays in forward."""
import ctypes
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from optimization_native_bridge import NativeArm,digest
from top_p_prefix_contract import CASE_SOURCES,dimensions,make_case,diagnostics


class FallbackAgreement:
    """Compare one prepared pair per case, retiring snapshots when either arm closes."""
    def __init__(self):
        self.pending={}

    def prepare(self,arm,spec):
        key=json.dumps(spec,sort_keys=True)
        state=self.pending.get(key)
        if state is None or arm in state['prepared'] or len(state['outputs'])==2:
            state=dict(prepared=set(),outputs={})
            self.pending[key]=state
        state['prepared'].add(arm)
        def retire():
            if self.pending.get(key) is state:self.pending.pop(key)
        return retire

    def validate(self,arm,spec,actual):
        state=self.pending[json.dumps(spec,sort_keys=True)]
        state['outputs'][arm]=tuple(t.detach().cpu().clone() for t in actual)
        if len(state['outputs'])<2:return None
        return all(torch.equal(a,b) for a,b in zip(state['outputs']['before'],state['outputs']['after']))


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):
        self.directory=Path(os.environ['TOP_P_PREFIX_DIRECTORY']).resolve()
        self.prepared={}
        self.fallback_agreement=FallbackAgreement()

    def prepare(self,arm,spec,fixture,output):
        import torch_npu
        rows,vocab,prefix,_=dimensions(spec)
        scores=fixture['scores']
        if scores.dtype!=torch.float16 or not scores.is_contiguous() or not bool(torch.isfinite(scores).all()):
            raise ValueError('finite contiguous FP16 scores required')
        binary=self.directory/f'{arm}.so';build_path=self.directory/f'build-{arm}/build.json'
        build=json.loads(build_path.read_text())
        if digest(binary)!=build['binary_sha256']:raise ValueError('binary provenance mismatch')
        library=ctypes.CDLL(str(binary))
        library.topp_configure.argtypes=[ctypes.c_uint32]*3+[ctypes.c_float]
        library.topp_configure.restype=ctypes.c_int
        if library.topp_configure(rows,vocab,prefix,spec['p']):raise ValueError('native shape guard')
        device=scores.npu();stride=(vocab+31)//32*32
        probability=torch.empty((rows,stride),device=device.device,dtype=torch.float32)
        indices=torch.empty((rows,stride),device=device.device,dtype=torch.int32)
        counts=torch.empty((rows,32),dtype=torch.int32,device=device.device)
        tags=torch.empty_like(counts)
        native=NativeArm(binary,[device,probability,indices,counts],tags,
            sources=[Path(__file__),*CASE_SOURCES,build_path,*[x['path'] for x in build['sources']]],
            source_revision=build['source_revision'],build_manifest=build,
            stream_provider=lambda:torch.npu.current_stream().npu_stream)
        native.metadata.update(arm=arm,shape=spec,launches_per_forward=1,diagnostic_output=str(output),
            workspace_gm_bytes=rows*stride*4,output_bytes=rows*stride*4+rows*256,
            local_buffer_bytes=6144,preallocation='all buffers before correctness; every used element initialized in forward',
            denominator='FP32 input-order scan of vector FP32 exponentials',prefix_scan='left-to-right scalar FP32',
            fallback='identical device full heapsort and crossing',torch=torch.__version__,torch_npu=torch_npu.__version__)
        def run():native.run();return indices[:,:vocab],counts[:,0],tags[:,0]
        item=SimpleNamespace(run=run,expected_kernels=native.expected_kernels,sources=native.sources,
            metadata=native.metadata,compilation_state=native.compilation_state,native=native,inputs=device,close=native.close)
        if spec.get('near_boundary'):
            retire=self.fallback_agreement.prepare(arm,spec)
            def close():
                try:native.close()
                finally:retire()
            item.close=close
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        torch.npu.synchronize()
        result=diagnostics(actual,fixture,arm)
        result['returned_counts']=actual[1].detach().cpu().tolist()
        result['returned_tags']=actual[2].detach().cpu().tolist()
        if not result['passed']:torch.save(tuple(t.cpu() for t in actual),Path(self.prepared[arm].metadata['diagnostic_output'])/'failed-actual.pt')
        if not torch.equal(self.prepared[arm].inputs.cpu(),fixture['scores']):result.update(passed=False,reason='input mutation')
        # The guard's uniform tied payload must match baseline support exactly.
        if spec.get('near_boundary'):
            agreement=self.fallback_agreement.validate(arm,spec,actual)
            if agreement is not None:
                result['exact_fallback_agreement']=agreement
                result['passed'] &= agreement
        return result


def factory():return Adapter()
