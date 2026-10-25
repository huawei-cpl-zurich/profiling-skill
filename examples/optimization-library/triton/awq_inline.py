"""Exact typed-source AWQ GEMM, prepared split-one component."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from awq_inline_contract import CASE_SOURCES,dimensions,validate_inputs,check_output
from awq_inline_types import classify

class Adapter:
    case_sources=CASE_SOURCES
    def __init__(self,*,compile_inspection=False):self.compile_inspection=compile_inspection;self.model=None;self.qualification_key=None;self.prepared={};self.bits={};self.pair_key=None
    def make_case(self,*args):raise RuntimeError('R09 compiler/type qualification failed; scientific CPU payload generation disabled')

    def _device_item(self,arm,spec,v,probe=False):
        m,k,n,g=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,triton
        from awq_inline_kernel import awq_dequantize_kernel,awq_gemm_kernel,materialized_gemm
        storage=torch.empty(m*n+2,dtype=torch.float16,device=v['A'].device);out=storage[1:-1].view(m,n)
        scratch=torch.empty(k*n+2,dtype=torch.float16,device=v['A'].device) if arm=='before' else None
        if not probe:
            storage[0]=-701;storage[-1]=-701
            if scratch is not None:scratch[0]=-701;scratch[-1]=-701
        grid=((m+15)//16*((n+31)//32),1);flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        if arm=='before':
            b=scratch[1:-1].view(k,n)
            calls=[(awq_dequantize_kernel,((n//8+15)//16,(k+15)//16),(v['Q'],v['S'],v['Z'],g,b,n//8,k),dict(BLOCK_SIZE_X=16,BLOCK_SIZE_Y=16,**flags)),
                (materialized_gemm,grid,(v['A'],b,out,m,n,k),dict(BM=16,BN=32,BK=32,**flags))]
        else:calls=[(awq_gemm_kernel,grid,(v['A'],v['Q'],out,v['Z'],v['S'],m,n,k,g),
                dict(BLOCK_SIZE_M=16,BLOCK_SIZE_N=32,BLOCK_SIZE_K=32,SPLIT_K=1,**flags))]
        compiled=[];attempts=[]
        def compile_only():
            if not compiled:
                pending=[]
                for kernel,grid,args,kw in calls:
                    artifact=kernel.warmup(*args,grid=grid,**kw);attempts.append(artifact);pending.append(artifact)
                compiled.extend(pending)
            return list(compiled)
        def run():
            if probe:raise RuntimeError('input-only probe cannot run')
            compile_only()
            for kernel,grid,args,kw in calls:kernel[grid](*args,**kw)
            return out
        def compilation_state():
            return [dict(hash=a.hash,binaries={key:hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest() for key,x in a.asm.items() if isinstance(x,(str,bytes))}) for a in compiled]
        def typed():
            a=compile_only()[-1]
            return classify(a.asm.get('ttir',''),a.asm.get('ttadapter',''))
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=compilation_state,typed=typed,
            attempted_artifacts=attempts,inputs=v,output_storage=storage,scratch=scratch,
            sources=[Path(__file__),Path(__file__).with_name('awq_inline_kernel.py'),Path(__file__).with_name('awq_inline_types.py'),*CASE_SOURCES],
            expected_kernels=[x[0].fn.__name__ for x in calls],metadata=dict(arm=arm,grids=[x[1] for x in calls],
                api_component_count=len(calls),flags=flags,tiles=dict(BM=16,BN=32,BK=32,SPLIT_K=1),
                declared_accumulator='C element FP16; tl.dot DEFAULT signature',scope='prepared split-one AWQ GEMM; private C/workspace reused; zeros allocation and degenerate split sum excluded BOTH'))

    def probe(self,arm,spec):
        if not self.compile_inspection:raise RuntimeError('R09 unqualified; explicit compile_inspection=True required')
        m,k,n,g=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('invalid arm')
        import torch,torch_npu
        v={name:torch.empty(shape,dtype=dtype,device='npu') for name,shape,dtype in
            [('A',(m,k),torch.float16),('Q',(k,n//8),torch.int32),('Z',(k//g,n//8),torch.int32),('S',(k//g,n),torch.float16)]}
        return self._device_item(arm,spec,v,True)

    def _qualify_historical(self,spec):
        records=[self.probe(arm,spec).typed() for arm in ('before','after')]
        if records[0]['model']!=records[1]['model']:raise ValueError('arm accumulator types differ')
        self.model=records[0]['model'];self.qualification_key=json.dumps(spec,sort_keys=True);return records

    def prepare(self,arm,spec,fixture,output):
        raise RuntimeError("R09 compiler/type qualification failed; numeric execution disabled")

    def _prepare_historical(self,arm,spec,fixture,output):
        dimensions(spec);validate_inputs(fixture['inputs'],spec)
        if self.model not in ('fp16','fp32') or self.qualification_key!=json.dumps(spec,sort_keys=True):raise RuntimeError('case-specific compile/type/engine qualification required before input moves')
        import torch_npu
        key=json.dumps(spec,sort_keys=True)
        if key!=self.pair_key or arm in self.prepared:self.prepared={};self.bits={};self.pair_key=key
        item=self._device_item(arm,spec,{n:x.npu() for n,x in fixture['inputs'].items()});item.metadata['model']=self.model;self.prepared[arm]=item;return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm];result,got=check_output(actual,fixture['expected'],item.metadata['model'])
        halos=all(bool((x[[0,-1]].cpu()==-701).all()) for x in (item.output_storage,item.scratch) if x is not None)
        immutable=all(torch.equal(x.cpu().view(torch.uint8),fixture['inputs'][n].view(torch.uint8)) for n,x in item.inputs.items())
        private=isinstance(actual,torch.Tensor) and all(actual.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() for x in item.inputs.values())
        pair=None;zero_signs=None
        if got is not None:
            self.bits[arm]=got.clone()
            if len(self.bits)==2:
                a,b=self.bits['before'],self.bits['after'];same=a.view(torch.int16)==b.view(torch.int16)
                pair=bool((same|((a==0)&(b==0))).all());zero_signs=int(((a==0)&(b==0)&~same).sum())
        result.update(canaries=halos,immutable=immutable,nonalias=private,paired_nonzero_bits=pair,paired_zero_sign_differences=zero_signs)
        result['passed']=result['passed'] and halos and immutable and private and pair is not False
        return result

def factory():return Adapter()
