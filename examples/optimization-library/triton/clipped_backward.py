"""Retained resource-failed clipped backward study; numeric execution is disabled."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import struct
import torch
from clipped_backward_contract import check_spec
from clipped_backward_tiles import _pick_block_sizes,_DTYPE_UB_BUDGET_BWD

FAILURE=('T04 failed study: candidate requires 214016 bytes of UB but 196608 bytes are available '
         'for the frozen configuration. No scientific numeric gate passed; execution is disabled. '
         'See experiments/optimization-wave008/T04-result.json.')


class StudyUnsupported(ValueError):
    """The recorded candidate failed resource qualification before correctness."""


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
            for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


class Adapter:
    case_sources=[Path(__file__).with_name('clipped_backward_contract.py')]

    def __init__(self,*,compile_inspection=False):self.compile_inspection=compile_inspection

    @staticmethod
    def make_case(spec):raise StudyUnsupported(FAILURE)

    def prepare(self,*args,**kwargs):raise StudyUnsupported(FAILURE)

    def capture(self,*args,**kwargs):raise StudyUnsupported(FAILURE)

    def validate(self,*args,**kwargs):raise StudyUnsupported(FAILURE)

    def compile_probe(self,arm,spec,output):
        """Explicit inspection: uninitialized prototypes only; never a numeric API."""
        if not self.compile_inspection:raise StudyUnsupported(FAILURE)
        if arm not in ('before','after'):raise ValueError('paired arm required')
        p=check_spec(spec);m,k=p['M'],p['K']
        import torch_npu
        from clipped_backward_kernels import _clipped_swiglu_bwd_kernel,_saturated_clipped_bwd_kernel
        inputs=dict(gup=torch.empty((m,2*k),dtype=torch.float32,device='npu:0'),
                    dy=torch.empty((m,k),dtype=torch.float32,device='npu:0'))
        buffer=torch.empty((m*2*k+64,),dtype=torch.float32,device='npu:0');out=buffer[32:-32].view(m,2*k)
        bm,bk=_pick_block_sizes(m,k,torch.float32,_DTYPE_UB_BUDGET_BWD)
        kernel=_clipped_swiglu_bwd_kernel if arm=='before' else _saturated_clipped_bwd_kernel
        args=(inputs['gup'],inputs['dy'],out,m,k,1.702,7.)
        kwargs=dict(BLOCK_M=bm,BLOCK_K=bk,LARGE_TENSOR=False,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        grid=((m+bm-1)//bm,(k+bk-1)//bk);compiled=[]
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,**kwargs,grid=grid);data=c.asm.get('npubin')
                if not isinstance(data,bytes) or not data.startswith(b'\x7fELF'):raise ValueError('missing actual npubin')
                offset=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
                sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
                table=sections[names];strings=data[table[4]:table[4]+table[5]]
                if not any(strings[t[0]:].split(b'\0',1)[0]==b'.text' and t[5]>0 for t in sections):raise ValueError('missing executable text')
                compiled.append(c)
            return compiled
        def run():raise StudyUnsupported(FAILURE)
        sources=[__file__,self.case_sources[0],Path(__file__).with_name('clipped_backward_kernels.py'),Path(__file__).with_name('clipped_backward_tiles.py')]
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),sources=sources,
            expected_kernels=[kernel.fn.__name__],prototype_inputs=inputs,prototype_output=out,
            metadata=dict(component='complete clipped SwiGLU backward',execution_enabled=False,prototype=True,
                configuration=dict(BLOCK_M=bm,BLOCK_K=bk),grid=grid,input_strides={n:list(v.stride()) for n,v in inputs.items()},
                scope='uninitialized compile-inspection storage; no expected fixture or numeric launch'))


def factory():raise StudyUnsupported(FAILURE)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('case','output','cache','provenance'):p.add_argument('--'+key,required=True)
    p.add_argument('--correctness-only',action='store_true');p.parse_args()
    factory()  # Reject before loading cases/provenance, generating expected data or importing backends.


if __name__=='__main__':raise SystemExit(main())
