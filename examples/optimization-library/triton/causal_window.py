"""Compile-inspection study only: numerical API unavailable on this source/backend."""
from pathlib import Path
from types import SimpleNamespace
import torch,hashlib,json
from causal_window_contract import parameters,block,check_inputs,input_case,equal


def backend():
    import torch_npu
    from causal_window_source import _causal_conv1d_update_kernel_no_cache_len_no_mtp
    from causal_window_kernels import _direct_state_window
    return 'npu:0',_causal_conv1d_update_kernel_no_cache_len_no_mtp,_direct_state_window


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def reject_numeric(*args,**kwargs):
    raise RuntimeError('numeric API unqualified: baseline requires missing triton.language.insert_slice; use explicit compile_probe inspection only')


class Adapter:
    case_sources=[Path(__file__).with_name('causal_window_contract.py')]
    make_case=staticmethod(reject_numeric)

    def __init__(self):self.state={};self.actual={}

    def compile_probe(self,arm,spec,output):
        folder=Path(output) if output is not None else None
        def retain(status,error=None,artifacts=()):
            if folder is None:return
            folder.mkdir(parents=True,exist_ok=True);attempt=1+sum(path.stem[8:].isdigit() for path in folder.glob('attempt-*.json'))
            for i,c in enumerate(artifacts):
                for key,value in c.asm.items():
                    if isinstance(value,str):value=value.encode()
                    if isinstance(value,bytes):(folder/f'attempt-{attempt}-kernel-{i}.{key}').write_bytes(value)
            paths=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('causal_window_source.py','causal_window_kernels.py','causal_window_screen.py')]]
            report=dict(status=status,numeric_qualification=False,device_forwards=0,arm=arm,spec=spec,error=error,compilation=identity(artifacts),source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
            (folder/f'attempt-{attempt}.json').write_text(json.dumps(report,indent=2)+'\n')
        try:item=self._prepare(arm,spec,input_case(spec))
        except Exception as error:
            retain('preparation-failed',dict(type=type(error).__name__,message=str(error)));raise
        original=item.compile_only
        def compile_only():
            try:artifacts=original()
            except Exception as error:
                retain('compile-unsupported',dict(type=type(error).__name__,message=str(error)));raise
            retain('compile-only',artifacts=artifacts);return artifacts
        def reject():raise RuntimeError('input-only compile probe cannot launch')
        item.run=reject;item.compile_only=compile_only;item.metadata.update(compile_only_probe=True,device_kernel_calls=0,numeric_qualification=False)
        return item

    def prepare(self,arm,spec,fixture,output):
        reject_numeric()

    def _prepare(self,arm,spec,cpu):
        p=parameters(spec);check_inputs(cpu,p)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,source,direct=backend();kernel=source if arm=='before' else direct;inputs={};buffers=[]
        for name,value in cpu.items():
            if name in ('x','state'):
                host=torch.full((value.numel()+64,),12345.,dtype=torch.float32);host[32:-32].view(value.shape).copy_(value)
                buffer=host.to(device);buffers.append(buffer);inputs[name]=buffer[32:-32].view(value.shape)
            else:inputs[name]=value.clone().to(device)
        x,state=inputs['x'],inputs['state'];compiled=[];grid=(p['B'],1,1)
        args=(x,state,inputs['weight'],inputs.get('bias'),inputs.get('indices'),x,-1)
        flags=dict(batch=p['B'],dim=p['D'],align_val=16,state_len=3,seq_len=1,width=4,out_len=1,x_batch_stride=x.stride(0),conv_batch_stride=state.stride(0),out_batch_stride=x.stride(0),DIM_BLOCK=block(p['D']),HAS_BIAS=p['bias'],SILU_ACTIVATION=p['silu'],IS_CONTINUOUS_BATCHING=p['indexed'],USE_PAD_SLOT=True,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,grid=grid,**flags)
                if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin']:raise RuntimeError('missing executable binary')
                compiled.append(c)
            return compiled
        def run():
            compile_only();kernel[grid](*args,**flags);return x
        if arm in self.state:self.state.clear();self.actual.clear()
        self.state[arm]=dict(inputs=inputs,buffers=buffers)
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),compiled=lambda:list(compiled),
            sources=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('causal_window_source.py','causal_window_kernels.py','causal_window_screen.py')]],expected_kernels=[kernel.fn.__name__],
            metadata=dict(arm=arm,component='fullseq1 causal_conv1d_update',DIM_BLOCK=block(p['D']),grid=grid,flags=flags,device_kernel_calls=1,
            input_strides={n:list(v.stride()) for n,v in inputs.items()},public_output_alias='x',mutable_inputs=['x','state'],
            profile_semantics='once-correctness-advanced x/state then TEN MORE evolving updates per arm; NO resets, NO postprofile inspection',
            allocator_scope='private independent per-arm x/state allocated/transferred ONCE atprepare; immutable weights; no allocator/fixed-token ordinary latency claim'))

    def validate(self,arm,actual,fixture,spec):
        p=parameters(spec);state=self.state[arm];inputs=state['inputs'];expected,new_state=fixture['expected']
        if not isinstance(actual,torch.Tensor) or actual is not inputs['x'] or actual.dtype!=torch.float32 or actual.shape!=expected.shape or actual.stride()!=expected.stride():return dict(passed=False,reason='source output must alias private x with source shape/dtype/stride')
        value=actual.detach().cpu();error=(value-expected).abs();allowed=2e-6+2e-5*expected.abs()
        ok=bool(torch.isfinite(value).all() and (error<=allowed).all()) if p['silu'] else equal(value,expected)
        state_exact=equal(inputs['state'],new_state,bits=True);ok &= state_exact
        for name in ('weight','bias','indices'):
            if name in inputs:ok &= equal(inputs[name],fixture['inputs'][name],bits=True)
        for buffer in state['buffers']:
            host=buffer.detach().cpu();ok &= bool((host[:32]==12345).all() and (host[-32:]==12345).all())
        self.actual[arm]=value.clone();other='before' if arm=='after' else 'after';result={}
        if other in self.actual:result['paired_exact_bits']=equal(value,self.actual[other],bits=True)
        result.update(passed=bool(ok),out_alias_x=True,state_all_bits=state_exact,max_absolute_error=float(error.max()),atol=2e-6,rtol=2e-5,non_silu_exact=not p['silu'])
        return result


def factory():return Adapter()


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache');p.add_argument('--provenance');p.add_argument('--correctness-only',action='store_true')
    p.add_argument('--inspect-compile',action='store_true');p.add_argument('--arm',choices=('before','after'),default='before');a=p.parse_args()
    if not a.inspect_compile:reject_numeric()  # Before cache/oracle/backend/input transfers or even case IO.
    spec=json.loads(Path(a.case).read_text());Adapter().compile_probe(a.arm,spec,a.output).compile_only()
    return 0


if __name__=='__main__':raise SystemExit(main())
