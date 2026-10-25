"""Unqualified compile-inspection mean study; default numerical workflow rejects."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,torch
from mean_inner_contract import parameters,strides,check_inputs,input_case,make_case,bits


def backend():
    import torch_npu
    from mean_inner_source import mean_kernel
    from mean_inner_kernels import _mean_inner
    return 'npu:0',mean_kernel,_mean_inner


def identity(compiled):
    return [dict(hash=str(c.hash),name=c.name,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def reject_numeric(*args,**kwargs):
    raise RuntimeError('numeric API unqualified: frozen paired FP32 bit gate failed (18 outputs in first case); explicit compile_probe inspection only')


class Adapter:
    case_sources=[Path(__file__).with_name('mean_inner_contract.py')]
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
            paths=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('mean_inner_source.py','mean_inner_kernels.py','mean_inner_screen.py')]]
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
        device,source,coalesced=backend();m,n,k=(p[t] for t in ('M','N','K'));buffers=[]
        def allocate(shape,initial=None):
            size=1
            for v in shape:size*=v
            host=torch.full((size+64,),12345.,dtype=torch.float32)
            if initial is not None:host[32:-32].view(shape).copy_(initial)
            buf=host.to(device);buffers.append(buf);return buf[32:-32].view(shape)
        transposed=p.get('physical_layout')=='MKN-backing'
        x=allocate((m,k,n) if transposed else (m,n,k),cpu['x'].transpose(1,2) if transposed else cpu['x'])
        if transposed:x=x.transpose(1,2)
        out=allocate((m,k));optimized=arm=='after' and k>1 and x.stride(2)==1;kernel=coalesced if optimized else source
        args=(x,out,*x.stride(),*out.stride(),m,n,k,1024)+( (8,) if optimized else ())
        grid=(m,(k+7)//8) if optimized else (m*k,);flags=dict(num_stages=1,multibuffer=False,enable_fp_fusion=False);compiled=[]
        def compile_only():
            if not compiled:
                c=kernel.warmup(*args,grid=grid,**flags)
                if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin']:raise RuntimeError('missing executable binary')
                compiled.append(c)
            return compiled
        def run():compile_only();kernel[grid](*args,**flags);return out
        if arm in self.state:self.state.clear();self.actual.clear()
        self.state[arm]=dict(x=x,out=out,buffers=buffers)
        return SimpleNamespace(run=run,compile_only=compile_only,compilation_state=lambda:identity(compiled),compiled=lambda:list(compiled),expected_kernels=[kernel.fn.__name__],
            sources=[Path(__file__),*self.case_sources,*[Path(__file__).with_name(n) for n in ('mean_inner_source.py','mean_inner_kernels.py','mean_inner_screen.py')]],
            metadata=dict(arm=arm,route='optimized' if optimized else 'reference',grid=grid,flags=flags,device_kernel_calls=1,input_strides=list(x.stride()),output_strides=list(out.stride()),
            K_TILE=8 if optimized else 1,BLOCK_SIZE=1024,reused_outputs=True,allocator_scope='private guarded output/inputs prepared ONCE outside BOTH APIs; one kernel each; no allocator or ordinary latency claim'))

    def validate(self,arm,actual,fixture,spec):
        p=parameters(spec);s=self.state[arm];e=fixture['expected']
        if not isinstance(actual,torch.Tensor) or actual is not s['out'] or actual.dtype!=torch.float32 or actual.shape!=e.shape or actual.stride()!=(p['K'],1) or actual.data_ptr()==s['x'].data_ptr():return dict(passed=False,reason='private contiguous FP32 MK output required')
        value=actual.detach().cpu();error=(value.double()-e).abs();bad=~torch.isfinite(value)|(error>2e-5+2e-5*e.abs());coords=bad.nonzero();diagnostics=[]
        for coord in coords[:8].tolist():
            idx=tuple(coord);diagnostics.append(dict(coordinate=coord,expected=float(e[idx]),actual=repr(float(value[idx])),absolute_error=repr(float(error[idx]))))
        immutable=bits(s['x'],fixture['inputs']['x']);canaries=all(bool((b[:32].cpu()==12345).all() and (b[-32:].cpu()==12345).all()) for b in s['buffers'])
        duplicate=p['M']<2 or bits(value[:1],value[1:2]);self.actual[arm]=value.clone();other='before' if arm=='after' else 'after';paired=other not in self.actual or bits(value,self.actual[other])
        paired_coords=(value.view(torch.int32)!=self.actual[other].view(torch.int32)).nonzero().tolist() if other in self.actual else []
        return dict(paired_mismatch_count=len(paired_coords),paired_first_mismatches=[dict(coordinate=coord,actual_bits=int(value.view(torch.int32)[tuple(coord)]),other_bits=int(self.actual[other].view(torch.int32)[tuple(coord)])) for coord in paired_coords[:8]],passed=bool(not coords.numel() and immutable and canaries and duplicate and paired),mismatch_count=len(coords),first_mismatches=diagnostics,nonfinite_count=int((~torch.isfinite(value)).sum()),
            max_absolute_error=float(error.max()),atol=2e-5,rtol=2e-5,input_immutable=immutable,canaries=canaries,duplicate_rows_bits=duplicate,paired_exact_bits=paired if other in self.actual else None,zero_sign_gate='paired int32 bits including signed zero; FP64 oracle numerical tolerance')


def factory():return Adapter()


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cache');parser.add_argument('--provenance');parser.add_argument('--correctness-only',action='store_true')
    parser.add_argument('--inspect-compile',action='store_true');parser.add_argument('--arm',choices=('before','after'),default='before');a=parser.parse_args()
    if not a.inspect_compile:reject_numeric()
    spec=json.loads(Path(a.case).read_text());Adapter().compile_probe(a.arm,spec,a.output).compile_only()
    return 0


if __name__=='__main__':raise SystemExit(main())
