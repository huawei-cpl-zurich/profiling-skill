"""Prepared GDN output component; precomputed state/metadata, not full GDN."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import hashlib,json,re,torch
from gdn_output_contract import parameters,layout,make_case,check_fixture,backing,halo,CANARY,ATOL,RTOL
HERE=Path(__file__).resolve().parent


def backend():
    import torch_npu
    import gdn_output_generated as kernels
    return 'npu:0',kernels


def guard(device):return torch.npu.device(device) if str(device).split(':')[0]=='npu' else nullcontext()
def identity(items):return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in items]


def typed_policy(c,expected):
    ir=c.asm.get('ttir','');ir=ir.decode() if isinstance(ir,bytes) else ir
    dots=[line for line in ir.splitlines() if re.search(r'\btt\.dot\b',line)]
    if len(dots)!=expected or any(line.count('xf16')<2 or 'xf32' not in line or 'xbf16' in line for line in dots):raise ValueError('actual F16 dot inputs/F32 accumulator-output required')
    return dict(dot_count=len(dots),dot_lines=dots,ttir_sha256=hashlib.sha256(ir.encode()).hexdigest(),source_boundary='gated score explicitly cast to V F16 before score-V dot')


class Adapter:
    case_sources=[HERE/'gdn_output_contract.py'];make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None
    def compile_probe(self,arm,spec,output):return self._prepare(arm,spec,None)
    def prepare(self,arm,spec,fixture,output):
        check_fixture(spec,fixture);return self._prepare(arm,spec,fixture['inputs'])
    def _prepare(self,arm,spec,cpu):
        t,nt=parameters(spec)
        if arm not in ('before','after'):raise ValueError('before/after required')
        device,k=backend();buffers=[];inputs={};compiled=[];raw=[];policies=[];closed=False
        def allocate(shape,dtype,value=None):
            h=128//torch.empty((),dtype=dtype,device='cpu').element_size();host=torch.full((__import__('math').prod(shape)+2*h,),CANARY,dtype=dtype,device='cpu')
            if value is not None:host[h:-h].view(shape).copy_(value)
            with guard(device):buffer=host.to(device)
            buffers.append(buffer);return buffer[h:-h].view(shape)
        for name,(shape,dtype) in layout(t).items():inputs[name]=allocate(shape,dtype,None if cpu is None else cpu[name])
        out=allocate((1,t,8,128),torch.float16);scratch=[allocate(out.shape,torch.float32) for _ in range(2)] if arm=='before' else []
        physical=[]
        for name,x in list(inputs.items())+[('out',out)]+list(zip(('state','local'),scratch)):
            fmt=None
            if x.device.type=='npu':
                import torch_npu
                with guard(device):fmt=int(torch_npu.get_npu_format(x));size=int(torch_npu.get_storage_size(x))
                if fmt!=2 or x.data_ptr()%128 or size<x.numel()+2*halo(x):raise ValueError('aligned actual ND physical storage required')
            if not x.is_contiguous() or x.storage_offset()!=halo(x) or backing(x).numel()!=x.numel()+2*halo(x):raise ValueError('owned dense full storage/128B halos required')
            physical.append(dict(name=name,shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),format=fmt))
        if len({x.untyped_storage().data_ptr() for x in [*inputs.values(),out,*scratch]})!=8+len(scratch):raise ValueError('device aliases')
        flags=dict(H=8,Hg=8,K=128,V=128,BT=64,BK=128,BV=128,IS_VARLEN=True,USE_G=True,num_warps=4,num_stages=2,enable_fp_fusion=False)
        def args(output):return (inputs['q'],inputs['k'],inputs['v'],inputs['h'],inputs['g'],output,inputs['cu'],inputs['offsets'],.5,t)
        calls=[(k.gdn_output_fused,args(out),(1,8),flags,3)] if arm=='after' else [(k.gdn_output_state,args(scratch[0]),(1,8),flags,1),(k.gdn_output_local,args(scratch[1]),(1,8),flags,2),(k.gdn_output_combine,(*scratch,out,t*8*128,256),((t*8*128+255)//256,),dict(num_warps=4,num_stages=2,enable_fp_fusion=False),0)]
        def compile_only():
            if closed:raise RuntimeError('prepared item closed')
            if not compiled:
                pending=[];proof=[]
                with guard(device):
                    for kernel,a,grid,kw,dots in calls:
                        c=kernel.warmup(*a,grid=grid,**kw);raw.append(c)
                        if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin'].startswith(b'\x7fELF'):raise ValueError('linked executable NPU ELF required')
                        proof.append(typed_policy(c,dots));pending.append(c)
                policies.extend(proof);compiled.extend(pending)
            return list(compiled)
        def run():
            if cpu is None:raise RuntimeError('input-only prototype cannot execute')
            compile_only()
            with guard(device):
                for kernel,a,grid,kw,_ in calls:kernel[grid](*a,**kw)
            return out
        state=dict(inputs=inputs,out=out,buffers=buffers,scratch=scratch)
        def close():
            nonlocal closed,out
            if closed:return
            with guard(device):closed=True;calls.clear();compiled.clear();raw.clear();inputs.clear();buffers.clear();scratch.clear();out=None;state.clear()
            if self.items.get(arm) is item:self.items.pop(arm);self.numeric.pop(arm,None)
        item=SimpleNamespace(run=run,close=close,compile_only=compile_only,compilation_state=lambda:identity(compiled),raw_artifacts=raw,type_policies=policies,state=state,
            expected_kernels=[kernel.fn.__name__ for kernel,*_ in calls],sources=[Path(__file__),*self.case_sources,HERE/'gdn_output_generated.py',HERE/'gdn_output_helper.py',HERE/'gdn_output_codegen.py',*sorted(p for p in (HERE/'vendor').rglob('*') if p.is_file())],
            metadata=dict(arm=arm,prototype=cpu is None,parameters=flags,physical=physical,prepared_device=str(device),api_kernels=len(calls),reused_outputs=True,scope='complete prepared output component only; state/offsets/commonH2D/allocations outside BOTH; all3 split stages vs exact1 fused stage charged'))
        if cpu is not None:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key or arm in self.items:self.numeric.clear();self.current=key
            self.items[arm]=item
        return item
    def capture(self,run,item,output):
        with guard(item.metadata['prepared_device']):
            import sys
            helpers=HERE.parents[3]/'skills/ascend-model-benchmarking/scripts'
            if str(helpers) not in sys.path:sys.path.insert(0,str(helpers))
            from profile_ten import capture_ten
            return capture_ten(run,output,expected_kernel=item.expected_kernels,metric_group='PipeUtilization')
    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];s=item.state;e=fixture['expected'];check_fixture(spec,fixture)
        if not isinstance(actual,torch.Tensor) or actual is not s['out'] or actual.dtype!=torch.float16 or actual.shape!=e.shape or not actual.is_contiguous():raise ValueError('private dense F16 output ABI required')
        with guard(item.metadata['prepared_device']):
            raws={x.untyped_storage().data_ptr():x.detach().cpu() for x in s['buffers']};raw=raws[actual.untyped_storage().data_ptr()];value=raw[halo(actual):-halo(actual)].view(e.shape).clone();diff=(value.double()-e.double()).abs();bad=(~torch.isfinite(value))|(diff>ATOL+RTOL*e.double().abs());coords=bad.nonzero()
            immutable=all(torch.equal(raws[x.untyped_storage().data_ptr()],backing(fixture['inputs'][name])) for name,x in s['inputs'].items());canary=all(bool((x[:128//x.element_size()]==CANARY).all() and (x[-128//x.element_size():]==CANARY).all()) for x in raws.values())
        other=self.numeric.get('after' if arm=='before' else 'before');self.numeric[arm]=value
        paired=None if other is None else bool(torch.isclose(value.double(),other.double(),atol=ATOL,rtol=RTOL).all());bits=None if other is None else bool(torch.equal(value.view(torch.int16),other.view(torch.int16)))
        return dict(passed=not len(coords) and immutable and canary and paired is not False,mismatch_count=len(coords),first_mismatches=[dict(coordinates=c.tolist(),actual=str(float(value[tuple(c)])),expected=float(e[tuple(c)])) for c in coords[:8]],max_absolute_error=float(diff.max()) if bool(torch.isfinite(value).all()) else None,nonfinite_count=int((~torch.isfinite(value)).sum()),canaries=canary,inputs_immutable=immutable,paired_numerical=paired,paired_all_bits_secondary=bits,CPU_all_bits_secondary=bool(torch.equal(value.view(torch.int16),e.view(torch.int16))),atol=ATOL,rtol=RTOL)


def factory():return Adapter()
