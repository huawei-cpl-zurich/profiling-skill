"""Prepared B=H1 dense featurewise KKT component; not public-wrapper qualification."""
import hashlib,json,re
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch
from kkt_contract import dimensions,shapes,guarded,backing,check_fixture,make_case,CANARY,ATOL,RTOL
HERE=Path(__file__).resolve().parent


def backend():
    import torch_npu
    import kkt_generated
    return 'npu:0',kkt_generated,torch_npu


def guard(device):return torch.npu.device(device) if str(device).split(':')[0]=='npu' else nullcontext()
def identity(compiled):return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest() for k,v in c.asm.items() if isinstance(v,(str,bytes))}) for c in compiled]


def ieee_policy(c,required):
    ir=c.asm.get('ttir','');ir=ir.decode() if isinstance(ir,bytes) else ir;lines=[x for x in ir.splitlines() if re.search(r'\btt\.dot\b',x)];default=getattr(getattr(c,'metadata',None),'default_dot_input_precision',None)
    if required and not lines:raise ValueError('compiled IEEE F32 dot required')
    origins=[]
    for line in lines:
        p=re.search(r'inputPrecision\s*=\s*([A-Za-z0-9_]+)',line)
        if line.count('xf32')<3 or 'xf16' in line or 'xbf16' in line or ('inputPrecision' in line and p is None) or (p and p[1]!='ieee') or (not p and default!='ieee'):raise ValueError('actual F32 IEEE precision required')
        origins.append('explicitIR' if p else 'actualCompilerMetadataDefaultIEEE')
    return dict(dot_lines=lines,origins=origins,compiler_default=default,raw_TTIR_sha256=hashlib.sha256(ir.encode()).hexdigest())


class Adapter:
    case_sources=[HERE/'kkt_contract.py'];make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None
    def compile_probe(self,arm,spec,output):return self._prepare(arm,spec,None)
    def prepare(self,arm,spec,fixture,output):check_fixture(spec,fixture);return self._prepare(arm,spec,fixture['inputs'])
    def _prepare(self,arm,spec,cpu):
        t,k,bt=dimensions(spec)
        if arm not in ('before','after'):raise ValueError('arm')
        device,module,npu=backend();stores=[];inputs={};compiled=[];raw_artifacts=[];policies=[];closed=False;probe=cpu is None
        def allocate(shape,value=None):
            n=__import__('math').prod(shape);raw=torch.full((n+64,),CANARY,dtype=torch.float32,device='cpu')
            if value is not None:raw[32:-32].view(shape).copy_(value)
            with guard(device):store=raw.to(device)
            stores.append(store);return store[32:-32].view(shape)
        for name,shape in shapes(t,k).items():inputs[name]=allocate(shape,None if probe else cpu[name])
        out=allocate((1,t,1,bt));physical=[]
        for name,x in list(inputs.items())+[('out',out)]:
            fmt=2 if npu is None else int(npu.get_npu_format(x))
            if not x.is_contiguous() or x.storage_offset()!=32 or backing(x).numel()!=x.numel()+64 or (npu is not None and x.data_ptr()%128) or fmt!=2:raise ValueError('aligned owned dense ND F32 physical storage required')
            physical.append(dict(name=name,shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),format=fmt))
        if len({x.untyped_storage().data_ptr() for x in [*inputs.values(),out]})!=4:raise ValueError('private nonalias inputs/output required')
        inter=module.kkt_full_inter if arm=='before' else module.kkt_lower_inter;args=(inputs['k'],inputs['gk'],inputs['beta'],out,None,None,t)
        common=dict(H=1,K=k,BT=bt,BC=16,BK=k,IS_VARLEN=False,num_warps=4,enable_fp_fusion=False)
        calls=[(module.kkt_zero,(out,t*bt),dict(BLOCK=256,num_warps=4),((t*bt+255)//256,)),(inter,args,dict(common,NC=bt//16,num_stages=3),((t+bt-1)//bt,(bt//16)**2,1)),(module.kkt_diagonal,args,common,((t+bt-1)//bt,bt//16,1))]
        def compile_only():
            if closed:raise RuntimeError('prepared item closed')
            if not compiled:
                pending=[];proof=[]
                with guard(device):
                    for index,(kernel,a,flags,grid) in enumerate(calls):
                        c=kernel.warmup(*a,grid=grid,**flags);pending.append(c);raw_artifacts.append(c)
                        if not isinstance(c.asm.get('npubin'),bytes) or not c.asm['npubin'].startswith(b'\x7fELF'):raise ValueError('actual linked NPU ELF required')
                        proof.append(ieee_policy(c,index==1))
                compiled.extend(pending);policies.extend(proof)
            return list(compiled)
        def run():
            if probe:raise RuntimeError('input-only prototype cannot run')
            with guard(device):
                compile_only()
                for kernel,a,flags,grid in calls:kernel[grid](*a,**flags)
            return out
        def close():
            nonlocal closed,out
            if closed:return
            closed=True;calls.clear();compiled.clear();inputs.clear();stores.clear();out=None;item.output=None
            if self.items.get(arm)is item:self.items.pop(arm);self.numeric.pop(arm,None)
        item=SimpleNamespace(run=run,close=close,compile_only=compile_only,compilation_state=lambda:identity(compiled),raw_artifacts=raw_artifacts,IEEE_policies=policies,
            sources=[Path(__file__),*self.case_sources,HERE/'kkt_generated.py',HERE/'kkt_codegen.py',*sorted(x for x in (HERE/'vendor').rglob('*') if x.is_file())],expected_kernels=[kernel.fn.__name__ for kernel,_,_,_ in calls],
            metadata=dict(arm=arm,prototype=probe,prepared_device=str(device),component='B=H1 dense featurewise KKT',parameters=spec.get('parameters',spec),physical_layouts=physical,device_kernel_calls=3,common_charged_zero=True,source_intra_common=True,allocator_excluded=True,compiler_managed_scratch='UNKNOWN if unreported',route='optimized' if arm=='after' else 'reference'),inputs=inputs,output=out,stores=stores)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key:self.items={};self.numeric={};self.current=key
            self.items[arm]=item
        return item
    def capture(self,run,item,output):
        with guard(item.metadata['prepared_device']):
            from profile_ten import capture_ten
            return capture_ten(run,output,expected_kernel=item.expected_kernels,metric_group='PipeUtilization')
    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];check_fixture(spec,fixture)
        if not isinstance(actual,torch.Tensor) or actual is not item.output or actual.dtype!=torch.float32 or actual.shape!=fixture['expected']['ideal'].shape:return dict(passed=False,reason='private complete F32 output required')
        with guard(item.metadata['prepared_device']):raw=item.stores[-1].detach().cpu();inputs={name:backing(x).detach().cpu() for name,x in item.inputs.items()}
        value=raw[32:-32].view(actual.shape);finite=bool(torch.isfinite(value).all());checks={}
        for name,expected in fixture['expected'].items():
            diff=(value.double()-expected.double()).abs();bad=(~torch.isfinite(value))|(diff>ATOL+RTOL*expected.double().abs());coords=bad.nonzero()
            checks[name]=dict(passed=not len(coords),mismatch_count=len(coords),max_absolute_error=float(diff.max())if finite else None,first_coordinates=coords[:8].tolist())
        halo=bool((raw[:32]==CANARY).all() and (raw[-32:]==CANARY).all());immutable=all(torch.equal(x,backing(fixture['inputs'][name]))for name,x in inputs.items());other=self.numeric.get('after'if arm=='before'else'before');self.numeric[arm]=value.clone()
        paired=other is None or bool(torch.isclose(value,other,atol=ATOL,rtol=RTOL).all());bits=None if other is None else bool(torch.equal(value.view(torch.int32),other.view(torch.int32)))
        return dict(passed=finite and all(x['passed']for x in checks.values())and halo and immutable and paired,outputs=checks,canaries=halo,inputs_immutable=immutable,paired_numeric=paired,paired_all_bits_secondary=bits,atol=ATOL,rtol=RTOL)


def factory():return Adapter()
