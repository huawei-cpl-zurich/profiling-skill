"""Prepared dense KDA final-dA two-stage tile comparison, not full backward."""
import hashlib,json,re
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch
from kda_da_contract import dimensions,backing,check_fixture,make_case,CANARY
HERE=Path(__file__).resolve().parent


def backend():
    import torch_npu
    import kda_da_generated
    return 'npu:0',kda_da_generated,torch_npu


def guard(device):return torch.npu.device(device)if str(device).split(':')[0]=='npu'else nullcontext()
def identity(compiled):return [dict(hash=str(c.hash),binaries={k:hashlib.sha256(v.encode()if isinstance(v,str)else v).hexdigest()for k,v in c.asm.items()if isinstance(v,(str,bytes))})for c in compiled]


def typed_policy(c,tile):
    text=c.asm.get('ttir','');text=text.decode()if isinstance(text,bytes)else text
    dots=[line for line in text.splitlines()if re.search(r'\btt\.dot\b',line)]
    if not dots or any(line.count('xf16')!=2 or line.count('xf32')!=1 for line in dots):raise ValueError('actual F16 operand / F32 accumulator dot required')
    if any(re.findall(r'tensor<(\d+)x(\d+)xf16',line)!=[(str(tile),str(tile))]*2 for line in dots):raise ValueError('actual selected square dot tile required')
    signature=re.search(r'tt\.func[^\n]*',text)
    if not signature or re.findall(r'!tt\.ptr<([^>,]+)',signature[0])[:3]!=['f16','f32','f32']:raise ValueError('actual A-half / P-or-tmp-F32 / output-F32 pointers required')
    casts=[line for line in text.splitlines()if 'arith.truncf' in line and 'xf32' in line and 'xf16' in line]
    if not casts:raise ValueError('actual F32 temporary/input cast to F16 before dot required')
    return dict(dot_lines=dots,cast_lines=casts,typed_signature=signature[0],raw_TTIR_sha256=hashlib.sha256(text.encode()).hexdigest())


class Adapter:
    case_sources=[HERE/'kda_da_contract.py'];make_case=staticmethod(make_case)
    def __init__(self):self.items={};self.numeric={};self.current=None
    def compile_probe(self,arm,spec,output):return self._prepare(arm,spec,None)
    def prepare(self,arm,spec,fixture,output):check_fixture(spec,fixture);return self._prepare(arm,spec,fixture['inputs'])
    def _prepare(self,arm,spec,cpu):
        shape=dimensions(spec)
        if arm not in('before','after'):raise ValueError('before/after required')
        device,module,npu=backend();b,t,h,bt=shape;tile=64 if arm=='before'else 32;stores=[];inputs={};compiled=[];policies=[];raw_artifacts=[];closed=False;probe=cpu is None
        def allocate(dtype,value=None):
            pad=128//torch.empty((),dtype=dtype,device='cpu').element_size();count=__import__('math').prod(shape);raw=torch.full((count+2*pad,),CANARY,dtype=dtype,device='cpu')
            if value is not None:raw[pad:-pad].view(shape).copy_(value)
            with guard(device):store=raw.to(device)
            stores.append(store);return store[pad:-pad].view(shape)
        inputs['A']=allocate(torch.float16,None if probe else cpu['A']);inputs['P']=allocate(torch.float32,None if probe else cpu['P']);tmp=allocate(torch.float32);out=allocate(torch.float32);physical=[]
        for name,x in list(inputs.items())+[('tmp',tmp),('out',out)]:
            pad=128//x.element_size();fmt=2 if npu is None else int(npu.get_npu_format(x))
            if not x.is_contiguous()or x.storage_offset()!=pad or backing(x).numel()!=x.numel()+2*pad or(npu is not None and x.data_ptr()%128)or fmt!=2:raise ValueError('owned aligned dense ND physical storage required')
            physical.append(dict(name=name,shape=list(x.shape),strides=list(x.stride()),offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),format=fmt))
        if len({x.untyped_storage().data_ptr()for x in[*inputs.values(),tmp,out]})!=4:raise ValueError('private nonalias inputs/tmp/output required')
        right=module.chunk_kda_bwd_kernel_dA_post_right;left=module.chunk_kda_bwd_kernel_dA_post_left;grid=((t+bt-1)//bt,b*h,(bt//tile)**2);flags=dict(H=h,BT=bt,BM=tile,BN=tile,BK=tile,NTILES_N=bt//tile,IS_VARLEN=False,num_warps=2,num_stages=1)
        calls=[(right,(inputs['A'],inputs['P'],tmp,None,None,t)),(left,(inputs['A'],tmp,out,None,None,t))]
        def compile_only():
            if closed:raise RuntimeError('prepared item closed')
            if not compiled:
                pending=[];proof=[]
                with guard(device):
                    for kernel,args in calls:
                        c=kernel.warmup(*args,grid=grid,**flags);raw_artifacts.append(c)
                        if not isinstance(c.asm.get('npubin'),bytes)or not c.asm['npubin'].startswith(b'\x7fELF'):raise ValueError('actual linked NPU ELF required')
                        proof.append(typed_policy(c,tile));pending.append(c)
                compiled.extend(pending);policies.extend(proof)
            return list(compiled)
        def run():
            if probe:raise RuntimeError('input-only prototype cannot run')
            with guard(device):
                compile_only()
                for kernel,args in calls:kernel[grid](*args,**flags)
            return out
        def close():
            nonlocal closed,out,tmp
            if closed:return
            closed=True;calls.clear();compiled.clear();raw_artifacts.clear();inputs.clear();stores.clear();out=None;tmp=None;item.output=None
            if self.items.get(arm)is item:self.items.pop(arm);self.numeric.pop(arm,None)
        item=SimpleNamespace(run=run,close=close,compile_only=compile_only,compiled=lambda:list(compiled),compilation_state=lambda:identity(compiled),typed_policies=policies,raw_artifacts=raw_artifacts,
            sources=[Path(__file__),*self.case_sources,HERE/'kda_da_generated.py',HERE/'kda_da_codegen.py',*sorted(x for x in(HERE/'vendor').rglob('*')if x.is_file())],expected_kernels=[kernel.fn.__name__ for kernel,_ in calls],
            metadata=dict(arm=arm,prototype=probe,prepared_device=str(device),component='dense final-dA two-stage component',parameters=spec.get('parameters',spec),physical_layouts=physical,device_kernel_calls=2,grid=grid,flags=flags,tile=tile,reused_output_and_F32_tmp=True,allocator_excluded=True,compiler_managed_scratch='UNKNOWN if unreported',route='optimized'if arm=='after'else'reference'),inputs=inputs,output=out,stores=stores)
        if not probe:
            key=json.dumps(spec,sort_keys=True)
            if self.current!=key:self.items={};self.numeric={};self.current=key
            if arm in self.items:self.numeric.clear()
            self.items[arm]=item
        return item
    def capture(self,run,item,output):
        with guard(item.metadata['prepared_device']):
            from profile_ten import capture_ten
            return capture_ten(run,output,expected_kernel=item.expected_kernels,metric_group='PipeUtilization')
    def validate(self,arm,actual,fixture,spec):
        item=self.items[arm];check_fixture(spec,fixture)
        if not isinstance(actual,torch.Tensor)or actual is not item.output or actual.dtype!=torch.float32 or tuple(actual.shape)!=dimensions(spec):return dict(passed=False,reason='private complete F32 output required')
        with guard(item.metadata['prepared_device']):raw=item.stores[-1].detach().cpu();all_raw=[x.detach().cpu()for x in item.stores];input_raw={name:backing(x).detach().cpu()for name,x in item.inputs.items()}
        value=raw[32:-32].view(actual.shape);expected=fixture['expected'];bad=(value!=expected)|~torch.isfinite(value);coords=bad.nonzero();finite=bool(torch.isfinite(value).all());halo=True
        for buf in all_raw:
            pad=128//buf.element_size();halo=halo and bool((buf[:pad]==CANARY).all()and(buf[-pad:]==CANARY).all())
        immutable=all(torch.equal(x,backing(fixture['inputs'][name]))for name,x in input_raw.items());other=self.numeric.get('after'if arm=='before'else'before');self.numeric[arm]=value.clone();paired=other is None or torch.equal(value,other);bits=None if other is None else bool(torch.equal(value.view(torch.int32),other.view(torch.int32)))
        return dict(passed=finite and not len(coords)and halo and immutable and paired,mismatch_count=len(coords),first_coordinates=coords[:8].tolist(),nonfinite_count=int((~torch.isfinite(value)).sum()),max_FP64_error=float((value.double()-fixture['expected_fp64']).abs().max())if finite else None,canaries=halo,inputs_immutable=immutable,paired_numeric=paired,paired_all_bits_secondary=bits,CPU_zero_sign_secondary=bool(torch.equal(value.view(torch.int32),expected.view(torch.int32))))


def factory():return Adapter()
