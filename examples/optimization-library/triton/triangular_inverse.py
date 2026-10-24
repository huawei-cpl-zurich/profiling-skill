"""F02: full-column substitution versus independent leaves and block merging.

All FP32 workspaces are allocated once; every forward initializes their entire
read footprint. No preparation launch, warmup, autotuning or tensor inspection.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from triangular_inverse_contract import CASE_SOURCES, chunks, diagnostics, dimensions, make_case, validate_inputs


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def triangular_forward(A, X, Starts, Lengths, Batches, T:tl.constexpr, H:tl.constexpr, BT:tl.constexpr, LEAF:tl.constexpr):
        task = tl.program_id(0)
        chunk, head = task//H, task%H
        col = tl.program_id(1)
        start = tl.load(Starts+chunk)
        length = tl.load(Lengths+chunk)
        batch = tl.load(Batches+chunk)
        r = tl.arange(0,BT)
        begin = col//LEAF*LEAF
        x = tl.full((BT,),0,tl.float32)
        for i in range(begin,begin+LEAF):
            a = tl.load(A+((batch*T+start+i)*H+head)*BT+r, (i<length)&(r<length)&(r<i), 0).to(tl.float32)
            solved = tl.where(i==col,1.,0.)-tl.sum(a*x,0)
            x = tl.where(r==i,solved,x)
        tl.store(X+(task*BT+r)*BT+col,x)

    @triton.jit
    def triangular_merge_left(A, X, Temp, Starts, Lengths, Batches, T:tl.constexpr,H:tl.constexpr,BT:tl.constexpr,S:tl.constexpr):
        task = tl.program_id(0)
        chunk, head = task//H, task%H
        group = tl.program_id(1)
        start = tl.load(Starts+chunk)
        length = tl.load(Lengths+chunk)
        batch = tl.load(Batches+chunk)
        r = tl.arange(0,S)
        base = group*(2*S)
        # Direct global pointer loads preserve the Cube operand contract.
        d = tl.load(X+(task*BT+base+S+r[:,None])*BT+base+S+r[None,:])
        c = tl.load(A+((batch*T+start+base+S+r[:,None])*H+head)*BT+base+r[None,:],
                    (base+S+r[:,None]<length)&(base+r[None,:]<length),0).to(tl.float32)
        product = tl.dot(d,c,input_precision='ieee')
        tl.store(Temp+(task*BT+base+S+r[:,None])*BT+base+r[None,:],product)

    @triton.jit
    def triangular_merge_right(X, XRight, Temp, Y, BT:tl.constexpr,S:tl.constexpr):
        task = tl.program_id(0)
        base = tl.program_id(1)*(2*S)
        r = tl.arange(0,S)
        d = tl.load(X+(task*BT+base+S+r[:,None])*BT+base+S+r[None,:])
        b = tl.load(XRight+(task*BT+base+r[:,None])*BT+base+r[None,:])
        middle = tl.load(Temp+(task*BT+base+S+r[:,None])*BT+base+r[None,:])
        lower = -tl.dot(middle,b,input_precision='ieee')
        tl.store(Y+(task*BT+base+r[:,None])*BT+base+r[None,:],b)
        tl.store(Y+(task*BT+base+S+r[:,None])*BT+base+S+r[None,:],d)
        tl.store(Y+(task*BT+base+S+r[:,None])*BT+base+r[None,:],lower)
        tl.store(Y+(task*BT+base+r[:,None])*BT+base+S+r[None,:],0.)

    @triton.jit
    def triangular_unpack(X, Out, Starts, Lengths, Batches,T:tl.constexpr,H:tl.constexpr,BT:tl.constexpr):
        task = tl.program_id(0)
        chunk, head = task//H,task%H
        start = tl.load(Starts+chunk)
        length = tl.load(Lengths+chunk)
        batch = tl.load(Batches+chunk)
        r = tl.arange(0,BT)
        col = tl.program_id(1)
        value = tl.load(X+(task*BT+r)*BT+col)
        value = tl.where((col<length)&(r>=col),value,0.)
        tl.store(Out+((batch*T+start+r)*H+head)*BT+col,value,r<length)

    return triangular_forward,triangular_merge_left,triangular_merge_right,triangular_unpack


def compiled_identity(compiled):
    rows=[]
    for artifact in compiled.values():
        asm = getattr(artifact,'asm',{})
        rows.append(dict(name=getattr(artifact,'name',None),hash=getattr(artifact,'hash',None),
            artifacts={key:hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
                       for key,value in asm.items() if isinstance(value,(str,bytes))}))
    return rows


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)

    def __init__(self):
        self.prepared={}

    def prepare(self,arm,spec,fixture,output):
        import torch
        import torch_npu
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        b,t,h,bt,_ = dimensions(spec)  # Reject unsupported BT before allocation/launch.
        validate_inputs(fixture['inputs'],spec)
        entries=chunks(spec)
        a=fixture['inputs']['a'].to('npu:0')
        tables=[torch.tensor(values,dtype=torch.int32,device='npu:0') for values in
                ([x[1] for x in entries],[x[2] for x in entries],[x[0] for x in entries])]
        count=len(entries)*h
        x=torch.empty((count,bt,bt),dtype=torch.float32,device='npu:0')
        y=torch.empty_like(x) if arm=='after' else None
        temp=torch.empty_like(x) if arm=='after' else None
        out=torch.empty_like(a,dtype=torch.float32)
        forward,left,right,unpack=kernels()
        compiled={}
        def keep(key,artifact):
            compiled[key]=artifact
        def run():
            keep('solve',forward[(count,bt)](a,x,*tables,t,h,bt,bt if arm=='before' else 16))
            source=x
            if arm=='after':
                destination=y
                size=16
                while size<bt:
                    keep(('left',size),left[(count,bt//(2*size))](a,source,temp,*tables,t,h,bt,size))
                    keep(('right',size),right[(count,bt//(2*size))](source,source,temp,destination,bt,size))
                    source,destination=destination,source
                    size*=2
            keep('unpack',unpack[(count,bt)](source,out,*tables,t,h,bt))
            return out
        selected=[forward,unpack] if arm=='before' or bt==16 else [forward,left,right,unpack]
        item=SimpleNamespace(run=run,inputs=dict(a=a),expected_kernels=[k.fn.__name__ for k in selected],
            sources=[Path(__file__),*CASE_SOURCES],compilation_state=lambda:compiled_identity(compiled),
            metadata=dict(language='triton',arm=arm,leaf=bt if arm=='before' else 16,
                triton_version=triton.__version__,torch_version=torch.__version__,shape=spec,
                workspace_bytes=sum(v.numel()*v.element_size() for v in (x,y,temp) if v is not None),
                launches_per_forward=2 if arm=='before' else 2+2*(bt.bit_length()-5),
                precision='FP16 input; FP32 solve/state/IEEE dot and output',
                compatibility='independent pointer-based hierarchy; no insert_slice extension'))
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        result=diagnostics(actual,fixture,spec)
        if not torch.equal(self.prepared[arm].inputs['a'].detach().cpu(),fixture['inputs']['a']):
            result.update(passed=False,reason='input mutated')
        return result


def factory():
    return Adapter()
