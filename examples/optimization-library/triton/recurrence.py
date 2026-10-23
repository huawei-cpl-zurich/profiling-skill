"""Wave001 family04: token scan versus chunk preparation plus state scan.

This is a narrow example, not a general GDN operator. No warmups/autotuning or
output/input inspection occur in forwards; all workspace is allocated once.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from recurrence_contract import CASE_SOURCES, diagnostics, dimensions, make_case, validate_inputs


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave001_delta_token(Q,K,V,Beta,G,Initial,Out,Final,Bounds,
                            HK:tl.constexpr,HV:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr):
        h=tl.program_id(0)%HV
        segment=tl.program_id(0)//HV
        kh=h//(HV//HK)
        d=tl.arange(0,DK)
        v=tl.arange(0,DV)
        state=tl.load(Initial+(segment*HV+h)*DK*DV+d[:,None]*DV+v[None,:])
        start=tl.load(Bounds+segment)
        end=tl.load(Bounds+segment+1)
        for token in range(start,end):
            key=tl.load(K+(token*HK+kh)*DK+d).to(tl.float32)
            query=tl.load(Q+(token*HK+kh)*DK+d).to(tl.float32)
            value=tl.load(V+(token*HV+h)*DV+v).to(tl.float32)
            beta=tl.load(Beta+token*HV+h)
            gate=tl.exp(tl.load(G+token*HV+h))
            state=state*gate
            prediction=tl.sum(state*key[:,None],0)
            delta=beta*(value-prediction)
            state=state+key[:,None]*delta[None,:]
            out=tl.sum(state*query[:,None],0)
            tl.store(Out+(token*HV+h)*DV+v,out)
        tl.store(Final+(segment*HV+h)*DK*DV+d[:,None]*DV+v[None,:],state)

    @triton.jit
    def wave001_delta_prepare(K,V,Beta,G,U,W,C,Starts,Lengths,
                              HK:tl.constexpr,HV:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,BT:tl.constexpr):
        chunk=tl.program_id(0)//HV
        h=tl.program_id(0)%HV
        kh=h//(HV//HK)
        start=tl.load(Starts+chunk)
        length=tl.load(Lengths+chunk)
        r=tl.arange(0,BT)
        d=tl.arange(0,DK)
        v=tl.arange(0,DV)
        valid=r<length
        key=tl.load(K+((start+r[:,None])*HK+kh)*DK+d[None,:],valid[:,None],0)
        value=tl.load(V+((start+r[:,None])*HV+h)*DV+v[None,:],valid[:,None],0)
        beta=tl.load(Beta+(start+r)*HV+h,valid,0)
        gate=tl.load(G+(start+r)*HV+h,valid,0)
        cumulative=tl.cumsum(gate,0)
        lower=r[:,None]>r[None,:]
        gram=tl.dot(key,tl.trans(key))
        decay=tl.exp(cumulative[:,None]-cumulative[None,:])
        a=tl.where(lower,gram*beta[:,None]*decay,0.)
        # Nilpotence of the strict lower BT32 matrix makes five terms enough
        # in doubling form: (I-A)(I+A^2)(I+A^4)(I+A^8)(I+A^16).
        identity=tl.where(r[:,None]==r[None,:],1.,0.)
        inverse=identity-a
        power=tl.dot(a.to(tl.float16),a.to(tl.float16))
        for level in tl.static_range(4):
            inverse=inverse+tl.dot(inverse.to(tl.float16),power.to(tl.float16))
            if level < 3:
                power=tl.dot(power.to(tl.float16),power.to(tl.float16))
        u=tl.dot(inverse.to(tl.float16),(value.to(tl.float32)*beta[:,None]).to(tl.float16))
        w=tl.dot(inverse.to(tl.float16),(key.to(tl.float32)*beta[:,None]*tl.exp(cumulative[:,None])).to(tl.float16))
        base=(chunk*HV+h)*BT
        tl.store(U+(base+r[:,None])*DV+v[None,:],u)
        tl.store(W+(base+r[:,None])*DK+d[None,:],w)
        tl.store(C+base+r,cumulative)

    @triton.jit
    def wave001_delta_chunk_scan(Q,K,Initial,U,W,C,Out,Final,Starts,Lengths,First,Count,
                                 HK:tl.constexpr,HV:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,BT:tl.constexpr):
        segment=tl.program_id(0)//HV
        h=tl.program_id(0)%HV
        kh=h//(HV//HK)
        r=tl.arange(0,BT)
        d=tl.arange(0,DK)
        v=tl.arange(0,DV)
        state=tl.load(Initial+(segment*HV+h)*DK*DV+d[:,None]*DV+v[None,:])
        first=tl.load(First+segment)
        count=tl.load(Count+segment)
        for chunk in range(first,first+count):
            start=tl.load(Starts+chunk)
            length=tl.load(Lengths+chunk)
            valid=r<length
            base=(chunk*HV+h)*BT
            cumulative=tl.load(C+base+r)
            last=tl.load(C+base+length-1)
            query=tl.load(Q+((start+r[:,None])*HK+kh)*DK+d[None,:],valid[:,None],0)
            key=tl.load(K+((start+r[:,None])*HK+kh)*DK+d[None,:],valid[:,None],0)
            u=tl.load(U+(base+r[:,None])*DV+v[None,:])
            w=tl.load(W+(base+r[:,None])*DK+d[None,:])
            delta=u-tl.dot(w.to(tl.float16),state.to(tl.float16))
            scores=tl.dot(query,tl.trans(key))
            scores=tl.where(r[:,None]>=r[None,:],scores*tl.exp(cumulative[:,None]-cumulative[None,:]),0.)
            carry=tl.dot((query.to(tl.float32)*tl.exp(cumulative[:,None])).to(tl.float16),state.to(tl.float16))
            out=carry+tl.dot(scores.to(tl.float16),delta.to(tl.float16))
            tl.store(Out+((start+r[:,None])*HV+h)*DV+v[None,:],out,valid[:,None])
            weighted=key.to(tl.float32)*tl.exp(last-cumulative[:,None])
            state=state*tl.exp(last)+tl.dot(tl.trans(weighted).to(tl.float16),delta.to(tl.float16))
        tl.store(Final+(segment*HV+h)*DK*DV+d[:,None]*DV+v[None,:],state)

    return wave001_delta_token,wave001_delta_prepare,wave001_delta_chunk_scan


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
        t,hk,hv,dk,dv,bounds=dimensions(spec)
        validate_inputs(fixture['inputs'],spec)
        inputs={k:v.npu() for k,v in fixture['inputs'].items()}
        out=torch.empty((t,hv,dv),dtype=torch.float16,device='npu')
        final=torch.empty_like(inputs['initial'])
        starts,lengths,first,count=[],[],[],[]
        for start,end in zip(bounds,bounds[1:]):
            first.append(len(starts))
            pieces=list(range(start,end,32))
            starts.extend(pieces)
            lengths.extend(min(32,end-p) for p in pieces)
            count.append(len(pieces))
        tables=[torch.tensor(values,dtype=torch.int32,device='npu') for values in (bounds,starts,lengths,first,count)]
        u=torch.empty((len(starts),hv,32,dv),dtype=torch.float32,device='npu') if arm=='after' else None
        w=torch.empty((len(starts),hv,32,dk),dtype=torch.float32,device='npu') if arm=='after' else None
        c=torch.empty((len(starts),hv,32),dtype=torch.float32,device='npu') if arm=='after' else None
        token,prep,scan=kernels()
        compiled={}
        def keep(artifact):
            compiled[id(artifact)]=artifact
        def run():
            if arm=='before':
                keep(token[((len(bounds)-1)*hv,)](inputs['q'],inputs['k'],inputs['v'],inputs['beta'],inputs['g'],inputs['initial'],out,final,tables[0],hk,hv,dk,dv))
            else:
                keep(prep[(len(starts)*hv,)](inputs['k'],inputs['v'],inputs['beta'],inputs['g'],u,w,c,tables[1],tables[2],hk,hv,dk,dv,32))
                keep(scan[((len(bounds)-1)*hv,)](inputs['q'],inputs['k'],inputs['initial'],u,w,c,out,final,tables[1],tables[2],tables[3],tables[4],hk,hv,dk,dv,32))
            return out,final
        def compilation_state():
            rows=[]
            for artifact in compiled.values():
                asm={key:hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
                     for key,value in getattr(artifact,'asm',{}).items() if isinstance(value,(str,bytes))}
                rows.append(dict(hash=getattr(artifact,'hash',None),artifacts=asm))
            return rows
        selected=[token] if arm=='before' else [prep,scan]
        item=SimpleNamespace(run=run,expected_kernels=[k.fn.__name__ for k in selected],
            sources=[Path(__file__),*CASE_SOURCES],inputs=inputs,compilation_state=compilation_state,
            metadata=dict(language='triton',triton_version=triton.__version__,torch_version=torch.__version__,
                          block=32,workspace_bytes=sum(x.numel()*x.element_size() for x in (u,w,c) if x is not None),
                          dot_inputs='FP16; FP32 accumulation/state',arm=arm,shape=spec))
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        result=diagnostics(actual,fixture)
        for name,value in self.prepared[arm].inputs.items():
            if not torch.equal(value.detach().cpu(),fixture['inputs'][name]):
                result.update(passed=False,reason='input mutation: '+name)
        return result


def factory():
    return Adapter()
