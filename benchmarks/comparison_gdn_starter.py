"""Original sequential Triton starter for the public packed delta-rule contract."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _sequential(Q,K,V,G,B,S,C,O,F,T:tl.constexpr,HK:tl.constexpr,HV:tl.constexpr,
                DK:tl.constexpr,DV:tl.constexpr,SCALE:tl.constexpr,NORM:tl.constexpr,
                BK:tl.constexpr,BV:tl.constexpr):
    tile=tl.program_id(0)
    head=tl.program_id(1)
    segment=tl.program_id(2)
    kh=head//(HV//HK)
    ki=tl.arange(0,BK)
    vi=tile*BV+tl.arange(0,BV)
    state=tl.load(S+((segment*HV+head)*DK+ki[:,None])*DV+vi[None,:],
                  (ki[:,None]<DK)&(vi[None,:]<DV),0).to(tl.float32)
    start=tl.load(C+segment)
    end=tl.load(C+segment+1)
    for token in range(start,end):
        q=tl.load(Q+(token*HK+kh)*DK+ki,ki<DK,0).to(tl.float32)
        k=tl.load(K+(token*HK+kh)*DK+ki,ki<DK,0).to(tl.float32)
        if NORM:
            q=q*tl.rsqrt(tl.sum(q*q,0)+1.e-6)
            k=k*tl.rsqrt(tl.sum(k*k,0)+1.e-6)
        v=tl.load(V+(token*HV+head)*DV+vi,vi<DV,0).to(tl.float32)
        g=tl.load(G+token*HV+head).to(tl.float32)
        beta=tl.load(B+token*HV+head).to(tl.float32)
        state=state*tl.exp(g)
        prediction=tl.sum(k[:,None]*state,0)
        delta=beta*(v-prediction)
        state=state+k[:,None]*delta[None,:]
        out=tl.sum((q*SCALE)[:,None]*state,0)
        tl.store(O+(token*HV+head)*DV+vi,out,vi<DV)
    tl.store(F+((segment*HV+head)*DK+ki[:,None])*DV+vi[None,:],state,
             (ki[:,None]<DK)&(vi[None,:]<DV))


class Model(torch.nn.Module):
    def forward(self,q,k,v,g,beta,initial_state,cu_seqlens,scale,use_qk_l2norm_in_kernel):
        _,tokens,hk,dk=q.shape
        hv,dv=v.shape[2:]
        output=torch.empty_like(v)
        final=torch.empty_like(initial_state,dtype=torch.float32)
        _sequential[(tr.cdiv(dv,32),hv,cu_seqlens.numel()-1)](
            q,k,v,g,beta,initial_state,cu_seqlens,output,final,tokens,hk,hv,dk,dv,
            scale,bool(use_qk_l2norm_in_kernel),tr.next_power_of_2(dk),32)
        return output,final
