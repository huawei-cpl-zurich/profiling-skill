"""Source-backed normalization placement study, uniform non-inplace FP32 scope.

Adapted from triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230,
moe/fused_recurrent.py. Removed optional modes; retained recurrence and mapping.
Copyright © 2025 Huawei Technologies Co., Ltd.
Copyright contributors to the vLLM project
Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
See recurrent_norm_LICENSE. No novelty or full vendor API compatibility claim.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recurrent_norm_contract import CASE_SOURCES, NAMES, diagnostics, dimensions as contract_dimensions, make_case as contract_make_case, validate_inputs


def normalize_spec(spec):
    # Accept only the fixed scope's explicit defaults; reject active modes.
    defaults = {'inplace_final_state':False,'is_kda':False,'is_beta_headwise':False,
                'is_varlen':False,'is_continuous_batching':False,'is_spec_decoding':False,
                'cu_seqlens':None,'ssm_state_indices':None,'num_accepted_tokens':None}
    allowed = {'parameters','seed','case_id','role','scenario_designation','hypothesis',
               'B','T','H','HV','K','V','zeros_and_epsilon'}
    if not isinstance(spec,dict):
        raise ValueError('case must be an object')
    result = dict(spec)
    if 'parameters' in result:
        if not isinstance(result['parameters'],dict):
            raise ValueError('parameters must be an object')
        result['parameters'] = dict(result['parameters'])
    for fields in ([result,result['parameters']] if 'parameters' in result else [result]):
        for key, default in defaults.items():
            if key in fields:
                if type(fields[key]) is not type(default) or fields[key] != default:
                    raise ValueError('unsupported optional recurrence mode: '+key)
                fields.pop(key)
    if set(result)-allowed:
        raise ValueError('unsupported optional recurrence mode or case field')
    if 'parameters' not in result:
        result['parameters'] = {k:result.pop(k) for k in ('B','T','H','HV','K','V','zeros_and_epsilon') if k in result}
    contract_dimensions(result)
    return result


def dimensions(spec):
    return contract_dimensions(normalize_spec(spec))


def make_case(spec):
    return contract_make_case(normalize_spec(spec))


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def normalize_qk(q, k, qn, kn, K: tl.constexpr, BK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0,BK)
        x = tl.load(q+row*K+offsets, offsets<K, 0).to(tl.float32)
        y = tl.load(k+row*K+offsets, offsets<K, 0).to(tl.float32)
        x = x / tl.sqrt(tl.sum(x*x)+1e-6)
        y = y / tl.sqrt(tl.sum(y*y)+1e-6)
        tl.store(qn+row*K+offsets,x,offsets<K)
        tl.store(kn+row*K+offsets,y,offsets<K)

    @triton.jit
    def fused_recurrent_gated_delta_rule_fwd_kernel(q,k,v,g,beta,o,h0,ht,scale,
            T: tl.constexpr,H: tl.constexpr,HV: tl.constexpr,K: tl.constexpr,V: tl.constexpr,
            BK: tl.constexpr,BV: tl.constexpr,USE_QK_L2NORM_IN_KERNEL: tl.constexpr):
        i_k,i_v,i_nh = tl.program_id(0),tl.program_id(1),tl.program_id(2)
        i_n,i_hv = i_nh//HV,i_nh%HV
        i_h = i_hv//(HV//H)
        bos = i_n*T
        o_k = i_k*BK+tl.arange(0,BK)
        o_v = i_v*BV+tl.arange(0,BV)
        p_q = q+(bos*H+i_h)*K+o_k
        p_k = k+(bos*H+i_h)*K+o_k
        p_v = v+(bos*HV+i_hv)*V+o_v
        p_beta = beta+bos*HV+i_hv
        p_g = g+bos*HV+i_hv
        p_o = o+(bos*HV+i_hv)*V+o_v
        mask_k,mask_v = o_k<K,o_v<V
        mask_h = mask_k[:,None]&mask_v[None,:]
        p_h0 = h0+i_n*HV*K*V+i_hv*K*V+o_k[:,None]*V+o_v[None,:]
        b_h = tl.zeros([BK,BV],tl.float32)
        b_h += tl.load(p_h0,mask_h,0).to(tl.float32)
        for i_t in range(0,T):
            b_q = tl.load(p_q,mask_k,0).to(tl.float32)
            b_k = tl.load(p_k,mask_k,0).to(tl.float32)
            b_v = tl.load(p_v,mask_v,0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q*b_q)+1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k*b_k)+1e-6)
            b_q = b_q*scale
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h*b_k[:,None],0)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_v *= b_beta
            b_h += b_k[:,None]*b_v[None,:]
            b_o = tl.sum(b_h*b_q[:,None],0)
            tl.store(p_o,b_o.to(p_o.dtype.element_ty),mask_v)
            p_q += H*K
            p_k += H*K
            p_o += HV*V
            p_v += HV*V
            p_g += HV
            p_beta += HV
        p_ht = ht+i_nh*K*V+o_k[:,None]*V+o_v[None,:]
        tl.store(p_ht,b_h.to(p_ht.dtype.element_ty),mask_h)
    return normalize_qk,fused_recurrent_gated_delta_rule_fwd_kernel


def identity(compiled):
    return dict(hash=compiled.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in compiled.asm.items() if isinstance(v,(str,bytes,bytearray))})


class Adapter:
    case_sources = CASE_SOURCES
    make_case = staticmethod(make_case)

    def __init__(self):
        self.prepared = {}

    def prepare(self,arm,spec,fixture,output):
        import torch
        import torch_npu
        import triton
        if arm not in ('before','after'):
            raise ValueError('arm must be before or after')
        b,t,h,hv,k,v = dimensions(spec)
        validate_inputs(fixture['inputs'],normalize_spec(spec))
        inputs = {n:x.npu() for n,x in fixture['inputs'].items()}
        # Guard every input/output storage; guards lie outside contiguous views.
        owners = []
        def guarded(shape):
            count = 1
            for dim in shape:
                count *= dim
            owner = torch.full((count+2,),31415.,dtype=torch.float32,device=inputs['q'].device)
            owners.append(owner)
            return owner[1:-1].view(shape)
        out, ht = guarded((b,t,hv,v)),guarded((b,hv,k,v))
        for n,x in list(inputs.items()):
            protected = guarded(tuple(x.shape))
            protected.copy_(x)
            inputs[n] = protected
        qn,kn = (guarded((b,t,h,k)),guarded((b,t,h,k))) if arm=='before' else (inputs['q'],inputs['k'])
        norm,rec = kernels()
        bk,bv = triton.next_power_of_2(k),min(triton.next_power_of_2(v),64)
        flags = dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        normargs = (inputs['q'],inputs['k'],qn,kn,k,bk)
        recargs = (qn,kn,inputs['v'],inputs['g'],inputs['beta'],out,inputs['h0'],ht,k**-.5,t,h,hv,k,v,bk,bv,arm=='after')
        grid = (1,triton.cdiv(v,bv),b*hv)
        compiled = []
        if arm=='before':
            compiled.append(norm.warmup(*normargs,grid=(b*t*h,),**flags))
        compiled.append(rec.warmup(*recargs,grid=grid,**flags))
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        paths = []
        for index,item in enumerate(compiled):
            for key,value in item.asm.items():
                if isinstance(value,(str,bytes,bytearray)):
                    path = output/f'kernel-{index}.{key}'
                    path.write_bytes(value.encode() if isinstance(value,str) else value)
                    paths.append(path)
        state = lambda: [identity(x) for x in compiled]
        def run():
            if arm=='before':
                norm[(b*t*h,)](*normargs,**flags)
            rec[grid](*recargs,**flags)
            return out,ht
        item = SimpleNamespace(run=run,expected_kernels=([norm.fn.__name__] if arm=='before' else [])+[rec.fn.__name__],
            sources=[Path(__file__),*CASE_SOURCES,*paths], compilation_state=state,
            metadata=dict(arm=arm,grid=grid,BK=bk,BV=bv,compiler_flags=flags,
                compiled=[dict(identity=identity(x),metadata=str(x.metadata)) for x in compiled],
                triton=triton.__version__,torch=torch.__version__,api_components=2 if arm=='before' else 1),
            inputs=inputs,owners=owners,outputs=(out,ht))
        self.prepared[arm] = item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        item = self.prepared[arm]
        result = diagnostics(actual,fixture)
        for n,x in item.inputs.items():
            if not torch.equal(x.cpu(),fixture['inputs'][n]):
                result.update(passed=False,reason='input mutation: '+n)
        if not all(bool((owner[[0,-1]].cpu()==31415.).all()) for owner in item.owners):
            result.update(passed=False,reason='output/intermediate canary changed')
        if isinstance(actual,(tuple,list)) and len(actual)==2:
            pointers = [x.untyped_storage().data_ptr() for x in (*actual,*item.inputs.values())]
            if len(set(pointers)) != len(pointers):
                result.update(passed=False,reason='output/input alias')
        return result


def main():
    import argparse
    root = HERE.parents[2]
    sys.path.insert(0,str(root/'scripts'))
    from optimization_pair import run_pair
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.case).read_text())
    dimensions(spec)  # Reject unsupported modes before generic cache creation.
    result = run_pair(Adapter(),spec,args.output,args.cache,json.loads(Path(args.provenance).read_text()))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
