# SPDX-License-Identifier: MIT
# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Derived from triton-ascend-kernels index/pack_unpack.py at
# 7acb9c40a6ca827e885d902b7eaee5d45457b230. See pack_selected_LICENSE.
"""Pad then overwrite versus select before one store; fixed 64x64 tiles."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from pack_selected_contract import CASE_SOURCES, dimensions, checked_inputs, make_case


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave007_pack(X, Y, Length, N:tl.constexpr, D:tl.constexpr, L:tl.constexpr,
                     PAD:tl.constexpr, SELECT:tl.constexpr, BT:tl.constexpr, BD:tl.constexpr):
        b=tl.program_id(0)
        t=tl.program_id(1)*BT+tl.arange(0,BT)
        d=tl.program_id(2)*BD+tl.arange(0,BD)
        start=0
        for i in range(b):
            start+=tl.load(Length+i)
        length=tl.load(Length+b)
        valid=(t<length)&(t<L)
        dst=Y+(b*L+t)[:,None]*D+d[None,:]
        src=X+(start+t)[:,None]*D+d[None,:]
        mask=(t<L)[:,None]&(d<D)[None,:]
        pad=tl.full((BT,BD),PAD,tl.float32)
        if not SELECT:
            tl.store(dst,pad,mask)
        x=tl.load(src,valid[:,None]&(d<D)[None,:],other=0)
        if SELECT:
            tl.store(dst,tl.where(valid[:,None],x,pad),mask)
        else:
            tl.store(dst,x,valid[:,None]&(d<D)[None,:])
    return wave007_pack


def identity(c):
    return dict(hash=c.hash,binaries={k:hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()
        for k,v in c.asm.items() if isinstance(v,(str,bytes,bytearray))})


class Adapter:
    case_sources=CASE_SOURCES
    make_case=staticmethod(make_case)
    def __init__(self): self.prepared={}
    def prepare(self,arm,spec,fixture,output):
        p=dimensions(spec); checked_inputs(fixture,p)
        if arm not in ('before','after'): raise ValueError('unknown arm')
        import torch, torch_npu, triton
        x,lengths=fixture['x'].npu(),fixture['lengths'].npu()
        b,l,d=len(p['lengths']),max(p['lengths'],default=0),p['D']
        count=b*l*d
        storage=torch.full((count+32,),12345.,device=x.device,dtype=torch.float32)
        out=storage[16:16+count].view(b,l,d)
        extra=torch.empty((0,0,d),device=x.device) if p.get('also_empty_batch') else None
        fn=kernels(); args=(x,out,lengths,sum(p['lengths']),d,l,p['pad_value'],arm=='after',64,64)
        grid=(b,triton.cdiv(l,64),triton.cdiv(d,64))
        options=dict(num_warps=4,num_stages=1,multibuffer=False,enable_fp_fusion=False)
        compiled=[fn.warmup(*args,grid=grid,**options)] if count else []
        artifacts=[]
        for c in compiled:
            for k,v in c.asm.items():
                if isinstance(v,str): v=v.encode()
                if isinstance(v,(bytes,bytearray)):
                    path=output/f'kernel.{k}';path.write_bytes(v);artifacts.append(path)
        def run():
            if count: fn[grid](*args,**options)
            return (out,extra) if extra is not None else out
        item=SimpleNamespace(run=run,inputs=(x,lengths),storage=storage,
            expected_kernels=[fn.fn.__name__],sources=[Path(__file__),*CASE_SOURCES,*artifacts],
            compilation_state=lambda:[identity(c) for c in compiled],
            metadata=dict(language='triton',arm=arm,parameters=p,grid=grid,block_t=64,block_d=64,
                compiler_options=options,kernel_launches_per_api=int(bool(count)),
                component_packs_per_api=2 if extra is not None else 1,
                triton_version=triton.__version__,torch_version=torch.__version__,compile_only=True,
                compiled_resources=[str(c.metadata) for c in compiled]))
        self.prepared[arm]=item
        return item
    def validate(self,arm,actual,fixture,spec):
        import torch
        item=self.prepared[arm]
        actuals=actual if isinstance(actual,tuple) else (actual,)
        expected=fixture['expected']; expected=expected if isinstance(expected,tuple) else (expected,)
        exact=all(a.dtype==e.dtype and a.shape==e.shape and torch.equal(a.cpu().view(torch.int32),e.view(torch.int32)) for a,e in zip(actuals,expected))
        immutable=all(torch.equal(x.cpu().view(torch.int32),fixture[k].view(torch.int32)) for x,k in zip(item.inputs,('x','lengths')))
        canary=bool((item.storage[:16]==12345.).all() and (item.storage[-16:]==12345.).all())
        nonalias=all(a.numel()==0 or all(a.data_ptr()!=x.data_ptr() for x in item.inputs) for a in actuals)
        return dict(passed=exact and immutable and canary and nonalias,bitwise_exact=exact,inputs_immutable=immutable,canaries=canary,nonalias=nonalias)


def main():
    import argparse, json, sys
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','cache','provenance'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    spec=json.loads(Path(args.case).read_text())
    dimensions(spec)  # Public rejection precedes generic fixture cache lookup.
    if args.validate_only:
        print(json.dumps(dict(status='supported',parameters=dimensions(spec))))
        return 0
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from optimization_pair import run_pair
    result=run_pair(Adapter(),spec,args.output,args.cache,json.loads(Path(args.provenance).read_text()))
    return 0 if result['status']=='complete' else 1


if __name__=='__main__': raise SystemExit(main())
