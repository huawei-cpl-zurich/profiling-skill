# Independent adaptation of Ascend/ops-rec token_mixing, commit
# e489c0bf6a2d5f2d05b3205fcacc01b62a908b10. Source copyright Huawei
# Technologies Co., Ltd., licensed under Apache License 2.0.
"""Fuse residual addition into LayerNorm, retaining the transpose in both arms."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from token_mixing_contract import CASE_SOURCES, INPUTS, check_inputs, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_token_transpose(X, XT, S: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        batch, row, col = i // (S*S), (i // S) % S, i % S
        v = tl.load(X + batch*S*S + col*S + row, i < TOTAL, 0)
        tl.store(XT + i, v, i < TOTAL)

    @triton.jit
    def wave005_token_add(X, XT, Z, S: tl.constexpr, BLOCK: tl.constexpr):
        c = tl.arange(0, BLOCK)
        i = tl.program_id(0)*S + c
        v = tl.load(X+i,c<S,0) + tl.load(XT+i,c<S,0)
        tl.store(Z+i,v,c<S)

    @triton.jit
    def wave005_token_norm(X, XT, W, Bias, Y, S: tl.constexpr, BLOCK: tl.constexpr, FUSED: tl.constexpr):
        c = tl.arange(0, BLOCK)
        i = tl.program_id(0)*S + c
        v = tl.load(X+i,c<S,0)
        if FUSED:
            v = v + tl.load(XT+i,c<S,0)
        mean = tl.sum(v,0) / S
        centered = tl.where(c<S, v-mean, 0.0)
        variance = tl.sum(centered*centered,0) / S
        y = ((v-mean) * tl.rsqrt(variance+1e-5)) * tl.load(W+c,c<S,0) + tl.load(Bias+c,c<S,0)
        tl.store(Y+i,y,c<S)
    return wave005_token_transpose, wave005_token_add, wave005_token_norm


def identity(binary):
    return dict(hash=binary.hash, binaries={key:hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()
        for key,value in binary.asm.items() if isinstance(value,(str,bytes,bytearray))})


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = CASE_SOURCES

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        import torch
        b,s,_,_ = parameters(spec)
        if arm not in ('before','after'):
            raise ValueError('expected before or after')
        check_inputs(fixture,b,s)
        import torch_npu
        import triton
        inputs = tuple(fixture[k].npu() for k in INPUTS)
        x,gamma,beta = inputs
        def empty():
            return torch.full((b,s,s),float('nan'),dtype=torch.float32,device=x.device)
        xt,y = empty(),empty()
        transpose,add,norm = kernels()
        block = triton.next_power_of_2(s)
        calls = [(transpose,(x,xt,s,b*s*s,256),(triton.cdiv(b*s*s,256),))]
        workspace = [xt]
        if arm == 'before':
            z = empty(); workspace.append(z)
            calls += [(add,(x,xt,z,s,block),(b*s,)),(norm,(z,xt,gamma,beta,y,s,block,False),(b*s,))]
        else:
            calls.append((norm,(x,xt,gamma,beta,y,s,block,True),(b*s,)))
        options = dict(enable_fp_fusion=False,multibuffer=False,num_stages=1)
        binaries,artifacts = [],[]
        for index,(kernel,args,grid) in enumerate(calls):
            binary = kernel.warmup(*args,grid=grid,**options); binaries.append(binary)
            for key,value in binary.asm.items():
                if isinstance(value,str): value=value.encode()
                if isinstance(value,(bytes,bytearray)):
                    path=output/f'{index}.{key}';path.write_bytes(value);artifacts.append(path)
        def run():
            for kernel,args,grid in calls: kernel[grid](*args,**options)
            return y
        item=SimpleNamespace(run=run,inputs=inputs,workspace=workspace,
            expected_kernels=[k.fn.__name__ for k,_,_ in calls],sources=[Path(__file__),*CASE_SOURCES,*artifacts],
            compilation_state=lambda:[identity(v) for v in binaries],
            metadata=dict(arm=arm,B=b,S=s,language='triton',eps=1e-5,torch_version=torch.__version__,triton_version=triton.__version__,
                workspace_logical_bytes=4*b*s*s*(2 if arm=='before' else 1),
                scope='preallocated buffers; transpose and every residual/statistic computation inside each API',
                arithmetic='FP32 residual and centered variance, FMA disabled'))
        self.prepared[arm]=item
        return item

    def validate(self, arm, actual, fixture, spec):
        import torch
        y=actual.cpu();expected=fixture['expected']
        valid=y.dtype==torch.float32 and y.shape==expected.shape and bool(torch.isfinite(y).all())
        error=(y.double()-expected).abs() if valid else None
        passed=valid and bool(torch.all(error<=1e-4+1e-4*expected.abs()))
        inputs=self.prepared[arm].inputs
        passed=passed and all(torch.equal(v.cpu(),fixture[k]) for k,v in zip(INPUTS,inputs))
        passed=passed and all(actual.data_ptr()!=v.data_ptr() for v in inputs)
        return dict(passed=bool(passed),max_abs_error=float(error.max()) if valid else None,atol=1e-4,rtol=1e-4,oracle='independent CPU FP64')
