# Independent formula adaptation of cann/ops-cv resize_bilinear_v2 documentation,
# commit 1e859a524a93d0a94b563cacaa067db6abbfd37c. Native device body not retrieved.
# Source documentation/interface: CANN Open Software License Agreement Version 2.0.
"""Recompute geometry per element versus reuse it across four-pixel channel tiles."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
from bilinear_contract import CASE_SOURCES, checked_input, make_case, parameters


def kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def wave005_bilinear_elements(X, Y, N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                                 C: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0)*B + tl.arange(0,B)
        q, c = i//C, i%C
        oy, ox, batch = q//OW%OH, q%OW, q//(OH*OW)
        sy = oy.to(tl.float32)*tl.full((),H/OH,tl.float32)
        sx = ox.to(tl.float32)*tl.full((),W/OW,tl.float32)
        fy, fx = tl.floor(sy), tl.floor(sx)
        y0, x0 = tl.minimum(tl.maximum(fy.to(tl.int32),0),H-1), tl.minimum(tl.maximum(fx.to(tl.int32),0),W-1)
        y1, x1 = tl.minimum(y0+1,H-1), tl.minimum(x0+1,W-1)
        wy, wx = sy-fy, sx-fx
        a00 = tl.load(X+((batch*H+y0)*W+x0)*C+c, i<N*OH*OW*C, 0)
        a01 = tl.load(X+((batch*H+y0)*W+x1)*C+c, i<N*OH*OW*C, 0)
        a10 = tl.load(X+((batch*H+y1)*W+x0)*C+c, i<N*OH*OW*C, 0)
        a11 = tl.load(X+((batch*H+y1)*W+x1)*C+c, i<N*OH*OW*C, 0)
        upper = a00*(1-wx)+a01*wx
        lower = a10*(1-wx)+a11*wx
        tl.store(Y+i,upper*(1-wy)+lower*wy,i<N*OH*OW*C)

    @triton.jit
    def wave005_bilinear_reused(X, Y, N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                               C: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr, P: tl.constexpr, BC: tl.constexpr):
        q = tl.program_id(0)*P + tl.arange(0,P)
        c = tl.arange(0,BC)
        oy, ox, batch = q//OW%OH, q%OW, q//(OH*OW)
        sy = oy.to(tl.float32)*tl.full((),H/OH,tl.float32)
        sx = ox.to(tl.float32)*tl.full((),W/OW,tl.float32)
        fy, fx = tl.floor(sy), tl.floor(sx)
        y0, x0 = tl.minimum(tl.maximum(fy.to(tl.int32),0),H-1), tl.minimum(tl.maximum(fx.to(tl.int32),0),W-1)
        y1, x1 = tl.minimum(y0+1,H-1), tl.minimum(x0+1,W-1)
        wy, wx = (sy-fy)[:,None], (sx-fx)[:,None]
        mask = (q[:,None]<N*OH*OW)&(c[None,:]<C)
        a00 = tl.load(X+((batch*H+y0)*W+x0)[:,None]*C+c[None,:], mask, 0)
        a01 = tl.load(X+((batch*H+y0)*W+x1)[:,None]*C+c[None,:], mask, 0)
        a10 = tl.load(X+((batch*H+y1)*W+x0)[:,None]*C+c[None,:], mask, 0)
        a11 = tl.load(X+((batch*H+y1)*W+x1)[:,None]*C+c[None,:], mask, 0)
        upper = a00*(1-wx)+a01*wx
        lower = a10*(1-wx)+a11*wx
        tl.store(Y+q[:,None]*C+c[None,:],upper*(1-wy)+lower*wy,mask)

    return wave005_bilinear_elements, wave005_bilinear_reused


def identity(compiled):
    binaries = {}
    for key, data in compiled.asm.items():
        if isinstance(data, str):
            data = data.encode()
        if isinstance(data, (bytes, bytearray)):
            binaries[key] = hashlib.sha256(data).hexdigest()
    return dict(hash=compiled.hash, binaries=binaries)


class Adapter:
    make_case = staticmethod(make_case)
    case_sources = CASE_SOURCES

    def __init__(self):
        self.prepared = {}

    def prepare(self, arm, spec, fixture, output):
        p = parameters(spec)  # Unsupported coordinate modes fail before any NPU import.
        if arm not in ('before','after'):
            raise ValueError('unknown arm')
        checked_input(fixture['x'],p)
        import torch
        import torch_npu
        import triton
        x = fixture['x'].npu()
        out = torch.full((p['N'],p['OH'],p['OW'],p['C']),float('nan'),dtype=torch.float32,device=x.device)
        before, after = kernels()
        shape = tuple(p[k] for k in ('N','H','W','C','OH','OW'))
        fn = before if arm=='before' else after
        args = (x,out,*shape,256) if arm=='before' else (x,out,*shape,4,triton.next_power_of_2(p['C']))
        grid = (triton.cdiv(p['N']*p['OH']*p['OW']*(p['C'] if arm=='before' else 1),256 if arm=='before' else 4),)
        options = dict(num_stages=1,multibuffer=False,enable_fp_fusion=False)
        compiled = fn.warmup(*args,grid=grid,**options)
        artifacts=[]
        for key,data in compiled.asm.items():
            if isinstance(data,str):data=data.encode()
            if isinstance(data,(bytes,bytearray)):
                path=output/('kernel.'+key);path.write_bytes(data);artifacts.append(path)
        def run():
            fn[grid](*args,**options)
            return out
        item=SimpleNamespace(run=run,inputs=(x,),expected_kernels=[fn.fn.__name__],
            sources=[Path(__file__),*CASE_SOURCES,*artifacts],compilation_state=lambda:[identity(compiled)],
            metadata=dict(arm=arm,parameters=p,language='triton',grid=grid,
                geometry_lanes=256 if arm=='before' else 4,channel_tile=None if arm=='before' else triton.next_power_of_2(p['C']),
                arithmetic='FP32 coordinates and two-stage bilinear interpolation; FP fusion disabled',
                triton_version=triton.__version__,torch_version=torch.__version__,
                preallocation='input copy/output outside API; all coordinate, weight and four-pixel gather work inside'))
        self.prepared[arm]=item
        return item

    def validate(self,arm,actual,fixture,spec):
        import torch
        y=actual.cpu()
        shape=fixture['expected'].shape
        error=(y.double()-fixture['expected']).abs()
        numerical=y.dtype==torch.float32 and y.shape==shape and bool(torch.isfinite(y).all()) and bool(torch.all(error<=1e-5+1e-5*fixture['expected'].abs()))
        immutable=torch.equal(self.prepared[arm].inputs[0].cpu(),fixture['x'])
        nonalias=actual.data_ptr()!=self.prepared[arm].inputs[0].data_ptr()
        return dict(passed=numerical and immutable and nonalias,numerical=numerical,inputs_immutable=immutable,
                    output_nonalias=nonalias,
                    max_absolute_error=float(error.max()),atol=1e-5,rtol=1e-5)
