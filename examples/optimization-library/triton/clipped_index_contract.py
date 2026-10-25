# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Apache-2.0; FP32 source picker adapted from MindSpeed 4efe2b675c2d51517c3fdd14c721020b8d41e241.
"""CPU FP64 clipped SwiGLU oracle and independent physical admission."""
from pathlib import Path
import torch


def dimensions(spec):
    p=spec.get('parameters',spec);m,k=p.get('M'),p.get('K')
    if (type(m) is not int or type(k) is not int or min(m,k)<=0
            or type(spec.get('seed')) is not int or spec['seed']<0
            or p.get('pattern') not in (None,'clamp_boundaries','zeros_signed_tail')):
        raise ValueError('unsupported clipped forward case')
    plan(m,k)
    return p,m,k


def pick_blocks(m,k):
    # Exact pinned FP32 source picker: cap1024, budget4096, floor512.
    bk=1
    while bk<k and bk<1024:bk*=2
    bk=max(512,min(bk,1024))
    bm=1 if m<=1 else 2 if m<=8 else 4 if m<=64 else 8 if m<=512 else 16
    cap=max(1,4096//bk);bm=max(1,min(bm,1<<(cap.bit_length()-1)))
    return bm,bk


def plan(m,k):
    if type(m) is not int or type(k) is not int or min(m,k)<=0:
        raise ValueError('positive integer dimensions required')
    bm,bk=pick_blocks(m,k);tile=bm*bk
    gm,gk=(m+bm-1)//bm,(k+bk-1)//bk;gf=(m*k+tile-1)//tile
    rowmax,colmax=gm*bm-1,gk*bk-1;flatmax=gf*tile-1
    max_input=max(rowmax*2*k+colmax+k,(flatmax//k)*2*k+flatmax%k+k)
    max_output=max(rowmax*k+colmax,flatmax)
    max_byte=4*(max(max_input,max_output)+1)-1
    if max(rowmax,colmax,2*k)>2**31-1 or max_byte>2**63-1:
        raise ValueError('padded address arithmetic exceeds supported integer route')
    return dict(block_m=bm,block_k=bk,tile=tile,flat_grid=[gf],source_grid=[gm,gk],
                max_padded_input_offset=max_input,max_padded_output_offset=max_output,
                max_padded_byte=max_byte,int64=max_byte>2**31-1,
                valid_input_elements=m*2*k,valid_output_elements=m*k)


def validate_input(v,spec):
    _,m,k=dimensions(spec)
    if (not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32
            or tuple(v.shape)!=(m,2*k) or not v.is_contiguous() or v.storage_offset()!=0
            or v.untyped_storage().nbytes()!=v.numel()*4 or not torch.isfinite(v).all()
            or not (v.abs()<=12).all()):
        raise ValueError('owned finite contiguous CPU FP32 GUP[M,2K] in [-12,12] required')
    nondyadic=v[(v*64)!=(v*64).trunc()]
    if nondyadic.numel():
        seven=torch.tensor(7.,dtype=torch.float32)
        limits=torch.stack((-seven,seven))
        boundaries=torch.cat((torch.nextafter(limits,torch.full_like(limits,float('inf'))),
                              torch.nextafter(limits,torch.full_like(limits,-float('inf')))))
        if spec.get('parameters',spec).get('pattern')!='clamp_boundaries' or not (nondyadic[:,None]==boundaries).any(-1).all():
            raise ValueError('dyadic 1/64 input or explicit clamp-boundary neighbors required')


def make_case(spec):
    p,m,k=dimensions(spec);rng=torch.Generator().manual_seed(spec['seed'])
    gup=torch.randint(-768,769,(m,2*k),generator=rng).float()/64
    if p.get('pattern')=='clamp_boundaries':
        seven=torch.tensor(7.,dtype=torch.float32)
        values=torch.stack((-seven,torch.nextafter(-seven,torch.tensor(-float('inf'))),
                            torch.nextafter(-seven,torch.tensor(float('inf'))),
                            seven,torch.nextafter(seven,torch.tensor(-float('inf'))),
                            torch.nextafter(seven,torch.tensor(float('inf'))),torch.tensor(0.)))
        gup.copy_(values.repeat((gup.numel()+6)//7)[:gup.numel()].view_as(gup))
    if p.get('pattern')=='zeros_signed_tail':
        gup[:,::2]=0.;gup[:,1::2]=-0.;gup[-1,k-1]=.5;gup[-1,-1]=-.5
    validate_input(gup,spec)
    g=gup[:,:k].double().clamp(max=7);u=gup[:,k:].double().clamp(-7,7)
    expected=(u+1)*(g*torch.sigmoid(g*1.702))
    return dict(inputs=dict(GUP=gup),expected=expected)


def check_output(actual,expected):
    if (not isinstance(actual,torch.Tensor) or actual.dtype!=torch.float32
            or actual.shape!=expected.shape or not actual.is_contiguous()):
        return dict(passed=False,reason='FP32 contiguous output shape required')
    got=actual.detach().cpu().double()
    return dict(passed=bool(torch.isfinite(got).all() and torch.allclose(got,expected,atol=3e-5,rtol=2e-5)),
                max_abs_error=float((got-expected).abs().max()),atol=3e-5,rtol=2e-5)


CASE_SOURCES=[Path(__file__)]
