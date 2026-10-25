# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025, DeepLink.
# SPDX-License-Identifier: MIT
# Pinned 7acb9c40a6ca827e885d902b7eaee5d45457b230; see rotary_trig_NOTICE.
# Common portability change: masked loads specify other=0.
import triton
import triton.language as tl

@triton.jit
def _fused_rotary_emb_kernel(
    Q,
    K,
    PostionIds,
    InvFreq,
    scaling_factor,
    OutQ,
    OutK,
    stride_bq,
    stride_sq,
    stride_hq: tl.constexpr,
    stride_dq: tl.constexpr,
    stride_bk,
    stride_sk,
    stride_hk: tl.constexpr,
    stride_dk: tl.constexpr,
    stride_bp,
    stride_sp,
    max_seq_len,
    BLOCK: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """fused rotary emb kernel."""
    batch_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)
    s_off = seq_block_id * BLOCK + tl.arange(0, BLOCK)[:, None]
    f_off = tl.arange(0, BLOCK_F)[None, :]
    s_mask = s_off < max_seq_len

    bp_off = stride_bp * batch_id
    p_off = bp_off + stride_sp * s_off

    sq_off = batch_id * stride_bq + s_off * stride_sq
    q0_off = sq_off + f_off * stride_dq
    q1_off = q0_off + BLOCK_F * stride_dq

    sk_off = batch_id * stride_bk + s_off * stride_sk
    k0_off = sk_off + f_off * stride_dk
    k1_off = k0_off + BLOCK_F * stride_dk

    inv_freq = tl.load(InvFreq + f_off).to(tl.float32)
    position_ids = tl.load(PostionIds + p_off, mask=s_mask, other=0).to(tl.float32)
    position_ids = position_ids / scaling_factor

    pos_freq = position_ids * inv_freq
    cos = tl.cos(pos_freq).to(Q.dtype.element_ty)
    sin = tl.sin(pos_freq).to(Q.dtype.element_ty)

    for h in range(BLOCK_HQ):
        q0 = tl.load(Q + q0_off + h * stride_hq, mask=s_mask, other=0)
        q1 = tl.load(Q + q1_off + h * stride_hq, mask=s_mask, other=0)
        q0_out = q0 * cos - q1 * sin
        tl.store(OutQ + q0_off + h * stride_hq, q0_out, mask=s_mask)
        q1_out = q1 * cos + q0 * sin
        tl.store(OutQ + q1_off + h * stride_hq, q1_out, mask=s_mask)

    for h in range(BLOCK_HK):
        k0 = tl.load(K + k0_off + h * stride_hk, mask=s_mask, other=0)
        k1 = tl.load(K + k1_off + h * stride_hk, mask=s_mask, other=0)
        k0_out = k0 * cos - k1 * sin
        tl.store(OutK + k0_off + h * stride_hk, k0_out, mask=s_mask)
        k1_out = k1 * cos + k0 * sin
        tl.store(OutK + k1_off + h * stride_hk, k1_out, mask=s_mask)


@triton.jit
def rotary_head_kernel(X,P,I,scale,O,SB,SS,SH:tl.constexpr,SD:tl.constexpr,PB,PS,S,BLOCK:tl.constexpr,F:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(2)
    s=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)[:,None]
    f=tl.arange(0,F)[None,:];mask=s<S
    position=tl.load(P+b*PB+s*PS,mask=mask,other=0).to(tl.float32)
    inv=tl.load(I+f).to(tl.float32)
    position=position/scale
    angle=position*inv
    cos=tl.cos(angle).to(X.dtype.element_ty);sin=tl.sin(angle).to(X.dtype.element_ty)
    offset=b*SB+s*SS+h*SH+f*SD
    x0=tl.load(X+offset,mask=mask,other=0);x1=tl.load(X+offset+F*SD,mask=mask,other=0)
    y0=x0*cos-x1*sin;y1=x1*cos+x0*sin
    tl.store(O+offset,y0,mask=mask);tl.store(O+offset+F*SD,y1,mask=mask)
