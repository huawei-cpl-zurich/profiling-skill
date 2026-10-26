# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import triton
import triton.language as tl

from mindspeed_ops.api.triton.utils import prepare_chunk_indices


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))

        p_gn = g + (min(T, i_t * BT + BT) - 1) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            # [BT, BV]
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            # [BT, BV]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))

            tl.debug_barrier()

        b_gk_exp = tl.math.exp2(b_g)
        b_dgk *= tl.math.exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], tl.math.exp2(b_gn[None, :] - b_g), 0)

        b_dw = -b_dw

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk

        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dv2_db(
    v,
    beta,
    A,
    dv,
    dv2,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh, i_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_nseq, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_nseq).to(tl.int32), tl.load(cu_seqlens + i_nseq + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    v += (bos * H + i_h) * V
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    db += bos * H + i_h

    o_m = i_m * BM + tl.arange(0, BM)
    o_t_m = i_t * BT + o_m

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_m * BM,), (BM,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (i_m * BM, i_t * BT), (BM, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where((o_m[:, None] < BT) & (i_t * BT + tl.arange(0, BT)[None, :] < T), b_A, 0)

    b_db = tl.zeros([BM], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT + i_m * BM, i_v * BV), (BM, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT + i_m * BM, i_v * BV), (BM, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dvb = tl.dot(b_A.to(b_dv.dtype), b_dv)
        b_dv2 = b_dvb * b_beta[:, None]
        b_db += tl.sum(b_dvb * b_v, axis=1)
        tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT + i_m * BM,), (BM,), (0,))
    b_db = tl.where(o_t_m < T, b_db, 0)
    tl.store(p_db, b_db, boundary_check=(0,))


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dk_dg_db(
    k,
    g,
    beta,
    A,
    dw,
    dk,
    dg,
    db,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh, i_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_nseq, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_nseq).to(tl.int32), tl.load(cu_seqlens + i_nseq + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    dw += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    db += bos * H + i_h

    o_m = i_m * BM + tl.arange(0, BM)
    o_t_m = i_t * BT + o_m

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_m * BM,), (BM,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (i_m * BM, i_t * BT), (BM, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where((o_m[:, None] < BT) & (i_t * BT + tl.arange(0, BT)[None, :] < T), b_A, 0)
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT + i_m * BM,), (BM,), (0,))
    b_db = tl.load(p_db, boundary_check=(0,))

    for i_k in range(tl.cdiv(K, BK)):
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dkgb = tl.dot(b_A, b_dw)

        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_dg = tl.load(p_dg, boundary_check=(0, 1))
        b_gk_exp = tl.math.exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        b_db += tl.sum(b_dkgb * b_kg, axis=1)
        b_dg += b_kg * b_dkgb * b_beta[:, None]
        b_dk += b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    b_db = tl.where(o_t_m < T, b_db, 0)
    tl.store(p_db, b_db, boundary_check=(0,))


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dA_pre(
    k,
    v,
    g,
    beta,
    dv,
    dw,
    dA_pre,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NTILES_N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_m, i_n = i_tile // NTILES_N, i_tile % NTILES_N

    if IS_VARLEN:
        i_nseq, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_nseq).to(tl.int32), tl.load(cu_seqlens + i_nseq + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    dv += (bos * H + i_h) * V
    dw += (bos * H + i_h) * K
    dA_pre += (bos * H + i_h) * BT

    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_t_m = i_t * BT + o_m
    o_t_n = i_t * BT + o_n
    acc = tl.zeros([BM, BN], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT + i_m * BM, i_v * BV), (BM, BV), (1, 0))
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT + i_n * BN, i_v * BV), (BN, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        acc += tl.dot(b_dv, tl.trans(b_v))

    for i_k in range(tl.cdiv(K, BK)):
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_n * BN, i_k * BK), (BN, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_n * BN, i_k * BK), (BN, BK), (1, 0))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_kg = b_k * tl.math.exp2(b_g)
        acc += tl.dot(b_dw, tl.trans(b_kg.to(b_dw.dtype)))

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_n * BN,), (BN,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    m_A = (o_t_m[:, None] > o_t_n[None, :]) & (o_t_m[:, None] < T) & (o_t_n[None, :] < T)
    acc = tl.where(m_A, acc * b_beta[None, :], 0)
    p_dA_pre = tl.make_block_ptr(dA_pre, (T, BT), (H * BT, 1), (i_t * BT + i_m * BM, i_n * BN), (BM, BN), (1, 0))
    tl.store(p_dA_pre, acc.to(p_dA_pre.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dA_post_right(
    A,
    dA_pre,
    dA_tmp,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NTILES_N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_m, i_n = i_tile // NTILES_N, i_tile % NTILES_N

    if IS_VARLEN:
        i_nseq, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_nseq).to(tl.int32), tl.load(cu_seqlens + i_nseq + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    A += (bos * H + i_h) * BT
    dA_pre += (bos * H + i_h) * BT
    dA_tmp += (bos * H + i_h) * BT

    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_t_m = i_t * BT + o_m
    o_t_n = i_t * BT + o_n

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i_k in range(tl.cdiv(BT, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_dA_pre = tl.make_block_ptr(dA_pre, (T, BT), (H * BT, 1), (i_t * BT + i_m * BM, i_k * BK), (BM, BK), (1, 0))
        p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (i_k * BK, i_t * BT + i_n * BN), (BK, BN), (0, 1))
        b_dA_pre = tl.load(p_dA_pre, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dA_pre = tl.where((o_t_m[:, None] < T) & (o_k[None, :] < BT), b_dA_pre, 0)
        b_A = tl.where((o_k[:, None] < BT) & (o_t_n[None, :] < T), b_A, 0)
        acc += tl.dot(b_dA_pre.to(b_A.dtype), b_A)

    p_dA_tmp = tl.make_block_ptr(dA_tmp, (T, BT), (H * BT, 1), (i_t * BT + i_m * BM, i_n * BN), (BM, BN), (1, 0))
    tl.store(p_dA_tmp, acc, boundary_check=(0, 1))


@triton.heuristics(
    {
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dA_post_left(
    A,
    dA_tmp,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NTILES_N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_m, i_n = i_tile // NTILES_N, i_tile % NTILES_N

    if IS_VARLEN:
        i_nseq, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_nseq).to(tl.int32), tl.load(cu_seqlens + i_nseq + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    A += (bos * H + i_h) * BT
    dA_tmp += (bos * H + i_h) * BT
    dA += (bos * H + i_h) * BT

    o_m = i_m * BM + tl.arange(0, BM)
    o_n = i_n * BN + tl.arange(0, BN)
    o_t_m = i_t * BT + o_m
    o_t_n = i_t * BT + o_n

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i_k in range(tl.cdiv(BT, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (i_m * BM, i_t * BT + i_k * BK), (BM, BK), (0, 1))
        p_dA_tmp = tl.make_block_ptr(dA_tmp, (T, BT), (H * BT, 1), (i_t * BT + i_k * BK, i_n * BN), (BK, BN), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dA_tmp = tl.load(p_dA_tmp, boundary_check=(0, 1))
        b_A = tl.where((o_m[:, None] < BT) & (i_t * BT + o_k[None, :] < T), b_A, 0)
        b_dA_tmp = tl.where((i_t * BT + o_k[:, None] < T) & (o_n[None, :] < BT), b_dA_tmp, 0)
        acc += tl.dot(b_A, b_dA_tmp.to(b_A.dtype))

    m_A = (o_t_m[:, None] > o_t_n[None, :]) & (o_t_m[:, None] < T) & (o_t_n[None, :] < T)
    acc = tl.where(m_A, -acc, 0)
    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT + i_m * BM, i_n * BN), (BM, BN), (1, 0))
    tl.store(p_dA, acc.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dw = torch.empty((B, T, H, K), dtype=A.dtype, device=q.device)
    dA_pre = torch.empty_like(A, dtype=torch.float)
    dA_tmp = torch.empty_like(A, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA_pre,
        dw=dw,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=32,
        num_warps=2,
        num_stages=1,
    )
    BDA = min(BT, 32)
    grid_rows = (NT, B * H, triton.cdiv(BT, BDA))
    chunk_kda_bwd_kernel_wy_dv2_db[grid_rows](
        v=v,
        beta=beta,
        A=A,
        dv=dv,
        dv2=dv2,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BM=BDA,
        BV=32,
        num_warps=2,
        num_stages=1,
    )
    chunk_kda_bwd_kernel_wy_dk_dg_db[grid_rows](
        k=k,
        g=g,
        beta=beta,
        A=A,
        dw=dw,
        dk=dk,
        dg=dg,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BM=BDA,
        BK=32,
        num_warps=2,
        num_stages=1,
    )
    grid_dA = (NT, B * H, triton.cdiv(BT, BDA) * triton.cdiv(BT, BDA))
    chunk_kda_bwd_kernel_wy_dA_pre[grid_dA](
        k=k,
        v=v,
        g=g,
        beta=beta,
        dv=dv,
        dw=dw,
        dA_pre=dA_pre,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BM=BDA,
        BN=BDA,
        BK=64,
        BV=64,
        NTILES_N=triton.cdiv(BT, BDA),
        num_warps=2,
        num_stages=1,
    )
    # Keep the public chunk_size unchanged, but tile the final dA = -A @ (dA_pre @ A)
    # postprocess so NPU backends do not need a full BT x BT matmul block in the fused kernel.
    chunk_kda_bwd_kernel_dA_post_right[grid_dA](
        A=A,
        dA_pre=dA_pre,
        dA_tmp=dA_tmp,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BM=BDA,
        BN=BDA,
        BK=BDA,
        NTILES_N=triton.cdiv(BT, BDA),
        num_warps=2,
        num_stages=1,
    )
    chunk_kda_bwd_kernel_dA_post_left[grid_dA](
        A=A,
        dA_tmp=dA_tmp,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        BM=BDA,
        BN=BDA,
        BK=BDA,
        NTILES_N=triton.cdiv(BT, BDA),
        num_warps=2,
        num_stages=1,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA
