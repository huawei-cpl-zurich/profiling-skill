# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Generated exact bodies; complete Apache2/MIT notices in vendor/.
import triton
import triton.language as tl

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


