# Copyright © 2026 Huawei Technologies Co., Ltd.
# Source: triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230.
# MIT license and third-party notices retained in compressor_pool_NOTICE.txt.
import triton
import triton.language as tl

@triton.jit
def _pooled_blocked(
    group_idx,
    start_pos,
    seg_row_base,
    cache_row,
    kv_ptr,
    score_ptr,
    cache_ptr,
    offs_h,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    HAS_RES,
):
    """单组分块三遍循环（max / 归一系数 / 加权和）。HAS_RES=0 时不访问
    cache；组内 token 偏移 >= 0 读投影矩阵，< 0 为段前残余，读 state_cache
    (cache_row, token_pos % CACHE_SIZE)。"""
    score_max = tl.full((HEAD_DIM,), -float("inf"), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                cache_ptr
                + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM
                + HEAD_DIM
                + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        score_max = tl.maximum(score_max, tl.max(score, axis=0))
    exp_sum = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                cache_ptr
                + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM
                + HEAD_DIM
                + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        exp_sum += tl.sum(tl.exp(score - score_max[None, :]), axis=0)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for chunk0 in range(0, RATIO_PAD, CHUNK_ROWS):
        rows = chunk0 + tl.arange(0, CHUNK_ROWS)
        row_valid = rows < RATIO
        token_pos = group_idx * RATIO + rows
        seg_off = token_pos - start_pos
        in_seg = seg_off >= 0
        score_seg = tl.load(
            score_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            score_cache = tl.load(
                cache_ptr
                + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM
                + HEAD_DIM
                + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            score = tl.where(in_seg[:, None], score_seg, score_cache)
        else:
            score = score_seg
        score = tl.where(row_valid[:, None], score, -float("inf"))
        prob = tl.exp(score - score_max[None, :]) / exp_sum[None, :]
        kv_seg = tl.load(
            kv_ptr + (seg_row_base + seg_off)[:, None] * HEAD_DIM + offs_h[None, :],
            mask=(row_valid & in_seg)[:, None],
            other=0.0,
        )
        if HAS_RES:
            slot = token_pos % CACHE_SIZE
            kv_cache = tl.load(
                cache_ptr
                + (cache_row * CACHE_SIZE + slot[:, None]) * 2 * HEAD_DIM
                + offs_h[None, :],
                mask=(row_valid & (seg_off < 0))[:, None],
                other=0.0,
            )
            kv_vals = tl.where(in_seg[:, None], kv_seg, kv_cache)
        else:
            kv_vals = kv_seg
        pooled += tl.sum(prob * kv_vals, axis=0)
    return pooled


@triton.jit
def _norm_store_1g(pooled, norm_w, eps: tl.constexpr, HEAD_DIM: tl.constexpr):
    """单组 (HEAD_DIM,) 向量的 pooled→bf16→RMSNorm，供各路径共用。"""
    pooled_bf16 = pooled.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(pooled_bf16 * pooled_bf16, axis=0) / HEAD_DIM
    return (pooled_bf16 * (1.0 / tl.sqrt(var + eps)) * norm_w).to(tl.bfloat16)


@triton.jit
def pool_general(KV, Score, Cache, Weight, Out, RATIO: tl.constexpr,
                 RATIO_PAD: tl.constexpr, START: tl.constexpr, HAS_RES: tl.constexpr):
    group = tl.program_id(0)
    h = tl.arange(0, 512)
    weight = tl.load(Weight + h).to(tl.float32)
    pooled = _pooled_blocked(group, START, 0, 0, KV, Score, Cache, h,
                             8, 512, RATIO, RATIO_PAD, 2, HAS_RES)
    value = _norm_store_1g(pooled, weight, 1e-6, 512)
    tl.store(Out + group * 512 + h, value)


@triton.jit
def pool_ratio2(KV, Score, Cache, Weight, Out, RATIO: tl.constexpr,
                RATIO_PAD: tl.constexpr, START: tl.constexpr, HAS_RES: tl.constexpr):
    group = tl.program_id(0)
    h = tl.arange(0, 512)
    offset = group * 2 * 512 + h
    s0 = tl.load(Score + offset)
    s1 = tl.load(Score + offset + 512)
    score_max = tl.maximum(s0, s1)
    e0 = tl.exp(s0 - score_max)
    e1 = tl.exp(s1 - score_max)
    exp_sum = e0 + e1
    kv0 = tl.load(KV + offset)
    kv1 = tl.load(KV + offset + 512)
    pooled = (e0 / exp_sum) * kv0 + (e1 / exp_sum) * kv1
    weight = tl.load(Weight + h).to(tl.float32)
    value = _norm_store_1g(pooled, weight, 1e-6, 512)
    tl.store(Out + group * 512 + h, value)
