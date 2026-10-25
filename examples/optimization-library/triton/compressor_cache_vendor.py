# Copyright © 2026 Huawei Technologies Co., Ltd.
# Verbatim selected source bodies: triton-ascend-kernels
# 7acb9c40a6ca827e885d902b7eaee5d45457b230 attention/compressor.py.
# Only the _proj_kernel autotune decorator is omitted; common fixed config.
# MIT/third-party notice: compressor_cache_NOTICE.txt.
import triton
import triton.language as tl

@triton.jit
def _proj2_kernel(
    out0_ptr,
    out1_ptr,
    x_ptr,
    w0_ptr,
    w1_ptr,
    M,
    IN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """双投影单 launch：x (M, IN_DIM) 分别 @ w0^T / w1^T，各写 (M, OUT_DIM)
    fp32。tile 空间 [0, 2T)（T 为单矩阵 tile 数）grid-stride 摊满全部核，
    c < T 走 w0、c >= T 走 w1。两矩阵的选择须用 if 分支、各持编译期固定
    指针（后端限制：运行时指针 select 会生成坏地址）。"""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    n_tile_count = OUT_DIM // BN
    m_tile_count = tl.cdiv(M, BM)
    total_tiles = m_tile_count * n_tile_count
    tiles_per_group = GROUP * n_tile_count
    for c in range(pid, 2 * total_tiles, num_progs):
        if c < total_tiles:
            tile_id = c
        else:
            tile_id = c - total_tiles
        swizzle_group = tile_id // tiles_per_group
        group_m_start = swizzle_group * GROUP
        group_m_tiles = (
            m_tile_count - group_m_start
            if (m_tile_count - group_m_start) < GROUP
            else GROUP
        )
        rank_in_group = tile_id - swizzle_group * tiles_per_group
        m_tile = group_m_start + rank_in_group % group_m_tiles
        n_tile = rank_in_group // group_m_tiles
        rows = m_tile * BM + offs_m
        cols = n_tile * BN + offs_n
        row_valid = rows < M
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        if c < total_tiles:
            for k0 in range(0, IN_DIM, BK):
                offs_k = k0 + tl.arange(0, BK)
                x = tl.load(
                    x_ptr + rows[:, None] * IN_DIM + offs_k[None, :],
                    mask=row_valid[:, None],
                    other=0.0,
                )
                w = tl.load(w0_ptr + offs_k[:, None] + cols[None, :] * IN_DIM)
                acc = tl.dot(x, w, acc)
            tl.store(
                out0_ptr + rows[:, None] * OUT_DIM + cols[None, :],
                acc,
                mask=row_valid[:, None],
            )
        else:
            for k0 in range(0, IN_DIM, BK):
                offs_k = k0 + tl.arange(0, BK)
                x = tl.load(
                    x_ptr + rows[:, None] * IN_DIM + offs_k[None, :],
                    mask=row_valid[:, None],
                    other=0.0,
                )
                w = tl.load(w1_ptr + offs_k[:, None] + cols[None, :] * IN_DIM)
                acc = tl.dot(x, w, acc)
            tl.store(
                out1_ptr + rows[:, None] * OUT_DIM + cols[None, :],
                acc,
                mask=row_valid[:, None],
            )


@triton.jit
def _proj_kernel(
    out_ptr,
    x_ptr,
    w_ptr,
    M,
    IN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """x (M, IN_DIM) @ w^T，w 为 (OUT_DIM, IN_DIM) 行主序，输出 fp32。
    tile 行基址用 int64 计算，避免大 M 时行号 × IN_DIM 溢出 int32。"""
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    n_tile_count = OUT_DIM // BN
    m_tile_count = tl.cdiv(M, BM)
    total_tiles = m_tile_count * n_tile_count
    tiles_per_group = GROUP * n_tile_count
    for tile_id in range(pid, total_tiles, num_progs):
        swizzle_group = tile_id // tiles_per_group
        group_m_start = swizzle_group * GROUP
        group_m_tiles = (
            m_tile_count - group_m_start
            if (m_tile_count - group_m_start) < GROUP
            else GROUP
        )
        rank_in_group = tile_id - swizzle_group * tiles_per_group
        m_tile = group_m_start + rank_in_group % group_m_tiles
        n_tile = rank_in_group // group_m_tiles
        rows = m_tile * BM + offs_m
        cols = n_tile * BN + offs_n
        row_valid = rows < M
        x_tile = x_ptr + (m_tile * BM).to(tl.int64) * IN_DIM
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, IN_DIM, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(
                x_tile + offs_m[:, None] * IN_DIM + offs_k[None, :],
                mask=row_valid[:, None],
                other=0.0,
            )
            w = tl.load(w_ptr + offs_k[:, None] + cols[None, :] * IN_DIM)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + rows[:, None] * OUT_DIM + cols[None, :],
            acc,
            mask=row_valid[:, None],
        )


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
def _pool_kernel(
    out_ptr,
    kv_ptr,
    score_ptr,
    cache_ptr,
    norm_w_ptr,
    meta_ptr,  # 拼包 [start_pos | used_len | out_row_offset | seg_row_base]
    block_table_ptr,
    total_tasks,
    TASKS_PER_BATCH,  # 每 batch 任务槽位上界 = 1 边界 + ceil(组槽位 / GROUPS_TILE)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    GROUPS_TILE: tl.constexpr,  # 干净任务每任务合并的组数
    eps: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
):
    """任务合并版池化。task==0 且段首有残余时为边界任务（残余组走
    _pooled_blocked）；其余任务处理 GROUPS_TILE 个连续组。"""
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    norm_w = tl.load(norm_w_ptr + offs_h).to(tl.float32)
    for task_id in range(pid, total_tasks, tl.num_programs(0)):
        batch = task_id // TASKS_PER_BATCH
        task = task_id - batch * TASKS_PER_BATCH
        start_pos = tl.load(meta_ptr + batch)  # [0, NUM_BATCH)
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)  # [NUM_BATCH, 2*NUM_BATCH)
        first_group = start_pos // RATIO
        residual = start_pos - first_group * RATIO  # 段前残余数，仅首组时可能 >0
        groups_in_batch = (start_pos + used_len) // RATIO - first_group
        has_res = residual > 0
        out_row_base = tl.load(
            meta_ptr + 2 * NUM_BATCH + batch
        )  # [2*NUM_BATCH, 3*NUM_BATCH)
        if (task == 0) & has_res & (groups_in_batch > 0):
            # 边界任务：段首残余组（每 batch 至多 1 个），cache 拼接
            seg_row_base = tl.load(
                meta_ptr + 3 * NUM_BATCH + batch
            )  # [3*NUM_BATCH, 4*NUM_BATCH)
            cache_row = tl.load(block_table_ptr + batch)
            pooled = _pooled_blocked(
                first_group,
                start_pos,
                seg_row_base,
                cache_row,
                kv_ptr,
                score_ptr,
                cache_ptr,
                offs_h,
                CACHE_SIZE,
                HEAD_DIM,
                RATIO,
                RATIO_PAD,
                CHUNK_ROWS,
                True,
            )
            norm_out = _norm_store_1g(pooled, norm_w, eps, HEAD_DIM)
            tl.store(out_ptr + out_row_base * HEAD_DIM + offs_h, norm_out)
        else:
            # 干净 tile：组 [g_lo, g_lo+GROUPS_TILE) 全部在段内；g_lo < 0
            # （纯积累段）须显式排除，否则向量路径负偏移越界。
            base = tl.where(has_res, 1, 0)
            g_lo = base + (task - base) * GROUPS_TILE
            if (g_lo >= 0) & (g_lo < groups_in_batch):
                seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + batch)
                if SINGLE_BLOCK:
                    g_offsets = tl.arange(0, GROUPS_TILE)
                    g_valid = (g_lo + g_offsets) < groups_in_batch
                    seg_first = (
                        first_group + g_lo + g_offsets
                    ) * RATIO - start_pos  # (GT,) 各组首行段内偏移
                    if RATIO_PAD == 2:
                        # r2 特化路径
                        tile_base = (seg_row_base + seg_first)[
                            :, None
                        ] * HEAD_DIM + offs_h[None, :]
                        s0 = tl.load(
                            score_ptr + tile_base, mask=g_valid[:, None], other=-1e30
                        )
                        s1 = tl.load(
                            score_ptr + tile_base + HEAD_DIM,
                            mask=g_valid[:, None],
                            other=-1e30,
                        )
                        score_max = tl.maximum(s0, s1)
                        e0 = tl.exp(s0 - score_max)
                        e1 = tl.exp(s1 - score_max)
                        exp_sum = e0 + e1
                        kv0 = tl.load(
                            kv_ptr + tile_base, mask=g_valid[:, None], other=0.0
                        )
                        kv1 = tl.load(
                            kv_ptr + tile_base + HEAD_DIM,
                            mask=g_valid[:, None],
                            other=0.0,
                        )
                        pooled = (e0 / exp_sum) * kv0 + (e1 / exp_sum) * kv1
                    else:
                        # 其他 ratio：三遍静态展开
                        score_max = tl.full(
                            (GROUPS_TILE, HEAD_DIM), -1e30, dtype=tl.float32
                        )
                        for jj in tl.static_range(RATIO_PAD):
                            row_ok = g_valid & (jj < RATIO)
                            seg_off = seg_first + jj
                            s_j = tl.load(
                                score_ptr
                                + (seg_row_base + seg_off)[:, None] * HEAD_DIM
                                + offs_h[None, :],
                                mask=row_ok[:, None],
                                other=-1e30,
                            )
                            score_max = tl.maximum(score_max, s_j)
                        exp_sum = tl.zeros((GROUPS_TILE, HEAD_DIM), dtype=tl.float32)
                        for jj in tl.static_range(RATIO_PAD):
                            row_ok = g_valid & (jj < RATIO)
                            seg_off = seg_first + jj
                            s_j = tl.load(
                                score_ptr
                                + (seg_row_base + seg_off)[:, None] * HEAD_DIM
                                + offs_h[None, :],
                                mask=row_ok[:, None],
                                other=-1e30,
                            )
                            exp_sum += tl.exp(s_j - score_max)
                        pooled = tl.zeros((GROUPS_TILE, HEAD_DIM), dtype=tl.float32)
                        for jj in tl.static_range(RATIO_PAD):
                            row_ok = g_valid & (jj < RATIO)
                            seg_off = seg_first + jj
                            s_j = tl.load(
                                score_ptr
                                + (seg_row_base + seg_off)[:, None] * HEAD_DIM
                                + offs_h[None, :],
                                mask=row_ok[:, None],
                                other=-1e30,
                            )
                            kv_j = tl.load(
                                kv_ptr
                                + (seg_row_base + seg_off)[:, None] * HEAD_DIM
                                + offs_h[None, :],
                                mask=row_ok[:, None],
                                other=0.0,
                            )
                            pooled += (tl.exp(s_j - score_max) / exp_sum) * kv_j
                    pooled_bf16 = pooled.to(tl.bfloat16).to(tl.float32)
                    var = tl.sum(pooled_bf16 * pooled_bf16, axis=1) / HEAD_DIM
                    norm_out = (
                        pooled_bf16
                        * (1.0 / tl.sqrt(var + eps))[:, None]
                        * norm_w[None, :]
                    )
                    out_rows = out_row_base + g_lo + g_offsets
                    tl.store(
                        out_ptr + out_rows[:, None] * HEAD_DIM + offs_h[None, :],
                        norm_out.to(tl.bfloat16),
                        mask=g_valid[:, None],
                    )
                else:
                    # 大 RATIO：每任务一组走分块版
                    cache_row = tl.load(block_table_ptr + batch)
                    pooled = _pooled_blocked(
                        first_group + g_lo,
                        start_pos,
                        seg_row_base,
                        cache_row,
                        kv_ptr,
                        score_ptr,
                        cache_ptr,
                        offs_h,
                        CACHE_SIZE,
                        HEAD_DIM,
                        RATIO,
                        RATIO_PAD,
                        CHUNK_ROWS,
                        False,
                    )
                    norm_out = _norm_store_1g(pooled, norm_w, eps, HEAD_DIM)
                    out_row = out_row_base + g_lo
                    tl.store(out_ptr + out_row * HEAD_DIM + offs_h, norm_out)


@triton.jit
def _pool_decode_kernel(
    out_ptr,
    kv_ptr,
    score_ptr,
    cache_ptr,
    norm_w_ptr,
    meta_ptr,
    block_table_ptr,
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    RATIO_PAD: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    eps: tl.constexpr,
):
    """单组段特化路径（max_used_len <= RATIO，decode 形态）：每 batch 至多
    1 个组，一核一 batch。"""
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    norm_w = tl.load(norm_w_ptr + offs_h).to(tl.float32)
    for b in range(pid, NUM_BATCH, tl.num_programs(0)):
        start_pos = tl.load(meta_ptr + b)
        used_len = tl.load(meta_ptr + NUM_BATCH + b)
        first_group = start_pos // RATIO
        residual = start_pos - first_group * RATIO
        groups = (start_pos + used_len) // RATIO - first_group
        if groups > 0:
            out_row_base = tl.load(meta_ptr + 2 * NUM_BATCH + b)
            seg_row_base = tl.load(meta_ptr + 3 * NUM_BATCH + b)
            cache_row = tl.load(block_table_ptr + b)
            # HAS_RES 须为编译期字面量
            if residual > 0:
                pooled = _pooled_blocked(
                    first_group,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    cache_ptr,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                    True,
                )
            else:
                pooled = _pooled_blocked(
                    first_group,
                    start_pos,
                    seg_row_base,
                    cache_row,
                    kv_ptr,
                    score_ptr,
                    cache_ptr,
                    offs_h,
                    CACHE_SIZE,
                    HEAD_DIM,
                    RATIO,
                    RATIO_PAD,
                    CHUNK_ROWS,
                    False,
                )
            norm_out = _norm_store_1g(pooled, norm_w, eps, HEAD_DIM)
            tl.store(out_ptr + out_row_base * HEAD_DIM + offs_h, norm_out)


# ============================================================================
# K3: state_cache 更新（只写每段最后 min(used_len, CACHE_SIZE) 个 token）
# ============================================================================


@triton.jit
def _cache_update_kernel(
    kv_ptr,
    score_ptr,
    cache_ptr,
    meta_ptr,
    block_table_ptr,
    total_write_slots,
    WRITE_SLOTS_PER_BATCH,  # 每 batch 写槽位上界 min(max_used_len, CACHE_SIZE)
    NUM_BATCH,
    CACHE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_h = tl.arange(0, HEAD_DIM)
    for task_id in range(pid, total_write_slots, tl.num_programs(0)):
        batch = task_id // WRITE_SLOTS_PER_BATCH
        write_idx = task_id - batch * WRITE_SLOTS_PER_BATCH
        used_len = tl.load(meta_ptr + NUM_BATCH + batch)
        tail_count = used_len if used_len < CACHE_SIZE else CACHE_SIZE
        if write_idx < tail_count:
            token_in_seg = (
                used_len - tail_count + write_idx
            )  # 最后 tail_count 个 token 中的第 write_idx 个
            proj_row = tl.load(meta_ptr + 3 * NUM_BATCH + batch) + token_in_seg
            token_pos = tl.load(meta_ptr + batch) + token_in_seg
            slot = token_pos % CACHE_SIZE
            cache_row = tl.load(block_table_ptr + batch)
            kv_vals = tl.load(kv_ptr + proj_row * HEAD_DIM + offs_h)
            score_vals = tl.load(score_ptr + proj_row * HEAD_DIM + offs_h)
            tl.store(
                cache_ptr + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM + offs_h,
                kv_vals,
            )
            tl.store(
                cache_ptr
                + (cache_row * CACHE_SIZE + slot) * 2 * HEAD_DIM
                + HEAD_DIM
                + offs_h,
                score_vals,
            )


