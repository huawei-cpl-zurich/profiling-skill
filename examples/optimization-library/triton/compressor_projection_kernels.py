# Copyright © 2026 Huawei Technologies Co., Ltd.
# MIT; adapted from Triton-Ascend-Kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230.
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


