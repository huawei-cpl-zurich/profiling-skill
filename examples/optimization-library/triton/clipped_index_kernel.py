# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Apache-2.0; see CLIPPED_INDEX_LICENSE.
# MindSpeed 4efe2b675c2d51517c3fdd14c721020b8d41e241.
# Modification: add conventional flat addressing comparison; source 2D math unchanged.
import triton
import triton.language as tl

@triton.jit
def _clipped_swiglu_flat_kernel(GUP, OUT, M, K, ALPHA: tl.constexpr,
        LIMIT: tl.constexpr, TILE: tl.constexpr, LARGE_TENSOR: tl.constexpr):
    pid = tl.program_id(0)
    if LARGE_TENSOR:
        pid = pid.to(tl.int64)
    j = pid * TILE + tl.arange(0, TILE)
    row = j // K
    col = j % K
    if LARGE_TENSOR:
        total = M.to(tl.int64) * K
    else:
        total = M * K
    mask = j < total
    gate = GUP + row * (2 * K) + col
    g = tl.load(gate, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gate + K, mask=mask, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    out = (u + 1.0) * (g * tl.sigmoid(g * ALPHA))
    tl.store(OUT + j, out.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _clipped_swiglu_kernel(
    GUP,
    OUT,
    M,
    K,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LARGE_TENSOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # row*2K overflows int32 once the tensor passes 2**31 elements; the launcher
    # flips LARGE_TENSOR on only then, so small tensors keep the int32 binary.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if LARGE_TENSOR:
        offs_m = offs_m.to(tl.int64)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # gate block [row, col]; up block = gate block + K cols (same row, second half)
    g_ptrs = GUP + offs_m[:, None] * (2 * K) + offs_k[None, :]
    u_ptrs = g_ptrs + K

    # inner fp32 compute (sigmoid etc. need high precision), quantize back on write
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

    g = tl.minimum(g, LIMIT)  # faithful to _apply_gate (gate only upper-clamped)
    u = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)

    out = (u + 1.0) * (g * tl.sigmoid(g * ALPHA))

    o_ptrs = OUT + offs_m[:, None] * K + offs_k[None, :]
    tl.store(o_ptrs, out.to(OUT.dtype.element_ty), mask=mask)

