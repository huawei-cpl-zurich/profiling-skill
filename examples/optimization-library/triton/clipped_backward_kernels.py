# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Apache-2.0; derived from MindSpeed 4efe2b675c2d51517c3fdd14c721020b8d41e241.
# Modified: candidate uniformly bypasses the source derivative graph only when all active gates exceed LIMIT.
import triton
import triton.language as tl

@triton.jit
def _clipped_swiglu_bwd_kernel(
    GUP,
    DY,
    DGUP,
    M,
    K,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LARGE_TENSOR: tl.constexpr,
):
    # backward of: g=min(gate,L), u=clamp(up,-L,L), out=(u+1)*g*sigmoid(g*alpha)
    # dout/dg = (u+1) * s * (1 + g*alpha*(1-s))   (0 where gate > L; kept at gate == L)
    # dout/du = g * s                              (0 where |up| > L; kept at |up| == L)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # see the forward kernel: int64 row offsets only for >2**31-element tensors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if LARGE_TENSOR:
        offs_m = offs_m.to(tl.int64)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # gate/up blocks live in the [gate | up] layout; dy block is the K-wide output.
    g_ptrs = GUP + offs_m[:, None] * (2 * K) + offs_k[None, :]
    u_ptrs = g_ptrs + K
    dy_ptrs = DY + offs_m[:, None] * K + offs_k[None, :]

    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    g_c = tl.minimum(g, LIMIT)  # faithful to _apply_gate (gate only upper-clamped)
    u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
    s = tl.sigmoid(g_c * ALPHA)
    glu = g_c * s

    dg = dy * (u_c + 1.0) * (s + g_c * ALPHA * s * (1.0 - s))
    du = dy * glu

    # clamp subgradient: zero strictly outside the clamp range, keep it at the
    # boundary (x == +-L) — matches torch's clamp backward exactly.
    dg = tl.where(g <= LIMIT, dg, 0.0)
    du = tl.where((u >= -LIMIT) & (u <= LIMIT), du, 0.0)

    dg_ptrs = DGUP + offs_m[:, None] * (2 * K) + offs_k[None, :]
    du_ptrs = dg_ptrs + K
    tl.store(dg_ptrs, dg.to(GUP.dtype.element_ty), mask=mask)
    tl.store(du_ptrs, du.to(GUP.dtype.element_ty), mask=mask)


@triton.jit
def _saturated_clipped_bwd_kernel(
    GUP,
    DY,
    DGUP,
    M,
    K,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LARGE_TENSOR: tl.constexpr,
):
    # backward of: g=min(gate,L), u=clamp(up,-L,L), out=(u+1)*g*sigmoid(g*alpha)
    # dout/dg = (u+1) * s * (1 + g*alpha*(1-s))   (0 where gate > L; kept at gate == L)
    # dout/du = g * s                              (0 where |up| > L; kept at |up| == L)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # see the forward kernel: int64 row offsets only for >2**31-element tensors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if LARGE_TENSOR:
        offs_m = offs_m.to(tl.int64)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    # gate/up blocks live in the [gate | up] layout; dy block is the K-wide output.
    g_ptrs = GUP + offs_m[:, None] * (2 * K) + offs_k[None, :]
    u_ptrs = g_ptrs + K
    dy_ptrs = DY + offs_m[:, None] * K + offs_k[None, :]

    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Inactive row/column lanes are the TRUE identity of the uniform test.
    all_saturated = tl.min(tl.min(tl.where(mask, g > LIMIT, True).to(tl.int32), axis=0), axis=0)
    if all_saturated:
        limit_f32 = tl.full((), LIMIT, tl.float32)
        s_limit = tl.sigmoid(limit_f32 * ALPHA)
        dg = tl.full((BLOCK_M, BLOCK_K), 0.0, tl.float32)
        du = dy * (limit_f32 * s_limit)
        du = tl.where((u >= -LIMIT) & (u <= LIMIT), du, 0.0)
    else:
        g_c = tl.minimum(g, LIMIT)  # faithful to _apply_gate (gate only upper-clamped)
        u_c = tl.maximum(tl.minimum(u, LIMIT), -LIMIT)
        s = tl.sigmoid(g_c * ALPHA)
        glu = g_c * s

        dg = dy * (u_c + 1.0) * (s + g_c * ALPHA * s * (1.0 - s))
        du = dy * glu

        # clamp subgradient: zero strictly outside the clamp range, keep it at the
        # boundary (x == +-L) — matches torch's clamp backward exactly.
        dg = tl.where(g <= LIMIT, dg, 0.0)
        du = tl.where((u >= -LIMIT) & (u <= LIMIT), du, 0.0)

    dg_ptrs = DGUP + offs_m[:, None] * (2 * K) + offs_k[None, :]
    du_ptrs = dg_ptrs + K
    tl.store(dg_ptrs, dg.to(GUP.dtype.element_ty), mask=mask)
    tl.store(du_ptrs, du.to(GUP.dtype.element_ty), mask=mask)
