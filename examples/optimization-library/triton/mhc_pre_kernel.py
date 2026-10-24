"""Portable source-style MindSpeed-Ops arch32 MHC backward adaptation.

Source: Ascend/MindSpeed-Ops 4efe2b675c2d51517c3fdd14c721020b8d41e241,
mindspeed_ops/arch32/triton/mhc/pre_bmm.py; Apache-2.0 (MHC_LICENSE).
The filtered source has no per-file copyright notice. GROUP=1, flattened FP32
inputs and common compiler flags are fixed in the host contract. Explicit zero
kernel replaces torch.zeros for deterministic portable device identity; this is
not an unmodified vendor whole-pipeline benchmark.
"""
import triton
import triton.language as tl


@triton.jit
def zero_gradient_kernel(P, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(P + i, 0.0, i < N)


@triton.jit
def hc_pre_bmm_bwd_fused_kernel(
    H_ptr,
    X_ptr,
    dY_ptr,
    dX_ptr,
    dH_ptr,
    BS,
    D,
    stride_h_bs: tl.constexpr,
    stride_h_n: tl.constexpr,
    stride_x_bs: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_dy_bs: tl.constexpr,
    stride_dy_d: tl.constexpr,
    stride_dx_bs: tl.constexpr,
    stride_dx_n: tl.constexpr,
    stride_dx_d: tl.constexpr,
    stride_dh_bs: tl.constexpr,
    stride_dh_n: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DIVISIBLE_D: tl.constexpr,
    EXCLUSIVE: tl.constexpr,
):
    pid_bs_blk = tl.program_id(0)
    pid_d_blk = tl.program_id(1)

    pid0 = pid_bs_blk * GROUP
    pids = pid0 + tl.arange(0, GROUP)
    mask_pid = pids < BS

    d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = tl.full((BLOCK_D,), True, tl.int1) if DIVISIBLE_D else (d < D)
    m = mask_pid[:, None] & mask_d[None, :]

    # ---- load dY once ----
    dY = tl.load(dY_ptr + pids[:, None] * stride_dy_bs + d[None, :] * stride_dy_d, mask=m, other=0.0).to(tl.float32)

    # ---- load H (for dX) ----
    h0 = tl.load(H_ptr + pids * stride_h_bs + 0 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    h1 = tl.load(H_ptr + pids * stride_h_bs + 1 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    h2 = tl.load(H_ptr + pids * stride_h_bs + 2 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    h3 = tl.load(H_ptr + pids * stride_h_bs + 3 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]

    # ---- write dX ----
    tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 0 * stride_dx_n + d[None, :] * stride_dx_d, dY * h0, mask=m)
    tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 1 * stride_dx_n + d[None, :] * stride_dx_d, dY * h1, mask=m)
    tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 2 * stride_dx_n + d[None, :] * stride_dx_d, dY * h2, mask=m)
    tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 3 * stride_dx_n + d[None, :] * stride_dx_d, dY * h3, mask=m)

    # ---- compute partial dH over this D tile ----
    # load X for each i, accumulate sum over D tile (axis=1 -> BLOCK_D)
    X_base = X_ptr + pids[:, None] * stride_x_bs + d[None, :] * stride_x_d

    x0 = tl.load(X_base + 0 * stride_x_n, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(X_base + 1 * stride_x_n, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X_base + 2 * stride_x_n, mask=m, other=0.0).to(tl.float32)
    x3 = tl.load(X_base + 3 * stride_x_n, mask=m, other=0.0).to(tl.float32)

    dh0 = tl.sum(x0 * dY, axis=1)  # (G,)
    dh1 = tl.sum(x1 * dY, axis=1)
    dh2 = tl.sum(x2 * dY, axis=1)
    dh3 = tl.sum(x3 * dY, axis=1)

    # Sole-writer dispatch changes only these four writes.
    if EXCLUSIVE:
        tl.store(dH_ptr + pids * stride_dh_bs + 0 * stride_dh_n, dh0, mask=mask_pid)
        tl.store(dH_ptr + pids * stride_dh_bs + 1 * stride_dh_n, dh1, mask=mask_pid)
        tl.store(dH_ptr + pids * stride_dh_bs + 2 * stride_dh_n, dh2, mask=mask_pid)
        tl.store(dH_ptr + pids * stride_dh_bs + 3 * stride_dh_n, dh3, mask=mask_pid)
    else:
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 0 * stride_dh_n, dh0, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 1 * stride_dh_n, dh1, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 2 * stride_dh_n, dh2, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 3 * stride_dh_n, dh3, mask=mask_pid)
