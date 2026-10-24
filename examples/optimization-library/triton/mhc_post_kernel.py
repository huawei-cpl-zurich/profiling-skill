# SPDX-License-Identifier: Apache-2.0
# Source-style adaptation of MindSpeedOps arch32 MHC forward kernels.
# Source pin: 4efe2b675c2d51517c3fdd14c721020b8d41e241.
import triton
import triton.language as tl
try:
    from triton.language.extra.cann.extension import extract_slice
    tl.extract_slice = extract_slice
    USE_SLICE = True
    SLICE_PROBE = 'extension extract_slice imported'
except ImportError as error:
    USE_SLICE = False
    SLICE_PROBE = repr(error)

@triton.jit
def hc_post_bmm1_fwd_kernel(
    H_post_ptr,
    h_out_ptr,
    h_post_ptr,
    BS,
    C,
    stride_hp_bs: tl.constexpr,
    stride_hp_n: tl.constexpr,
    stride_ho_bs: tl.constexpr,
    stride_ho_c: tl.constexpr,
    stride_y_bs: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_y_c: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DIVISIBLE_C: tl.constexpr,
):
    pid_bs_blk = tl.program_id(0)
    pid_c_blk = tl.program_id(1)

    # ---- bs tile ----
    pid0 = pid_bs_blk * GROUP
    pids = pid0 + tl.arange(0, GROUP)
    mask_pid = pids < BS

    # ---- c tile ----
    c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

    m = mask_pid[:, None] & mask_c[None, :]

    # ---- load h_out once (G,BC) ----
    hout = tl.load(h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c, mask=m, other=0.0).to(
        tl.float32
    )

    # ---- load H_post scalars (G,1) ----
    hp0 = tl.load(H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp1 = tl.load(H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp2 = tl.load(H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
    hp3 = tl.load(H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]

    # ---- store h_post[bs, i, c] ----
    Y_base = h_post_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
    tl.store(Y_base + 0 * stride_y_n, hout * hp0, mask=m)
    tl.store(Y_base + 1 * stride_y_n, hout * hp1, mask=m)
    tl.store(Y_base + 2 * stride_y_n, hout * hp2, mask=m)
    tl.store(Y_base + 3 * stride_y_n, hout * hp3, mask=m)


@triton.jit
def hc_post_bmm2_fwd_kernel(
    H_ptr,
    X_ptr,
    Y_ptr,
    ADD_ptr,
    FUSED: tl.constexpr,
    USE_SLICE: tl.constexpr,
    stride_h_bs: tl.constexpr,
    stride_h_n: tl.constexpr,
    stride_h_k: tl.constexpr,
    stride_x_bs: tl.constexpr,
    stride_x_k: tl.constexpr,
    stride_x_c: tl.constexpr,
    stride_y_bs: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_y_c: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C: tl.constexpr,
):
    pid_bs_blk = tl.program_id(0)
    pid_c_blk = tl.program_id(1)

    pid0 = pid_bs_blk * GROUP
    pids = pid0 + tl.arange(0, GROUP)

    c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C
    X_base = X_ptr + pids[:, None] * stride_x_bs + c[None, :] * stride_x_c
    x0 = tl.load(X_base + 0 * stride_x_k, mask=c_mask[None, :]).to(tl.float32)
    x1 = tl.load(X_base + 1 * stride_x_k, mask=c_mask[None, :]).to(tl.float32)
    x2 = tl.load(X_base + 2 * stride_x_k, mask=c_mask[None, :]).to(tl.float32)
    x3 = tl.load(X_base + 3 * stride_x_k, mask=c_mask[None, :]).to(tl.float32)

    if USE_SLICE:
        k = tl.arange(0, 4)
        h0 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h1 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h2 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h3 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
    
        h00 = tl.extract_slice(h0, [0, 0], [GROUP, 1], [1, 1])
        h01 = tl.extract_slice(h0, [0, 1], [GROUP, 1], [1, 1])
        h02 = tl.extract_slice(h0, [0, 2], [GROUP, 1], [1, 1])
        h03 = tl.extract_slice(h0, [0, 3], [GROUP, 1], [1, 1])
    
        h10 = tl.extract_slice(h1, [0, 0], [GROUP, 1], [1, 1])
        h11 = tl.extract_slice(h1, [0, 1], [GROUP, 1], [1, 1])
        h12 = tl.extract_slice(h1, [0, 2], [GROUP, 1], [1, 1])
        h13 = tl.extract_slice(h1, [0, 3], [GROUP, 1], [1, 1])
    
        h20 = tl.extract_slice(h2, [0, 0], [GROUP, 1], [1, 1])
        h21 = tl.extract_slice(h2, [0, 1], [GROUP, 1], [1, 1])
        h22 = tl.extract_slice(h2, [0, 2], [GROUP, 1], [1, 1])
        h23 = tl.extract_slice(h2, [0, 3], [GROUP, 1], [1, 1])
    
        h30 = tl.extract_slice(h3, [0, 0], [GROUP, 1], [1, 1])
        h31 = tl.extract_slice(h3, [0, 1], [GROUP, 1], [1, 1])
        h32 = tl.extract_slice(h3, [0, 2], [GROUP, 1], [1, 1])
        h33 = tl.extract_slice(h3, [0, 3], [GROUP, 1], [1, 1])
    
    else:
        h00 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + 0 * stride_h_k).to(tl.float32)
        h01 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + 1 * stride_h_k).to(tl.float32)
        h02 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + 2 * stride_h_k).to(tl.float32)
        h03 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + 3 * stride_h_k).to(tl.float32)
        h10 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + 0 * stride_h_k).to(tl.float32)
        h11 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + 1 * stride_h_k).to(tl.float32)
        h12 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + 2 * stride_h_k).to(tl.float32)
        h13 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + 3 * stride_h_k).to(tl.float32)
        h20 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + 0 * stride_h_k).to(tl.float32)
        h21 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + 1 * stride_h_k).to(tl.float32)
        h22 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + 2 * stride_h_k).to(tl.float32)
        h23 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + 3 * stride_h_k).to(tl.float32)
        h30 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + 0 * stride_h_k).to(tl.float32)
        h31 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + 1 * stride_h_k).to(tl.float32)
        h32 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + 2 * stride_h_k).to(tl.float32)
        h33 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + 3 * stride_h_k).to(tl.float32)

    y0 = tl.fma(x0, h00, tl.fma(x1, h01, tl.fma(x2, h02, x3 * h03)))
    y1 = tl.fma(x0, h10, tl.fma(x1, h11, tl.fma(x2, h12, x3 * h13)))
    y2 = tl.fma(x0, h20, tl.fma(x1, h21, tl.fma(x2, h22, x3 * h23)))
    y3 = tl.fma(x0, h30, tl.fma(x1, h31, tl.fma(x2, h32, x3 * h33)))

    # Explicit FP32 boundary shared with the GM intermediate in the baseline.
    y0 = y0.to(tl.float32)
    y1 = y1.to(tl.float32)
    y2 = y2.to(tl.float32)
    y3 = y3.to(tl.float32)
    if FUSED:
        add_base = ADD_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
        y0 = y0 + tl.load(add_base + 0 * stride_y_n, mask=c_mask[None, :])
        y1 = y1 + tl.load(add_base + 1 * stride_y_n, mask=c_mask[None, :])
        y2 = y2 + tl.load(add_base + 2 * stride_y_n, mask=c_mask[None, :])
        y3 = y3 + tl.load(add_base + 3 * stride_y_n, mask=c_mask[None, :])
    Y_base = Y_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
    tl.store(Y_base + 0 * stride_y_n, y0, mask=c_mask[None, :])
    tl.store(Y_base + 1 * stride_y_n, y1, mask=c_mask[None, :])
    tl.store(Y_base + 2 * stride_y_n, y2, mask=c_mask[None, :])
    tl.store(Y_base + 3 * stride_y_n, y3, mask=c_mask[None, :])



@triton.jit
def transpose_user_h(H, T, BS: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, inner = k // 16, k % 16
    value = tl.load(H + b * 16 + (inner % 4) * 4 + inner // 4, k < BS * 16)
    tl.store(T + k, value, k < BS * 16)

@triton.jit
def add_outputs(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + k, k < N)
    b = tl.load(B + k, k < N)
    tl.store(Y + k, a + b, k < N)
