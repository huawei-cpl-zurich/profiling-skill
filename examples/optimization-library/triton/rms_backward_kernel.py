# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 LinkedIn Corporation
# Source: triton-ascend-ops 7acb9c40a6ca827e885d902b7eaee5d45457b230.
# MIT and accompanying third-party terms: RMS_BACKWARD_LICENSE / RMS_BACKWARD_NOTICES.
import triton
import triton.language as tl
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
    W_row = W_row + offset

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask, other=0.0)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row[None, :]).to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row[None, :]
        else:
            m = dY_row * W_row[None, :]

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols)
            * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None]
            * X_row
        )

        # calculate the gradient of W
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]).to(X_dtype), 0)
        else:
            # here X_row is already in fp32 (see previous if block)
            dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


@triton.jit
def zero_dx(out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, 0., i < N)


@triton.jit
def sum_dw(partial, out, D: tl.constexpr, CORES: tl.constexpr,
           ROWS: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, ROWS)
    v = tl.load(partial + r[:, None] * D + c[None, :],
                (r[:, None] < CORES) & (c[None, :] < D), other=0.)
    tl.store(out + c, tl.sum(v, 0), c < D)
