# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# MIT; source pin 7acb9c40a6ca827e885d902b7eaee5d45457b230.
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Pointers
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD (cast will occur as needed based on out_ptr dtype)
    d_mask = off_d[None, :] < D
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask, other=0)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

@triton.jit
def _pack_seq_kernel_prefix(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    starts_ptr,
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # Pointers
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD (cast will occur as needed based on out_ptr dtype)
    d_mask = off_d[None, :] < D
    pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask, other=0)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)

@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask, other=0)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)

@triton.jit
def _unpack_seq_triton_kernel_prefix(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    starts_ptr,
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    in_start = tl.load(starts_ptr + pid_b)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask, other=0)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)

@triton.jit
def _exclusive_prefix(lengths, starts, B: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    length = tl.load(lengths + i, i < B, other=0)
    inclusive = tl.cumsum(length, 0)
    tl.store(starts, 0)
    tl.store(starts + i + 1, inclusive, i < B)
