# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import triton
import triton.language as tl

from mindspeed_ops.utils import is_arch35


@triton.jit
def _inplace_rope_fwd_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    total,
    S,
    H,
    nope_dim: tl.constexpr,
    Rd: tl.constexpr,
    half: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    D = nope_dim + Rd
    SH = S * H
    num_blocks = total // BLOCK_ROWS
    pair = tl.arange(0, half)

    for block in range(pid, num_blocks, num_progs):
        row_start = block * BLOCK_ROWS
        b = row_start // SH
        s = (row_start % SH) // H
        base = b * S + s

        x_bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(total, D),
            strides=(D, 1),
            offsets=(row_start, nope_dim),
            block_shape=(BLOCK_ROWS, Rd),
            order=(1, 0),
        )
        xr = tl.load(x_bp)
        ck = tl.load(cos_ptr + base * half + pair)
        sk = tl.load(sin_ptr + base * half + pair)

        xp = tl.reshape(xr, (BLOCK_ROWS, half, 2))
        xe, xo = tl.split(xp)
        e = xe.to(tl.float32)
        o = xo.to(tl.float32)
        c = ck.to(tl.float32)[None, :]
        ss = sk.to(tl.float32)[None, :]

        o0 = (e * c - o * ss).to(xr.dtype)
        o1 = (o * c + e * ss).to(xr.dtype)
        res = tl.reshape(tl.join(o0, o1), (BLOCK_ROWS, Rd))
        tl.store(x_bp, res)


@triton.jit
def _inplace_rope_bwd_kernel(
    dy_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    dx_ptr,
    dcos_ptr,
    dsin_ptr,
    total,
    S,
    H,
    nope_dim: tl.constexpr,
    Rd: tl.constexpr,
    half: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    D = nope_dim + Rd
    SH = S * H
    num_blocks = total // BLOCK_ROWS
    pair = tl.arange(0, half)

    for block in range(pid, num_blocks, num_progs):
        row_start = block * BLOCK_ROWS
        b = row_start // SH
        s = (row_start % SH) // H
        base = b * S + s

        # Copy the nope segment: dx[..., :nope_dim] = dy[..., :nope_dim] (fused into the kernel to avoid a separate copy kernel)
        for nope_start in range(0, nope_dim, BLOCK_NOPE):
            nope_bp = tl.make_block_ptr(
                base=dy_ptr,
                shape=(total, D),
                strides=(D, 1),
                offsets=(row_start, nope_start),
                block_shape=(BLOCK_ROWS, BLOCK_NOPE),
                order=(1, 0),
            )
            nope = tl.load(nope_bp)
            nope_bp_dx = tl.make_block_ptr(
                base=dx_ptr,
                shape=(total, D),
                strides=(D, 1),
                offsets=(row_start, nope_start),
                block_shape=(BLOCK_ROWS, BLOCK_NOPE),
                order=(1, 0),
            )
            tl.store(nope_bp_dx, nope)

        dy_bp = tl.make_block_ptr(
            base=dy_ptr,
            shape=(total, D),
            strides=(D, 1),
            offsets=(row_start, nope_dim),
            block_shape=(BLOCK_ROWS, Rd),
            order=(1, 0),
        )
        # Saved rope slice, flattened as [total, Rd]
        x_bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(total, Rd),
            strides=(Rd, 1),
            offsets=(row_start, 0),
            block_shape=(BLOCK_ROWS, Rd),
            order=(1, 0),
        )
        dx_bp = tl.make_block_ptr(
            base=dx_ptr,
            shape=(total, D),
            strides=(D, 1),
            offsets=(row_start, nope_dim),
            block_shape=(BLOCK_ROWS, Rd),
            order=(1, 0),
        )

        dyr = tl.load(dy_bp)
        xr = tl.load(x_bp)
        ck = tl.load(cos_ptr + base * half + pair)
        sk = tl.load(sin_ptr + base * half + pair)

        yp = tl.reshape(dyr, (BLOCK_ROWS, half, 2))
        dye, dyo = tl.split(yp)
        xp = tl.reshape(xr, (BLOCK_ROWS, half, 2))
        xe, xo = tl.split(xp)

        de = dye.to(tl.float32)
        do = dyo.to(tl.float32)
        e = xe.to(tl.float32)
        o = xo.to(tl.float32)
        c = ck.to(tl.float32)[None, :]
        ss = sk.to(tl.float32)[None, :]

        # dx = R^T(dy)
        dxe = de * c + do * ss
        dxo = -de * ss + do * c
        dx_out = tl.reshape(tl.join(dxe, dxo), (BLOCK_ROWS, Rd)).to(dyr.dtype)
        tl.store(dx_bp, dx_out)

        # dcos = dye*xe + dyo*xo ; dsin = -dye*xo + dyo*xe (reduced over the head dim)
        dcos_partial = de * e + do * o
        dsin_partial = -de * o + do * e
        dcos_block = tl.sum(dcos_partial, axis=0)
        dsin_block = tl.sum(dsin_partial, axis=0)

        tl.atomic_add(dcos_ptr + base * half + pair, dcos_block)
        tl.atomic_add(dsin_ptr + base * half + pair, dsin_block)


def _pick_block_rows(total: int, H: int) -> int:
    max_br = 128 if is_arch35() else 64
    for br in (128, 64, 32, 16, 8, 4, 2, 1):
        if br <= max_br and total % br == 0 and H % br == 0:
            return br
    return 1


def inplace_rope_fwd_inplace(x, cos_half, sin_half, block_rows=None):
    """In-place forward interleaved RoPE on the trailing ``Rd`` dims of ``x``.

    ``x`` has shape ``[B, S, H, D]`` with ``D = nope_dim + Rd``; ``cos_half`` and
    ``sin_half`` have shape ``[B, S, half]`` with ``Rd = 2 * half``. The rope dims
    (``x[..., -Rd:]``) are rotated in-place; the ``nope`` prefix is left untouched.
    Returns ``x``.
    """
    B, S, H, D = x.shape
    half = cos_half.shape[-1]
    Rd = half * 2
    nope_dim = D - Rd
    total = B * S * H
    assert (half & (half - 1)) == 0, f"half must be a power of 2, got {half}"
    assert Rd % 2 == 0 and nope_dim > 0
    if block_rows is None:
        block_rows = _pick_block_rows(total, H)
    assert total % block_rows == 0 and H % block_rows == 0

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_ROWS"]),)

    _inplace_rope_fwd_kernel[grid](
        x,
        cos_half,
        sin_half,
        total,
        S,
        H,
        nope_dim,
        Rd,
        half,
        block_rows,
        num_stages=2,
    )
    return x


def inplace_rope_bwd(dy, x_rope, cos_half, sin_half, block_rows=None):
    """Backward of interleaved RoPE.

    Returns ``(dx, dcos, dsin)`` where ``dx`` is the gradient of ``x``,
    ``dcos``/``dsin`` are the ``[B, S, half]`` fp32 gradients of ``cos_half`` and
    ``sin_half``. ``x_rope`` is the saved rope slice ``[B, S, H, Rd]``.
    """
    B, S, H, D = dy.shape
    half = cos_half.shape[-1]
    Rd = half * 2
    nope_dim = D - Rd
    total = B * S * H
    assert (half & (half - 1)) == 0, f"half must be a power of 2, got {half}"
    assert Rd % 2 == 0 and nope_dim > 0
    if block_rows is None:
        block_rows = _pick_block_rows(total, H)
    assert total % block_rows == 0 and H % block_rows == 0

    dx = dy.new_empty(dy.shape)
    dcos = torch.zeros(B, S, half, dtype=torch.float32, device=dy.device)
    dsin = torch.zeros(B, S, half, dtype=torch.float32, device=dy.device)

    BLOCK_NOPE = 64
    assert nope_dim % BLOCK_NOPE == 0, "nope_dim must be divisible by BLOCK_NOPE"

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_ROWS"]),)

    _inplace_rope_bwd_kernel[grid](
        dy,
        x_rope,
        cos_half,
        sin_half,
        dx,
        dcos,
        dsin,
        total,
        S,
        H,
        nope_dim,
        Rd,
        half,
        block_rows,
        BLOCK_NOPE,
        num_stages=2,
    )
    return dx, dcos, dsin
