# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Source-derived backward bodies; original source retained adjacent.
import triton
import triton.language as tl

@triton.jit
def rope_backward_zero(dc, ds, size:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(dc+i,0.,i<size)
    tl.store(ds+i,0.,i<size)

@triton.jit
def rope_backward_atomic(
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


@triton.jit
def rope_backward_owned(
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

        tl.store(dcos_ptr + base * half + pair, dcos_block)
        tl.store(dsin_ptr + base * half + pair, dsin_block)

