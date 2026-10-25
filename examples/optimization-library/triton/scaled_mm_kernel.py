# Copyright © 2026 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Adapted from triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230.
# MIT licensed; see scaled_mm_NOTICE.txt for retained third-party notices.
import triton
import triton.language as tl

@triton.jit
def _store_scaled_result(accumulator, c_ptr, bias_ptr, mode_ptr, mode, pid,
                         M, N, pid_m, pid_n, stride_cm, stride_cn,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # Common private proof cost: one int32 store per program per API.
    tl.store(mode_ptr + pid, mode)
    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    mode_ptr,
    FACTOR: tl.constexpr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # These are the same required source scale loads, before broadcasting.
    sa = tl.load(scale_a_ptrs[:, None], masks_scale_am[:, None], other=0)
    sb = tl.load(scale_b_ptrs[:, None], masks_scale_bn[:, None], other=0)
    accumulator = accumulator.to(tl.float32)
    if FACTOR and (BLOCK_SIZE_SCALE_A == 1 and BLOCK_SIZE_SCALE_B == 1):
        av = tl.sum(tl.sum(sa, 0), 0)
        bv = tl.sum(tl.sum(sb, 0), 0)
        abits = av.to(tl.int32, bitcast=True)
        bbits = bv.to(tl.int32, bitcast=True)
        # Positive normal powers of two with unbiased exponent in [-8,8].
        valid_a = ((abits & 0x807fffff) == 0) & (abits >= (119 << 23)) & (abits <= (135 << 23))
        valid_b = ((bbits & 0x807fffff) == 0) & (bbits >= (119 << 23)) & (bbits <= (135 << 23))
        # Store in each branch to avoid an unsupported matrix-valued merge.
        if valid_a & valid_b:
            factored = accumulator * (av * bv)
            _store_scaled_result(factored, c_ptr, bias_ptr, mode_ptr, 1, pid,
                                 M, N, pid_m, pid_n, stride_cm, stride_cn,
                                 BLOCK_SIZE_M, BLOCK_SIZE_N)
        else:
            fallback = sa.broadcast_to((BLOCK_SIZE_M, 1)) * accumulator
            fallback = sb.broadcast_to((BLOCK_SIZE_N, 1)).T * fallback
            _store_scaled_result(fallback, c_ptr, bias_ptr, mode_ptr, 2, pid,
                                 M, N, pid_m, pid_n, stride_cm, stride_cn,
                                 BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        sequential = sa.broadcast_to((BLOCK_SIZE_M, 1)) * accumulator
        sequential = sb.broadcast_to((BLOCK_SIZE_N, 1)).T * sequential
        _store_scaled_result(sequential, c_ptr, bias_ptr, mode_ptr, 2 if FACTOR else 0, pid,
                             M, N, pid_m, pid_n, stride_cm, stride_cn,
                             BLOCK_SIZE_M, BLOCK_SIZE_N)
