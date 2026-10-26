# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch

from mindspeed_ops.arch32.triton.inplace_rope import (
    inplace_rope_fwd_inplace,
    inplace_rope_bwd,
)

__all__ = ["inplace_rope"]


def _validate_inputs(x, cos_half, sin_half):
    if x.dim() != 4:
        raise ValueError(f"x must be 4D [B, S, H, D], got shape {tuple(x.shape)}")
    if cos_half.dim() != 3 or sin_half.dim() != 3:
        raise ValueError(
            f"cos_half/sin_half must be 3D [B, S, half], got {tuple(cos_half.shape)} / {tuple(sin_half.shape)}"
        )

    for name, t in (("x", x), ("cos_half", cos_half), ("sin_half", sin_half)):
        if t.device.type != "npu":
            raise RuntimeError(f"{name} must be on NPU, got {t.device.type}")

    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not cos_half.is_contiguous() or not sin_half.is_contiguous():
        raise ValueError("cos_half and sin_half must be contiguous")

    B, S, H, D = x.shape
    half = cos_half.shape[-1]

    if sin_half.shape != cos_half.shape:
        raise ValueError(f"sin_half shape {tuple(sin_half.shape)} != cos_half shape {tuple(cos_half.shape)}")
    if not (cos_half.dtype.is_floating_point and sin_half.dtype.is_floating_point):
        raise ValueError("cos_half/sin_half must be floating point")

    Rd = half * 2
    nope_dim = D - Rd
    if nope_dim <= 0:
        raise ValueError(f"x last dim D={D} must be larger than rope dim Rd={Rd}")
    if (half & (half - 1)) != 0:
        raise ValueError(f"half must be a power of 2, got {half}")
    return B, S, H, nope_dim, half, Rd


class _InplaceRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos_half, sin_half):
        B, S, H, nope_dim, half, Rd = _validate_inputs(x, cos_half, sin_half)
        ctx.save_for_backward(x[..., -Rd:].contiguous(), cos_half, sin_half)
        inplace_rope_fwd_inplace(x, cos_half, sin_half)
        return x

    @staticmethod
    def backward(ctx, dy):
        x_rope, cos_half, sin_half = ctx.saved_tensors
        dx, dcos, dsin = inplace_rope_bwd(dy, x_rope, cos_half, sin_half)
        return dx, dcos, dsin


def inplace_rope(x: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """Interleaved rotary position embedding (DeepSeek V4 style).

    Applies the interleaved RoPE rotation to the trailing ``Rd`` dims of ``x``,
    leaving the ``nope`` prefix unchanged. Equivalent to
    ``torch_npu.npu_rotary_mul(..., rotary_mode="interleave")`` on the rope slice.

    Args:
        x: Input tensor with shape ``[B, S, H, D]``, where ``D = nope_dim + Rd``.
        cos_half: Cosine tensor with shape ``[B, S, half]``, ``Rd = 2 * half``.
            Typically fp32 for higher accumulation precision.
        sin_half: Sine tensor with the same shape as ``cos_half``.

    Returns:
        A new tensor with the same shape/dtype as ``x``. The trailing ``Rd`` dims
        are rotated; the ``nope`` prefix is copied unchanged. Backward yields
        ``dx`` and the ``dcos``/``dsin`` gradients of ``cos_half``/``sin_half``.
    """
    return _InplaceRopeFunction.apply(x, cos_half, sin_half)
