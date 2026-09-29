"""Self-contained frozen Triton matmul-add candidate used for calibration."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


KERNEL_NAME = "streaming_matmul_add_kernel_mix_aic"


@triton.jit
def streaming_matmul_add_kernel(
    lhs,
    rhs,
    bias,
    output,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, k, BLOCK_K):
        lhs_offsets = rows[:, None] * k + start + reduction[None, :]
        rhs_offsets = (start + reduction[:, None]) * n + cols[None, :]
        lhs_tile = tl.load(
            lhs + lhs_offsets,
            mask=(rows[:, None] < m) & (start + reduction[None, :] < k),
            other=0.0,
        )
        rhs_tile = tl.load(
            rhs + rhs_offsets,
            mask=(start + reduction[:, None] < k) & (cols[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(lhs_tile, rhs_tile)
    result = accumulator + tl.load(bias + cols, mask=cols < n, other=0.0)[None, :]
    offsets = rows[:, None] * n + cols[None, :]
    tl.store(output + offsets, result, mask=(rows[:, None] < m) & (cols[None, :] < n))


class Model(nn.Module):
    def forward(self, lhs, rhs, bias):
        m, k = lhs.shape
        _, n = rhs.shape
        output = torch.empty((m, n), device=lhs.device, dtype=torch.float32)
        block_m = block_n = block_k = 32
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        streaming_matmul_add_kernel[grid](
            lhs, rhs, bias, output, m, n, k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return output
