"""Self-contained frozen Triton matmul-add candidate used for calibration."""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu
import triton.runtime.driver as driver


KERNEL_NAME = "streaming_matmul_add_kernel_mix_aic"


@triton.jit
def streaming_matmul_add_kernel(
    lhs, rhs, bias, output,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_m: tl.constexpr = triton.cdiv(m, BLOCK_M)
    blocks_n: tl.constexpr = triton.cdiv(n, BLOCK_N)
    total_blocks: tl.constexpr = blocks_m * blocks_n
    blocks_per_core: tl.constexpr = triton.cdiv(total_blocks, num_cores)
    reduction = tl.arange(0, BLOCK_K)

    # Optimization point #3: each Cube core owns one contiguous tile range.
    for local_block in range(0, blocks_per_core):
        block_idx = pid * blocks_per_core + local_block
        block_m = block_idx // blocks_n
        block_n = block_idx - block_m * blocks_n
        rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_block = block_idx < total_blocks
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, k, BLOCK_K):
            lhs_offsets = rows[:, None] * k + start + reduction[None, :]
            rhs_offsets = (start + reduction[:, None]) * n + cols[None, :]
            lhs_tile = tl.load(
                lhs + lhs_offsets,
                mask=valid_block & (rows[:, None] < m) &
                     (start + reduction[None, :] < k),
                other=0.0,
            )
            rhs_tile = tl.load(
                rhs + rhs_offsets,
                mask=valid_block & (start + reduction[:, None] < k) &
                     (cols[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(lhs_tile, rhs_tile)
        result = accumulator + tl.load(
            bias + cols, mask=valid_block & (cols < n), other=0.0
        )[None, :]
        offsets = rows[:, None] * n + cols[None, :]
        tl.store(
            output + offsets,
            result,
            mask=valid_block & (rows[:, None] < m) & (cols[None, :] < n),
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch_npu.npu.current_device()
        properties = driver.active.utils.get_device_properties(device)
        self.cube_core_num = properties["num_aicore"]

    def forward(self, lhs, rhs, bias):
        m, k = lhs.shape
        _, n = rhs.shape
        output = torch.empty((m, n), device=lhs.device, dtype=torch.float32)
        total_blocks = triton.cdiv(m, 32) * triton.cdiv(n, 32)
        grid_size = min(total_blocks, self.cube_core_num)
        grid = (grid_size,)
        streaming_matmul_add_kernel[grid](
            lhs, rhs, bias, output, m, n, k,
            num_cores=grid_size,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        )
        return output
