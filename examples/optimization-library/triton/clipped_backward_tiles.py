# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Apache-2.0; MindSpeed 4efe2b675c2d51517c3fdd14c721020b8d41e241.
import torch

_DTYPE_BLOCK_K_CAP = {
    torch.bfloat16: 4096,
    torch.float16: 4096,
    torch.float32: 1024,
}
# max BLOCK_M * BLOCK_K (fp32 elements) per tile, by dtype — all dtypes share the
# 4096-elem cap (bf16/fp16 1x4096, fp32 4x1024). fp32's budget was a conservative
# 2048 before; 4x1024 halves the grid and is 14%~36% faster across LLM shapes
# (this backend is program-scheduling bound, ~85ns/program).
_DTYPE_UB_BUDGET = {
    torch.bfloat16: 4096,
    torch.float16: 4096,
    torch.float32: 4096,
}
# Backward kernel holds 5 fp32 tiles (g, u, dy in + dg, du out) vs 3 for forward.
# 5 tiles * 4096 elems * 4B = 80KiB stays well under the 192KiB UB (verified on
# Ascend910; 8192-elem tiles fail to compile regardless of dtype or multibuffer).
# The real perf lever is program count — measured ~85ns/program scheduling cost on
# this backend, so halving the grid (2048 -> 4096-elem tiles) nearly halves runtime.
_DTYPE_UB_BUDGET_BWD = {
    torch.bfloat16: 4096,
    torch.float16: 4096,
    torch.float32: 4096,
}


def _pick_block_sizes(m: int, k: int, dtype: torch.dtype, ub_budget: dict | None = None) -> tuple[int, int]:
    """Pick (BLOCK_M, BLOCK_K) by data size and dtype — manual tiering, no autotune.

    - BLOCK_K: cover K in one tile when feasible, but never exceed the per-dtype UB
      ceiling (bf16/fp16 4096, fp32 1024). Power-of-two.
    - BLOCK_M: grow with M, capped so BLOCK_M * BLOCK_K stays within the dtype UB
      budget (fp32 budget is far tighter than bf16).
    """
    k_cap = _DTYPE_BLOCK_K_CAP.get(dtype, 1024)
    budget = (ub_budget or _DTYPE_UB_BUDGET).get(dtype, 2048)

    # BLOCK_K: smallest power-of-2 >= K, capped by the dtype ceiling.
    block_k = 1
    while block_k < k and block_k < k_cap:
        block_k *= 2
    block_k = min(block_k, k_cap)
    block_k = max(512, block_k)  # floor at 512 for tiny K

    # BLOCK_M: grow with M, capped by UB budget (block_m * block_k <= budget).
    if m <= 1:
        block_m = 1
    elif m <= 8:
        block_m = 2
    elif m <= 64:
        block_m = 4
    elif m <= 512:
        block_m = 8
    else:
        block_m = 16

    ub_cap = max(1, budget // block_k)
    ub_cap_pow2 = 1 << (ub_cap.bit_length() - 1) if ub_cap >= 1 else 1
    block_m = max(1, min(block_m, ub_cap_pow2))

    return block_m, block_k
