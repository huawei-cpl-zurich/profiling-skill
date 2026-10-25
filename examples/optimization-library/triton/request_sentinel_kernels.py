# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py
import triton
import triton.language as tl
from request_sentinel_source import expand_kernel

@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_unreplaced_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def replace_tokens_kernel(output, replace_from, replace_to, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(output + idx, mask=idx < N, other=0)
    value = tl.where(value == replace_from, replace_to, value)
    tl.store(output + idx, value, mask=idx < N)
