# Copyright © 2026 Huawei Technologies Co., Ltd.
# Copyright 2023-2024 SGLang Team
# MIT; adapted from triton-ascend-kernels 7acb9c40a6ca827e885d902b7eaee5d45457b230.
import triton
import triton.language as tl

@triton.jit
def wave007_logsoftmax_streaming(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles multiple rows due to fixed grid size.
    """
    # Get the block index
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    # Calculate the range of rows this block should handle
    rows_per_block = (n_rows + num_blocks - 1) // num_blocks
    start_row = pid * rows_per_block
    end_row = tl.minimum((pid + 1) * rows_per_block, n_rows)

    # Loop over the rows assigned to this block
    for row_idx in range(start_row, end_row):
        # Compute base pointers for input and output rows
        row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride

        # Step 1: Find maximum value in the row for numerical stability
        max_val = -float("inf")
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            # Load values
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

            # Update maximum
            max_val = tl.max(tl.maximum(vals, max_val))

        # Step 2: Compute sum of exp(x - max_val)
        sum_exp = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            # Load values
            vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

            # Compute exp(x - max_val) and accumulate
            exp_vals = tl.exp(vals - max_val)
            sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

        # Compute log(sum_exp)
        log_sum_exp = tl.log(sum_exp)

        # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            # Load values
            vals = tl.load(row_start_ptr + col_idx, mask=mask)

            # Compute log_softmax
            output_vals = vals - max_val - log_sum_exp

            # Store results
            tl.store(output_row_start_ptr + col_idx, output_vals, mask=mask)



@triton.jit
def wave007_logsoftmax_resident(input_ptr, output_ptr, input_row_stride,
                               output_row_stride, n_rows, n_cols,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    rows_per_block = (n_rows + num_blocks - 1) // num_blocks
    start_row = pid * rows_per_block
    end_row = tl.minimum((pid + 1) * rows_per_block, n_rows)
    for row_idx in range(start_row, end_row):
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(input_ptr + row_idx * input_row_stride + col_idx,
                       mask=mask, other=-float("inf"))
        max_val = tl.max(vals)
        exp_vals = tl.exp(vals - max_val)
        sum_exp = tl.sum(tl.where(mask, exp_vals, 0.0))
        output_vals = vals - max_val - tl.log(sum_exp)
        tl.store(output_ptr + row_idx * output_row_stride + col_idx,
                 output_vals, mask=mask)
