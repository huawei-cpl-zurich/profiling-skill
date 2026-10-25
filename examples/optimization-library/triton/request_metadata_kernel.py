# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: MIT
# Source pin 7acb9c40a6ca827e885d902b7eaee5d45457b230; see request_metadata_NOTICE.
import triton
import triton.language as tl

@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    rows_per_prog: tl.constexpr,
    tiles_per_row: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    row_blk_id = tl.program_id(0)

    for row_id in range(0, rows_per_prog):
        token_id = row_blk_id * rows_per_prog + row_id
        if token_id < num_tokens:
            # Load request id for this token (no mask: grid is exact)
            req = tl.load(req_id_ptr + token_id)

            is_prefill = False
            if HAS_PREFILL:
                prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
                is_prefill = prefill_req_id.to(tl.float32) >= 0
                workspace_start = tl.load(
                    workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
                )

            # tile index along columns
            for tile_id in range(0, tiles_per_row):
                # Each program covers BLOCK_N consecutive columns
                indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

                # Load token indices for this tile
                ti_ptr = (
                    token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
                )
                tok = tl.load(ti_ptr)  # int32

                # Only token == -1 should propagate as -1
                is_invalid_tok = tok.to(tl.float32) < 0

                # Compute block id and in-block offset
                block_id = tok // BLOCK_SIZE
                inblock_off = tok % BLOCK_SIZE

                # Guard block_table access
                valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
                bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
                is_invalid_tok |= ~valid_block
                base = tl.load(bt_ptr, mask=valid_block & (not is_prefill), other=0)
                out_val = base * BLOCK_SIZE + inblock_off

                # Override with prefill output if prefill is enabled
                if HAS_PREFILL:
                    prefill_out = workspace_start + tok
                    out_val = tl.where(is_prefill, prefill_out, out_val)
                out_val = tl.where(is_invalid_tok, -1, out_val)

                # Store results
                out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
                tl.store(out_ptr_ij, out_val)


# Educational row/tile grid; vector mapping expressions preserved verbatim.
@triton.jit
def request_tile_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    rows_per_prog: tl.constexpr,
    tiles_per_row: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    req = tl.load(req_id_ptr + token_id)

    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id.to(tl.float32) >= 0
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )


    tile_id = tl.program_id(1)
    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load token indices for this tile
    ti_ptr = (
        token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    )
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok.to(tl.float32) < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & (not is_prefill), other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)
