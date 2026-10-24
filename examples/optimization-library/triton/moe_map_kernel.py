# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# MindSpeedOps 4efe2b675c2d51517c3fdd14c721020b8d41e241,
# mindspeed_ops/arch32/triton/sort_chunks_by_idx.py:27-64.
import triton
import triton.language as tl


@triton.jit
def moe_map_before(split_sizes_ptr,cumsum_ptr,inverse_sorted_indices_ptr,output_cumsum_ptr,dst_rows_ptr,
                   num_splits,num_tokens,BLOCK_SIZE:int,SUB_BLOCK_SIZE:tl.constexpr,IDX_LOAD_WIDTH:tl.constexpr):
    pid=tl.program_id(0)
    load_split_offset=tl.arange(0,IDX_LOAD_WIDTH)
    input_split_sizes=tl.load(split_sizes_ptr+load_split_offset,mask=load_split_offset<num_splits,other=0).to(tl.float32)
    input_split_sizes_cumsum=tl.cumsum(input_split_sizes)
    for off in range(0,BLOCK_SIZE,SUB_BLOCK_SIZE):
        token_offsets=pid*BLOCK_SIZE+off+tl.arange(0,SUB_BLOCK_SIZE)
        token_offsets_cmp=token_offsets.to(tl.float32)
        input_split_sizes_mask=tl.where(input_split_sizes_cumsum[None,:]<=token_offsets_cmp[:,None],1,0)
        input_chunk_indices=tl.sum(input_split_sizes_mask,axis=-1)
        cumsum_mask=input_chunk_indices<(num_splits+1)
        input_split_sizes_presums=tl.load(cumsum_ptr+input_chunk_indices,mask=cumsum_mask,other=0)
        inv_mask=input_chunk_indices<num_splits
        output_chunk_indices=tl.load(inverse_sorted_indices_ptr+input_chunk_indices,mask=inv_mask,other=0)
        output_chunk_mask=output_chunk_indices<(num_splits+1)
        output_presums=tl.load(output_cumsum_ptr+output_chunk_indices,mask=output_chunk_mask,other=0)
        dst_rows=output_presums+token_offsets_cmp-input_split_sizes_presums
        tl.store(dst_rows_ptr+token_offsets,dst_rows,mask=token_offsets<num_tokens)


@triton.jit
def moe_map_after(split_sizes_ptr,cumsum_ptr,inverse_sorted_indices_ptr,output_cumsum_ptr,dst_rows_ptr,
                  num_splits,num_tokens,BLOCK_SIZE:int,SUB_BLOCK_SIZE:tl.constexpr,IDX_LOAD_WIDTH:tl.constexpr):
    pid=tl.program_id(0)
    load_split_offset=tl.arange(0,IDX_LOAD_WIDTH)
    input_split_sizes=tl.load(split_sizes_ptr+load_split_offset,mask=load_split_offset<num_splits,other=0)
    input_split_sizes_cumsum=tl.cumsum(input_split_sizes)
    for off in range(0,BLOCK_SIZE,SUB_BLOCK_SIZE):
        token_offsets=pid*BLOCK_SIZE+off+tl.arange(0,SUB_BLOCK_SIZE)
        token_offsets_cmp=token_offsets
        input_split_sizes_mask=tl.where(input_split_sizes_cumsum[None,:]<=token_offsets_cmp[:,None],1,0)
        input_chunk_indices=tl.sum(input_split_sizes_mask,axis=-1)
        cumsum_mask=input_chunk_indices<(num_splits+1)
        input_split_sizes_presums=tl.load(cumsum_ptr+input_chunk_indices,mask=cumsum_mask,other=0)
        inv_mask=input_chunk_indices<num_splits
        output_chunk_indices=tl.load(inverse_sorted_indices_ptr+input_chunk_indices,mask=inv_mask,other=0)
        output_chunk_mask=output_chunk_indices<(num_splits+1)
        output_presums=tl.load(output_cumsum_ptr+output_chunk_indices,mask=output_chunk_mask,other=0)
        dst_rows=output_presums+token_offsets_cmp-input_split_sizes_presums
        tl.store(dst_rows_ptr+token_offsets,dst_rows,mask=token_offsets<num_tokens)
