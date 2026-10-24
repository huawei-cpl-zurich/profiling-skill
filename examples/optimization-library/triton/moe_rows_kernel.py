# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MindSpeedOps 4efe2b675c2d51517c3fdd14c721020b8d41e241,
# mindspeed_ops/arch32/triton/sort_chunks_by_idx.py (_sort_chunks_by_map_kernel).
import triton
import triton.language as tl

try:
    get_element = tl.get_element
    GET_ELEMENT_SOURCE = 'triton.language.get_element'
except AttributeError:
    try:
        from triton.language.extra.cann.extension import get_element
        GET_ELEMENT_SOURCE = 'triton.language.extra.cann.extension.get_element'
    except ImportError:
        get_element = None
        GET_ELEMENT_SOURCE = 'unavailable; portable scalar map load'


@triton.jit
def moe_rows_before(X,Y,M,P,Q,N:tl.constexpr,D:tl.constexpr,BS:tl.constexpr,BLOCK_TOKENS:tl.constexpr,
                    PROBS:tl.constexpr,FORWARD:tl.constexpr,USE_ELEMENT:tl.constexpr):
    start = tl.program_id(0)*BS
    h = tl.arange(0,D)
    if USE_ELEMENT:
        offsets = tl.arange(0,BLOCK_TOKENS)
        mapping = tl.load(M+start+offsets,(offsets<BS)&(start+offsets<N),0).to(tl.int64)
    for j in range(BLOCK_TOKENS):
        i = start+j
        if j<BS and i<N:
            if USE_ELEMENT: mapped = get_element(mapping,(j,)).to(tl.int64)
            else: mapped = tl.load(M+i).to(tl.int64)
            if FORWARD: src,dst = i,mapped
            else: src,dst = mapped,i
            value = tl.load(X+src*D+h)
            tl.store(Y+dst*D+h,value)
            if PROBS:
                prob = tl.load(P+src)
                tl.store(Q+dst,prob)


@triton.jit
def moe_rows_after(X,Y,M,P,Q,N:tl.constexpr,D:tl.constexpr,BS:tl.constexpr,BLOCK_TOKENS:tl.constexpr,
                   PROBS:tl.constexpr,FORWARD:tl.constexpr,USE_ELEMENT:tl.constexpr):
    start = tl.program_id(0)*BS
    h = tl.arange(0,D)
    r = tl.arange(0,4)
    for off in range(0,BLOCK_TOKENS,4):
        i = start+off+r
        mask = (off+r<BS)&(i<N)
        mapped = tl.load(M+i,mask,0).to(tl.int64)
        if FORWARD: src,dst = i,mapped
        else: src,dst = mapped,i
        value = tl.load(X+src[:,None]*D+h[None,:],mask[:,None],0)
        tl.store(Y+dst[:,None]*D+h[None,:],value,mask[:,None])
        if PROBS:
            prob = tl.load(P+src,mask,0)
            tl.store(Q+dst,prob,mask)
