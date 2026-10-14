"""Frozen fully fused Triton BSA candidate used only for route qualification."""

import torch
import triton
import triton.language as tl


def blockmask_access_plan(heads, blocks):
    """Describe the bounded block-mask access made by each official head."""
    if blocks <= 0 or blocks != heads // 3:
        raise ValueError("qualification layout requires one third block-sparse heads")
    sparse_start = heads - 2 * blocks
    sparse_end = sparse_start + blocks
    return [
        (head, min(max(head - sparse_start, 0), blocks - 1),
         sparse_start <= head < sparse_end)
        for head in range(heads)
    ]


@triton.jit
def block_sparse_attention_kernel(
    q_ptr, k_ptr, v_ptr, cuq_ptr, cuk_ptr, head_type_ptr,
    stream_ptr, blockmask_ptr, out_ptr,
    total_q: tl.constexpr, total_k: tl.constexpr,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    B: tl.constexpr, NBLK: tl.constexpr,
    NROW: tl.constexpr, NCOL: tl.constexpr,
    SCALE: tl.constexpr, CAUSAL: tl.constexpr,
    EXACT: tl.constexpr, MAX_K: tl.constexpr,
    BLOCK_K: tl.constexpr, DENSE_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // H
    head = pid - q_idx * H
    d = tl.arange(0, D)
    batch = tl.zeros((), tl.int32)
    for ib in range(B):
        batch = tl.where(q_idx >= tl.load(cuq_ptr + ib + 1), ib + 1, batch)
    batch = tl.minimum(batch, B - 1)
    q_start = tl.load(cuq_ptr + batch)
    q_end = tl.load(cuq_ptr + batch + 1)
    k_start = tl.load(cuk_ptr + batch)
    k_end = tl.load(cuk_ptr + batch + 1)
    q_local = q_idx - q_start
    sq = q_end - q_start
    sk = k_end - k_start

    q = tl.load(q_ptr + (q_idx * H + head) * D + d).to(tl.float32)
    htype = tl.load(head_type_ptr + head)
    block_rank = head - DENSE_HEADS
    sparse_head = (block_rank >= 0) & (block_rank < NBLK)
    safe_block_rank = tl.maximum(0, tl.minimum(block_rank, NBLK - 1))
    sink = tl.load(stream_ptr + 2 * head)
    local = tl.load(stream_ptr + 2 * head + 1)
    kv_head = head // (H // HK)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)

    for k0 in range(0, MAX_K, BLOCK_K):
        ko = k0 + tl.arange(0, BLOCK_K)
        valid = ko < sk
        k_abs = k_start + ko
        k_tile = tl.load(
            k_ptr + (k_abs[:, None] * HK + kv_head) * D + d[None, :],
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)
        score = tl.sum(k_tile * q[None, :], axis=1) * SCALE

        keep = valid
        if CAUSAL:
            keep = keep & (ko <= q_local + sk - sq)
        qb = q_local // 128
        kb = ko // 128
        bm_off = (((batch * NBLK + safe_block_rank) * NROW + qb) * NCOL + kb)
        bm = tl.load(
            blockmask_ptr + bm_off,
            mask=valid & sparse_head,
            other=0,
        ).to(tl.int1)
        keep = keep & ((htype != 1) | bm)

        if EXACT:
            exact_keep = (ko <= tl.minimum(q_local + sk - sq, sk)) & ~(
                (ko < q_local + sk - sq - (local - 1)) & (ko >= sink)
            )
            keep = keep & ((htype != -1) | exact_keep)
        else:
            qblock = q_local // 128
            if CAUSAL:
                start = tl.maximum((sq - sk) // 128, 0)
                mr = (tl.maximum(sk - sq, 0) + 127) // 128 + 1 + qblock - start
                valid_row = qblock >= start
            else:
                mr = NCOL
                valid_row = True
            lo = tl.maximum(mr - local, 0)
            hi = tl.minimum(mr, NCOL)
            stream_keep = valid_row & (((kb >= lo) & (kb < hi)) | (kb < sink))
            keep = keep & ((htype != -1) | stream_keep)

        masked_score = tl.where(keep, score, -float("inf"))
        tile_m = tl.max(masked_score, axis=0)
        new_m = tl.maximum(m_i, tile_m)
        finite = new_m != -float("inf")
        alpha = tl.where(finite, tl.exp(m_i - new_m), 1.0)
        p = tl.where(keep & finite, tl.exp(score - new_m), 0.0)
        v_tile = tl.load(
            v_ptr + (k_abs[:, None] * HK + kv_head) * D + d[None, :],
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = new_m

    result = tl.where(l_i > 0.0, acc / l_i, 0.0)
    tl.store(out_ptr + (q_idx * H + head) * D + d, result)


class Model:
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, softmax_scale,
                is_causal, exact_streaming):
        total_q, heads, depth = q.shape
        total_k, kv_heads, _ = k.shape
        batch = cu_seqlens_q.shape[0] - 1
        blocks = base_blockmask.shape[1]
        rows = base_blockmask.shape[2]
        columns = base_blockmask.shape[3]
        layout = blockmask_access_plan(heads, blocks)
        dense_heads = next(head for head, _rank, should_load in layout if should_load)
        scale = float(softmax_scale) if softmax_scale is not None else depth ** -0.5
        output = torch.empty_like(q)
        block_sparse_attention_kernel[(total_q * heads,)](
            q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
            streaming_info, base_blockmask, output,
            total_q=total_q, total_k=total_k, H=heads, HK=kv_heads, D=depth,
            B=batch, NBLK=blocks, NROW=rows, NCOL=columns,
            SCALE=scale, CAUSAL=bool(is_causal), EXACT=bool(exact_streaming),
            MAX_K=columns * 128, BLOCK_K=128, DENSE_HEADS=dense_heads,
        )
        return output

    __call__ = forward
