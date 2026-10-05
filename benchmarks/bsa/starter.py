"""Treatment-neutral, full-domain Triton forward reference for packed BSA."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bsa_reference_fwd(
    Q, K, V, O, HMT, SINFO, RANK, BMASK,
    QS, QE, KS, KE,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    SCALE: tl.constexpr, CAUSAL: tl.constexpr, EXACT: tl.constexpr,
    QST: tl.constexpr, QSH: tl.constexpr, QSD: tl.constexpr,
    KST: tl.constexpr, KSH: tl.constexpr, KSD: tl.constexpr,
    VST: tl.constexpr, VSH: tl.constexpr, VSD: tl.constexpr,
    OST: tl.constexpr, OSH: tl.constexpr, OSD: tl.constexpr,
    MBS: tl.constexpr, MBH: tl.constexpr, MBR: tl.constexpr, MBC: tl.constexpr,
    SEQ: tl.constexpr, NROW: tl.constexpr, NCOL: tl.constexpr,
    K_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 32,
    BLOCK_D: tl.constexpr = 128,
):
    q_len = QE - QS
    k_len = KE - KS
    head = tl.program_id(1)
    kv_head = head // (H // HK)
    mask_type = tl.load(HMT + head)
    mask_rank = tl.load(RANK + head)
    sink = tl.load(SINFO + 2 * head)
    local = tl.load(SINFO + 2 * head + 1)

    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    dim = tl.arange(0, BLOCK_D)
    q = tl.load(Q + (QS + row[:, None]) * QST + head * QSH
                + dim[None, :] * QSD,
                (row[:, None] < q_len) & (dim[None, :] < D), other=0)
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0, tl.float32)
    acc = tl.full((BLOCK_M, BLOCK_D), 0, tl.float32)

    for start in range(0, tl.cdiv(K_LEN, BLOCK_N)):
        key = start * BLOCK_N + col
        k = tl.load(K + (KS + key[:, None]) * KST + kv_head * KSH
                    + dim[None, :] * KSD,
                    (key[:, None] < k_len) & (dim[None, :] < D), other=0)
        score = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))) * SCALE
        keep = (row[:, None] < q_len) & (key[None, :] < k_len)
        if CAUSAL:
            keep = keep & (key[None, :] <= row[:, None] + k_len - q_len)
        if mask_type == 1:
            mask_row = row // 128
            mask_col = key // 128
            block = tl.load(BMASK + SEQ * MBS + mask_rank * MBH
                            + mask_row[:, None] * MBR + mask_col[None, :] * MBC,
                            (row[:, None] < q_len) & (key[None, :] < k_len)
                            & (mask_row[:, None] < NROW)
                            & (mask_col[None, :] < NCOL), other=0)
            keep = keep & block
        elif mask_type == -1:
            if EXACT:
                diagonal = row[:, None] + k_len - q_len
                keep = keep & (key[None, :] <= tl.minimum(diagonal, k_len))
                keep = keep & ~((key[None, :] < diagonal - (local - 1))
                                & (key[None, :] >= sink))
            else:
                row_block = row // 128
                key_block = key // 128
                if CAUSAL:
                    first_row = tl.maximum((q_len - k_len) // 128, 0)
                    right = tl.cdiv(tl.maximum(k_len - q_len, 0), 128) + 1 \
                        + row_block - first_row
                    valid_row = row_block >= first_row
                else:
                    right = tl.full((BLOCK_M,), tl.cdiv(k_len, 128), tl.int32)
                    valid_row = tl.full((BLOCK_M,), True, tl.int1)
                window = (key_block[None, :] >= tl.maximum(right[:, None] - local, 0)) \
                    & (key_block[None, :] < tl.minimum(right[:, None], tl.cdiv(k_len, 128)))
                keep = keep & valid_row[:, None] \
                    & (window | (key_block[None, :] < sink))

        score = tl.where(keep, score, -float("inf"))
        tile_max = tl.max(score, 1)
        new_m = tl.maximum(m, tile_max)
        safe_m = tl.where(new_m == -float("inf"), 0, new_m)
        alpha = tl.where(l > 0, tl.exp(m - safe_m), 0)
        p = tl.exp(score - safe_m[:, None])
        new_l = l * alpha + tl.sum(p, 1)
        v = tl.load(V + (KS + key[:, None]) * VST + kv_head * VSH
                    + dim[None, :] * VSD,
                    (key[:, None] < k_len) & (dim[None, :] < D), other=0)
        # fp16 dot inputs retain the fp32 online normalization and accumulator.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m = new_m
        l = new_l

    out = acc / tl.where(l[:, None] > 0, l[:, None], 1)
    tl.store(O + (QS + row[:, None]) * OST + head * OSH
             + dim[None, :] * OSD, out,
             (row[:, None] < q_len) & (dim[None, :] < D))


class Model(nn.Module):
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                head_mask_type, streaming_info, base_blockmask,
                softmax_scale, is_causal, exact_streaming):
        total_q, h, d = q.shape
        total_k, hk, _ = k.shape
        assert v.shape == k.shape and q.dtype == k.dtype == v.dtype
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert h % hk == 0 and d <= 128
        batch = cu_seqlens_q.numel() - 1
        assert cu_seqlens_k.numel() == batch + 1
        assert head_mask_type.shape == (h,)
        assert streaming_info.shape == (2 * h,)
        assert not exact_streaming or is_causal

        # Small metadata is read once, including packed boundaries and sparse ranks.
        meta = torch.cat((cu_seqlens_q, cu_seqlens_k, head_mask_type)).tolist()
        cuq = meta[:batch + 1]
        cuk = meta[batch + 1:2 * batch + 2]
        types = meta[2 * batch + 2:]
        assert cuq[-1] == total_q and cuk[-1] == total_k
        ranks = []
        sparse_count = 0
        for kind in types:
            ranks.append(sparse_count)
            sparse_count += kind == 1
        assert base_blockmask.shape == (
            batch, sparse_count,
            max(triton.cdiv(cuq[b + 1] - cuq[b], 128) for b in range(batch)),
            max(triton.cdiv(cuk[b + 1] - cuk[b], 128) for b in range(batch)),
        )
        rank_tensor = torch.tensor(ranks, dtype=torch.int32, device=q.device)
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        scale = softmax_scale if softmax_scale is not None else d ** -0.5
        for b in range(batch):
            qs, qe = cuq[b:b + 2]
            ks, ke = cuk[b:b + 2]
            if qe == qs:
                continue
            _bsa_reference_fwd[(triton.cdiv(qe - qs, 16), h)](
                q, k, v, out, head_mask_type, streaming_info, rank_tensor,
                base_blockmask, qs, qe, ks, ke, h, hk, d, scale,
                is_causal, exact_streaming,
                *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                *base_blockmask.stride(), b,
                base_blockmask.shape[2], base_blockmask.shape[3], ke - ks,
                num_warps=4,
            )
        return out
