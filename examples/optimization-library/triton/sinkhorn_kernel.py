"""MindSpeedOps arch32 forward, pinned 4efe2b675c2d51517c3fdd14c721020b8d41e241.
Apache-2.0, see SINKHORN_LICENSE. GROUP=32 is a bounded source adaptation.
Common sigmoid expansion preserves source formula on namespaces lacking sigmoid.
Width4 changes only live comb width; stride8 and all padding writes remain.
"""
import triton
import triton.language as tl

@triton.jit
def hc_split_sinkhorn_forward_prepost_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    batch_seq_size,
    hc_mult,
    eps,
    feat_dim,
    BLOCK_HC: tl.constexpr,
    GROUP: tl.constexpr,
):
    # program handles GROUP batch_seq entries
    pid0 = tl.program_id(0) * GROUP
    pids = pid0 + tl.arange(0, GROUP)  # (G,)
    pid_mask = pids < batch_seq_size  # (G,)

    # scales
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)

    # base pre/post (loaded once per program)
    ar4 = tl.arange(0, BLOCK_HC)  # (4,)
    base_pre = tl.load(hc_base_ptr + ar4)  # (4,)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)  # (4,)

    # offsets for each pid
    pid_feat_off = pids[:, None] * feat_dim  # (G,1)
    pid_hc_off = pids[:, None] * hc_mult  # (G,1)

    # mixes_pre/post: shape (G,4)
    mixes_pre = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    mixes_post = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)

    # compute
    pre = (1.0 / (1.0 + tl.exp(-(mixes_pre * scale_pre + base_pre[None, :])))) + eps
    post = 2.0 * (1.0 / (1.0 + tl.exp(-(mixes_post * scale_post + base_post[None, :]))))

    # store
    tl.store(pre_ptr + pid_hc_off + ar4[None, :], pre, mask=pid_mask[:, None])
    tl.store(post_ptr + pid_hc_off + ar4[None, :], post, mask=pid_mask[:, None])

@triton.jit
def hc_split_sinkhorn_forward_comb_kernel(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    comb_ptr,
    batch_seq_size,
    hc_mult,
    eps,
    feat_dim,
    BLOCK_HC: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr,
    GROUP: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    # program handles GROUP batch_seq entries
    pid0 = tl.program_id(0) * GROUP
    pids = pid0 + tl.arange(0, GROUP)  # (G,)
    pid_mask = pids < batch_seq_size  # (G,)

    # scale comb
    scale_comb = tl.load(hc_scale_ptr + 2).to(tl.float32)

    # comb base: 4x8 padded
    r = tl.arange(0, BLOCK_HC)[:, None]  # (4,1)
    c = tl.arange(0, BLOCK_ALIGN)[None, :]  # (1,8)
    col_mask = c < BLOCK_HC
    idx2d = r * BLOCK_HC + c  # (4,8)

    base_comb = tl.load(hc_base_ptr + (2 * hc_mult) + idx2d, mask=col_mask, other=0.0).to(tl.float32)  # (4,8)

    # offsets for each pid
    pid_feat_off = pids[:, None] * feat_dim  # (G,1)
    pid_comb_off = pids[:, None] * 32  # (G,1)

    # mixes comb load: shape (G,4,8)
    mixes_comb = tl.load(
        mixes_ptr + pid_feat_off[:, :, None] + (2 * hc_mult) + idx2d[None, :, :],
        mask=pid_mask[:, None, None] & col_mask[None, :, :],
        other=0.0,
    ).to(tl.float32)  # (G,4,8)

    # comb logits: (G,4,8)
    comb = mixes_comb * scale_comb + base_comb[None, :, :]

    # softmax over last dim
    very_neg = -1.0e20
    comb_for_max = tl.where(col_mask[None, :, :], comb, very_neg)
    row_max = tl.max(comb_for_max, axis=2)  # (G,4)

    comb = tl.exp(comb - row_max[:, :, None])
    comb = tl.where(col_mask[None, :, :], comb, 0.0)

    # softmax normalization
    row_sum0 = tl.sum(comb, axis=2)  # (G,4)
    comb = comb / row_sum0[:, :, None]

    # entry-wise eps smoothing
    comb = comb + eps
    comb = tl.where(col_mask[None, :, :], comb, 0.0)

    # 2) initial col normalize once
    col_sum0 = tl.sum(comb, axis=1)  # (G,8)
    comb = comb / (col_sum0[:, None, :] + eps)

    for _ in tl.static_range(0, SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb, axis=2)  # (G,4)
        comb = comb / (row_sum[:, :, None] + eps)

        col_sum = tl.sum(comb, axis=1)  # (G,8)
        comb = comb / (col_sum[:, None, :] + eps)

    # clear padding at end
    comb = tl.where(col_mask[None, :, :], comb, 0.0)

    if BLOCK_ALIGN == 8:
        lin = tl.arange(0, 32)
        comb_flat = tl.reshape(comb, (GROUP, 32))
        tl.store(comb_ptr + pid_comb_off + lin[None, :], comb_flat, mask=pid_mask[:, None])
    else:
        offsets = pids[:, None, None] * 32 + r[None, :, :] * 8 + c[None, :, :]
        tl.store(comb_ptr + offsets, comb, mask=pid_mask[:, None, None])
        tl.store(comb_ptr + offsets + 4, 0.0, mask=pid_mask[:, None, None])
