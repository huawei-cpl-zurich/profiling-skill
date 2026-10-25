"""Launch fusion of unchanged pinned MindSpeedOps 8-lane kernel bodies.
Apache-2.0; attribution and source pin are in SINKHORN_LICENSE.
"""
import triton
import triton.language as tl
from sinkhorn_kernel import (hc_split_sinkhorn_forward_prepost_kernel as prepost,
                             hc_split_sinkhorn_forward_comb_kernel as comb)

@triton.jit
def hc_sinkhorn_forward_fused_kernel(mixes, scale, base, pre, post, backing,
        batch_seq_size, hc_mult, eps, feat_dim, BLOCK_HC: tl.constexpr,
        BLOCK_ALIGN: tl.constexpr, GROUP: tl.constexpr, SINKHORN_ITERS: tl.constexpr):
    # Complete the original pre/post body including stores before entering comb.
    prepost(mixes, scale, base, pre, post, batch_seq_size, hc_mult, eps,
            feat_dim, BLOCK_HC, GROUP)
    comb(mixes, scale, base, backing, batch_seq_size, hc_mult, eps,
         feat_dim, BLOCK_HC, BLOCK_ALIGN, GROUP, SINKHORN_ITERS)
