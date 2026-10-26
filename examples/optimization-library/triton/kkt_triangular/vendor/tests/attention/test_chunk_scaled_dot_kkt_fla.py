# Copyright © 2026 Huawei Technologies Co., Ltd.

import torch
import pytest

from triton_ascend_kernels.attention.fla import chunk_scaled_dot_kkt_fla_fwd


def pytorch_chunk_scaled_dot_kkt(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Compute beta * tril(K @ K^T) * exp(g_i - g_j) per chunk.

    Args:
        k: [B, H, T, K]
        beta: [B, T, H]
        g: [B, T, H]
    Returns:
        A: [B, T, H, BT]
    """
    B, H, T, K = k.shape
    BT = chunk_size
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    beta_h = beta.transpose(1, 2).to(torch.float32)
    g_h = g.transpose(1, 2).to(torch.float32)
    k_f = k.to(torch.float32)

    for b in range(B):
        for h in range(H):
            for i_t in range((T + BT - 1) // BT):
                start = i_t * BT
                end = min(start + BT, T)
                length = end - start
                k_chunk = k_f[b, h, start:end]
                beta_chunk = beta_h[b, h, start:end]
                g_chunk = g_h[b, h, start:end]

                dot = k_chunk @ k_chunk.T
                rows = torch.arange(length, device=k.device)
                tril = (rows[:, None] > rows[None, :]).to(torch.float32)
                g_diff = (g_chunk[:, None] - g_chunk[None, :]).clamp(-50.0, 50.0)
                a = dot * tril * torch.exp(g_diff) * beta_chunk[:, None]
                A[b, start:end, h, :length] = a
    return A


@pytest.mark.parametrize(
    "batch, seq_len, head_num, head_dim, chunk_size",
    [(1, 64, head_num, head_dim, 64) for head_num in [4, 8] for head_dim in [64, 128]],
)
def test_chunk_scaled_dot_kkt_fla(batch, seq_len, head_num, head_dim, chunk_size):
    # Validate the float32 single-chunk path that matches current kernel behavior.
    device = "npu"
    dtype = torch.float32
    torch.manual_seed(42)
    k = torch.randn(batch, head_num, seq_len, head_dim, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, head_num, device=device, dtype=dtype).sigmoid()
    g = torch.randn(batch, seq_len, head_num, device=device, dtype=torch.float32)

    triton_out = chunk_scaled_dot_kkt_fla_fwd(
        k=k.clone(),
        g=g.clone(),
        beta=beta.clone(),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    pytorch_out = pytorch_chunk_scaled_dot_kkt(
        k=k, beta=beta, g=g, chunk_size=chunk_size
    )

    torch.testing.assert_close(triton_out, pytorch_out, atol=0.01, rtol=0.01)
