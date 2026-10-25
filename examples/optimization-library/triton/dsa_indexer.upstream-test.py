# Copyright © 2026 Huawei Technologies Co., Ltd.

import torch
import pytest

import numpy as np
from triton_ascend_kernels.attention import dsa_lightning_indexer_forward


def forward_torch(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
):
    b, m, h, d = q.shape
    n = k_cache.shape[1]
    index_score = torch.zeros((b, m, n), dtype=torch.float32, device=q.device)

    for batch_id in range(b):
        for m_id in range(m):
            q_slice = q[batch_id, m_id].to(torch.float32)
            k_slice = k_cache[batch_id].to(torch.float32)
            relu_out = torch.maximum(
                torch.matmul(
                    q_slice.to(torch.float32), k_slice.to(torch.float32).transpose(0, 1)
                ),
                torch.tensor(0),
            )
            weight_out = relu_out * weights[batch_id, m_id].unsqueeze(-1)
            reduce_out = torch.sum(weight_out, dim=0)
            index_score[batch_id, m_id] = reduce_out
    return index_score


@pytest.mark.parametrize(
    "batch, q_seq, k_seq, q_headnum, dim, dtype",
    [
        (
            batch,
            q_seq,
            k_seq,
            64,
            128,
            dtype,
        )
        for batch in [1, 2, 4, 5]
        for q_seq, k_seq in [[1, 1024], [1024, 1024], [1, 4096], [4096, 4096]]
        for dtype in [torch.bfloat16, torch.float16]
    ],
)
def test_indexer_and_dump(batch, q_seq, k_seq, q_headnum, dim, dtype):
    device = "npu"
    query = torch.tensor(
        np.random.uniform(-10, 10, (batch, q_seq, q_headnum, dim)),
        device=device,
        dtype=dtype,
    )
    weights = torch.tensor(
        np.random.uniform(-1, 1, (batch, q_seq, q_headnum)),
        device=device,
        dtype=torch.float32,
    )
    k_cache = torch.tensor(
        np.random.uniform(-10, 10, (batch, k_seq, dim)),
        device=device,
        dtype=dtype,
    )

    pytorch_out = forward_torch(query, weights, k_cache)
    triton_out = dsa_lightning_indexer_forward(query, weights, k_cache)

    torch.testing.assert_close(triton_out, pytorch_out, atol=0.01, rtol=0.01)
