"""Model adapter for the frozen streaming matmul-add calibration kernel."""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


_SOURCE = Path(__file__).parents[1] / "streaming_matmul_add.py"
_SPEC = importlib.util.spec_from_file_location("frozen_streaming_matmul_add", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen calibration kernel from {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Model(nn.Module):
    def forward(self, lhs, rhs, bias):
        m, k = lhs.shape
        _, n = rhs.shape
        output = torch.empty((m, n), device=lhs.device, dtype=torch.float32)
        block_m = block_n = block_k = 32
        grid = (_MODULE.triton.cdiv(m, block_m), _MODULE.triton.cdiv(n, block_n))
        _MODULE.streaming_matmul_add_kernel[grid](
            lhs, rhs, bias, output, m, n, k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return output
