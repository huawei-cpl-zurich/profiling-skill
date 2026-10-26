# Copyright © 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 sgl-project
# Based on vLLM: https://github.com/vllm-project/vllm
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import triton
import triton.language as tl

@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
