"""Deterministic torch reference for the streaming matmul-add control."""

import json
import os

import torch
import torch.nn as nn


SEED = 20260925


class Model(nn.Module):
    def forward(self, lhs, rhs, bias):
        return lhs.float().matmul(rhs.float()) + bias.float()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def get_input_groups():
    groups = []
    for case_index, case in enumerate(_load_cases()):
        generator = torch.Generator()
        generator.manual_seed(SEED + case_index)
        lhs = torch.randn((case["m"], case["k"]), generator=generator,
                          dtype=torch.float16)
        rhs = torch.randn((case["k"], case["n"]), generator=generator,
                          dtype=torch.float16)
        bias = torch.randn((case["n"],), generator=generator,
                           dtype=torch.float16)
        groups.append([lhs, rhs, bias])
    return groups
