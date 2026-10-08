#!/usr/bin/env python3
"""Small correctness-checked kernels for compiler and profiler attribution."""

from __future__ import annotations

import argparse
import json

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


SEED = 20261008
CASES = (
    {
        "name": "copy",
        "constructs": ["triton_load", "triton_store"],
        "kernel_name": "pipe_probe_copy",
        "expected_status": "unknown",
    },
    {
        "name": "elementwise",
        "constructs": ["triton_load", "triton_elementwise", "triton_store"],
        "kernel_name": "pipe_probe_elementwise",
        "expected_status": "unknown",
    },
    {
        "name": "dot",
        "constructs": ["triton_load", "triton_dot", "triton_store"],
        "kernel_name": "pipe_probe_dot",
        "expected_status": "unknown",
    },
)


if triton is not None:
    @triton.jit
    def pipe_probe_copy(source, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        values = tl.load(source + offsets, mask=offsets < n)
        tl.store(output + offsets, values, mask=offsets < n)

    @triton.jit
    def pipe_probe_elementwise(source, output, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        values = tl.load(source + offsets, mask=offsets < n)
        tl.store(output + offsets, values * 1.5 + 2.0, mask=offsets < n)

    @triton.jit
    def pipe_probe_dot(lhs, rhs, output, block: tl.constexpr):
        rows = tl.arange(0, block)[:, None]
        cols = tl.arange(0, block)[None, :]
        reduction = tl.arange(0, block)
        a = tl.load(lhs + rows * block + reduction[None, :])
        b = tl.load(rhs + reduction[:, None] * block + cols)
        tl.store(output + rows * block + cols, tl.dot(a, b))
else:
    pipe_probe_copy = pipe_probe_elementwise = pipe_probe_dot = None


def contract() -> dict:
    return {"schema_version": 1, "seed": SEED, "cases": list(CASES)}


def run_case(name: str) -> dict:
    import torch
    import torch_npu  # noqa: F401

    if triton is None:
        raise RuntimeError("triton is not installed")

    torch.manual_seed(SEED)
    if name in {"copy", "elementwise"}:
        source = torch.randn((256,), dtype=torch.float32, device="npu")
        output = torch.empty_like(source)
        kernel = pipe_probe_copy if name == "copy" else pipe_probe_elementwise
        kernel[(1,)](source, output, n=256, block=256)
        expected = source if name == "copy" else source * 1.5 + 2.0
    elif name == "dot":
        lhs = torch.randn((32, 32), dtype=torch.float16, device="npu")
        rhs = torch.randn((32, 32), dtype=torch.float16, device="npu")
        output = torch.empty((32, 32), dtype=torch.float32, device="npu")
        pipe_probe_dot[(1,)](lhs, rhs, output, block=32)
        expected = lhs.float() @ rhs.float()
    else:
        raise ValueError(f"unknown case {name!r}")
    torch.npu.synchronize()
    passed = torch.allclose(output, expected, rtol=2e-2, atol=2e-2)
    selected = next(case for case in CASES if case["name"] == name)
    return {
        "status": "success" if passed else "failure",
        "case": selected,
        "correct": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--case", choices=tuple(case["name"] for case in CASES))
    args = parser.parse_args()
    if args.contract == (args.case is not None):
        parser.error("choose exactly one of --contract or --case")
    payload = contract() if args.contract else run_case(args.case)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status", "success") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
