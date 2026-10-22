#!/usr/bin/env python3
"""Small, numerically checked explicit-simulator controls; no profiling setup."""
import argparse
import json

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def a3_timeline_add(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets)
    tl.store(out + offsets, value + 1.0)


@triton.jit
def a3_timeline_phase(x, out, BLOCK: tl.constexpr, ROUNDS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets, volatile=True)
    for _ in range(ROUNDS):
        value = value * 1.00001 + 0.01
    tl.store(out + offsets, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("add", "phase"), required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    # CPU-created inputs avoid unrelated initialization kernels in the model.
    source = torch.arange(128, dtype=torch.float32)
    x = source.npu()
    output = torch.empty_like(x)
    if args.control == "add":
        a3_timeline_add[(1,)](x, output, BLOCK=128)
        expected = source + 1.0
    else:
        a3_timeline_phase[(1,)](x, output, BLOCK=128, ROUNDS=32, enable_fp_fusion=False)
        expected = source
        for _ in range(32):
            expected = expected * 1.00001 + 0.01
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
    print(json.dumps({"control": args.control, "correctness": True}), flush=True)


if __name__ == "__main__":
    main()
