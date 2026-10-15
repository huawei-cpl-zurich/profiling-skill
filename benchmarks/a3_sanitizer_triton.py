"""Deliberate overlapping/disjoint program writes; sanitizer qualification only."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def a3_sanitizer_write(Y, OVERLAP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    destination = offsets if OVERLAP else pid * BLOCK + offsets
    tl.store(Y + destination, tl.full((BLOCK,), 1, tl.int32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mutant', action='store_true')
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=False)
    y = torch.zeros(8192, dtype=torch.int32, device='npu')
    compiled = a3_sanitizer_write[(2,)](y, args.mutant, 4096)
    torch.npu.synchronize()
    artifacts = []
    for name, value in compiled.asm.items():
        if isinstance(value, str):
            destination = args.artifacts / (name + '.txt')
            destination.write_text(value)
            artifacts.append(dict(stage=name, sha256=hashlib.sha256(value.encode()).hexdigest()))
    expected = torch.ones_like(y)
    if args.mutant:
        expected[4096:] = 0
    correct = torch.equal(y.cpu(), expected.cpu())
    print(json.dumps(dict(correctness=correct, mutant=args.mutant, artifacts=artifacts)))
    return int(not correct)


if __name__ == '__main__':
    raise SystemExit(main())
