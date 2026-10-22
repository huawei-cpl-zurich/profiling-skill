#!/usr/bin/env python3
"""Correctness-checked A3 copy, gather and matrix controls; timing is external."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def describe(case, variant, size, matrix_size):
    if case == 'matmul':
        return dict(case=case, variant=variant, shape=[matrix_size] * 3,
                    dtype='float16', output_dtype='float32', tile=[32, 32, 64],
                    requested_num_stages=1 if variant == 'baseline' else 2,
                    logical_minimum_bytes=8 * matrix_size**2,
                    useful_flops=2 * matrix_size**3,
                    actual_buffering='verify emitted compiler artifacts')
    stride = 2 if case == 'copy' and variant == 'candidate' else 1
    indirect = case == 'gather' and variant == 'candidate'
    return dict(case=case, variant=variant, elements=size, stride=stride,
                indirect=indirect, dtype='float32', block=256,
                logical_input_bytes=4 * size, logical_output_bytes=4 * size,
                logical_index_bytes=4 * size if indirect else 0,
                input_allocation_bytes=8 * size,
                accessed_input_span_bytes=((size - 1) * stride + 1) * 4,
                physical_traffic='unmeasured')


def retain(compiled, directory):
    if directory is None:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, value in compiled.asm.items():
        path = directory / ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)
        content = value if isinstance(value, bytes) else str(value).encode()
        path.write_bytes(content)
        result[name] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest())
    return result


def run(args):
    global tl
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def paired_memory(x, indices, out, N: tl.constexpr, STRIDE: tl.constexpr,
                      INDIRECT: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        address = tl.load(indices + offset, offset < N, 0) if INDIRECT else offset * STRIDE
        value = tl.load(x + address, offset < N, 0)
        tl.store(out + offset, value, offset < N)

    @triton.jit
    def paired_matmul(a, b, out, N: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        reduction = tl.arange(0, BLOCK_K)
        acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
        for k in range(0, tl.cdiv(N, BLOCK_K)):
            kk = k * BLOCK_K + reduction
            av = tl.load(a + rows[:, None] * N + kk[None, :],
                         (rows[:, None] < N) & (kk[None, :] < N), 0)
            bv = tl.load(b + kk[:, None] * N + cols[None, :],
                         (kk[:, None] < N) & (cols[None, :] < N), 0)
            acc += tl.dot(av, bv)
        tl.store(out + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < N) & (cols[None, :] < N))

    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    config = describe(args.case, args.variant, args.size, args.matrix_size)
    if args.case == 'matmul':
        n = args.matrix_size
        # CPU initialization avoids unrelated device initialization kernels.
        a_cpu = ((torch.arange(n * n).reshape(n, n) % 13) - 6).half() / 8
        b_cpu = ((torch.arange(n * n).reshape(n, n) % 11) - 5).half() / 8
        expected = a_cpu.float() @ b_cpu.float()
        a, b = a_cpu.to(device), b_cpu.to(device)
        output = torch.empty((n, n), dtype=torch.float32, device=device)
        def launch():
            return paired_matmul[(triton.cdiv(n, 32), triton.cdiv(n, 32))](
                a, b, output, n, 32, 64, num_stages=config['requested_num_stages'])
    else:
        n = args.size
        source = (torch.arange(2 * n, dtype=torch.int32) % 1024).float()
        index = torch.arange(n, dtype=torch.int32)
        expected = source[index.long() * config['stride']]
        x, indices = source.to(device), index.to(device)
        output = torch.empty(n, dtype=torch.float32, device=device)
        def launch():
            return paired_memory[(triton.cdiv(n, 256),)](
                x, indices, output, n, config['stride'], config['indirect'], 256)
    compiled = None
    for _ in range(args.iterations):
        compiled = launch()
    torch.npu.synchronize()
    actual = output.cpu()
    correct = (torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
               if args.case == 'matmul' else torch.equal(actual, expected))
    versions = {}
    for name in ('torch', 'torch-npu', 'triton', 'triton-ascend'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(schema_version=1, correctness=bool(correct), config=config,
                iterations=args.iterations, device_name=torch.npu.get_device_name(args.device),
                versions=versions, compiler_artifacts=retain(compiled, args.artifacts),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selector='discover exact exported name with BasicInfo')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--case', choices=('copy', 'gather', 'matmul'))
    parser.add_argument('--variant', choices=('baseline', 'candidate'), default='baseline')
    parser.add_argument('--size', type=int, default=1048576)
    parser.add_argument('--matrix-size', type=int, default=512)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--artifacts', type=Path)
    args = parser.parse_args()
    if min(args.size, args.matrix_size, args.iterations) <= 0 or args.device < 0:
        parser.error('sizes and iterations must be positive; device must be nonnegative')
    if args.manifest:
        print(json.dumps([describe(case, variant, args.size, args.matrix_size)
                          for case in ('copy', 'gather', 'matmul')
                          for variant in ('baseline', 'candidate')], sort_keys=True))
        return 0
    if args.case is None:
        parser.error('--case is required unless --manifest')
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['correctness'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
