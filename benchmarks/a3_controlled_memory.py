#!/usr/bin/env python3
"""Controlled read sweeps: useful bytes, allocation and address span are explicit."""
import argparse
import json
from pathlib import Path
import random

STRIDES = (1, 2, 4, 8, 16, 32, 64)


def cases(n=1048576):
    if n <= 0:
        raise ValueError('size must be positive')
    result = []
    for family in ('count', 'span'):
        for stride in STRIDES:
            result.append(dict(id=f'{family}/{stride}/{n}', kind='direct', stride=stride,
                               n=n if family == 'count' else (n + stride - 1) // stride,
                               allocation_elements=64 * n if family == 'count' else n))
    for pattern in ('contiguous', 'local32', 'global'):
        result.append(dict(id=f'indexed/{pattern}/{n}', kind='indexed', pattern=pattern,
                           n=n, stride=1, allocation_elements=64 * n))
    return result


def indices(case):
    if case['kind'] == 'direct':
        return range(0, case['n'] * case['stride'], case['stride'])
    result = list(range(case['n']))
    rng = random.Random(20261017)
    if case['pattern'] == 'global':
        rng.shuffle(result)
    elif case['pattern'] == 'local32':
        for start in range(0, case['n'], 32):
            part = result[start:start + 32]
            rng.shuffle(part)
            result[start:start + 32] = part
    elif case['pattern'] != 'contiguous':
        raise ValueError('unknown index pattern')
    return result


def footprint(case):
    # Every pattern is a permutation, or a unique arithmetic progression.
    span = ((case['n'] - 1) * case['stride'] + 1) * 4
    return dict(useful_input_bytes=case['n'] * 4, index_bytes=case['n'] * 4
                if case['kind'] == 'indexed' else 0, output_bytes=case['n'] * 4,
                allocation_bytes=case['allocation_elements'] * 4, accessed_span_bytes=span,
                distinct_input_bytes=case['n'] * 4, grid=[(case['n'] + 255) // 256],
                block=256, physical_traffic='unmeasured')


def run(case, artifact_root):
    global tl
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl
    from a3_profile_memory import triton_distributions
    from compile_evidence import retain_compiler_evidence

    @triton.jit
    def a3_memory_probe(x, index, out, n: tl.constexpr, stride: tl.constexpr,
                        indirect: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        address = tl.load(index + offset, offset < n, 0) if indirect else offset * stride
        value = tl.load(x + address, offset < n, 0)
        tl.store(out + offset, value, offset < n)

    # CPU initialization/reference; no auxiliary device kernels contaminate capture.
    source = (torch.arange(case['allocation_elements'], dtype=torch.int32) % 1024).float()
    index = torch.tensor(indices(case), dtype=torch.int32)
    expected = source[index.long()]
    x, idx = source.npu(), index.npu()
    out = torch.empty(case['n'], device='npu', dtype=torch.float32)
    compiled = a3_memory_probe[tuple(footprint(case)['grid'])](
        x, idx, out, case['n'], case['stride'], case['kind'] == 'indexed', 256)
    torch.npu.synchronize()
    correct = torch.equal(out.cpu(), expected)
    return dict(case=case, footprint=footprint(case), correctness=correct,
                compiler=retain_compiler_evidence(compiled.asm, artifact_root, case['id'], 'probe', 1),
                environment=dict(torch=torch.__version__, triton_module=triton.__version__,
                                 triton_distributions=triton_distributions(),
                                 device_name=torch.npu.get_device_name(0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1048576)
    parser.add_argument('--case')
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--artifacts', type=Path, default=Path('compiler-evidence'))
    args = parser.parse_args()
    matrix = cases(args.size)
    if args.manifest:
        print(json.dumps([dict(case=c, footprint=footprint(c)) for c in matrix]))
        return 0
    selected = [c for c in matrix if c['id'] == args.case]
    if len(selected) != 1:
        parser.error('unknown --case for selected --size')
    result = run(selected[0], args.artifacts)
    print(json.dumps(result), flush=True)
    return 0 if result['correctness'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
