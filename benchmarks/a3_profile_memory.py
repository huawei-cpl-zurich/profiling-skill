#!/usr/bin/env python3
"""Read-only memory controls; logical traffic is not measured physical traffic."""
import argparse
import json
from pathlib import Path
import random


def cases():
    result = []
    for n in (4096, 1048576):
        for stride in (1, 2, 4, 8, 16, 32, 64):
            result.append(dict(id=f'strided/{stride}/{n}', kind='strided', stride=stride, n=n))
        for pattern in ('contiguous', 'local32', 'global', 'hotset64'):
            result.append(dict(id=f'irregular/{pattern}/{n}', kind='irregular', pattern=pattern, n=n))
    return result


def indices(case):
    n = case['n']
    if case['kind'] == 'strided':
        return range(0, n * case['stride'], case['stride'])
    rng = random.Random(20261015)
    result = list(range(n))
    if case['pattern'] == 'global':
        rng.shuffle(result)
    elif case['pattern'] == 'local32':
        for start in range(0, n, 32):
            group = result[start:start + 32]
            rng.shuffle(group)
            result[start:start + 32] = group
    elif case['pattern'] == 'hotset64':
        result = [rng.randrange(64) for _ in range(n)]
    elif case['pattern'] != 'contiguous':
        raise ValueError('unknown access pattern')
    return result


def footprint(case, addresses):
    allocation = case['n'] * case.get('stride', 1)
    if case.get('pattern') == 'hotset64':
        allocation = 64
    return dict(input_allocation_bytes=allocation * 4,
                accessed_span_bytes=(max(addresses) - min(addresses) + 1) * 4,
                distinct_input_bytes=len(set(addresses)) * 4,
                logical_input_read_bytes=case['n'] * 4,
                output_write_bytes=case['n'] * 4,
                index_read_bytes=case['n'] * 4 if case['kind'] == 'irregular' else 0,
                physical_traffic='unmeasured')


def run(case, artifact_root):
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl
    from compile_evidence import retain_compiler_evidence

    @triton.jit
    def a3_memory_probe(x, index, out, n: tl.constexpr, stride: tl.constexpr,
                        indirect: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        address = tl.load(index + offset, offset < n, 0) if indirect else offset * stride
        value = tl.load(x + address, offset < n, 0)
        tl.store(out + offset, value, offset < n)

    address = indices(case)
    traffic = footprint(case, address)
    # CPU-generated bounded exactly representable values; no initialization kernels.
    source = torch.arange(traffic['input_allocation_bytes'] // 4, dtype=torch.int32)
    source = (source % 1024).to(torch.float32)
    index = torch.tensor(address, dtype=torch.int32)
    expected = source[index.long()]
    x, idx = source.npu(), index.npu()
    out = torch.empty(case['n'], dtype=torch.float32, device='npu')
    compiled = a3_memory_probe[(triton.cdiv(case['n'], 256),)](
        x, idx, out, case['n'], case.get('stride', 1), case['kind'] == 'irregular', 256)
    torch.npu.synchronize()
    correct = torch.equal(out.cpu(), expected)
    artifacts = retain_compiler_evidence(compiled.asm, artifact_root, case['id'], 'probe', 1)
    return dict(case=case, correctness=correct, footprint=traffic, compiler=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[c['id'] for c in cases()])
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--artifacts', default='compiler-evidence')
    args = parser.parse_args()
    if args.manifest:
        print(json.dumps(cases()))
        return 0
    if not args.case:
        parser.error('--case is required unless --manifest')
    result = run(next(c for c in cases() if c['id'] == args.case), Path(args.artifacts))
    print(json.dumps(result), flush=True)
    return 0 if result['correctness'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
