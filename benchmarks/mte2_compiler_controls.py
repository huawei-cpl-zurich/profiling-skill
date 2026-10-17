"""Small correctness/capture controls; not a performance measurement harness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def kernels():
    # JIT source resolution uses module globals; keep device imports lazy while
    # avoiding closure-only aliases that differ between Triton versions.
    global triton, tl
    import triton
    import triton.language as tl
    @triton.jit
    def mask_control(x, out, N: tl.constexpr, PREDICATE: tl.constexpr,
                     BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        selected = i % 4 == 0
        if PREDICATE:
            value = tl.load(x + i, (i < N) & selected, other=0)
        else:
            value = tl.load(x + i, i < N, other=0)
            value = tl.where(selected, value, 0.)
        tl.store(out + i, value, i < N)

    @triton.jit
    def buffering_control(a, b, out, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for step in range(triton.cdiv(K, BK)):
            kk = step * BK + k
            av = tl.load(a + m[:, None] * K + kk[None, :],
                         (m[:, None] < M) & (kk[None, :] < K), other=0.)
            bv = tl.load(b + kk[:, None] * N + n[None, :],
                         (kk[:, None] < K) & (n[None, :] < N), other=0.)
            acc += tl.dot(av, bv)
        tl.store(out + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < N))
    return mask_control, buffering_control


def run(family, variant, output):
    import torch
    import torch_npu  # noqa: F401
    import triton
    mask, buffering = kernels()
    torch.manual_seed(0)
    device = 'npu:0'  # logical selected-device mapping belongs to remote runtime
    if family == 'mask':
        cpu_x = torch.randn(1 << 20, dtype=torch.float32)
        expected = torch.where(torch.arange(cpu_x.numel()) % 4 == 0, cpu_x, 0.)
        x = cpu_x.to(device)
        out = torch.empty_like(x)
        constants = {'N': x.numel(), 'PREDICATE': variant == 'after', 'BLOCK': 1024}
        options = {}
        compiled = mask[(triton.cdiv(x.numel(), 1024),)](x, out, **constants)
    else:
        cpu_a = torch.randn((128, 512), dtype=torch.float16)
        cpu_b = torch.randn((512, 128), dtype=torch.float16)
        expected = cpu_a.float() @ cpu_b.float()
        a, b = cpu_a.to(device), cpu_b.to(device)
        out = torch.empty((128, 128), device=device, dtype=torch.float32)
        constants = dict(M=128, N=128, K=512, BM=32, BN=64, BK=32)
        options = {'num_stages': 1 if variant == 'before' else 2}
        compiled = buffering[(4, 2)](a, b, out, **constants, **options)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected, atol=0.05, rtol=0.01)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for stage, value in compiled.asm.items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        path = output / (stage + ('.txt' if isinstance(value, str) else '.bin'))
        path.write_bytes(data)
        artifacts.append({'stage': stage, 'path': str(path),
                          'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    result = {'family': family, 'variant': variant, 'correctness': 'passed',
              'specialization': constants, 'options': options,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'target': str(triton.runtime.driver.active.get_current_target()),
              'compiler_metadata': str(compiled.metadata), 'triton_version': triton.__version__,
              'artifacts': artifacts,
              'limitation': 'num_stages is a compiler request, not proof of overlap; '
                            'collect timing separately without forced compilation.'}
    (output / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def manifest():
    return {'cases': [dict(id=family + '-' + variant, family=family, variant=variant,
                           specialization=({'N': 1 << 20, 'BLOCK': 1024,
                                             'PREDICATE': variant == 'after'} if family == 'mask'
                            else dict(M=128, N=128, K=512, BM=32, BN=64, BK=32)),
                           options=({} if family == 'mask' else
                                    {'num_stages': 1 if variant == 'before' else 2}))
                      for family in ('mask', 'buffering') for variant in ('before', 'after')]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--case', choices=[item['id'] for item in manifest()['cases']])
    parser.add_argument('--family', choices=('mask', 'buffering'))
    parser.add_argument('--variant', choices=('before', 'after'))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.manifest:
        print(json.dumps(manifest()))
        raise SystemExit(0)
    if args.case:
        args.family, args.variant = args.case.rsplit('-', 1)
    if not args.family or not args.variant or not args.output:
        parser.error('provide --case (or --family and --variant) and fresh --output')
    run(args.family, args.variant, args.output)
