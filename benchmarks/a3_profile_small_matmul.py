"""One-program, single-dot Cube controls; native inputs are host-prepacked."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def cases():
    return [dict(id=f'{impl}/{m}x{n}x{k}', implementation=impl, m=m, n=n, k=k,
                 selector='a3_small_mmad' if impl == 'native' else None)
            for impl in ('native', 'triton')
            for m, n, k in ((16, 16, 16), (32, 32, 32), (64, 64, 64),
                            (128, 128, 128), (16, 64, 32), (64, 16, 32))]


def offset(r, c, rows, cols, layout):
    if layout not in ('ZZ', 'ZN', 'NZ') or rows % 16 or cols % 16:
        raise ValueError('expected aligned ZZ/ZN/NZ matrix')
    block = (c // 16 * (rows // 16) + r // 16 if layout == 'NZ'
             else r // 16 * (cols // 16) + c // 16)
    inner = c % 16 * 16 + r % 16 if layout == 'ZN' else r % 16 * 16 + c % 16
    return block * 256 + inner


def values(rows, cols, salt):
    return [((r * 3 + c * 5 + salt) % 17 - 8) / 8
            for r in range(rows) for c in range(cols)]


def reference(a, b, m, n, k):
    return [sum(a[r * k + q] * b[q * n + c] for q in range(k))
            for r in range(m) for c in range(n)]


def command(executable, case, artifacts):
    if case['implementation'] == 'native':
        return [str(executable), *[str(case[d]) for d in ('m', 'n', 'k')]]
    return [sys.executable, str(Path(__file__).resolve()), '--triton-case', case['id'],
            '--artifacts', str(artifacts)]


def run_triton(case, artifacts):
    global tl
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl
    from compile_evidence import retain_compiler_evidence

    @triton.jit
    def a3_small_dot(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        r = tl.arange(0, M)
        s = tl.arange(0, N)
        q = tl.arange(0, K)
        av = tl.load(a + r[:, None] * K + q[None, :])
        bv = tl.load(b + q[:, None] * N + s[None, :])
        out = tl.dot(av, bv, out_dtype=tl.float32)
        tl.store(c + r[:, None] * N + s[None, :], out)

    m, n, k = (case[d] for d in ('m', 'n', 'k'))
    av, bv = values(m, k, 3), values(k, n, 7)
    expected = torch.tensor(reference(av, bv, m, n, k), dtype=torch.float32).reshape(m, n)
    a = torch.tensor(av, dtype=torch.float16).reshape(m, k).npu()
    b = torch.tensor(bv, dtype=torch.float16).reshape(k, n).npu()
    c = torch.empty((m, n), dtype=torch.float32, device='npu')
    compiled = a3_small_dot[(1,)](a, b, c, m, n, k)
    torch.npu.synchronize()
    actual = c.cpu()
    return dict(correctness=torch.equal(actual, expected),
                max_abs_error=float((actual - expected).abs().max()), case=case,
                compiler=retain_compiler_evidence(compiled.asm, artifacts, case['id'], 'dot', 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--case', action='append', choices=[c['id'] for c in cases()])
    parser.add_argument('--triton-case', choices=[c['id'] for c in cases() if c['implementation'] == 'triton'])
    parser.add_argument('--artifacts', type=Path, default=Path('compiler-evidence'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    if args.manifest:
        print(json.dumps(cases()))
        return 0
    if args.triton_case:
        result = run_triton(next(c for c in cases() if c['id'] == args.triton_case), args.artifacts)
        print(json.dumps(result))
        return int(not result['correctness'])
    if args.output is None:
        parser.error('--output required')
    from a3_native_build import build
    from a3_memory_batch import execute, run_case
    from profile_device import bounded_run, wait_for_device
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = [c for c in cases() if not args.case or c['id'] in args.case]
    executable = (build(args.output / 'build', source=Path(__file__).with_name('a3_small_matmul.asc'))
                  if any(c['implementation'] == 'native' for c in selected) else None)
    def probe(env):
        p = bounded_run([sys.executable, '-c', 'import torch,torch_npu; '
                         'torch.empty(1,device="npu"); torch.npu.synchronize()'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return p.returncode == 0, p.stderr[-1000:]
    physical, env, lease = wait_for_device(probe)
    summary = dict(physical_device=physical, logical_device=0, cells={})
    try:
        for case in selected:
            folder = args.output / case['id'].replace('/', '-')
            argv = command(executable, case, folder / 'compiler-evidence')
            try:
                if args.correctness_only:
                    p = execute(argv, folder, env, timeout=300)
                    records = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
                    if p.returncode or not any(r.get('correctness') is True for r in records):
                        raise RuntimeError('numerical verification failed; retained command logs')
                    result = records[-1]
                else:
                    result = run_case(argv, folder, env, case['selector'])
            except Exception as exc:
                result = dict(failure=str(exc), type=type(exc).__name__)
            summary['cells'][case['id']] = result
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(dict(case=case['id'], result=result)), flush=True)
            if 'failure' in result:
                break
    finally:
        lease.close()
    return int(any('failure' in r for r in summary['cells'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
