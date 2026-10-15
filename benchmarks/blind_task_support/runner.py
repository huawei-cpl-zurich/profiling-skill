"""Numerical contract and execution entrypoint for a standalone task."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch


def inputs_and_reference(spec, seed, check=0):
    generator = torch.Generator().manual_seed(seed)
    task, n = spec['id'], spec['n']
    x = torch.randn(n, generator=generator, dtype=torch.float32)
    inputs = {'x': x}
    if task == 'task01':
        work = (64, 0, 256, 17)[check % 4]
        inputs['work'] = torch.tensor([work], dtype=torch.int32)
        expected = x * (1 + work * 1e-6) * 1.5 + 2
    elif task == 'task02':
        indices = torch.arange(n, dtype=torch.int32) if check % 2 == 0 else torch.randperm(n, generator=generator).int()
        inputs['index'] = indices
        expected = x[indices.long()]
    elif task == 'task03':
        inputs = {k: torch.randn((n, n), generator=generator) for k in ('lhs', 'rhs')}
        expected = inputs['lhs'] @ inputs['rhs']
    elif task == 'task04':
        selected = ((torch.arange(n) * 17 % 4) == 0) if check == 0 else torch.rand(n, generator=generator) < 0.35
        inputs['selected'] = selected
        inputs['repetitions'] = torch.tensor([(128, 1, 32)[check % 3]], dtype=torch.int32)
        expected = torch.where(selected, x * 1.5 + 2, 0)
    elif task == 'task05':
        expected = x * 1.5 + 2
    elif task == 'task06':
        state, values = seed & 0xffffffff, []
        for _ in range(2 * n):
            state = (1664525 * state + 1013904223) & 0xffffffff
            values.append((state % 2049 - 1024) / 64)
        inputs = {'x': torch.tensor(values, dtype=torch.float32)}
        expected = torch.cat((inputs['x'][:n] + inputs['x'][n:] + 32, torch.full((n,), 32.)))
    else:
        raise ValueError('unknown task')
    return inputs, expected


def native_run(root, args):
    source = root / f'{args.implementation}.asc'
    identity = hashlib.sha256(source.read_bytes() + (root / 'native_host.asc').read_bytes()).hexdigest()[:16]
    folder = root / 'build' / identity
    folder.mkdir(parents=True, exist_ok=True)
    executable = folder / 'kernel'
    if not executable.exists():
        command = ['bisheng', str(root / 'native_host.asc'), '-I', str(root), '-O2', '-g',
                   '--npu-arch=dav-2201', '-lascendcl', '-o', str(executable),
                   f'-DIMPLEMENTATION="{args.implementation}.asc"']
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=folder)
        (folder / 'compile.json').write_text(json.dumps({'command': command, 'returncode': result.returncode}))
        (folder / 'compile.stdout').write_text(result.stdout)
        (folder / 'compile.stderr').write_text(result.stderr)
        if result.returncode:
            sys.stderr.write(result.stderr)
            raise RuntimeError('native compilation failed; retained diagnostics in ' + str(folder))
    for check in range(args.checks):
        subprocess.run([str(executable), str(args.seed + check), str(args.launches), str(args.device)], check=True, timeout=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=['baseline', 'candidate'], default='candidate')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checks', type=int, default=1)
    parser.add_argument('--launches', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if min(args.checks, args.launches) < 1:
        parser.error('checks and launches must be positive')
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'task.json').read_text())
    if spec['language'] == 'ascendc':
        if args.cpu_only:
            inputs, expected = inputs_and_reference(spec, args.seed)
            print(json.dumps({'correctness': True, 'native_oracle': 'fixed host', 'elements': expected.numel()}))
        else:
            native_run(root, args)
        return
    module = None
    if not args.cpu_only:
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
        module_spec = importlib.util.spec_from_file_location('implementation', root / f'{args.implementation}.py')
        module = importlib.util.module_from_spec(module_spec)
        sys.modules['implementation'] = module
        module_spec.loader.exec_module(module)
    for check in range(args.checks):
        inputs, expected = inputs_and_reference(spec, args.seed + check, check)
        if module is not None:
            device_inputs = {k: v.npu() for k, v in inputs.items()}
            output = torch.empty_like(expected, device='npu')
            for _ in range(args.launches):
                compiled = module.launch(device_inputs, output)
            torch.npu.synchronize()
            actual = output.cpu()
            if spec['id'] == 'task02':
                assert torch.equal(actual, expected), 'indexed copy differs from CPU reference'
            else:
                torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
            if compiled is not None and hasattr(compiled, 'asm'):
                artifacts = root / 'compiler-evidence' / args.implementation
                artifacts.mkdir(parents=True, exist_ok=True)
                for stage, data in compiled.asm.items():
                    if isinstance(data, str) and stage.replace('_', '').isalnum():
                        (artifacts / (stage + '.txt')).write_text(data)
        print(json.dumps({'correctness': True, 'seed': args.seed + check,
                          'check': check, 'selector': 'task_kernel', 'elements': expected.numel()}), flush=True)


if __name__ == '__main__':
    main()
