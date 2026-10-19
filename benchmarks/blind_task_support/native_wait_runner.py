"""Trusted standalone native compiler/contract; candidate contains only a kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from a3_native_build import build


def oracle(kind, variant, seed):
    n, values = variant['n'], []
    for _ in range(2*n):
        seed = (1664525*seed+1013904223) & 0xffffffff
        values.append((seed % 2049-1024)/64)
    rounds = variant['rounds']
    if kind == 1:
        result = []
        for b in range(2):
            result += [x+(32 if variant['imbalanced'] and b == 0 else 0) for x in values[b*n:(b+1)*n]]
            result += [rounds+x+(32 if variant['imbalanced'] and b == 1 else 0)
                       for x in values[(1-b)*n:(2-b)*n]]
        return result
    return ([values[i]+values[n+i]+rounds for i in range(n)] + [rounds]*n) if kind == 2 else \
        values[:n]+[rounds*variant['repetitions']]*n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('baseline', 'candidate'), default='candidate')
    parser.add_argument('--checks', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--launches', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--variant')
    parser.add_argument('--event-timing', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = json.loads((root/'task.json').read_text())
    if min(args.checks, args.launches) < 1 or args.device < 0:
        parser.error('positive checks/launches and nonnegative device required')
    if args.variant and args.variant not in spec['variants']:
        parser.error('unknown variant')
    source = root/(args.implementation+'.asc')
    identity = hashlib.sha256(source.read_bytes()+(root/'native_host.asc').read_bytes()+
                              (root/'task.json').read_bytes()).hexdigest()
    if not args.cpu_only:
        folder = root/'build'/identity
        exe = folder/'a3_sync_control'
        if not exe.exists():
            build(folder, source=root/'native_host.asc', extra_flags=(
                '-I', str(root), f'-DIMPLEMENTATION="{source.name}"', f'-DTASK_KIND={spec["native_wait_kind"]}'))
    variants = list(spec['variants'])
    for check in range(args.checks):
        name = args.variant or (variants[check % len(variants)] if args.checks > 1 else spec['default_variant'])
        v = spec['variants'][name]
        if args.cpu_only:
            print(json.dumps(dict(oracle_only=True, variant=name, expected=oracle(spec['native_wait_kind'], v, args.seed+check))))
        else:
            subprocess.run([str(exe), str(args.seed+check), *[str(v[k]) for k in
                           ('n', 'rounds', 'repetitions', 'imbalanced')], str(args.launches),
                           str(args.device), str(int(args.event_timing))], check=True, timeout=300)


if __name__ == '__main__':
    main()
