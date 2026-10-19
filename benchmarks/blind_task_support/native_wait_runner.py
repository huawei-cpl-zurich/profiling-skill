"""Trusted standalone native compiler/contract; candidate contains only a kernel."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from a3_native_build import build


def seeded_inputs(kind, variant, seed):
    n, values = variant['n'], []
    payload = max(2, variant['repetitions']) if kind == 3 else 2
    vectors = payload+(2 if kind == 1 else 1)
    for i in range(vectors*n):
        seed = (1664525*seed+1013904223) & 0xffffffff
        values.append((seed % 65535+1)/65536 if kind == 3 or i >= payload*n else (seed % 2049-1024)/64)
    return np.array(values, dtype=np.float32).reshape(vectors, n)


def oracle(kind, variant, seed):
    values = seeded_inputs(kind, variant, seed)
    w = values[2:].copy() if kind == 1 else values[-1].copy()
    for rep in range(variant['repetitions'] if kind == 3 else 1):
        for _ in range(variant['rounds']):
            negative = np.multiply(w, np.float32(-1))
            complement = np.add(negative, np.float32(1))
            product = np.multiply(w, complement)
            w = np.multiply(product, np.float32(4))
        if kind == 3:
            w = np.multiply(np.add(w, values[rep]), np.float32(.25))
    if kind == 1:
        result = []
        for b in range(2):
            result += np.add(values[b], np.float32(32 if variant['imbalanced'] and b == 0 else 0)).tolist()
            peer = np.add(values[1-b], np.float32(32 if variant['imbalanced'] and b == 1 else 0))
            result += np.add(w[b], peer).tolist()
        return result
    return np.concatenate((np.add(np.add(values[0], values[1]), w) if kind == 2 else
                           values[variant['repetitions']-1], w)).tolist()


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
