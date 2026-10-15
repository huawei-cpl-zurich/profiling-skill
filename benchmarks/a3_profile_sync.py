"""Safe paired synchronization placement controls; no unmatched waits."""
import argparse
import json
from pathlib import Path


def cases():
    result = []
    for family in ('mte2-vector', 'vector-mte3'):
        for rounds in (1, 32):
            for placement in ('overlap', 'early-wait', 'late-set'):
                result.append(dict(id=f'{family}/{rounds}/{placement}', family=family,
                                   rounds=rounds, placement=placement, expected_increment=rounds,
                                   selector='a3_intra_probe', blocks=1))
    for mode in (0, 2):
        for imbalanced in (False, True):
            for placement in ('overlap', 'early-wait'):
                result.append(dict(id=f'mode{mode}/{int(imbalanced)}/{placement}',
                                   family=f'mode{mode}', rounds=32, placement=placement,
                                   imbalanced=imbalanced, selector=f'a3_inter{mode}_probe', blocks=2))
    return result


def command(executable, case):
    family = {'mte2-vector': 0, 'vector-mte3': 1, 'mode0': 2, 'mode2': 3}[case['family']]
    placement = {'overlap': 0, 'early-wait': 1, 'late-set': 2}[case['placement']]
    return [str(executable), str(family), str(placement), str(case['rounds']),
            str(int(case.get('imbalanced', False)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--case', action='append', choices=[c['id'] for c in cases()])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    if args.manifest:
        print(json.dumps(cases()))
        return 0
    if args.output is None:
        parser.error('--output required')
    import subprocess
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from a3_native_build import build
    from a3_memory_batch import run_case, execute
    from profile_device import bounded_run, wait_for_device
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    executable = build(args.output / 'build')
    def probe(env):
        p = bounded_run([sys.executable, '-c', 'import torch,torch_npu; '
                         'torch.empty(1,device="npu"); torch.npu.synchronize()'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return p.returncode == 0, p.stderr[-1000:]
    physical, env, lease = wait_for_device(probe)
    summary = {'physical_device': physical, 'logical_device': 0, 'cells': {}}
    try:
        for case in cases():
            if args.case and case['id'] not in args.case:
                continue
            folder = args.output / case['id'].replace('/', '-')
            try:
                if args.correctness_only:
                    p = execute(command(executable, case), folder, env, timeout=300)
                    records = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
                    if p.returncode or not any(r.get('correctness') is True for r in records):
                        raise RuntimeError('native numerical verification failed')
                    result = records[-1]
                else:
                    result = run_case(command(executable, case), folder, env, case['selector'])
            except Exception as exc:
                result = {'failure': str(exc), 'type': type(exc).__name__}
            summary['cells'][case['id']] = result
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps({'case': case['id'], 'result': result}), flush=True)
    finally:
        lease.close()
    return int(any('failure' in value for value in summary['cells'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
