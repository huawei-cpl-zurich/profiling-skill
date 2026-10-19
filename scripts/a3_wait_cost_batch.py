#!/usr/bin/env python3
"""Hardware-only phase controls; timing/counters infer benefits, not a timeline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import a3_native_build
import a3_memory_batch as base
from a3_controlled_memory_batch import checkpoint, checked_output, memory_capture, probe, save

MODES = ('transfer', 'compute', 'serialized', 'overlap')
SELECTOR = 'a3_wait_cost_probe'


def cases(canary=False):
    return [dict(id=f'{n}/{rounds}/{reps}/{mode}', n=n, rounds=rounds,
                 repetitions=reps, mode=mode, blocks=1)
            for n in ((1024,) if canary else (1024, 8192))
            for rounds in ((1,) if canary else (1, 32))
            for reps in ((1,) if canary else (1, 16)) for mode in MODES]


def expected(case):
    """Independent host specification for both observable output regions."""
    n = case['n']
    return ([0 if case['mode'] == 'compute' else i % 17 for i in range(n)] +
            [0 if case['mode'] == 'transfer' else case['rounds'] * case['repetitions']] * n)


def command(executable, case, timing=False):
    return [str(executable), str(MODES.index(case['mode'])), str(case['n']),
            str(case['rounds']), str(case['repetitions']), str(int(timing))]


def conclusions(cells):
    groups = {}
    for cell in cells.values():
        c = cell['case']
        groups.setdefault('/'.join(c['id'].split('/')[:3]), {})[c['mode']] = cell
    result = {}
    for key, group in groups.items():
        if not all(m in group and 'event_timing' in group[m] for m in MODES):
            continue
        durations = {m: group[m]['event_timing']['median_us'] for m in MODES}
        result[key] = dict(event_medians_us=durations,
                           paired_speedup=durations['serialized'] / durations['overlap'],
                           isolated_stage_sum_us=durations['transfer'] + durations['compute'],
                           isolated_stage_max_us=max(durations['transfer'], durations['compute']),
                           interpretation='isolated stages repeat launch/prologue/epilogue overhead; '
                           'sum/max are qualitative references, not exact predicted bounds; '
                           'paired timing does not locate stalls or prove simultaneous pipe overlap')
    return result


def run(selected, output, env, *, physical, resume=False):
    if not selected:
        raise ValueError('at least one complete group required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    root = Path(__file__).resolve().parents[1]
    source = root / 'benchmarks/a3_wait_cost_control.asc'
    sources = [source, Path(__file__), root / 'benchmarks/a3_native_build.py',
               root / 'scripts/a3_memory_batch.py', root / 'scripts/a3_controlled_memory_batch.py',
               root / 'scripts/profile_device.py']
    config = dict(cases=selected, sources={str(p.relative_to(root)):
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  samples=dict(BasicInfo=5, PipeUtilization=3, Memory=3),
                  event_timing=dict(warmup=3, launch_count=7))
    manifest = output / 'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError('resume configuration/source mismatch')
    save(manifest, config)
    build_folder = output / 'build'
    def compile_native():
        exe = a3_native_build.build(build_folder, source=source)
        return dict(executable=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    built = checkpoint(output / 'build-checkpoint', dict(source=config['sources']), compile_native)
    executable = Path(built['executable'])
    if hashlib.sha256(executable.read_bytes()).hexdigest() != built['binary_sha256']:
        raise RuntimeError('retained executable changed')
    summary = dict(cells={}, physical_device=physical, logical_device=0, build=built,
                   capture_order=[], event_timing='uninstrumented ACL stream event elapsed time',
                   BasicInfo='diagnostic profiler duration, not canonical uninstrumented latency')
    save(output / 'summary.json', summary)
    for case in selected:
        folder = output / case['id'].replace('/', '-')
        cell = dict(case=case, captures=[])
        summary['cells'][case['id']] = cell
        for phase, timing in (('correctness', False), ('event-timing', True)):
            argv = command(executable, case, timing)
            def execute():
                return dict(checked_output(base.execute(argv, folder / phase, env, timeout=300)),
                            physical_device=physical, captured_at_unix=time.time())
            record = checkpoint(folder / phase, dict(command=argv), execute)
            if timing:
                if record.get('warmup') != 3 or record.get('launch_count') != 7 or \
                        len(record.get('samples_us', [])) != 7:
                    raise RuntimeError('native event timing sample count mismatch')
                cell['event_timing'] = base.summarize(record['samples_us'])
                cell['event_evidence'] = record
            else:
                cell['correctness'] = record
        save(output / 'summary.json', summary)
    for metric, count in config['samples'].items():
        for round_id in range(count):
            order = selected[round_id % len(selected):] + selected[:round_id % len(selected)]
            for case in order:
                cell = summary['cells'][case['id']]
                folder = output / case['id'].replace('/', '-') / f'{metric}-{round_id}'
                argv = command(executable, case)
                def collect():
                    evidence = memory_capture(argv, folder, env, SELECTOR) if metric == 'Memory' \
                        else base.capture(argv, folder, env, metric, SELECTOR)
                    return dict(evidence, physical_device=physical, captured_at_unix=time.time())
                evidence = checkpoint(folder, dict(command=argv, metric=metric, selector=SELECTOR), collect)
                cell['captures'].append(evidence)
                summary['capture_order'].append(dict(case=case['id'], metric=metric, round=round_id,
                                                      physical_device=evidence['physical_device']))
                samples = [base.duration(c['basic_rows'][0]) for c in cell['captures']
                           if c['metric'] == 'BasicInfo']
                if samples:
                    cell['profile_timing'] = base.summarize(samples)
                summary['comparisons'] = conclusions(summary['cells'])
                save(output / 'summary.json', summary)
                print(json.dumps(summary['capture_order'][-1]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--group', help='n/rounds/repetitions; run all four modes together')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    selected = cases(args.canary)
    if args.group:
        selected = [c for c in selected if c['id'].rsplit('/', 1)[0] == args.group]
        if not selected:
            parser.error('unknown or excluded group')
    physical, env, lease = base.wait_for_device(probe)
    try:
        run(selected, args.output.resolve(), env, physical=physical, resume=args.resume)
    finally:
        lease.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
