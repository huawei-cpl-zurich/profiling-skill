#!/usr/bin/env python3
"""One retained job, one idle-device lease, sequential compact captures."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_device import bounded_run, wait_for_device

SUCCESS = 'Profiling running finished. All task success.'


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError('timing samples must be finite and positive')
    median = statistics.median(samples)
    spread = (max(samples) - min(samples)) / median
    return dict(samples_us=samples, median_us=median, range_over_median=spread,
                low_confidence=spread > .05)


def basic_rows(report, selector=None):
    rows = []
    for path in sorted(report.rglob('*OpBasicInfo*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    selected = [row for row in rows if any(
        ('kernel' in key.lower() or 'op' in key.lower()) and 'name' in key.lower()
        and (value == selector if selector else 'a3_memory_probe' in value)
        for key, value in row.items() if value)]
    if not selected:
        raise RuntimeError('exact selector missing from BasicInfo export')
    return selected


def duration(row):
    fields = [value for key, value in row.items()
              if 'duration' in key.lower() and '(us)' in key.lower().replace(' ', '')]
    if len(fields) != 1:
        raise RuntimeError('ambiguous or missing microsecond duration column')
    return float(fields[0])


def validate_capture(code, transcript):
    if code or SUCCESS not in transcript or 'data collection failed' in transcript.lower():
        raise RuntimeError('profiler success/complete collection not established: ' + transcript[-2000:])


def execute(argv, folder, env, *, timeout=600):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'command.json').write_text(json.dumps(argv))
    try:
        result = bounded_run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, timeout=timeout)
    except Exception as exc:
        (folder / 'failure.json').write_text(json.dumps({'error': str(exc), 'type': type(exc).__name__}))
        raise
    (folder / 'stdout.log').write_text(result.stdout)
    (folder / 'stderr.log').write_text(result.stderr)
    (folder / 'exit.json').write_text(json.dumps({'code': result.returncode}))
    return result


def capture(command, folder, env, metric, selector=None):
    wrapper = folder.parent / (folder.name + '-application.sh')
    wrapper.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join(command) + '\n')
    wrapper.chmod(0o700)
    argv = ['msprof', 'op', '--application=' + str(wrapper), '--output=' + str(folder / 'report'),
            '--aic-metrics=' + metric, '--warm-up=0', '--launch-count=1', '--kill=off']
    if selector:
        argv.append('--kernel-name=' + selector)
    result = execute(argv, folder, env)
    validate_capture(result.returncode, result.stdout + result.stderr)
    rows = basic_rows(folder / 'report', selector)
    if len(rows) != 1:
        raise RuntimeError('expected exactly one selected kernel launch')
    evidence = {'metric': metric, 'basic_rows': rows, 'selector': selector}
    if metric == 'PipeUtilization':
        evidence['pipe_rows'] = []
        for path in sorted((folder / 'report').rglob('*PipeUtilization*.csv')):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                evidence['pipe_rows'].extend(dict(row) for row in csv.DictReader(stream))
        if not evidence['pipe_rows']:
            raise RuntimeError('missing pipe rows despite profiler exit zero')
    (folder / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    return evidence


def run_case(command, folder, env, selector=None):
    """Also accepts a native executable command; caller owns the device lease."""
    folder.mkdir(parents=True, exist_ok=False)
    result = execute(command, folder / 'correctness', env, timeout=300)
    if result.returncode:
        raise RuntimeError('workload correctness process failed: ' + result.stderr[-2000:])
    if command[0] == sys.executable:
        records = []
        for line in result.stdout.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        if not any(isinstance(row, dict) and row.get('correctness') is True for row in records):
            raise RuntimeError('workload omitted explicit correctness evidence')
    discovery = capture(command, folder / 'discovery', env, 'BasicInfo', selector)
    if selector is None:
        row = discovery['basic_rows'][0]
        selector = next(value for key, value in row.items() if 'name' in key.lower()
                        and ('kernel' in key.lower() or 'op' in key.lower()))
    timings, pipes = [], []
    for index in range(5):
        evidence = capture(command, folder / f'basic-{index}', env, 'BasicInfo', selector)
        timings.append(duration(evidence['basic_rows'][0]))
    for index in range(3):
        pipes.append(capture(command, folder / f'pipe-{index}', env, 'PipeUtilization', selector))
    summary = dict(selector=selector, timing=summarize(timings), pipe_captures=pipes)
    (folder / 'result.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from a3_profile_memory import cases
    selected = cases()
    if args.pilot:
        selected = [c for c in selected if c['id'] in (
            'strided/1/4096', 'strided/16/4096', 'irregular/global/4096')]
    if args.case:
        selected = [c for c in selected if c['id'] in args.case]
        if {c['id'] for c in selected} != set(args.case):
            parser.error('unknown or excluded --case')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    def probe(env):
        result = bounded_run([sys.executable, '-c', 'import torch,torch_npu; '
                              'torch.empty(1,device="npu"); torch.npu.synchronize()'],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=30)
        return result.returncode == 0, result.stderr[-1000:]
    physical, env, lease = wait_for_device(probe)
    summary = dict(physical_device=physical, logical_device=0, cells={})
    try:
        for case in selected:
            folder = args.output / case['id'].replace('/', '-')
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'benchmarks'
                                          / 'a3_profile_memory.py'), '--case', case['id'],
                       '--artifacts', str(folder / 'compiler-evidence')]
            try:
                summary['cells'][case['id']] = run_case(command, folder, env)
            except Exception as exc:
                summary['cells'][case['id']] = {'failure': str(exc), 'type': type(exc).__name__}
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps({'case': case['id'], 'result': summary['cells'][case['id']]}), flush=True)
    finally:
        lease.close()
    return int(any('failure' in cell for cell in summary['cells'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
