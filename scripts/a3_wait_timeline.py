#!/usr/bin/env python3
"""Paired native wait controls; acquire simulator evidence before expansion."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import a3_memory_batch as base
from a3_controlled_memory_batch import checked_output, memory_capture, probe, save
from a3_native_build import build
from a3_profile_sync import cases, command
from summarize_a3_simulator import parse_trace


def pairs():
    selected = [c for c in cases() if c['placement'] != 'late-set' and c['rounds'] == 32]
    result = {}
    for case in selected:
        name = case['family'] + (('/imbalanced' if case['imbalanced'] else '/balanced')
                                 if 'imbalanced' in case else '')
        result.setdefault(name, []).append(case)
    return result


def instruction_evidence(result, capture):
    """Read all retained rows, not the normalizer's 256-event display sample."""
    simulation = result.get('simulation', {})
    if not result.get('clean_acquisition_gate_passed'):
        return dict(status='unavailable', reason='clean exact-kernel simulation required')
    controls, counts = [], Counter()
    for source in simulation['sources']:
        path = capture / 'report' / source['name']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise RuntimeError('retained simulator source changed')
        rows, _ = parse_trace(json.loads(raw), path)
        for event in rows:
            counts[event['core'], event['pipeline'], event['kind']] += 1
            if event['kind'] == 'control':
                controls.append(dict(event, source_trace=source['name']))
    return dict(status='observed', controls=controls,
                counts=[dict(core=c, pipeline=p, kind=k, count=n)
                        for (c, p, k), n in sorted(counts.items())],
                source_attribution='unavailable: source placement alone does not bind instruction PCs',
                clock='simulator-trace-microseconds',
                cross_core_overlap='unavailable: no independent common-clock proof')


def simulation(executable, case, output, env, soc):
    output.mkdir(parents=True, exist_ok=False)
    wrapper = output / 'application.sh'
    wrapper.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.join(command(executable, case)) + '\n')
    wrapper.chmod(0o700)
    capture = output / 'capture'
    argv = [sys.executable, str(ROOT / 'scripts/a3_timeline_probe.py'),
            '--application', str(wrapper.resolve()), '--output', str(capture.resolve()),
            '--kernel-name', case['selector'], '--language', 'ascendc', '--mode', 'simulation',
            '--soc-version', soc, '--timeout', '300']
    process = base.execute(argv, output / 'worker', env, timeout=620)
    result_path = capture / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.is_file() else dict(
        status='worker_failed', clean_acquisition_gate_passed=False)
    # A nonzero outer worker is never promoted by a stale/partial result file.
    if process.returncode:
        result['clean_acquisition_gate_passed'] = False
    result['worker_exit'] = process.returncode
    result['instruction_evidence'] = instruction_evidence(result, capture)
    save(output / 'summary.json', result)
    return result


def measured_capture(cmd, folder, env, metric, selector):
    if metric != 'Memory':
        return base.capture(cmd, folder, env, metric, selector, timeout=300)
    # memory_capture owns the narrow unsupported-metric classification. Supply
    # its existing capture interface with the same bound, without global patches.
    return memory_capture(cmd, folder, env, selector, timeout=300)


def run_pair(selected, executable, output, env, soc, *, canary=False,
             simulate=simulation, capture=measured_capture, execute=base.execute):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', cases={}, capture_order=[],
                  simulator_not_hardware=True)
    def persist():
        save(output / 'summary.json', result)
    persist()
    try:
        # Native correctness and one native-application simulator acquisition
        # qualify the pair before issuing the remaining profiling matrix.
        for case in selected[:1] if canary else selected:
            folder = output / case['placement']
            folder.mkdir()
            cmd = command(executable, case)
            correctness = checked_output(execute(cmd, folder / 'correctness', env, timeout=300))
            cell = dict(case=case, correctness=correctness, captures=[])
            result['cases'][case['id']] = cell
            cell['simulation'] = simulate(executable, case, folder / 'simulation', env, soc)
            persist()
            if not cell['simulation']['clean_acquisition_gate_passed']:
                result.update(status='acquisition_unavailable',
                              reason='native simulator canary failed; pair expansion stopped')
                persist()
                return result
        if canary:
            result['status'] = 'canary_passed'
            persist()
            return result
        for metric, rounds in [('BasicInfo', 5), ('PipeUtilization', 3), ('Memory', 3)]:
            unavailable = False
            for index in range(rounds):
                order = selected[index % len(selected):] + selected[:index % len(selected)]
                for case in order:
                    if metric == 'Memory' and unavailable:
                        continue
                    folder = output / case['placement'] / f'{metric}-{index}'
                    record = capture(command(executable, case), folder, env, metric, case['selector'])
                    unavailable = record.get('availability') == 'unsupported'
                    cell = result['cases'][case['id']]
                    cell['captures'].append(record)
                    result['capture_order'].append(dict(case=case['id'], metric=metric, round=index))
                    if metric == 'BasicInfo':
                        samples = [base.duration(c['basic_rows'][0]) for c in cell['captures']
                                   if c['metric'] == 'BasicInfo']
                        cell['timing'] = base.summarize(samples)
                    if unavailable:
                        result['memory_availability'] = 'unsupported for selected target/capture'
                    persist()
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed', failure=dict(type=type(exc).__name__, message=str(exc)))
    persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--family', choices=tuple(pairs()), action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--soc-version')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    if args.manifest:
        print(json.dumps(pairs()))
        return 0
    if not args.output or not args.soc_version:
        parser.error('--output and observed --soc-version required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = build(output / 'build')
    physical, env, lease = base.wait_for_device(probe)
    summary = dict(physical_device=physical, logical_device=0,
                   executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(), pairs={})
    try:
        for name in args.family or pairs():
            summary['pairs'][name] = run_pair(pairs()[name], executable,
                output / name.replace('/', '-'), env, args.soc_version, canary=args.canary)
            save(output / 'summary.json', summary)
            print(json.dumps(dict(family=name, status=summary['pairs'][name]['status'])), flush=True)
    finally:
        lease.close()
    return int(any(p['status'] not in {'complete', 'canary_passed'} for p in summary['pairs'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
