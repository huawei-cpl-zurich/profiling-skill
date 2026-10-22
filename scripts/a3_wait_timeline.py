#!/usr/bin/env python3
"""Paired native wait controls; hardware evidence first, simulation optional."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import a3_memory_batch as base
from a3_controlled_memory_batch import checked_output, checkpoint, memory_capture, probe, save
from a3_native_build import build
from a3_profile_sync import cases, command
from summarize_a3_simulator import parse_trace


def pairs(rounds=32):
    if rounds not in (1, 32):
        raise ValueError('native repetition controls support 1 or 32 rounds')
    selected = [dict(c) for c in cases() if c['placement'] != 'late-set' and
                (c['rounds'] == rounds if not c['family'].startswith('mode') else True)]
    result = {}
    for case in selected:
        if case['rounds'] != rounds:
            case.update(rounds=rounds, id=case['id'] + f'/rounds-{rounds}')
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


def run_pair(selected, executable, output, env, soc=None, *, canary=False, include_simulation=False,
             simulate=simulation, capture=measured_capture, execute=base.execute):
    if include_simulation and not soc:
        raise ValueError('requested simulation requires observed --soc-version')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', cases={}, capture_order=[],
                  simulator_not_hardware=True,
                  timing_kind='hardware msprof op diagnostic duration, not uninstrumented timing')
    def persist():
        save(output / 'summary.json', result)
    persist()
    try:
        # Check BOTH variants before any performance capture, including canaries.
        for case in selected:
            folder = output / case['placement']
            folder.mkdir()
            cmd = command(executable, case)
            correctness = checkpoint(folder / 'correctness', dict(command=cmd, case=case),
                lambda: checked_output(execute(cmd, folder / 'correctness', env, timeout=300)))
            cell = dict(case=case, correctness=correctness, captures=[],
                        simulation=dict(status='not_requested'))
            result['cases'][case['id']] = cell
            persist()
        samples = [('BasicInfo', 1)] if canary else [('BasicInfo', 5), ('PipeUtilization', 3), ('Memory', 3)]
        for metric, rounds in samples:
            unavailable = False
            for index in range(rounds):
                order = selected[index % len(selected):] + selected[:index % len(selected)]
                for case in order:
                    if metric == 'Memory' and unavailable:
                        continue
                    folder = output / case['placement'] / f'{metric}-{index}'
                    cmd = command(executable, case)
                    request = dict(command=cmd, metric=metric, selector=case['selector'], case=case)
                    def acquire():
                        observed = capture(cmd, folder, env, metric, case['selector'])
                        return dict(observed, command=cmd, artifact_root=str(folder),
                                    physical_device_mask=env.get('ASCEND_RT_VISIBLE_DEVICES'),
                                    captured_at_unix=time.time())
                    record = checkpoint(folder, request, acquire)
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
        result['status'] = 'canary_passed' if canary else 'complete'
        persist()  # Hardware results survive optional simulator timeout/failure.
        if include_simulation:
            for case in selected:
                cell = result['cases'][case['id']]
                try:
                    cell['simulation'] = simulate(executable, case,
                        output / case['placement'] / 'simulation', env, soc)
                    cell['simulation']['availability'] = ('available' if
                        cell['simulation'].get('clean_acquisition_gate_passed') else 'unavailable')
                except Exception as exc:
                    cell['simulation'] = dict(status='unavailable', clean_acquisition_gate_passed=False,
                                             failure=dict(type=type(exc).__name__, message=str(exc)))
                persist()
    except Exception as exc:
        result.update(status='failed', failure=dict(type=type(exc).__name__, message=str(exc)))
    persist()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--family', choices=tuple(pairs()), action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--soc-version')
    parser.add_argument('--simulation', action='store_true', help='optional diagnostic after hardware captures')
    parser.add_argument('--rounds', type=int, choices=(1, 32), action='append')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args(argv)
    repetitions = list(dict.fromkeys(args.rounds or [32]))
    families = list(dict.fromkeys(args.family or pairs()))
    if args.manifest:
        print(json.dumps({f'{name}/rounds-{r}': group for r in repetitions for name, group in pairs(r).items()}))
        return 0
    if not args.output or (args.simulation and not args.soc_version):
        parser.error('--output required; requested simulation also requires observed --soc-version')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = build(output / 'build')
    physical, env, lease = base.wait_for_device(probe)
    summary = dict(physical_device=physical, logical_device=0,
                   executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(), pairs={})
    try:
        for rounds in repetitions:
            for name in families:
                key = f'{name}/rounds-{rounds}'
                summary['pairs'][key] = run_pair(pairs(rounds)[name], executable,
                    output / f'rounds-{rounds}' / name.replace('/', '-'), env, args.soc_version,
                    canary=args.canary, include_simulation=args.simulation)
                save(output / 'summary.json', summary)
                print(json.dumps(dict(family=key, status=summary['pairs'][key]['status'])), flush=True)
    finally:
        lease.close()
    return int(any(p['status'] not in {'complete', 'canary_passed'} for p in summary['pairs'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
