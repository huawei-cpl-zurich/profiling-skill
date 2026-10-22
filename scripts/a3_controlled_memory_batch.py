#!/usr/bin/env python3
"""Interleaved A3 captures with exact selectors and completed-boundary resume."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import a3_memory_batch as base
from a3_controlled_memory import cases, footprint


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def checked_output(result):
    if result.returncode:
        raise RuntimeError('workload process failed: ' + result.stderr[-2000:])
    records = []
    for line in result.stdout.splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    valid = [r for r in records if isinstance(r, dict) and r.get('correctness') is True]
    if len(valid) != 1:
        raise RuntimeError('expected exactly one correctness result')
    return valid[0]


def memory_capture(command, folder, env, selector, *, timeout=None):
    try:
        evidence = base.capture(command, folder, env, 'Memory', selector,
                                **({} if timeout is None else {'timeout': timeout}))
    except RuntimeError:
        transcript = '\n'.join(p.read_text() for p in folder.glob('*.log'))
        # Only a named metric rejection is a capability result. Runtime/compiler,
        # partial collection and transport errors must propagate as failures.
        if not re.search(r'\bMemory\s+(?:(?:metric|group)\s+)?(?:is\s+)?'
                         r'(?:not supported|unsupported)\b|'
                         r'\b(?:unsupported|invalid|unrecognized)[^\n]{0,80}'
                         r'(?:--aic-metrics|\bmetric(?:\s+group)?\b)[^\n]{0,40}\bMemory\b',
                         transcript, re.IGNORECASE):
            raise
        return dict(metric='Memory', availability='unsupported', diagnostic=transcript[-4000:])
    exports = []
    for path in sorted((folder / 'report').rglob('*Memory*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            names = [v for k, v in row.items() if 'name' in k.lower()
                     and ('kernel' in k.lower() or 'op' in k.lower())]
            if names and any(v != selector for v in names):
                raise RuntimeError('Memory export has a different selector')
        exports.append(dict(path=str(path.relative_to(folder)), rows=rows))
    if not exports or not any(e['rows'] for e in exports):
        raise RuntimeError('Memory capture succeeded but no nonempty Memory CSV was exported')
    return dict(evidence, availability='available', exports=exports,
                units='raw column names retained; byte factors and denominators unvalidated')


def checkpoint(folder, request, action):
    done = folder / 'completed.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record['request'] != request:
            raise RuntimeError('completed capture request changed: ' + str(folder))
        for name, digest in record['files'].items():
            path = folder / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('completed capture evidence changed: ' + str(path))
        validate_compiler_artifacts(record['result'])
        return record['result']
    if folder.exists():
        raise RuntimeError('incomplete capture retained; inspect logs before explicit retry: ' + str(folder))
    result = action()
    validate_compiler_artifacts(result)
    folder.mkdir(parents=True, exist_ok=True)
    files = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob('*') if p.is_file()}
    save(done, dict(request=request, result=result, files=files))
    return result


def validate_compiler_artifacts(result):
    # Compiler artifacts live beside correctness/, not inside its phase subtree.
    # Their existing retained pointers and digests are the authoritative binding.
    for artifact in result.get('compiler', []):
        path = Path(artifact['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise RuntimeError('referenced compiler evidence missing or changed: ' + str(path))


def interleaved(selected, output, env, *, physical, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'benchmarks/a3_controlled_memory.py',
               root / 'benchmarks/a3_profile_memory.py',
               root / 'scripts/a3_memory_batch.py', root / 'scripts/profile_device.py',
               root / 'benchmarks/compile_evidence.py']
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    configuration = dict(cases=selected, sources=source_hashes,
                         samples=dict(BasicInfo=5, PipeUtilization=3, Memory=3))
    manifest = output / 'configuration.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != configuration:
            raise RuntimeError('resume configuration/source mismatch')
    else:
        save(manifest, configuration)
    # A new resumed session may select another equivalent device. Keep each
    # capture's actual physical provenance rather than rewriting old records.
    summary = dict(cells={}, capture_order=[])
    commands = {}
    for case in selected:
        folder = output / case['id'].replace('/', '-')
        command = [sys.executable, str(root / 'benchmarks/a3_controlled_memory.py'),
                   '--size', case['id'].split('/')[-1], '--case', case['id'],
                   '--artifacts', str(folder / 'compiler-evidence')]
        commands[case['id']] = command
        correctness = checkpoint(folder / 'correctness', dict(command=command, phase='correctness'),
                                 lambda: checked_output(base.execute(command, folder / 'correctness', env,
                                                                    timeout=300)))
        discovery = checkpoint(folder / 'discovery', dict(command=command, metric='BasicInfo'),
                               lambda: base.capture(command, folder / 'discovery', env, 'BasicInfo'))
        row = discovery['basic_rows'][0]
        selector = next(v for k, v in row.items() if 'name' in k.lower()
                        and ('kernel' in k.lower() or 'op' in k.lower()))
        summary['cells'][case['id']] = dict(case=case, footprint=footprint(case), selector=selector,
                                            correctness=correctness, captures=[])
    save(output / 'summary.json', summary)
    for metric, count in configuration['samples'].items():
        for round_id in range(count):
            # Rotate each round to avoid one case always occupying the warm/cold edge.
            order = selected[round_id % len(selected):] + selected[:round_id % len(selected)]
            for case in order:
                cell = summary['cells'][case['id']]
                command = commands[case['id']]
                folder = output / case['id'].replace('/', '-') / f'{metric}-{round_id}'
                request = dict(command=command, metric=metric, selector=cell['selector'])
                def capture():
                    observed = memory_capture(command, folder, env, cell['selector']) if metric == 'Memory' \
                        else base.capture(command, folder, env, metric, cell['selector'])
                    return dict(observed, physical_device=physical, captured_at_unix=time.time())
                result = checkpoint(folder, request, capture)
                cell['captures'].append(result)
                summary['capture_order'].append(dict(case=case['id'], metric=metric, round=round_id,
                                                      physical_device=result['physical_device']))
                samples = [base.duration(c['basic_rows'][0]) for c in cell['captures']
                           if c['metric'] == 'BasicInfo']
                if samples:
                    cell['timing'] = base.summarize(samples)
                    cell['useful_input_GB_per_s'] = cell['footprint']['useful_input_bytes'] / \
                        cell['timing']['median_us'] / 1000
                save(output / 'summary.json', summary)
                print(json.dumps(summary['capture_order'][-1]), flush=True)
    return summary


def probe(env, *, executable=sys.executable, expression=None):
    expression = expression or ('import torch,torch_npu; '
                                'torch.empty(1,device="npu"); torch.npu.synchronize()')
    result = base.bounded_run([executable, '-c', expression], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=30)
    return result.returncode == 0, result.stderr[-1000:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1048576)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    selected = cases(args.size)
    if args.pilot:
        ids = {f'count/1/{args.size}', f'count/16/{args.size}', f'indexed/global/{args.size}'}
        selected = [c for c in selected if c['id'] in ids]
    if args.case:
        selected = [c for c in selected if c['id'] in args.case]
        if {c['id'] for c in selected} != set(args.case):
            parser.error('unknown or excluded --case')
    physical, env, lease = base.wait_for_device(probe)
    try:
        interleaved(selected, args.output.resolve(), env, physical=physical, resume=args.resume)
    finally:
        lease.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
