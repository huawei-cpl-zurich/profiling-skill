#!/usr/bin/env python3
"""Prepare and run private, in-job natural-task qualification; no transport client."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tarfile


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def discover_kernel(report):
    rows = []
    for path in report.rglob('*OpBasicInfo*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError('complete discovery requires one compute row; observed '+str(len(rows)))
    names = {value for key, value in rows[0].items() if value and 'name' in key.lower()
             and ('kernel' in key.lower() or 'op' in key.lower())}
    if len(names) != 1:
        raise ValueError('discovery omitted an unambiguous exported kernel name')
    return names.pop(), rows


def prepare(source_root, task, output):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    exporter = runpy.run_path(str(source_root/'benchmarks/natural_before_tasks.py'))
    alternatives = exporter['qualification_variants'](task)
    if len(alternatives) != 2 or not all(isinstance(source, str) for source in alternatives):
        raise ValueError('qualification requires exactly two scoped source alternatives')
    for name in ('a3_memory_batch.py', 'profile_device.py'):
        if not (source_root/'scripts'/name).is_file():
            raise FileNotFoundError(name)
    output.mkdir(parents=True, exist_ok=False)
    exporter['export'](task, output/'variant-0')
    for index, source in enumerate(alternatives, 1):
        shutil.copytree(output/'variant-0', output/f'variant-{index}')
        (output/f'variant-{index}/candidate.py').write_text(source)
    for name in ('a3_memory_batch.py', 'profile_device.py'):
        shutil.copyfile(source_root/'scripts'/name, output/name)
    shutil.copyfile(__file__, output/'qualify_natural_tasks.py')
    manifest = dict(task=task, variants={f'variant-{index}':
        hashlib.sha256((output/f'variant-{index}/candidate.py').read_bytes()).hexdigest()
        for index in range(3)})
    save(output/'qualification.json', manifest)
    return manifest


def run_variant(bundle, variant, output, env, helpers):
    folder = bundle/variant
    command = [sys.executable, str(folder/'runner.py'), '--implementation=candidate']
    output.mkdir()
    stage = 'candidate'
    result = dict(variant=variant, source_sha256=hashlib.sha256((folder/'candidate.py').read_bytes()).hexdigest())
    try:
        checked = helpers['execute'](command+['--checks=4'], output/'correctness', env, timeout=300)
        records = []
        for line in checked.stdout.splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        result['correctness'] = records
        if checked.returncode or len(records) != 4 or not all(r.get('correctness') is True for r in records):
            raise RuntimeError('four seeded correctness checks did not pass: '+checked.stderr[-6000:])
        stage = 'profiling'
        wrapper = output/'application.sh'
        wrapper.write_text('#!/bin/sh\nexec '+shlex.join(command)+'\n')
        wrapper.chmod(0o700)
        discovery = output/'discovery'
        argv = ['msprof', 'op', '--application='+str(wrapper), '--output='+str(discovery/'report'),
                '--aic-metrics=BasicInfo', '--warm-up=0', '--launch-count=5000', '--kill=off']
        captured = helpers['execute'](argv, discovery, env)
        helpers['validate_capture'](captured.returncode, captured.stdout+captured.stderr)
        # No selector filter in acquisition: reject auxiliary compute launches.
        selector, rows = discover_kernel(discovery/'report')
        result['discovered_compute_rows'] = rows
        result['selector'] = selector
        timings, pipes = [], []
        for index in range(5):
            item = helpers['capture'](command, output/f'basic-{index}', env, 'BasicInfo', selector)
            timings.append(helpers['duration'](item['basic_rows'][0]))
        for index in range(3):
            pipes.append(helpers['capture'](command, output/f'pipe-{index}', env, 'PipeUtilization', selector))
        result.update(state='completed', timing=helpers['summarize'](timings), pipe_captures=pipes)
    except Exception as error:
        result.update(state='failed', failure_stage=stage, error=str(error), error_type=type(error).__name__)
    save(output/'result.json', result)
    return result


def compact_archive(bundle, output):
    """Keep all text diagnostics/CSV/IR, not vendor binary profiling trees."""
    archive_path = output.with_suffix('.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for base in (output, *(bundle/f'variant-{i}/compiler-evidence' for i in range(3))):
            if base.exists():
                for path in base.rglob('*'):
                    if path.is_file() and path.suffix in ('.json', '.log', '.csv', '.txt', '.mlir', '.ll', '.s'):
                        archive.add(path, arcname=str(path.relative_to(bundle)))
    return archive_path


def run(bundle, output, *, helpers=None, device=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle/'qualification.json').read_text())
    if not output.is_relative_to(bundle):
        raise ValueError('output must be inside the prepared bundle')
    for name, expected in manifest['variants'].items():
        if hashlib.sha256((bundle/name/'candidate.py').read_bytes()).hexdigest() != expected:
            raise ValueError('qualification source changed: '+name)
    output.mkdir(parents=True, exist_ok=False)
    helpers = helpers or runpy.run_path(str(bundle/'a3_memory_batch.py'))
    device_helpers = runpy.run_path(str(bundle/'profile_device.py')) if device is None else None
    lease = None
    summary = dict(task=manifest['task'], variants={})
    try:
        if device is None:
            def probe(env):
                p = device_helpers['bounded_run']([sys.executable, '-c',
                    'import torch,torch_npu; torch.empty(1,device="npu"); torch.npu.synchronize()'],
                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                return p.returncode == 0, p.stderr[-1000:]
            physical, env, lease = device_helpers['wait_for_device'](probe)
        else:
            physical, env = device
        summary.update(physical_device=physical, logical_device=0)
        version = helpers['execute'](['msprof', '--version'], output/'profiler-version', env, timeout=30)
        summary['profiler_version'] = dict(returncode=version.returncode, stdout=version.stdout, stderr=version.stderr)
        for name in manifest['variants']:
            result = run_variant(bundle, name, output/name, env, helpers)
            summary['variants'][name] = result
            save(output/'summary.json', summary)
            print(json.dumps(dict(task=manifest['task'], variant=name, result=result)), flush=True)
            if result['state'] != 'completed':
                break
        summary['state'] = ('completed' if len(summary['variants']) == 3 and
            all(r['state'] == 'completed' for r in summary['variants'].values()) else 'failed')
    except Exception as error:
        summary.update(state='failed', failure_stage='environment', error=str(error), error_type=type(error).__name__)
    finally:
        if lease:
            lease.close()
        save(output/'summary.json', summary)
        compact_archive(bundle, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='verb', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--source-root', type=Path, required=True)
    prepare_parser.add_argument('--task', required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--bundle', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_root, args.task, args.output) if args.verb == 'prepare' else run(args.bundle, args.output)
    print(json.dumps(result), flush=True)
    return int(args.verb == 'run' and result['state'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
