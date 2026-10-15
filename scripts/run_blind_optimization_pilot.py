#!/usr/bin/env python3
"""Freeze and run six paired optimization tasks in four balanced waves."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

HELPERS = ('blind_optimization_launcher.py', 'production_launcher.py',
           'blind_optimization_tools.py', 'profile_device.py', 'a3_memory_batch.py')
GUIDANCE = ('SKILL.md', *('references/' + name for name in (
    'triton-pipe-attribution.md', 'a2-a3-msprof-op.md', 'a3-performance-reasoning.md',
    'acquisition-workflow.md', 'a3-timeline-and-correctness.md',
    'triton-pipe-attribution.json', 'a3-pipe-activity-model.json',
    'pipe-saturation-contract.md', 'a5-pipe-capacity-model.json',
    'a5-metric-semantics.md', 'a5-timeline-semantics.md')))
WAVES = (((1, True), (2, False), (3, True)), ((1, False), (2, True), (3, False)),
         ((4, False), (5, True), (6, False)), ((4, True), (5, False), (6, True)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(source, output, model, forbidden):
    source, output = source.resolve(), output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (*forbidden, source)):
        raise ValueError('output must be outside source and forbidden workspaces')
    if not all(isinstance(model.get(key), str) and model[key].strip()
               for key in ('name', 'reasoning_effort')):
        raise ValueError('explicit model name and reasoning_effort required')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    snapshots = [(source / 'scripts' / name, output / 'runtime' / name) for name in HELPERS]
    snapshots += [(source / name, output / 'skill' / name) for name in GUIDANCE]
    snapshots += [(source / 'benchmarks' / name, output / 'exporter' / name) for name in (
        'blind_before_tasks.py', 'blind_task_support/runner.py', 'blind_task_support/native_host.asc')]
    for original, target in snapshots:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    exporter = runpy.run_path(str(output / 'exporter/blind_before_tasks.py'))
    (output / 'bundles').mkdir()
    for index in range(1, 7):
        task = f'task{index:02}'
        exporter['export'](task, output / 'bundles' / task)
        write_json(output / f'{task}-tool.json', [sys.executable,
                   str(output / 'runtime/blind_optimization_tools.py'),
                   '--bundle', str(output / 'bundles' / task)])
    launcher = runpy.run_path(str(output / 'runtime/blind_optimization_launcher.py'))
    (output / 'prompt.md').write_text(launcher['PROMPT'])
    write_json(output / 'model.json', model)
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True)
    write_json(output / 'manifest.json', dict(schema_version=1, source_revision=revision.stdout.strip(),
               files=hashes, waves=WAVES, wall_budget_seconds=600, model=model,
               forbidden=[str(p.resolve()) for p in forbidden]))
    return output


def run_cohort(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError('changed frozen input: ' + name)
    # Exclusive creation prevents an accidental second launch, including after interruption.
    with (output / 'started.json').open('x') as stream:
        json.dump(dict(started=time.time()), stream)

    def run(cell):
        index, guidance = cell
        task = f'task{index:02}'
        name = task + ('-guidance' if guidance else '-control')
        command = [sys.executable, str(output / 'runtime/blind_optimization_launcher.py'),
                   '--bundle', str(output / 'bundles' / task), '--prompt', str(output / 'prompt.md'),
                   '--output', str(output / name), '--model-json', str(output / 'model.json'),
                   '--tool-command-json', str(output / f'{task}-tool.json')]
        for path in manifest['forbidden']:
            command += ['--forbidden', path]
        if guidance:
            command += ['--skill', str(output / 'skill')]
        result = dict(cell=name, command=command, started=time.time())
        try:
            with (output / f'{name}-controller.log').open('x') as stream:
                result['controller_exit'] = subprocess.run(command, stdout=stream,
                                                           stderr=subprocess.STDOUT).returncode
            receipt = output / name / 'receipt.json'
            result['receipt'] = json.loads(receipt.read_text()) if receipt.exists() else None
        except (OSError, ValueError) as error:
            result.update(controller_exit=74, error=str(error))
        result['ended'] = time.time()
        write_json(output / f'{name}-result.json', result)
        return result

    results = []
    for index, wave in enumerate(WAVES, 1):
        with ThreadPoolExecutor(max_workers=3) as pool:
            cells = list(pool.map(run, wave))
        write_json(output / f'wave-{index}.json', cells)
        results.extend(cells)
    return int(any(cell['controller_exit'] for cell in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-json', type=Path, required=True)
    parser.add_argument('--forbidden-workspace', type=Path, action='append', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = prepare(args.source_root, args.output, json.loads(args.model_json.read_text()),
                     args.forbidden_workspace)
    return 0 if args.prepare_only else run_cohort(output)


if __name__ == '__main__':
    raise SystemExit(main())
