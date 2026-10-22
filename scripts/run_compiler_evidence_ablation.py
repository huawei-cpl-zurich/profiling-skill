#!/usr/bin/env python3
"""Twelve fresh offline diagnoses: three tasks by four supplied-evidence arms."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipe_evidence_diagnostic import DiagnosticLauncher, digest, seal_result
from blind_optimization_launcher import positive_budget, write_json

ARMS = ('timing', 'pipes', 'compiler', 'both')
PUBLIC = {'task.json', 'TASK.md', 'baseline.py', 'candidate-a.py', 'candidate-b.py',
          'timing.json', 'environment.json'}
PROMPT = '''Assess the supplied Triton baseline and two proposed candidate changes without editing inputs or executing kernels. Read TASK.md, environment.json, baseline.py, candidate-a.py, candidate-b.py and timing.json. Use available evidence and guidance as you find useful. Give an evidence-supported diagnosis, choose which candidate to test next (a/b/neither/indeterminate), predict both candidates' latency direction, and explain uncertainty and a discriminating next measurement. Distinguish observed evidence from hypotheses; do not assume a hardware bottleneck or final instruction scheduling that the evidence cannot establish.
Write answer.json with nonempty strings diagnosis, uncertainty, next_measurement; decision a/b/neither/indeterminate; predictions containing exactly two objects with candidate a or b, latency improve/regress/indeterminate, nonempty execution_change and a nonempty evidence list of file/metric citations. Commit answer.json on your assigned branch; do not merge. Inputs must remain unchanged. This is an offline diagnostic: no live runtime tools are available. Read execution.json for the deadline and reserve the final 30 seconds for the answer and commit.
'''
RUNTIME = ('run_compiler_evidence_ablation.py', 'run_pipe_evidence_diagnostic.py',
           'blind_optimization_launcher.py', 'production_launcher.py')


def safe_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('invalid declared input path')
    path = root / name
    if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('input must be a regular file without symlinks: ' + name)
    return path


def checked_inputs(packets, skill):
    manifest = json.loads((packets / 'manifest.json').read_text())
    tasks = [f'task{i:02}' for i in range(1, 4)]
    if manifest.get('schema_version') != 1 or manifest.get('tasks') != tasks or not isinstance(manifest.get('files'), dict):
        raise ValueError('expected schema 1, three tasks, and files hash map')
    files = [safe_file(packets, 'manifest.json')]
    evidence = manifest.get('evidence', {})
    for task in tasks:
        public = packets / 'public' / task
        declaration = json.loads(safe_file(public, 'task.json').read_text()).get('public_files')
        if not isinstance(declaration, list) or len(declaration) != len(PUBLIC) or set(declaration) != PUBLIC:
            raise ValueError('unexpected public file declaration')
        files.extend(safe_file(public, name) for name in sorted(PUBLIC))
        environment = json.loads((public / 'environment.json').read_text())
        if not all(isinstance(environment.get(k), str) and environment[k].strip() for k in ('soc', 'compiler', 'profiler')):
            raise ValueError('explicit SoC, compiler and profiler metadata required')
        for kind in ('pipes', 'compiler'):
            names = evidence.get(task, {}).get(kind)
            if not isinstance(names, list) or not names or len(names) != len(set(names)):
                raise ValueError('nonempty distinct evidence file list required')
            files.extend(safe_file(packets / kind / task, name) for name in names)
    for path in files[1:]:
        if manifest['files'].get(path.relative_to(packets).as_posix()) != digest(path):
            raise ValueError('packet hash mismatch: ' + str(path))
    skill_files = sorted(skill.rglob('*'))
    if not (skill / 'SKILL.md').is_file() or any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in skill_files):
        raise ValueError('regular frozen skill snapshot required')
    skill_files = [safe_file(skill, p.relative_to(skill).as_posix()) for p in skill_files if p.is_file()]
    return tasks, evidence, files, skill_files


def run(packets, skill, model, output, *, budget=300, forbidden=(), resume=False,
        launcher=DiagnosticLauncher, launcher_kwargs=None):
    packets, skill, output = (Path(p).resolve() for p in (packets, skill, output))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (packets, skill)):
        raise ValueError('output must not overlap inputs')
    if any(output.is_relative_to(Path(p).resolve()) for p in forbidden):
        raise ValueError('output must be outside forbidden workspace')
    if type(budget) is not int or budget <= 0 or not all(isinstance(model.get(k), str) and model[k] for k in ('name', 'reasoning_effort')):
        raise ValueError('explicit model and positive budget required')
    tasks, evidence, files, skill_files = checked_inputs(packets, skill)
    identity = {p.relative_to(packets).as_posix(): digest(p) for p in files}
    skill_identity = {p.relative_to(skill).as_posix(): digest(p) for p in skill_files}
    runtime = {name: digest(Path(__file__).with_name(name)) for name in RUNTIME}
    manifest = {'tasks': tasks, 'arms': list(ARMS), 'budget_seconds': budget,
                'model': model, 'files': identity, 'skill_files': skill_identity,
                'runtime_files': runtime, 'prompt': PROMPT}
    if output.exists():
        if not resume:
            raise ValueError('output exists; inspect or resume undispatched cells')
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('resume inputs differ from frozen manifest')
    else:
        output.mkdir(parents=True, mode=0o700)
        for path in files:
            target = output / 'frozen' / 'packets' / path.relative_to(packets)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copytree(skill, output / 'frozen' / 'skill')
        (output / 'frozen' / 'prompt.md').write_text(PROMPT)
        write_json(output / 'manifest.json', manifest)
    frozen = output / 'frozen'
    if (frozen / 'prompt.md').read_text() != PROMPT:
        raise ValueError('frozen prompt changed')
    for name, sha in identity.items():
        if digest(safe_file(frozen / 'packets', name)) != sha:
            raise ValueError('frozen packet changed')
    skill_items = list((frozen / 'skill').rglob('*'))
    if any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in skill_items):
        raise ValueError('frozen skill file types changed')
    current_skill = {p.relative_to(frozen / 'skill').as_posix(): digest(safe_file(frozen / 'skill', p.relative_to(frozen / 'skill').as_posix()))
                     for p in skill_items if p.is_file()}
    if current_skill != skill_identity:
        raise ValueError('frozen skill changed')

    def cell(pair):
        task, arm = pair
        root = output / f'{task}-{arm}'
        root.mkdir(exist_ok=True)
        bundle, stage = root / 'bundle', root / 'agent'
        result_path = root / 'result.json'
        try:
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['valid']:
                    sealed = seal_result(stage / 'workspace', bundle, result['receipt'])
                    if any(sealed[k] != result[k] for k in sealed):
                        raise ValueError('retained committed answer changed')
                return {'task': task, 'arm': arm, 'complete': result['valid'], 'error': result.get('error')}
            if bundle.exists() or stage.exists():
                raise ValueError('partial cell exists; inspect, never duplicate')
            shutil.copytree(frozen / 'packets' / 'public' / task, bundle)
            names = sorted(PUBLIC)
            for kind in ('pipes', 'compiler'):
                if arm in (kind, 'both'):
                    for name in evidence[task][kind]:
                        target = bundle / kind / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(frozen / 'packets' / kind / task / name, target)
                        names.append(kind + '/' + name)
            declaration = json.loads((bundle / 'task.json').read_text())
            declaration['public_files'] = names
            write_json(bundle / 'task.json', declaration)
            if (frozen / 'prompt.md').read_text() != PROMPT or any(digest(Path(__file__).with_name(n)) != sha for n, sha in runtime.items()):
                raise ValueError('frozen prompt or runtime changed')
            instance = launcher(stage, bundle, frozen / 'prompt.md', model,
                                [sys.executable, str(Path(__file__).with_name('run_pipe_evidence_diagnostic.py')), 'offline'],
                                skill=frozen / 'skill', timeout=budget, forbidden=tuple(forbidden), **(launcher_kwargs or {}))
            receipt = instance.launch()
            result = {'valid': True, 'receipt': receipt, **seal_result(instance.workspace, bundle, receipt)}
        except Exception as error:
            # Preserve partial work and do not turn observation/retry into a second model call.
            result = {'valid': False, 'error': str(error)}
        write_json(result_path, result)
        return {'task': task, 'arm': arm, 'complete': result['valid'], 'error': result.get('error')}

    with ThreadPoolExecutor(max_workers=3) as pool:
        cells = list(pool.map(cell, [(task, arm) for task in tasks for arm in ARMS]))
    result = {'completed': sum(c['complete'] for c in cells), 'cells': cells}
    write_json(output / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packets', 'skill', 'model-json', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--budget-seconds', type=positive_budget, default=300)
    parser.add_argument('--forbidden', type=Path, action='append', default=[])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.packets, args.skill, json.loads(args.model_json.read_text()), args.output,
                 budget=args.budget_seconds, forbidden=args.forbidden, resume=args.resume)
    print(json.dumps(result))
    return 0 if result['completed'] == 12 else 1


if __name__ == '__main__':
    raise SystemExit(main())
