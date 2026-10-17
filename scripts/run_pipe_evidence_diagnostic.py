#!/usr/bin/env python3
"""Paired supplied-evidence diagnostic: six tasks, three arms, two fresh contexts."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind_optimization_launcher import BlindLauncher, positive_budget, write_json

ARMS = ('timing', 'pipes', 'pipes-skill')
PUBLIC = {'task.json', 'TASK.md', 'baseline.py', 'candidate-a.py', 'candidate-b.py', 'timing.json'}
PIPES = {'pipe-manifest.json', 'pipe-0.csv', 'pipe-1.csv', 'pipe-2.csv'}
COMMON = '''Assess the supplied Triton baseline and two proposed candidate changes without editing source or executing kernels. Read TASK.md, baseline.py, candidate-a.py, candidate-b.py and timing.json. Use available evidence and guidance as you find useful. Give an evidence-supported diagnosis, choose which candidate to test next (a/b/neither/indeterminate), predict both candidates' latency direction, and explain uncertainty and a discriminating next measurement. Do not assume a bottleneck that evidence cannot establish.
Write answer.json with nonempty strings diagnosis, uncertainty, next_measurement; decision a/b/neither/indeterminate; predictions containing exactly two objects with candidate a or b, latency improve/regress/indeterminate, nonempty execution_change and a nonempty evidence list of file/metric citations. Commit answer.json on your assigned branch; do not merge. Inputs must remain unchanged. This is an offline diagnostic: no live runtime tools are available. Reserve the final 30 seconds for the answer and commit; execution.json records your deadline.
'''
BEFORE = COMMON + 'This is the initial assessment. Only currently supplied evidence is available.\n'
AFTER = COMMON + 'Read your sealed initial answer in before-answer.json. Reassess it using the currently supplied evidence; explicitly explain any changed or unchanged conclusions.\n'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_answer(answer):
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    if not isinstance(answer, dict) or not all(text(answer.get(k)) for k in ('diagnosis', 'uncertainty', 'next_measurement')):
        raise ValueError('incomplete answer text')
    if answer.get('decision') not in ('a', 'b', 'neither', 'indeterminate'):
        raise ValueError('invalid decision')
    predictions = answer.get('predictions')
    if not isinstance(predictions, list) or len(predictions) != 2 or any(not isinstance(p, dict) for p in predictions):
        raise ValueError('two predictions required')
    if sorted(p.get('candidate', '') for p in predictions) != ['a', 'b']:
        raise ValueError('predict each candidate once')
    for p in predictions:
        if p.get('latency') not in ('improve', 'regress', 'indeterminate') or not text(p.get('execution_change')):
            raise ValueError('invalid prediction')
        if not isinstance(p.get('evidence'), list) or not p['evidence'] or not all(text(v) for v in p['evidence']):
            raise ValueError('prediction citations required')


class DiagnosticLauncher(BlindLauncher):
    def base_command(self):
        command = super().base_command()
        # BlindLauncher normally leaves candidate.* writable for optimization.
        # Here candidate proposals are evidence, not editable implementations.
        files = json.loads((self.bundle / 'task.json').read_text())['public_files']
        mounts = []
        for name in files:
            mounts += ['--ro-bind', str(self.workspace / name), '/workspace/' + name]
        return command + mounts


def checked_files(packets, skill):
    packet_manifest = json.loads((packets / 'manifest.json').read_text())
    tasks = packet_manifest['tasks']
    if tasks != [f'task{i:02}' for i in range(1, 7)]:
        raise ValueError('expected task01 through task06')
    files = [packets / 'manifest.json']
    for task in tasks:
        public = packets / 'public' / task
        declaration = json.loads((public / 'task.json').read_text())['public_files']
        if not isinstance(declaration, list) or len(declaration) != len(PUBLIC) or set(declaration) != PUBLIC:
            raise ValueError('unexpected public files')
        files += [public / n for n in sorted(PUBLIC)]
        files += [packets / 'pipes' / task / n for n in sorted(PIPES)]
    if not (skill / 'SKILL.md').is_file():
        raise ValueError('skill snapshot missing')
    skill_files = list(skill.rglob('*'))
    for path in [*files, *skill_files]:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('symlink inputs unsupported')
        if path in files and not path.is_file():
            raise ValueError('missing packet file: ' + str(path))
    if 'files' in packet_manifest:
        for path in files[1:]:
            if packet_manifest['files'].get(str(path.relative_to(packets))) != digest(path):
                raise ValueError('packet differs from exporter manifest: ' + str(path))
    return tasks, files, [p for p in skill_files if p.is_file()]


def seal_result(workspace, bundle, receipt):
    if receipt.get('exit_code') or receipt.get('timed_out'):
        raise ValueError('agent failed or timed out; partial work retained')
    for name in json.loads((bundle / 'task.json').read_text())['public_files']:
        if digest(workspace / name) != digest(bundle / name):
            raise ValueError('source evidence changed: ' + name)
    data = (workspace / 'answer.json').read_bytes()
    validate_answer(json.loads(data))
    committed = subprocess.check_output(['git', '-C', str(workspace), 'show', 'HEAD:answer.json'], stderr=subprocess.PIPE)
    if data != committed:
        raise ValueError('answer.json must match committed bytes')
    return {'answer_sha256': hashlib.sha256(data).hexdigest(),
            'commit': subprocess.check_output(['git', '-C', str(workspace), 'rev-parse', 'HEAD'], text=True).strip()}


def run(packets, skill, model, output, *, budget=240, forbidden=(), launcher=DiagnosticLauncher,
        launcher_kwargs=None, resume=False):
    packets, skill, output = (Path(p).resolve() for p in (packets, skill, output))
    if type(budget) is not int or budget <= 0 or not all(isinstance(model.get(k), str) and model[k] for k in ('name', 'reasoning_effort')):
        raise ValueError('explicit model and positive budget required')
    tasks, files, skill_files = checked_files(packets, skill)
    identities = {str(p.relative_to(packets)): digest(p) for p in files}
    skill_identity = {str(p.relative_to(skill)): digest(p) for p in skill_files}
    manifest = {'tasks': tasks, 'arms': ARMS, 'budget_seconds': budget, 'model': model,
                'files': identities, 'skill_files': skill_identity,
                'prompts': {'before': hashlib.sha256(BEFORE.encode()).hexdigest(), 'after': hashlib.sha256(AFTER.encode()).hexdigest()},
                'runtime_files': {name: digest(Path(__file__).with_name(name)) for name in
                                  ('run_pipe_evidence_diagnostic.py', 'blind_optimization_launcher.py', 'production_launcher.py')}}
    manifest = json.loads(json.dumps(manifest))
    if output.exists():
        if not resume:
            raise ValueError('output already exists; inspect results or explicitly resume undispatched boundaries')
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('resume inputs differ from frozen manifest')
    else:
        if any(output.is_relative_to(Path(p).resolve()) for p in forbidden):
            raise ValueError('output must be outside forbidden workspace')
        output.mkdir(parents=True, mode=0o700)
        frozen = output / 'frozen'
        frozen.mkdir()
        for path in files:
            dest = frozen / 'packets' / path.relative_to(packets)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        shutil.copytree(skill, frozen / 'skill')
        (frozen / 'before.md').write_text(BEFORE)
        (frozen / 'after.md').write_text(AFTER)
        write_json(output / 'manifest.json', manifest)
    frozen = output / 'frozen'
    for name, sha in identities.items():
        if digest(frozen / 'packets' / name) != sha:
            raise ValueError('frozen packet changed')
    for name, sha in skill_identity.items():
        if digest(frozen / 'skill' / name) != sha:
            raise ValueError('frozen skill changed')

    def verify_runtime():
        for name, sha in manifest['runtime_files'].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError('runtime source changed before dispatch: ' + name)
        for name, content in [('before.md', BEFORE), ('after.md', AFTER)]:
            if (frozen / name).read_text() != content:
                raise ValueError('frozen prompt changed')

    def trajectory(cell):
        task, arm = cell
        root = output / f'{task}-{arm}'
        root.mkdir(exist_ok=True)
        previous = None
        for stage in (1, 2):
            result_path = root / f'stage{stage}-result.json'
            stage_root = root / f'stage{stage}'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            elif stage_root.exists():
                return {'task': task, 'arm': arm, 'complete': False, 'error': 'stage exists without terminal result; inspect, never duplicate'}
            else:
                bundle = root / f'stage{stage}-bundle'
                if bundle.exists():
                    return {'task': task, 'arm': arm, 'complete': False, 'error': 'partial boundary; inspect rather than replace'}
                shutil.copytree(frozen / 'packets' / 'public' / task, bundle)
                names = list(PUBLIC)
                if stage == 2:
                    shutil.copy2(previous, bundle / 'before-answer.json')
                    names.append('before-answer.json')
                    if arm != 'timing':
                        for name in sorted(PIPES):
                            shutil.copy2(frozen / 'packets' / 'pipes' / task / name, bundle / name)
                            names.append(name)
                declaration = json.loads((bundle / 'task.json').read_text())
                declaration['public_files'] = sorted(names)
                write_json(bundle / 'task.json', declaration)
                try:
                    verify_runtime()
                    instance = launcher(stage_root, bundle, frozen / ('before.md' if stage == 1 else 'after.md'), model,
                                        [sys.executable, str(Path(__file__).resolve()), 'offline'],
                                        skill=frozen / 'skill' if stage == 2 and arm == 'pipes-skill' else None,
                                        timeout=budget, forbidden=tuple(forbidden), **(launcher_kwargs or {}))
                    receipt = instance.launch()
                    result = {'valid': True, 'receipt': receipt, **seal_result(instance.workspace, bundle, receipt)}
                except Exception as error:
                    result = {'valid': False, 'error': str(error)}
                write_json(result_path, result)
            if not result['valid']:
                return {'task': task, 'arm': arm, 'complete': False, 'error': result['error']}
            previous = stage_root / 'workspace' / 'answer.json'
            if digest(previous) != result['answer_sha256']:
                return {'task': task, 'arm': arm, 'complete': False, 'error': 'sealed answer changed'}
        return {'task': task, 'arm': arm, 'complete': True}

    with ThreadPoolExecutor(max_workers=3) as pool:
        cells = list(pool.map(trajectory, [(task, arm) for task in tasks for arm in ARMS]))
    result = {'completed': sum(c['complete'] for c in cells), 'cells': cells}
    write_json(output / 'results.json', result)
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'offline':
        if sys.argv[-1] in ('help', '--help', '-h'):
            print('Offline supplied-evidence assessment. Read task and available evidence files; write and commit answer.json. Live execution is unavailable.')
            return 0
        print('Live execution unavailable: supplied-evidence diagnostic; no kernel, profiling, or remote operations are dispatched.', file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packets', 'skill', 'model-json', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--budget-seconds', type=positive_budget, default=240)
    parser.add_argument('--forbidden', type=Path, action='append', default=[])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.packets, args.skill, json.loads(args.model_json.read_text()), args.output,
                 budget=args.budget_seconds, forbidden=args.forbidden, resume=args.resume)
    print(json.dumps(result))
    return 0 if result['completed'] == 18 else 1


if __name__ == '__main__':
    raise SystemExit(main())
