#!/usr/bin/env python3
"""Paired real coding sessions using the existing isolated actor/tool boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics

from blind_optimization_launcher import BlindLauncher, LaunchError, execute, write_json

MEASUREMENT = 'profiled_device_operation_span_us'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skill_files(path):
    path = Path(path).resolve()
    if not (path / 'SKILL.md').is_file() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('regular skill snapshot with SKILL.md required')
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


class CodingLauncher(BlindLauncher):
    extra_managed_options = {"--config"}

    def __init__(self, *args, guidance, max_tool_calls, **kwargs):
        super().__init__(*args, skill=None, **kwargs)
        self.guidance, self.max_tool_calls = guidance, max_tool_calls
        self.tool_calls = 0

    def prepare(self):
        for name, path in self.guidance.items():
            if not re.fullmatch('[a-z][a-z0-9-]*', name):
                raise ValueError('guidance names must be skill directory names')
            skill_files(path)
        super().prepare()
        for name, source in self.guidance.items():
            shutil.copytree(source, self.frozen / 'guidance' / name)

    def base_command(self):
        command = super().base_command()
        for name in sorted(self.guidance):
            command += ['--ro-bind', str(self.frozen / 'guidance' / name),
                        '/workspace/.agents/skills/' + name]
        return command

    def preflight(self):
        checks = ['test -r /codex-home/auth.json', 'test ! -e /codex-home/skills',
                  'test ! -e /codex-home/plugins', 'command -v experiment-tool']
        checks += ['test -r /workspace/.agents/skills/' + name + '/SKILL.md'
                   for name in self.guidance]
        result = execute([*self.base_command(), 'sh', '-ceu', ';'.join(checks)], self.root, 30)
        write_json(self.root / 'isolation.json', result)
        if result['returncode']:
            raise LaunchError('coding isolation preflight failed: ' + result['stderr'])
        return result

    def agent_command(self):
        command = super().agent_command()
        extra = ['--ephemeral', '--skip-git-repo-check', '-c', 'web_search="disabled"']
        for feature in ('apps', 'plugins', 'browser_use', 'computer_use', 'image_generation',
                        'multi_agent', 'hooks'):
            extra += ['--disable', feature]
        return [*command[:-1], *extra, command[-1]]

    def after_agent(self, result):
        result.update(tool_calls=self.tool_calls, max_tool_calls=self.max_tool_calls)
        usage = []
        for line in result['stdout'].splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get('type') == 'turn.completed':
                usage.append(event.get('usage', {}))
        result['token_usage'] = usage
        return result


def adjudicate(report, task, candidate_sha256):
    if (report.get('schema') != 'optimization-coding-evaluation-v1'
            or report.get('candidate_sha256') != candidate_sha256
            or report.get('measurement_kind') != MEASUREMENT):
        raise ValueError('trusted evaluation identity or measurement boundary mismatch')
    rows = report.get('cases')
    expected = task['expected_case_ids']
    if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows) or len(rows) != len(expected)
            or {r.get('id') for r in rows} != set(expected)):
        raise ValueError('terminal evaluation must retain every expected case exactly once')
    if report.get('outcome') == 'infrastructure_failure':
        return dict(solved=None, comparison_valid=False, outcome='infrastructure_failure', cases=rows)
    if report.get('outcome') not in (None, 'success', 'candidate_failure'):
        raise ValueError('unknown trusted terminal outcome')
    ratios = {}
    for row in rows:
        if (report.get('outcome') == 'candidate_failure' or row.get('correct') is not True
                or row.get('identity_verified') is not True):
            return dict(solved=False, comparison_valid=True, outcome='candidate_failure',
                        reason='correctness or identity failure', cases=rows)
        for field in ('baseline_samples_us', 'candidate_samples_us'):
            values = row.get(field)
            if (not isinstance(values, list) or len(values) != 10 or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
                raise ValueError('exactly ten finite positive samples per implementation required')
        ratios[row['id']] = (statistics.mean(row['baseline_samples_us']) /
                             statistics.mean(row['candidate_samples_us']))
    geomean = math.exp(statistics.mean(math.log(ratios[key]) for key in
                    task.get('performance_case_ids', expected)))
    all_geomean = math.exp(statistics.mean(math.log(ratio) for ratio in ratios.values()))
    solved = (geomean >= task['minimum_geomean_speedup']
              and min(ratios.values()) >= 1 / task['maximum_case_regression'])
    return dict(solved=solved, comparison_valid=True, outcome='measured',
                reason='performance gate passed' if solved else 'performance gate failed',
                geomean_speedup=geomean, primary_geomean_speedup=geomean,
                all_case_geomean_speedup=all_geomean, minimum_case_speedup=min(ratios.values()),
                case_speedups=ratios, cases=rows)


def freeze(manifest, output):
    """Reserve once and freeze public files/guidance before any actor dispatch."""
    if (type(manifest.get('budget_seconds')) is not int or manifest['budget_seconds'] <= 0
            or type(manifest.get('max_tool_calls')) is not int or manifest['max_tool_calls'] <= 0):
        raise ValueError('positive identical wall and tool-call budgets required')
    tasks = manifest['tasks']
    if any(not isinstance(manifest['model'].get(k), str) or not manifest['model'][k]
           for k in ('name', 'reasoning_effort')):
        raise ValueError('explicit shared model configuration required')
    if not tasks or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('distinct nonempty task suite required')
    common, library = manifest['common_skills'], manifest['library_skill']
    if library['name'] in common:
        raise ValueError('treatment and common guidance must differ')
    skills = {**common, library['name']: library['path']}
    for name, path in skills.items():
        if not re.fullmatch('[a-z][a-z0-9-]*', name):
            raise ValueError('invalid skill name')
        skill_files(path)
    for task in tasks:
        source = Path(task['bundle']).resolve()
        public = json.loads((source / 'task.json').read_text())['public_files']
        if (not isinstance(public, list) or 'candidate.py' not in public
                or not Path(task['prompt']).is_file()):
            raise ValueError('public candidate.py and prompt required')
        for name in public:
            if not isinstance(name, str):
                raise ValueError('public filenames must be strings')
            relative = Path(name)
            path = source / relative
            if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.is_file():
                raise ValueError('regular declared public file required')
        for key in ('minimum_geomean_speedup', 'maximum_case_regression'):
            if type(task[key]) not in (int, float) or not math.isfinite(task[key]) or task[key] < 1:
                raise ValueError('finite predeclared performance thresholds >=1 required')
        if not task['expected_case_ids'] or len(set(task['expected_case_ids'])) != len(task['expected_case_ids']):
            raise ValueError('distinct terminal case identities required')
        primary = task.get('performance_case_ids', task['expected_case_ids'])
        if (not isinstance(primary, list) or not primary or len(set(primary)) != len(primary)
                or not set(primary).issubset(task['expected_case_ids'])):
            raise ValueError('distinct nonempty performance cases must be expected cases')
        for key in ('tool_command', 'evaluator_command'):
            if not task[key] or not all(isinstance(arg, str) and arg for arg in task[key]):
                raise ValueError('trusted command argv required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'original-manifest.json', manifest)
    frozen = json.loads(json.dumps(manifest))
    for name, source in skills.items():
        destination = output / 'frozen' / 'guidance' / name
        shutil.copytree(source, destination)
        if name in common:
            frozen['common_skills'][name] = str(destination)
        else:
            frozen['library_skill']['path'] = str(destination)
    for index, task in enumerate(frozen['tasks']):
        destination = output / 'frozen' / f'task-{index:03d}'
        destination.mkdir(parents=True)
        source = Path(task['bundle']).resolve()
        public = json.loads((source / 'task.json').read_text())['public_files']
        for name in public:
            relative = Path(name)
            path = source / relative
            if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.is_file():
                raise ValueError('regular declared public file required')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        # The launcher reads the public allowlist; private bundle contents never enter it.
        shutil.copyfile(source / 'task.json', destination / 'task.json')
        task['bundle'] = str(destination)
        prompt = destination / 'prompt.md'
        prompt.write_text(Path(task['prompt']).read_text())
        task['prompt'] = str(prompt)
    inventory = {str(p.relative_to(output)): sha(p) for p in (output / 'frozen').rglob('*') if p.is_file()}
    write_json(output / 'input-digests.json', inventory)
    write_json(output / 'frozen-manifest.json', frozen)
    return frozen


def evaluate(command, request, session):
    result = execute(command, session, 600,
                     (session / 'evaluation.stdout', session / 'evaluation.stderr'),
                     input_text=json.dumps(request))
    if result['returncode'] or result['timed_out']:
        raise RuntimeError('trusted terminal evaluator failed; diagnostics retained')
    return json.loads(result['stdout'])


def development_failures(session, tool_command=None):
    """Inspect only host-owned dispatched reports; help and budget refusals are free."""
    journal = session / 'commands.jsonl'
    failures = []
    for index, line in enumerate(journal.read_text().splitlines() if journal.exists() else []):
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError('invalid host journal row')
            if not entry.get('dispatched'):
                continue
            arguments = entry.get('arguments')
            command = entry.get('command', [])
            if arguments is None and tool_command is not None:
                prefix = [*tool_command, '--workspace']
                if command[:len(prefix)] == prefix and len(command) > len(prefix):
                    arguments = command[len(prefix) + 1:]
            is_help = (isinstance(arguments, list) and all(isinstance(a, str) for a in arguments)
                       and (arguments[:1] == ['help'] or any(a in ('-h', '--help') for a in arguments)))
            if is_help and entry.get('returncode') == 0 and not entry.get('timed_out'):
                continue
            report = json.loads(entry['stdout'])
            if (not isinstance(report, dict) or report.get('outcome') not in
                    ('success', 'candidate_failure') or entry.get('timed_out')):
                raise ValueError('infrastructure or unknown backend outcome')
        except (ValueError, KeyError, TypeError) as error:
            failures.append(dict(journal_row=index + 1, reason=str(error)))
    return failures


def run_suite(manifest, output, *, launcher=CodingLauncher, evaluator=evaluate):
    output = Path(output).resolve()
    frozen = freeze(manifest, output)
    jobs = [(index, condition) for index in range(len(frozen['tasks'])) for condition in (False, True)]
    random.Random(frozen.get('seed', 0)).shuffle(jobs)
    results = []
    for index, condition in jobs:
        task = frozen['tasks'][index]
        session = output / f'session-{len(results):03d}'
        receipt = dict(task_id=task['id'], with_library=condition, status='dispatch-intent')
        # Reserve dispatch intent outside the launcher's new session directory.
        receipt_path = output / f'{session.name}-receipt.json'
        write_json(receipt_path, receipt)
        try:
            guidance = dict(frozen['common_skills'])
            if condition:
                guidance[frozen['library_skill']['name']] = frozen['library_skill']['path']
            actor = launcher(session, Path(task['bundle']), Path(task['prompt']), frozen['model'],
                             task['tool_command'], guidance=guidance,
                             max_tool_calls=frozen['max_tool_calls'], timeout=frozen['budget_seconds'])
            receipt['actor'] = actor.launch()
            candidate = session / 'workspace/candidate.py'
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError('final candidate must be a regular candidate.py')
            source_hash = sha(candidate)
            submitted = session / 'submitted-candidate.py'
            shutil.copyfile(candidate, submitted)
            request = dict(action='evaluate', phase='terminal', task_id=task['id'],
                           candidate_path=str(submitted), candidate_sha256=source_hash,
                           output_directory=str(session / 'terminal-evidence'))
            raw = evaluator(task['evaluator_command'], request, session)
            if sha(submitted) != source_hash:
                raise ValueError('submitted candidate changed during terminal evaluation')
            write_json(session / 'terminal-report.json', raw)
            result = adjudicate(raw, task, source_hash)
            issues = development_failures(session, task['tool_command'])
            # A harness-enforced wall expiry submits the last candidate by protocol.
            expired = receipt['actor'].get('timed_out') is True
            actor_failed = bool(receipt['actor'].get('returncode')) and not expired
            receipt.update(status='complete' if result['comparison_valid'] and not issues and not actor_failed
                           else 'invalid-comparison',
                           actor_budget_expiry_eligible=expired, actor_infrastructure_failure=actor_failed,
                           candidate_sha256=source_hash, terminal_result=result, result=dict(result),
                           development_infrastructure_failures=issues)
        except Exception as error:
            receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(receipt_path, receipt)
        results.append(receipt)
    invalid = {r['task_id'] for r in results if r['status'] != 'complete'}
    for position, receipt in enumerate(results):
        receipt['paired_comparison_valid'] = receipt['task_id'] not in invalid
        if not receipt['paired_comparison_valid']:
            if receipt['status'] == 'complete':
                receipt['status'] = 'invalid-comparison'
            receipt.setdefault('result', {}).update(solved=None, comparison_valid=False,
                                                    reason='paired infrastructure failure')
        write_json(output / f'session-{position:03d}-receipt.json', receipt)
    write_json(output / 'results.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = run_suite(json.loads(args.manifest.read_text()), args.output)
    return int(any(row['status'] != 'complete' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
