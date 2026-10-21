#!/usr/bin/env python3
"""Opt-in alternating optimization roles using the existing isolated launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from blind_optimization_launcher import BlindLauncher, LaunchError, write_json
from actor_analyzer_contract import build_prompt, validate_analysis


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_hash(workspace):
    path = workspace / 'candidate.py'
    return digest(path) if path.is_file() and not path.is_symlink() else None


def save(path, value):
    temporary = path.with_suffix('.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def correct(result):
    records = result.get('correctness')
    return result.get('state') == 'completed' and (records is True or
        isinstance(records, list) and bool(records) and all(
            isinstance(item, dict) and item.get('correctness') is True for item in records))


def managed_observation(operation, handle):
    client = (Path.home() / '.agents/skills/remote-access/scripts/cpl-remote').resolve(strict=True)
    result = subprocess.run([str(client), '--json', operation, handle],
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)


def cleanup_phase(phase, remote=managed_observation):
    """Stop only this phase's work; uncertainty never permits a new dispatch."""
    ownership = phase / 'owned-tools.json'
    if not ownership.exists():
        return {'state': 'terminal', 'calls': []}
    config = json.loads(ownership.read_text())
    calls, uncertain = [], False
    for path in sorted(Path(config['directory']).glob('*/receipt.json')):
        if path.parent.name in config['existing']:
            continue
        try:
            receipt = json.loads(path.read_text())
            if receipt.get('state') in ('completed', 'failed', 'cancelled'):
                continue
            handle = receipt.get('handle') or receipt.get('upload', {}).get('handle')
            if not handle:
                if receipt.get('dispatch_intent'):
                    uncertain = True
                    calls.append({'call_id': path.parent.name, 'state': 'handle-unknown'})
                    continue
                # No workload dispatch was attempted; abort the prepared call.
                result = {'state': 'failed', 'error': 'phase ended before dispatch'}
            else:
                result = remote('observe', handle)
                if result.get('state') not in ('completed', 'failed', 'cancelled'):
                    result = remote('cancel', handle)
                calls.append({'handle': handle, 'result': result})
            if result.get('state') in ('completed', 'failed', 'cancelled'):
                receipt['state'] = result['state']
                save(path, receipt)
            else:
                uncertain = True
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            uncertain = True
            calls.append({'call_id': path.parent.name, 'error': str(error)})
    result = {'state': 'observation-required' if uncertain else 'terminal', 'calls': calls}
    save(phase / 'cleanup.json', result)
    return result


def bridge(config_path, arguments):
    """Keep managed Tools receipts and source identity outside the agent mount."""
    config = json.loads(config_path.read_text())
    workspace, phase = Path(config['workspace']), config_path.parent
    pending_calls = []
    for candidate in Path(config['campaign']).glob('**/tool-state/*/receipt.json'):
        pending = json.loads(candidate.read_text())
        pending_calls.append((candidate, pending))
    # Tools already journals dispatch intent before transport. Never start another
    # remote workload while an earlier operation needs observation/recovery.
    if arguments and arguments[0] not in ('help', 'observe'):
        for receipt, pending in pending_calls:
            if pending.get('state') not in ('completed', 'failed', 'cancelled'):
                print(json.dumps({'state': 'observation-required', 'call_id': pending['call_id'],
                                  'handle': pending.get('handle'),
                                  'next': 'experiment-tool observe ' + pending['call_id']}))
                return 75
    if arguments[:1] == ['observe'] and len(arguments) > 1:
        for receipt, pending in pending_calls:
            if pending.get('call_id') == arguments[1]:
                parent = receipt.parent.parent.parent
                workspace = parent / ('workspace' if parent == Path(config['campaign']) else 'candidate-snapshot')
                break
    before = candidate_hash(workspace)
    result = subprocess.run([*config['command'], '--workspace', str(workspace), *arguments],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    try:
        payload = json.loads(result.stdout)
        if before and correct(payload) and payload.get('source_sha256') == before == candidate_hash(workspace):
            with (phase / 'qualifications.jsonl').open('a') as stream:
                stream.write(json.dumps(payload) + '\n')
    except ValueError:
        pass
    return result.returncode


class RoleLauncher(BlindLauncher):
    def __init__(self, campaign, phase, role, budget, commit):
        self.campaign, self.role, self.commit = campaign, role, commit
        prompt = phase.parent / (phase.name + '.prompt.md')
        prompt.write_text(build_prompt(role, budget, int(phase.name.split('-')[1]),
                                      campaign.instruction, candidate_commit=commit))
        super().__init__(phase, campaign.bundle, prompt, campaign.model,
                         ['unused'], skill=campaign.skill, timeout=budget,
                         auth_home=campaign.auth_home)

    def prepare(self):
        super().prepare()
        # Bootstrap is the launcher's own newly-created directory, never a user
        # worktree. Both roles instead see one persistent experiment history.
        shutil.rmtree(self.workspace)
        self.workspace = self.campaign.workspace
        if self.role == 'analyzer':
            self.workspace = self.root / 'candidate-snapshot'
            subprocess.run(['git', 'clone', '--no-hardlinks', str(self.campaign.workspace),
                            str(self.workspace)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(self.workspace), 'checkout', '--detach', self.commit],
                           check=True, capture_output=True)
        config = self.root / 'tool-config.json'
        save(config, {'workspace': str(self.workspace), 'command': self.campaign.tool_command,
                      'campaign': str(self.campaign.root)})
        self.tool_command = [sys.executable, str(Path(__file__).resolve()), 'tool', str(config)]
        self.reports = self.root / 'reports'
        self.reports.mkdir()
        directory = self.workspace.parent / 'tool-state'
        save(self.root / 'owned-tools.json', {'directory': str(directory),
                                              'existing': [p.name for p in directory.glob('*')]})

    def launch(self, preflight_only=False):
        done, cleanup_lock = threading.Event(), threading.Lock()
        def watchdog():
            while not done.wait(.1):
                deadline = getattr(self, 'deadline_monotonic', None)
                if deadline is not None and time.monotonic() >= deadline:
                    with cleanup_lock:
                        cleanup_phase(self.root)
                    return
        observer = threading.Thread(target=watchdog, daemon=True)
        observer.start()
        try:
            return super().launch(preflight_only)
        finally:
            done.set()
            observer.join()
            with cleanup_lock:
                if self.root.exists():
                    cleanup_phase(self.root)

    def base_command(self):
        command = super().base_command()
        retained, index = [], 0
        while index < len(command):
            if command[index] == '--ro-bind' and index + 2 < len(command):
                destination = command[index + 2]
                relative = destination.removeprefix('/workspace/')
                if destination.startswith('/workspace/') and relative not in self.campaign.public_files \
                        and not relative.startswith('.agents/'):
                    index += 3
                    continue
            if command[index:index + 3] == ['--bind', str(self.workspace), '/workspace'] \
                    and self.role == 'analyzer':
                retained += ['--ro-bind', str(self.workspace), '/workspace']
                index += 3
                continue
            retained.append(command[index])
            index += 1
        # Both roles can record work without mutating immutable public API files.
        return retained + ['--bind', str(self.reports), '/analysis']


class Campaign:
    def __init__(self, output, bundle, model, tool_command, instruction, *, skill=None,
                 rounds=4, actor_budget=1200, analyzer_budget=600, actor_only=False,
                 auth_home=None, runner=None):
        if min(rounds, actor_budget, analyzer_budget) <= 0:
            raise ValueError('rounds and budgets must be positive')
        self.root, self.bundle = Path(output).resolve(), Path(bundle).resolve()
        self.workspace = self.root / 'workspace'
        self.model, self.tool_command, self.instruction = model, tool_command, instruction
        self.skill, self.auth_home = skill, auth_home
        self.rounds, self.actor_budget, self.analyzer_budget = rounds, actor_budget, analyzer_budget
        self.actor_only, self.runner = actor_only, runner or self.launch_role
        self.public_files = json.loads((self.bundle / 'task.json').read_text())['public_files']
        if not isinstance(self.public_files, list) or 'candidate.py' not in self.public_files:
            raise ValueError('public_files must include candidate.py')
        for name in self.public_files:
            source = self.bundle / name
            if Path(name).is_absolute() or '..' in Path(name).parts or source.is_symlink() \
                    or not source.is_file() or not source.resolve().is_relative_to(self.bundle):
                raise ValueError('invalid public bundle path')
        # candidate is the only mutable supplied source; newly created logs remain writable.
        self.public_files = [name for name in self.public_files if name != 'candidate.py']
        self.config = dict(bundle=str(self.bundle), model=model, tool_command=tool_command,
                           instruction=instruction, rounds=rounds, actor_budget=actor_budget,
                           analyzer_budget=analyzer_budget, actor_only=actor_only,
                           skill=str(skill) if skill else None)

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.workspace), *arguments], text=True)

    def prepare(self):
        if (self.root / 'state.json').exists():
            state = json.loads((self.root / 'state.json').read_text())
            if state['configuration'] != self.config:
                raise LaunchError('resume configuration changed; preserve the original campaign')
            return state
        if self.root.exists():
            raise LaunchError('unrecognized existing campaign output')
        self.workspace.mkdir(parents=True)
        for name in ['candidate.py', *self.public_files]:
            source = self.bundle / name
            if Path(name).is_absolute() or '..' in Path(name).parts or source.is_symlink():
                raise ValueError('invalid public bundle path')
            destination = self.workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        self.git('init', '-b', 'experiment/actor-analyzer')
        self.git('config', 'user.name', 'Experiment Agent')
        self.git('config', 'user.email', 'experiment@localhost')
        (self.workspace / '.git/info/exclude').write_text('execution.json\n.agents/\n*.tar.gz\n')
        self.commit('Starting implementation')
        state = {'state': 'running', 'phases': {}, 'verified': None, 'configuration': self.config,
                 'configured_actor_seconds': self.rounds * self.actor_budget,
                 'configured_analyzer_seconds': 0 if self.actor_only else self.rounds * self.analyzer_budget}
        save(self.root / 'state.json', state)
        return state

    def commit(self, message):
        self.git('add', '--all')
        self.git('commit', '--allow-empty', '-m', message)
        return self.git('rev-parse', 'HEAD').strip()

    def qualify(self, phase, result):
        """Trusted tool integration helper, never exposed inside the sandbox."""
        if candidate_hash(self.workspace) and correct(result) \
                and result.get('source_sha256') == candidate_hash(self.workspace):
            with (phase / 'qualifications.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')

    def launch_role(self, campaign, phase, role, budget, commit):
        return RoleLauncher(self, phase, role, budget, commit).launch()

    def finish_phase(self, phase, role, entry, state):
        archive = self.workspace / 'experiments' / phase.name
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('commands.jsonl', 'receipt.json', 'agent.jsonl', 'agent.stderr',
                     'qualifications.jsonl', 'cleanup.json'):
            if (phase / name).is_file():
                shutil.copyfile(phase / name, archive / name)
        if role == 'actor':
            commit = self.commit(phase.name + ': candidate and execution evidence')
            if (phase / 'qualifications.jsonl').exists():
                records = [json.loads(line) for line in (phase / 'qualifications.jsonl').read_text().splitlines()]
                if any(correct(item) and item.get('source_sha256') is not None
                       and item.get('source_sha256') == candidate_hash(self.workspace)
                       for item in records):
                    state['verified'] = commit
            entry['candidate_commit'] = commit
        else:
            report_path = phase / 'reports/analysis.json'
            if not report_path.exists():
                report_path = phase / 'analysis.json'
            trusted = phase / 'candidate-snapshot/evidence'
            if trusted.exists():
                shutil.copytree(trusted, phase / 'reports/evidence', dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('*.tar.gz'))
            if (phase / 'reports').exists():
                shutil.copytree(phase / 'reports', archive / 'analysis', dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('*.tar.gz'))
            elif report_path.exists():
                shutil.copyfile(report_path, archive / 'raw-analysis.json')
            try:
                report = validate_analysis(json.loads(report_path.read_text()), entry['candidate_commit'])
                verified_source = subprocess.check_output(['git', '-C', str(self.workspace), 'show',
                                                           entry['candidate_commit'] + ':candidate.py'])
                expected_source = hashlib.sha256(verified_source).hexdigest()
                entry['evidence_sha256'] = {}
                for evidence in report['evidence']:
                    path = phase / 'reports' / evidence['path']
                    original = phase / 'candidate-snapshot' / evidence['path']
                    if not path.is_file() or path.is_symlink() or not original.is_file() \
                            or digest(path) != digest(original):
                        raise ValueError('analysis cites missing evidence: ' + evidence['path'])
                    entry['evidence_sha256'][evidence['path']] = digest(path)
                    if evidence['kind'] == 'profile':
                        summary = json.loads(path.read_text())
                        if summary.get('source_sha256') != expected_source or not correct(summary) \
                                or not any(item.get('pipe_rows') for item in summary.get('captures', [])):
                            raise ValueError('profile citation lacks acquired candidate pipe evidence')
                shutil.copyfile(report_path, self.workspace / 'analysis.json')
                entry['analysis_valid'] = True
            except (ValueError, OSError) as error:
                entry.update(analysis_valid=False, analysis_error=str(error))
            self.commit(phase.name + ': performance analysis and diagnostics')

    def run(self):
        self.root.parent.mkdir(parents=True, exist_ok=True)
        # The kernel releases this lock on interruption; persistent phase intents
        # distinguish observation from permission to dispatch another agent.
        import fcntl
        with self.root.with_name(self.root.name + '.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self.prepare()
            if state['state'] == 'completed':
                return state
            for index in range(1, self.rounds + 1):
                for role in (('actor',) if self.actor_only else ('actor', 'analyzer')):
                    name = f'round-{index}-{role}'
                    phase = self.root / name
                    entry = state['phases'].get(name)
                    if entry:
                        if entry['state'] != 'dispatch-intent':
                            if entry.get('cleanup', {}).get('state') == 'observation-required':
                                entry['cleanup'] = cleanup_phase(phase)
                                save(self.root / 'state.json', state)
                                if entry['cleanup']['state'] != 'terminal':
                                    return {**state, 'state': 'observation-required', 'pending_phase': name}
                            continue
                        if not (phase / 'receipt.json').exists():
                            return {**state, 'state': 'observation-required', 'pending_phase': name}
                        result = json.loads((phase / 'receipt.json').read_text())
                    else:
                        commit = state['verified'] if role == 'analyzer' else None
                        if role == 'analyzer' and not commit:
                            state['phases'][name] = {'state': 'skipped-no-verified-candidate'}
                            save(self.root / 'state.json', state)
                            continue
                        entry = {'state': 'dispatch-intent', 'candidate_commit': commit}
                        state['phases'][name] = entry
                        save(self.root / 'state.json', state)
                        # Runner creates its own phase root; test runners use it directly.
                        if self.runner != self.launch_role:
                            phase.mkdir()
                        budget = self.actor_budget if role == 'actor' else self.analyzer_budget
                        result = self.runner(self, phase, role, budget, commit)
                        save(phase / 'receipt.json', result)
                    entry['cleanup'] = cleanup_phase(phase)
                    self.finish_phase(phase, role, entry, state)
                    entry.update(state='terminal', result=result)
                    save(self.root / 'state.json', state)
                    if entry['cleanup']['state'] != 'terminal':
                        return {**state, 'state': 'observation-required', 'pending_phase': name}
            state['state'] = 'completed'
            save(self.root / 'state.json', state)
            return state


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'tool':
        # ToolHandler supplies --workspace; the bridge owns the exact workspace.
        arguments = sys.argv[3:]
        if arguments[:1] == ['--workspace']:
            arguments = arguments[2:]
        return bridge(Path(sys.argv[2]), arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'bundle', 'model-json', 'tool-command-json', 'instruction'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--skill', type=Path)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--actor-budget', type=int, default=1200)
    parser.add_argument('--analyzer-budget', type=int, default=600)
    parser.add_argument('--actor-only', action='store_true')
    args = parser.parse_args()
    result = Campaign(args.output, args.bundle, json.loads(args.model_json.read_text()),
                      json.loads(args.tool_command_json.read_text()), args.instruction.read_text(),
                      skill=args.skill, rounds=args.rounds, actor_budget=args.actor_budget,
                      analyzer_budget=args.analyzer_budget, actor_only=args.actor_only).run()
    print(json.dumps(result, indent=2))
    return 0 if result['state'] == 'completed' else 75


if __name__ == '__main__':
    raise SystemExit(main())
