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
import tempfile
import threading
import time

from blind_optimization_launcher import BlindLauncher, LaunchError, write_json, execute
from actor_analyzer_contract import build_prompt, validate_analysis
import analyzer_history


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_hash(workspace):
    path = workspace / 'candidate.py'
    return digest(path) if path.is_file() and not path.is_symlink() else None


def directory_manifest(directory):
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('pinned inputs must not contain symlinks')
    return {str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def skill_manifest(directory):
    if not (directory / 'SKILL.md').is_file():
        raise ValueError('skill needs SKILL.md and no symlinks')
    return directory_manifest(directory)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def correct(result):
    records = result.get('correctness')
    return result.get('state') == 'completed' and (records is True or
        isinstance(records, list) and bool(records) and all(
            isinstance(item, dict) and item.get('correctness') is True for item in records))


def citation_identities(summary):
    captures = summary.get('captures', [])
    kernels = {summary.get('selector')}
    kernels.update(summary.get('discovered_names', []))
    kernels.update(item.get('selector') for item in captures)
    records = summary.get('correctness', [])
    if isinstance(records, list):
        kernels.update(item.get('selector') for item in records)
    kernels.discard(None)
    # Legacy standalone runners identify seeded cases by check index, while
    # Model adapters export explicit semantic case IDs. Do not invent task aliases.
    cases = summary.get('cases')
    if cases is None:
        cases = [item['check'] for item in records if 'check' in item] if isinstance(records, list) else []
    return sorted(kernels), sorted(str(item) for item in cases)


def validate_citation(evidence, summary, expected_source, artifact):
    """Bind all claimed identities to the trusted call, never the report itself."""
    if not isinstance(summary, dict) or summary.get('source_sha256') != expected_source \
            or summary.get('implementation', 'candidate') != 'candidate' or not correct(summary):
        raise ValueError('citation lacks successful frozen candidate provenance')
    kernels, cases = citation_identities(summary)
    if not set(evidence['kernels']) <= set(kernels) or not set(evidence['cases']) <= set(cases):
        raise ValueError('citation kernel/case claims are not supported by the trusted call')
    captures = summary.get('captures', [])
    kind = evidence['kind']
    if kind == 'profile' and not any(item.get('pipe_rows') for item in captures):
        raise ValueError('profile citation lacks acquired candidate pipe evidence')
    if kind == 'timing' and not any(summary.get(key) for key in ('timing', 'latency', 'busy_time')):
        raise ValueError('timing citation lacks acquired timing evidence')
    if kind == 'compiler':
        inventory = summary.get('compiler_artifacts', {})
        if not inventory:
            raise ValueError('compiler citation lacks inventoried artifacts')
        parts = Path(evidence['path']).parts[2:]
        if parts != ('results', 'summary.json') and (
                len(parts) != 3 or parts[:2] != ('compiler-evidence', 'candidate')
                or inventory.get(parts[-1]) != digest(artifact)):
            raise ValueError('compiler citation does not match the candidate artifact inventory')


def retain_result(phase, result):
    """Journal trusted tool identity now; qualify final candidate bytes later."""
    source = result.get('source_sha256')
    if correct(result) and isinstance(source, str) and len(source) == 64 \
            and all(character in '0123456789abcdef' for character in source):
        with (phase / 'qualifications.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')


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
    paths = sorted(Path(config['directory']).glob('*/receipt.json'))
    phase_worker = Path(config['directory']) / 'phase.json'
    if phase_worker.exists():
        paths.insert(0, phase_worker)
    for path in paths:
        if path.parent.name in config['existing']:
            continue
        try:
            receipt = json.loads(path.read_text())
            if path == phase_worker and receipt.get('remote_folder') and receipt.get('handle'):
                from model_campaign_warm import shutdown_owned_phase
                stopped = shutdown_owned_phase(path)
                calls.append({'worker_shutdown': stopped})
                if stopped['state'] != 'terminal':
                    uncertain = True
                    continue
                receipt = json.loads(path.read_text())  # Preserve shutdown handle and acknowledgment.
            if path != phase_worker and receipt.get('state') in ('completed', 'failed', 'cancelled'):
                continue
            if receipt.get('dispatch_intent') and not receipt.get('handle'):
                uncertain = True
                calls.append({'call_id': path.parent.name, 'state': 'handle-unknown'})
                continue
            handle = receipt.get('handle') or receipt.get('upload', {}).get('handle')
            if not handle:
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
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            uncertain = True
            calls.append({'call_id': path.parent.name, 'error': str(error)})
    result = {'state': 'observation-required' if uncertain else 'terminal', 'calls': calls}
    save(phase / 'cleanup.json', result)
    return result


def bridge(config_path, arguments):
    """Keep managed Tools receipts and source identity outside the agent mount."""
    if arguments[:1] == ['stop']:
        raise ValueError('phase shutdown is a trusted controller operation')
    config = json.loads(config_path.read_text())
    if config.get('report_repair_only'):
        print(json.dumps(dict(state='failed', failure_stage='report-repair',
                             error='All remote/device tools are disabled during report repair')))
        return 2
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
    result = subprocess.run([*config['command'], '--workspace', str(workspace), *arguments],
                            capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    try:
        payload = json.loads(result.stdout)
        retain_result(phase, payload)
    except ValueError:
        pass
    return result.returncode


class RoleLauncher(BlindLauncher):
    def __init__(self, campaign, phase, role, budget, commit):
        self.campaign, self.role, self.commit = campaign, role, commit
        campaign.verify_inputs()
        prompt = phase.parent / (phase.name + '.prompt.md')
        text = build_prompt(role, budget, int(phase.name.split('-')[1]),
                            campaign.instruction, candidate_commit=commit)
        if role == 'actor' and campaign.single_kernel:
            text += ('\nFor this single-kernel task, check/ir are cheap correctness/compiler probes only. '
                     'Finish with timing (or pipes if useful) returning '
                     'launch_contract.profile_stream_verified=true to qualify the final candidate. '
                     'Timing verification does not require diagnostic profiling analysis.\n')
        if campaign.analysis_contract == 2:
            text += ('\nAnalysis contract 2: read /history/context.json and retained round records. '
                     'The controller supplies exact source_diff, identities and comparison requirements.\n')
            if role == 'analyzer':
                text += (Path(__file__).resolve().parents[1] / 'prompts/actor-analyzer/history.md').read_text()
            else:
                text += ('Read /history/context.json latest_analysis status before using prior reports. '
                         'Read the latest valid report and its selected comparison reports and candidate.py files; '
                         'record which findings you considered, including agreement/disagreement.\n')
        prompt.write_text(text)
        super().__init__(phase, campaign.bundle, prompt, campaign.model,
                         ['unused'], skill=campaign.skill or (
                             campaign.root / 'profiling-skills/ascend-profiling'
                             if 'ascend-profiling' in campaign.profiling_skills else None), timeout=budget,
                         auth_home=campaign.auth_home)
        self.finish_reserve_seconds = min(120, budget // 5)

    def prepare(self):
        if getattr(self, 'resume_before_agent', False):
            self.workspace = self.root / 'candidate-snapshot'
            self.frozen, self.state = self.root / 'frozen', self.root / 'codex-state'
            self.wrapper, self.journal = self.root / 'experiment-tool', self.root / 'commands.jsonl'
            self.reports = self.root / 'reports'
            expected = subprocess.check_output(['git', '-C', str(self.campaign.workspace),
                                                 'show', self.commit + ':candidate.py'])
            if candidate_hash(self.workspace) != hashlib.sha256(expected).hexdigest():
                raise LaunchError('retained analyzer snapshot differs from frozen candidate')
            for path in (self.frozen / 'prompt.md', self.wrapper, self.state / 'auth.json', self.reports):
                if not path.exists():
                    raise LaunchError('incomplete retained analyzer preparation')
            if digest(self.frozen / 'prompt.md') != digest(self.prompt_source):
                raise LaunchError('retained analyzer prompt changed')
            if self.skill_source and skill_manifest(self.frozen / 'skill') != skill_manifest(self.skill_source):
                raise LaunchError('retained analyzer skill changed')
            for name, expected in self.campaign.config.get('profiling_skills', {}).items():
                if name != 'ascend-profiling' and skill_manifest(self.frozen / 'profiling-skills' / name) != expected:
                    raise LaunchError('retained analyzer profiling inputs changed')
            config = json.loads((self.root / 'tool-config.json').read_text())
            if config != dict(workspace=str(self.workspace), command=self.campaign.tool_command,
                              campaign=str(self.campaign.root)):
                raise LaunchError('retained analyzer tool configuration changed')
            self.socket_temporary = tempfile.TemporaryDirectory(prefix='blind-tool-')
            self.socket_dir = Path(self.socket_temporary.name)
            self.tool_command = [sys.executable, str(Path(__file__).resolve()), 'tool',
                                 str(self.root / 'tool-config.json')]
            return
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
            if self.campaign.analysis_contract == 2:
                ctx = json.loads((self.campaign.history / 'context.json').read_text())
                if ctx['repeated_round'] is not None:
                    evidence = self.campaign.history / f"round-{ctx['repeated_round']}/evidence"
                    if evidence.exists():
                        shutil.copytree(evidence, self.workspace / 'evidence', dirs_exist_ok=True)
            if self.skill_source:
                (self.workspace / '.agents/skills/ascend-profiling').mkdir(parents=True)
        config = self.root / 'tool-config.json'
        save(config, {'workspace': str(self.workspace), 'command': self.campaign.tool_command,
                      'campaign': str(self.campaign.root)})
        self.tool_command = [sys.executable, str(Path(__file__).resolve()), 'tool', str(config)]
        self.reports = self.root / 'reports'
        self.reports.mkdir()
        for name in self.campaign.profiling_skills:
            (self.workspace / '.agents/skills' / name).mkdir(parents=True, exist_ok=True)
            if name == 'ascend-profiling':  # Parent launcher already snapshots this mount.
                if skill_manifest(self.frozen / 'skill') != self.campaign.config['profiling_skills'][name]:
                    raise LaunchError('profiling snapshot changed during role preparation')
                continue
            shutil.copytree(self.campaign.root / 'profiling-skills' / name,
                            self.frozen / 'profiling-skills' / name)
            if skill_manifest(self.frozen / 'profiling-skills' / name) != self.campaign.config['profiling_skills'][name]:
                raise LaunchError('profiling snapshot changed during role preparation')
        if self.role == 'actor':
            for name in self.campaign.coding_skills:
                shutil.copytree(self.campaign.root / 'coding-skills' / name,
                                self.frozen / 'coding-skills' / name)
                if skill_manifest(self.frozen / 'coding-skills' / name) != self.campaign.config['coding_skills'][name]:
                    raise LaunchError('coding snapshot changed during role preparation')
        directory = self.workspace.parent / 'tool-state'
        save(self.root / 'owned-tools.json', {'directory': str(directory),
                                              'existing': [p.name for p in directory.glob('*')]})

    def before_agent(self):
        if self.role == 'analyzer' and self.campaign.analysis_contract == 2:
            from analyzer_profile_acquisition import acquire
            acquire(self, execute, save)

    def after_agent(self, result):
        if self.role == 'analyzer' and self.campaign.analysis_contract == 2:
            from analyzer_report_repair import repair
            import blind_optimization_launcher as runtime
            return repair(self, result, runtime.execute, save)
        return result

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
            result = super().launch(preflight_only)
            # Analyzers write a report, never candidate commits. Their report is
            # validated independently by finish_phase before any batch gate.
            if (self.role == 'analyzer' and result.get('returncode') == 0
                    and result.get('timed_out') is False):
                result['exit_code'] = 0
                save(self.root / 'receipt.json', result)
            return result
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
        for name in self.campaign.profiling_skills:
            if name != 'ascend-profiling':
                retained += ['--ro-bind', str(self.frozen / 'profiling-skills' / name),
                             '/workspace/.agents/skills/' + name]
        if self.role == 'actor':
            for name in self.campaign.coding_skills:
                retained += ['--ro-bind', str(self.frozen / 'coding-skills' / name),
                             '/workspace/.agents/skills/' + name]
        if self.campaign.analysis_contract == 2:
            retained += ['--ro-bind', str(self.campaign.history), '/history']
        if getattr(self, 'report_repair_only', False):
            return retained + ['--ro-bind', str(self.root / 'report-repair'), '/report-repair',
                '--ro-bind', str(self.reports), '/analysis',
                '--bind', str(self.reports / 'analysis.json'), '/analysis/analysis.json']
        return retained + ['--bind', str(self.reports), '/analysis']


class Campaign:
    def __init__(self, output, bundle, model, tool_command, instruction, *, skill=None,
                 rounds=4, actor_budget=1200, analyzer_budget=600, actor_only=False,
                 auth_home=None, runner=None, coding_skills=(), profiling_skills=(), analysis_contract=2):
        if min(rounds, actor_budget, analyzer_budget) <= 0:
            raise ValueError('rounds and budgets must be positive')
        self.root, self.bundle = Path(output).resolve(), Path(bundle).resolve()
        self.workspace = self.root / 'workspace'
        self.history = self.root / 'history'
        self.analysis_contract = analysis_contract
        self.model, self.tool_command, self.instruction = model, tool_command, instruction
        self.profiling_source = Path(skill).resolve() if skill else None
        self.skill = self.root / 'profiling-skill' if skill else None
        self.auth_home = auth_home
        self.rounds, self.actor_budget, self.analyzer_budget = rounds, actor_budget, analyzer_budget
        self.actor_only, self.runner = actor_only, runner or self.launch_role
        self.profiling_skills = {}
        for directory in profiling_skills:
            directory = Path(directory).resolve()
            if directory.name in self.profiling_skills or (skill and directory.name == 'ascend-profiling'):
                raise ValueError('profiling skill names must be unique')
            skill_manifest(directory)
            self.profiling_skills[directory.name] = directory
        self.coding_skills = {}
        for directory in coding_skills:
            directory = Path(directory).resolve()
            if directory.name in self.coding_skills or directory.name == 'ascend-profiling' \
                    or directory.name in self.profiling_skills \
                    or not (directory / 'SKILL.md').is_file() \
                    or any(p.is_symlink() for p in directory.rglob('*')):
                raise ValueError('coding skill needs unique name, SKILL.md, and no symlinks')
            self.coding_skills[directory.name] = directory
        task = json.loads((self.bundle / 'task.json').read_text())
        self.single_kernel = task.get('single_kernel') is True
        self.public_files = task['public_files']
        if not isinstance(self.public_files, list) or 'candidate.py' not in self.public_files:
            raise ValueError('public_files must include candidate.py')
        for name in self.public_files:
            source = self.bundle / name
            if Path(name).is_absolute() or '..' in Path(name).parts or source.is_symlink() \
                    or not source.is_file() or not source.resolve().is_relative_to(self.bundle):
                raise ValueError('invalid public bundle path')
        # candidate is the only mutable supplied source; newly created logs remain writable.
        self.public_files = [name for name in self.public_files if name != 'candidate.py']
        self.config = dict(bundle=str(self.bundle), bundle_manifest=directory_manifest(self.bundle),
                           model=model, tool_command=tool_command,
                           instruction=instruction, rounds=rounds, actor_budget=actor_budget,
                           analyzer_budget=analyzer_budget, actor_only=actor_only,
                           skill=str(self.profiling_source) if skill else None,
                           profiling_manifest=skill_manifest(self.profiling_source) if skill else None,
                           coding_skills={name: skill_manifest(directory)
                               for name, directory in self.coding_skills.items()})
        # Omit the new key for legacy campaigns so their resume configuration is unchanged.
        if self.profiling_skills:
            self.config['profiling_skills'] = {name: skill_manifest(directory)
                for name, directory in self.profiling_skills.items()}

        existing = self.root / 'state.json'
        if existing.exists():
            self.analysis_contract = json.loads(existing.read_text())['configuration'].get('analysis_contract', 1)
        if self.analysis_contract == 2:
            self.config['analysis_contract'] = 2
        elif self.analysis_contract != 1:
            raise ValueError('unsupported analysis contract')

    def verify_inputs(self):
        if directory_manifest(self.bundle) != self.config['bundle_manifest']:
            raise LaunchError('task bundle changed; preserve original campaign inputs')
        if self.skill and skill_manifest(self.skill) != self.config['profiling_manifest']:
            raise LaunchError('frozen profiling snapshot changed; preserve original evidence')
        for name, expected in self.config['coding_skills'].items():
            if skill_manifest(self.root / 'coding-skills' / name) != expected:
                raise LaunchError('coding snapshot changed; preserve original evidence')
        for name, expected in self.config.get('profiling_skills', {}).items():
            if skill_manifest(self.root / 'profiling-skills' / name) != expected:
                raise LaunchError('profiling snapshot changed; preserve original evidence')

    def qualified(self, result):
        return correct(result) and (not self.single_kernel or
            result.get('launch_contract', {}).get('profile_stream_verified') is True)

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.workspace), *arguments], text=True)

    def prepare(self):
        if (self.root / 'state.json').exists():
            state = json.loads((self.root / 'state.json').read_text())
            if state['configuration'] != self.config:
                raise LaunchError('resume configuration changed; preserve the original campaign')
            self.verify_inputs()
            return state
        if self.root.exists():
            raise LaunchError('unrecognized existing campaign output')
        self.workspace.mkdir(parents=True)
        if self.analysis_contract == 2:
            self.history.mkdir()
        if self.skill:
            shutil.copytree(self.profiling_source, self.skill)
            if skill_manifest(self.skill) != self.config['profiling_manifest']:
                raise LaunchError('profiling source changed during snapshot creation')
        for name, directory in self.coding_skills.items():
            shutil.copytree(directory, self.root / 'coding-skills' / name)
        for name, directory in self.profiling_skills.items():
            shutil.copytree(directory, self.root / 'profiling-skills' / name)
        self.verify_inputs()
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
        retain_result(phase, result)

    def launch_role(self, campaign, phase, role, budget, commit):
        launcher = RoleLauncher(self, phase, role, budget, commit)
        launcher.resume_before_agent = phase.exists()
        return launcher.launch()

    def validate_phase_analysis(self, phase, commit):
        """Read-only authoritative validation shared by repair and publication."""
        report_path = phase / 'reports/analysis.json'
        if not report_path.exists():
            report_path = phase / 'analysis.json'
        repair_result = phase / 'report-repair/terminal.json'
        if (phase / 'report-repair/dispatch.json').exists():
            if not repair_result.is_file():
                raise ValueError('report repair has no terminal result; observation required')
            terminal = json.loads(repair_result.read_text())
            if terminal['returncode'] != 0 or terminal['timed_out']:
                raise ValueError('report repair did not finish successfully')
        acquisitions = {}
        report = validate_analysis(json.loads(report_path.read_text()), commit)
        verified_source = subprocess.check_output(['git', '-C', str(self.workspace), 'show',
                                                   commit + ':candidate.py'])
        expected_source = hashlib.sha256(verified_source).hexdigest()
        snapshot_source = phase / 'candidate-snapshot/candidate.py'
        if snapshot_source.is_file() and digest(snapshot_source) != expected_source:
            raise ValueError('frozen analyzer candidate source changed')
        evidence_sha256 = {}
        errors = []
        for evidence in report['evidence']:
            try:
                original = phase / 'candidate-snapshot' / evidence['path']
                retained = phase / 'trusted-evidence' / Path(evidence['path']).relative_to('evidence')
                path = retained if retained.exists() else original
                if not path.is_file() or path.is_symlink() or not original.is_file() \
                        or digest(path) != digest(original):
                    raise ValueError('analysis cites missing evidence: ' + evidence['path'])
                evidence_sha256[evidence['path']] = digest(path)
                relative_summary = Path('evidence') / Path(evidence['path']).parts[1] / 'results/summary.json'
                original_summary = phase / 'candidate-snapshot' / relative_summary
                retained_summary = phase / 'trusted-evidence' / relative_summary.relative_to('evidence')
                summary_path = retained_summary if retained_summary.exists() else original_summary
                if not summary_path.is_file() or summary_path.is_symlink() \
                        or not original_summary.is_file() or digest(summary_path) != digest(original_summary):
                    raise ValueError('citation lacks its authoritative call summary')
                evidence_sha256[str(relative_summary)] = digest(summary_path)
                summary = json.loads(summary_path.read_text())
                validate_citation(evidence, summary, expected_source, path)
                receipt_path = phase / 'tool-state' / relative_summary.parts[1] / 'receipt.json'
                receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
                acquisitions[str(relative_summary)] = analyzer_history.measurements(summary, receipt,
                    hashlib.sha256(json.dumps(self.config['bundle_manifest'], sort_keys=True).encode()).hexdigest())
            except (ValueError, OSError) as error:
                errors.append(evidence['path'] + ': ' + str(error))
        if self.analysis_contract == 2:
            try:
                ctx = json.loads((self.history / 'context.json').read_text())
                attempts = [json.loads(path.read_text()) for path in (phase / 'tool-state').glob('*/receipt.json')]
                attempted = any(item.get('request', {}).get('verb', item.get('request', {}).get('operation')) == 'pipes'
                                and item.get('request', {}).get('source_sha256') == expected_source
                                for item in attempts)
                required_path = phase / 'required-profile.json'
                required = json.loads(required_path.read_text()) if required_path.is_file() else {}
                attempted |= (required.get('source_sha256') == expected_source
                              and required.get('command_dispatched') is True
                              and required.get('profile_attempt_arguments') == ['pipes'])
                if ctx['status'] != 'unchanged' and not attempted:
                    raise ValueError('new verified kernel requires a recorded supplied-tool profile attempt')
                if required.get('state') == 'completed' and required.get('evidence'):
                    acquired_path = phase / 'candidate-snapshot' / required['evidence']
                    acquired = json.loads(acquired_path.read_text())
                    if any(item.get('pipe_rows') for item in acquired.get('captures', [])) and not any(
                            item['kind'] == 'profile' and item['path'] == required['evidence'] for item in report['evidence']):
                        raise ValueError('analysis must cite the controller-acquired profile')
                if ctx['status'] == 'unchanged' and not attempted:
                    repeated = json.loads((self.history / f"round-{ctx['repeated_round']}/record.json").read_text())
                    acquisitions = {str(i): item for i, item in enumerate(repeated['acquisitions'])}
                report = analyzer_history.finalize(report, ctx, self.history, list(acquisitions.values()))
            except (ValueError, OSError) as error:
                errors.append('history/semantics: ' + str(error))
        if errors:
            raise ValueError('; '.join(errors))
        return report, evidence_sha256, acquisitions

    def finish_phase(self, phase, role, entry, state):
        if role == 'analyzer' and self.analysis_contract == 2:
            retained = self.history / ('round-' + phase.name.split('-')[1])
            if (retained / 'record.json').exists():
                record = json.loads((retained / 'record.json').read_text())
                if record['candidate_commit'] != entry['candidate_commit']:
                    raise LaunchError('immutable history disagrees with pending phase')
                if digest(retained / 'candidate.py') != record['source_sha256']:
                    raise LaunchError('retained candidate identity changed')
                entry.update({key: record[key] for key in
                              ('analysis_valid', 'analysis_error', 'evidence_sha256')})
                handoff = json.loads((retained / 'analysis.json').read_text()) if record['analysis_valid'] else dict(
                    contract_version=2, candidate_commit=entry['candidate_commit'], status='invalid',
                    reason=record['analysis_error'])
                save(self.workspace / 'analysis.json', handoff)
                self.commit(phase.name + ': recovered retained performance analysis')
                return
        archive = self.workspace / 'experiments' / phase.name
        archive.mkdir(parents=True, exist_ok=True)
        for name in ('commands.jsonl', 'receipt.json', 'agent.jsonl', 'agent.stderr',
                     'qualifications.jsonl', 'cleanup.json', 'required-profile.json',
                     'role-execution.json', 'model-dispatch.json'):
            if (phase / name).is_file():
                shutil.copyfile(phase / name, archive / name)
        if role == 'actor':
            commit = self.commit(phase.name + ': candidate and execution evidence')
            if (phase / 'qualifications.jsonl').exists():
                records = [json.loads(line) for line in (phase / 'qualifications.jsonl').read_text().splitlines()]
                if any(self.qualified(item) and item.get('source_sha256') is not None
                       and item.get('source_sha256') == candidate_hash(self.workspace)
                       for item in records):
                    state['verified'] = commit
            previous = state.get('verified')
            if previous and previous != commit:
                verified_source = subprocess.check_output(['git', '-C', str(self.workspace),
                    'show', previous + ':candidate.py'])
                if candidate_hash(self.workspace) == hashlib.sha256(verified_source).hexdigest():
                    state['verified'] = commit
                    entry['carried_forward_from'] = previous
            entry['candidate_commit'] = commit
            entry['candidate_verified'] = state.get('verified') == commit
        else:
            report_path = phase / 'reports/analysis.json'
            if not report_path.exists():
                report_path = phase / 'analysis.json'
            trusted = phase / 'candidate-snapshot/evidence'
            if trusted.exists():
                shutil.copytree(trusted, phase / 'trusted-evidence', dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('*.tar.gz'))
                shutil.copytree(phase / 'trusted-evidence', archive / 'trusted-evidence', dirs_exist_ok=True)
            if (phase / 'report-repair').exists():
                shutil.copytree(phase / 'report-repair', archive / 'report-repair', dirs_exist_ok=True)
            if (phase / 'reports').exists():
                shutil.copytree(phase / 'reports', archive / 'analysis', dirs_exist_ok=True,
                                symlinks=True, ignore=shutil.ignore_patterns('*.tar.gz'))
            elif report_path.exists():
                shutil.copyfile(report_path, archive / 'raw-analysis.json')
            acquisitions = {}
            try:
                report, entry['evidence_sha256'], acquisitions = self.validate_phase_analysis(phase, entry['candidate_commit'])
                save(self.workspace / 'analysis.json', report)
                entry['analysis_valid'] = True
            except (ValueError, OSError) as error:
                entry.update(analysis_valid=False, analysis_error=str(error))
            if self.analysis_contract == 2:
                index = int(phase.name.split('-')[1])
                folder = self.history / f'round-{index}'
                staging = Path(tempfile.mkdtemp(prefix='.round-', dir=self.history))
                final_folder, folder = folder, staging
                source = subprocess.check_output(['git', '-C', str(self.workspace), 'show',
                                                  entry['candidate_commit'] + ':candidate.py'])
                (folder / 'candidate.py').write_bytes(source)
                record = dict(round=index, candidate_commit=entry['candidate_commit'],
                              source_sha256=hashlib.sha256(source).hexdigest(),
                              analysis_valid=entry['analysis_valid'],
                              analysis_error=entry.get('analysis_error'),
                              report_repair=json.loads((phase / 'report-repair/validation.json').read_text())
                                  if (phase / 'report-repair/validation.json').is_file() else None,
                              required_profile=json.loads((phase / 'required-profile.json').read_text())
                                  if (phase / 'required-profile.json').is_file() else None,
                              evidence_sha256=entry.get('evidence_sha256', {}),
                              acquisitions=list(acquisitions.values()) if entry['analysis_valid'] else [],
                              bottlenecks=report['bottlenecks'] if entry['analysis_valid'] else [])
                if entry['analysis_valid']:
                    save(folder / 'analysis.json', report)
                if (phase / 'trusted-evidence').exists():
                    shutil.copytree(phase / 'trusted-evidence', folder / 'evidence')
                save(folder / 'record.json', record)
                os.rename(folder, final_folder)  # Publish once; never overwrite a retained round.
                # Compatibility handoff must never silently retain a stale latest report.
                if not entry['analysis_valid']:
                    save(self.workspace / 'analysis.json', dict(contract_version=2,
                        candidate_commit=entry['candidate_commit'], status='invalid',
                        reason=entry['analysis_error']))
            self.commit(phase.name + ': performance analysis and diagnostics')

    def run(self, stop_after_round=None):
        if stop_after_round is not None and not 1 <= stop_after_round <= self.rounds:
            raise ValueError("checkpoint round outside configured campaign")
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
                    self.verify_inputs()
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
                            recoverable = (role == 'analyzer' and self.analysis_contract == 2
                                and (phase / 'role-execution.json').is_file()
                                and not (phase / 'model-dispatch.json').exists())
                            if not recoverable:
                                return {**state, 'state': 'observation-required', 'pending_phase': name}
                            result = self.runner(self, phase, role, self.analyzer_budget, entry['candidate_commit'])
                            save(phase / 'receipt.json', result)
                        else:
                            result = json.loads((phase / 'receipt.json').read_text())
                    else:
                        commit = state['verified']
                        if role == 'actor' and commit:
                            verified = subprocess.check_output(['git', '-C', str(self.workspace),
                                'show', commit + ':candidate.py'])
                            if candidate_hash(self.workspace) != hashlib.sha256(verified).hexdigest():
                                candidate = self.workspace / 'candidate.py'
                                if candidate.is_symlink():
                                    candidate.unlink()
                                candidate.write_bytes(verified)
                                # Keep failed experiments and analysis history; restore
                                # only the candidate rather than rewinding the branch.
                                self.commit('Restore verified candidate before ' + name)
                        if role == 'analyzer' and not commit:
                            state['phases'][name] = {'state': 'skipped-no-verified-candidate'}
                            save(self.root / 'state.json', state)
                            continue
                        if self.analysis_contract == 2:
                            source = subprocess.check_output(['git', '-C', str(self.workspace),
                                'show', (commit or self.git('rev-parse', 'HEAD').strip()) + ':candidate.py'])
                            ctx = analyzer_history.context(self.history, source, commit, index)
                            prior = ctx['history']
                            ctx['latest_analysis'] = prior[-1] if prior else None
                            save(self.history / 'context.json', ctx)
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
                if stop_after_round == index and index < self.rounds:
                    state['state'] = 'checkpoint'
                    save(self.root / 'state.json', state)
                    return state
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
    parser.add_argument('--profiling-skill', type=Path, action='append', default=[])
    parser.add_argument('--coding-skill', type=Path, action='append', default=[])
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--stop-after-round', type=int)
    parser.add_argument('--actor-budget', type=int, default=1200)
    parser.add_argument('--analyzer-budget', type=int, default=600)
    parser.add_argument('--actor-only', action='store_true')
    args = parser.parse_args()
    result = Campaign(args.output, args.bundle, json.loads(args.model_json.read_text()),
                      json.loads(args.tool_command_json.read_text()), args.instruction.read_text(),
                      skill=args.skill, rounds=args.rounds, actor_budget=args.actor_budget,
                      analyzer_budget=args.analyzer_budget, actor_only=args.actor_only,
                      coding_skills=args.coding_skill, profiling_skills=args.profiling_skill).run(stop_after_round=args.stop_after_round)
    print(json.dumps(result, indent=2))
    return 0 if result['state'] in ('completed', 'checkpoint') else 75


if __name__ == '__main__':
    raise SystemExit(main())
