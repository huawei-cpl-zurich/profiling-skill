from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from actor_analyzer_controller import Campaign, RoleLauncher, bridge, digest, cleanup_phase


def make_campaign(tmp_path, runner, **options):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('value = 0\n')
    (bundle / 'API.md').write_text('Public API')
    (bundle / 'private.py').write_text('secret')
    (bundle / 'task.json').write_text(json.dumps({'public_files': ['candidate.py', 'API.md']}))
    options.setdefault('analysis_contract', 1)
    return Campaign(tmp_path / 'run', bundle, {'name': 'test', 'reasoning_effort': 'low'},
                    ['unused'], 'Optimize public API', runner=runner, **options)


def test_four_rounds_preserve_history_handoff_and_frozen_candidate(tmp_path):
    seen = []
    def runner(campaign, phase, role, budget, commit):
        seen.append((role, budget, commit))
        if role == 'actor':
            n = len([x for x in seen if x[0] == 'actor'])
            if n > 1:
                assert (campaign.workspace / 'analysis.json').is_file()
                assert commit == seen[-2][2]
            (campaign.workspace / 'candidate.py').write_text(f'value = {n}\n')
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            assert commit == campaign.git('rev-parse', 'HEAD').strip()
            report = {'candidate_commit': commit, 'profiling': {'status': 'unavailable', 'reason': 'test'},
                      'evidence': [], 'observations': [], 'bottlenecks': [], 'ceilings': [], 'unknowns': []}
            (phase / 'analysis.json').write_text(json.dumps(report))
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner)
    result = campaign.run()
    assert result['state'] == 'completed'
    assert [x[:2] for x in seen] == [('actor', 1200), ('analyzer', 600)] * 4
    assert not (campaign.workspace / 'private.py').exists()
    assert len(campaign.git('log', '--format=%H').splitlines()) >= 9
    assert campaign.run() == result
    assert len(seen) == 8


def test_invalid_latest_uses_verified_revision_and_no_verified_skips(tmp_path):
    analysis = []
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            (campaign.workspace / 'candidate.py').write_text(str(len(analysis)))
            if not analysis:
                campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                        'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            analysis.append(commit)
        return {'exit_code': 124}
    campaign = make_campaign(tmp_path, runner, rounds=2)
    campaign.run()
    assert len(analysis) == 2 and analysis[0] == analysis[1]
    other = tmp_path / 'other'
    other.mkdir()
    campaign = make_campaign(other, lambda *args: {'exit_code': 1}, rounds=1)
    result = campaign.run()
    assert result['phases']['round-1-analyzer']['state'] == 'skipped-no-verified-candidate'


def test_interrupted_dispatch_is_not_repeated(tmp_path):
    calls = []
    def crash(*args):
        calls.append(1)
        raise RuntimeError('observer disconnected')
    campaign = make_campaign(tmp_path, crash, rounds=1)
    with pytest.raises(RuntimeError):
        campaign.run()
    assert campaign.run()['state'] == 'observation-required'
    assert len(calls) == 1


def test_next_actor_restores_verified_source_without_erasing_failed_history(tmp_path):
    observed = []
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            observed.append((campaign.workspace / 'candidate.py').read_text())
            if len(observed) == 1:
                (campaign.workspace / 'candidate.py').write_text('verified = 1\n')
                campaign.qualify(phase, dict(state='completed', correctness=True,
                    source_sha256=digest(campaign.workspace / 'candidate.py')))
            elif len(observed) == 2:
                (campaign.workspace / 'candidate.py').write_text('broken = True\n')
                (campaign.workspace / 'failed-diagnostics.txt').write_text('compile failed')
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=3)
    state = campaign.run()
    assert observed == ['value = 0\n', 'verified = 1\n', 'verified = 1\n']
    failed = state['phases']['round-2-actor']['candidate_commit']
    assert campaign.git('show', failed + ':candidate.py') == 'broken = True\n'
    assert (campaign.workspace / 'failed-diagnostics.txt').read_text() == 'compile failed'
    assert 'Restore verified candidate before round-3-actor' in campaign.git('log', '--format=%s')
    assert campaign.run() == state


def test_actor_only_has_no_analyzer_cost(tmp_path):
    roles = []
    campaign = make_campaign(tmp_path, lambda c, p, r, b, v: roles.append(r) or {'exit_code': 0},
                             actor_only=True, rounds=2)
    result = campaign.run()
    assert roles == ['actor', 'actor']
    assert result['configured_analyzer_seconds'] == 0


def test_bridge_rejects_malformed_source_identity_and_routes_pending_observation(tmp_path, capsys):
    campaign = make_campaign(tmp_path, None, rounds=1)
    campaign.prepare()
    phase = campaign.root / 'round-1-actor'
    phase.mkdir()
    helper = tmp_path / 'fake-tool.py'
    helper.write_text('import json,sys\nprint(json.dumps({"state":"completed","correctness":True,'
                      '"source_sha256":"stale","arguments":sys.argv[1:]}))\n')
    config = phase / 'tool-config.json'
    config.write_text(json.dumps({'campaign': str(campaign.root), 'workspace': str(campaign.workspace),
                                 'command': [sys.executable, str(helper)]}))
    assert bridge(config, ['check']) == 0
    assert not (phase / 'qualifications.jsonl').exists()
    receipt = campaign.root / 'tool-state/pending/receipt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'state': 'running', 'call_id': 'pending', 'handle': 'retained'}))
    assert bridge(config, ['pipes']) == 75
    assert bridge(config, ['observe', 'pending']) == 0
    assert 'retained' in capsys.readouterr().out


def test_actual_analyzer_boundary_candidate_git_readonly_reports_writable(tmp_path):
    import shutil
    from blind_optimization_launcher import execute
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('Bubblewrap and Codex runtime required')
    campaign = make_campaign(tmp_path, None, rounds=1)
    campaign.prepare()
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    campaign.auth_home = auth
    phase = campaign.root / 'round-1-analyzer'
    launcher = RoleLauncher(campaign, phase, 'analyzer', 600, campaign.git('rev-parse', 'HEAD').strip())
    launcher.prepare()
    code = '''from pathlib import Path
for path in ('/workspace/candidate.py', '/workspace/.git/config'):
    try: Path(path).write_text('tampered')
    except OSError: pass
    else: raise AssertionError(path + ' writable')
Path('/analysis/analysis.json').write_text('{}')
assert not Path('/workspace/private.py').exists()
'''
    try:
        result = execute([*launcher.base_command(), 'python3', '-c', code], phase, 30)
        assert result['returncode'] == 0, result
        assert (phase / 'reports/analysis.json').is_file()
        assert (campaign.workspace / 'candidate.py').read_text() == 'value = 0\n'
    finally:
        launcher.socket_temporary.cleanup()


def test_failed_analysis_cannot_forge_profile_evidence(tmp_path):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            (phase / 'reports').mkdir()
            (phase / 'reports/forged.json').write_text('{"pipe_rows":[{}]}')
            report = {'candidate_commit': commit, 'profiling': {'status': 'acquired'},
                      'evidence': [{'path': 'forged.json', 'kind': 'profile', 'kernels': ['x'], 'cases': ['1']}],
                      'observations': [], 'bottlenecks': [], 'ceilings': [], 'unknowns': []}
            (phase / 'reports/analysis.json').write_text(json.dumps(report))
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    result = campaign.run()
    assert result['phases']['round-1-analyzer']['analysis_valid'] is False
    assert not (campaign.workspace / 'analysis.json').exists()


def test_resume_refuses_changed_budgets_without_dispatch(tmp_path):
    calls = []
    campaign = make_campaign(tmp_path, lambda *a: calls.append(1) or {'exit_code': 0}, rounds=1)
    campaign.run()
    campaign.config['actor_budget'] = 1
    with pytest.raises(Exception, match='configuration changed'):
        campaign.run()
    assert len(calls) == 1  # analyzer was skipped because no candidate was verified


def test_cleanup_only_cancels_phase_owned_jobs_and_blocks_unknown_handles(tmp_path):
    phase = tmp_path / 'phase'
    phase.mkdir()
    state = tmp_path / 'tool-state'
    for name in ('old', 'new', 'unknown'):
        folder = state / name
        folder.mkdir(parents=True)
        (folder / 'receipt.json').write_text(json.dumps({'call_id': name, 'state': 'running',
            'dispatch_intent': True, **({'handle': name} if name != 'unknown' else
            {'upload': {'state': 'completed', 'handle': 'completed-transfer'}})}))
    (phase / 'owned-tools.json').write_text(json.dumps({'directory': str(state), 'existing': ['old']}))
    calls = []
    def remote(operation, handle):
        calls.append((operation, handle))
        return {'state': 'cancelled' if operation == 'cancel' else 'running'}
    result = cleanup_phase(phase, remote)
    assert calls == [('observe', 'new'), ('cancel', 'new')]
    assert result['state'] == 'observation-required'
    assert json.loads((state / 'new/receipt.json').read_text())['state'] == 'cancelled'
    assert json.loads((state / 'old/receipt.json').read_text())['state'] == 'running'


def test_invalid_analysis_raw_report_is_committed_without_replacing_handoff(tmp_path):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            reports = phase / 'reports'
            reports.mkdir()
            if phase.name == 'round-1-analyzer':
                report = {'candidate_commit': commit, 'profiling': {'status': 'unavailable', 'reason': 'test'},
                          'evidence': [], 'observations': [], 'bottlenecks': [], 'ceilings': [], 'unknowns': []}
                (reports / 'analysis.json').write_text(json.dumps(report))
            else:
                (reports / 'analysis.json').write_text('{incomplete')
                (reports / 'failure.log').write_text('remote observer interrupted')
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=2)
    result = campaign.run()
    assert not result['phases']['round-2-analyzer']['analysis_valid']
    assert json.loads((campaign.workspace / 'analysis.json').read_text())['candidate_commit'] == \
        result['phases']['round-1-analyzer']['candidate_commit']
    assert campaign.git('show', 'HEAD:experiments/round-2-analyzer/analysis/analysis.json') == '{incomplete'
    assert 'interrupted' in campaign.git('show', 'HEAD:experiments/round-2-analyzer/analysis/failure.log')


def test_watchdog_cleans_up_at_original_deadline_before_agent_returns(tmp_path, monkeypatch):
    import threading
    import time
    import actor_analyzer_controller as controller
    campaign = make_campaign(tmp_path, None, rounds=1)
    campaign.prepare()
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    campaign.auth_home = auth
    phase = campaign.root / 'round-1-actor'
    launcher = RoleLauncher(campaign, phase, 'actor', 1, None)
    cancelled = threading.Event()
    def fake_launch(self, preflight_only=False):
        self.root.mkdir()
        self.deadline_monotonic = time.monotonic() + .02
        assert cancelled.wait(1), 'remote cleanup waited for agent return'
        return {'exit_code': 124}
    monkeypatch.setattr(controller.BlindLauncher, 'launch', fake_launch)
    monkeypatch.setattr(controller, 'cleanup_phase', lambda p: cancelled.set() or {'state': 'terminal'})
    assert launcher.launch()['exit_code'] == 124


def test_uncertain_cleanup_prevents_analyzer_dispatch(tmp_path, monkeypatch):
    import actor_analyzer_controller as controller
    roles = []
    def runner(campaign, phase, role, budget, commit):
        roles.append(role)
        campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                'source_sha256': digest(campaign.workspace / 'candidate.py')})
        return {'exit_code': 0}
    monkeypatch.setattr(controller, 'cleanup_phase', lambda p: {'state': 'observation-required'})
    campaign = make_campaign(tmp_path, runner, rounds=1)
    assert campaign.run()['state'] == 'observation-required'
    assert campaign.run()['state'] == 'observation-required'
    assert roles == ['actor']


@pytest.mark.parametrize('role,actor_only', [('actor', False), ('actor', True), ('analyzer', False)])
def test_explicit_coding_skill_snapshot_actor_only_readonly(tmp_path, role, actor_only):
    import shutil
    from blind_optimization_launcher import execute
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('Bubblewrap and Codex required')
    skill = tmp_path / 'triton-guarded-kernel'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('Pinned coding guidance')
    profiling = tmp_path / 'profiling-source'
    profiling.mkdir()
    (profiling / 'SKILL.md').write_text('Pinned profiling guidance')
    campaign = make_campaign(tmp_path, None, rounds=1, coding_skills=[skill], actor_only=actor_only,
                             skill=profiling)
    campaign.prepare()
    (skill / 'SKILL.md').write_text('Changed after preparation')
    (profiling / 'SKILL.md').write_text('Profiling changed after preparation')
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    campaign.auth_home = auth
    phase = campaign.root / ('round-1-' + role)
    launcher = RoleLauncher(campaign, phase, role, 600, campaign.git('rev-parse', 'HEAD').strip())
    launcher.prepare()
    code = f'''from pathlib import Path
p = Path('/workspace/.agents/skills/triton-guarded-kernel/SKILL.md')
assert Path('/workspace/.agents/skills/ascend-profiling/SKILL.md').read_text() == 'Pinned profiling guidance'
assert p.exists() == {role == 'actor'}
assert not Path({str(skill)!r}).exists()
if p.exists():
    assert p.read_text() == 'Pinned coding guidance'
    try: p.write_text('tampered')
    except OSError: pass
    else: raise AssertionError('coding skill writable')
'''
    try:
        result = execute([*launcher.base_command(), 'python3', '-c', code], phase, 30)
        assert result['returncode'] == 0, result
    finally:
        launcher.socket_temporary.cleanup()


def test_coding_skill_symlink_rejected_before_campaign_creation(tmp_path):
    skill = tmp_path / 'coding'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('guidance')
    (skill / 'alias.md').symlink_to(skill / 'SKILL.md')
    with pytest.raises(ValueError, match='no symlinks'):
        make_campaign(tmp_path, None, coding_skills=[skill])
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('role', ['actor', 'analyzer'])
@pytest.mark.parametrize('names', [('cannbot-profiler',), ('cannbot-profiler', 'ascend-profiling'),
                                 ('ascend-profiling',)])
def test_named_profilers_real_isolated_cli_with_actor_only_coding(tmp_path, role, names):
    import shutil
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('Bubblewrap and Codex required')
    profilers = []
    for name in names:
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'SKILL.md').write_text(name + ' pinned')
        profilers.append(directory)
    coding = tmp_path / 'coding'
    coding.mkdir()
    (coding / 'SKILL.md').write_text('coding pinned')
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    campaign = make_campaign(tmp_path, None, rounds=1, profiling_skills=profilers,
                             coding_skills=[coding], auth_home=auth)
    campaign.prepare()
    for directory in profilers:
        (directory / 'SKILL.md').write_text('changed source')
    phase = campaign.root / ('round-1-' + role)
    budget = 600 if role == 'actor' else 240
    launcher = RoleLauncher(campaign, phase, role, budget, campaign.git('rev-parse', 'HEAD').strip())
    runtime = tmp_path / 'fake-runtime'
    (runtime / 'bin').mkdir(parents=True)
    (runtime / 'lib/node_modules').mkdir(parents=True)
    (runtime / 'bin/node').touch()
    launcher.codex = runtime / 'bin/codex'
    launcher.codex.write_text('''#!/usr/bin/python3
import json, subprocess
from pathlib import Path
skills = Path('/workspace/.agents/skills')
names = ''' + repr(names) + '''
for name in names:
    source = skills / name / 'SKILL.md'
    assert source.read_text() == name + ' pinned'
    try: source.write_text('tampered')
    except OSError: pass
    else: raise AssertionError('profiling skill writable')
assert (skills / 'coding/SKILL.md').exists() == ''' + repr(role == 'actor') + '''
assert (skills / 'ascend-profiling/SKILL.md').exists() == ('ascend-profiling' in names)
execution = json.loads(Path('/workspace/execution.json').read_text())
assert execution['wall_budget_seconds'] == ''' + str(budget) + '''
assert execution['finish_reserve_seconds'] == ''' + str(min(120, budget // 5)) + '''
if ''' + repr(role == 'actor') + ''':
    Path('notes.md').write_text('Skills read and evidence recorded')
    subprocess.run(['git', 'add', 'notes.md'], check=True)
    subprocess.run(['git', 'commit', '-m', 'Record experiment'], check=True)
else:
    Path('/analysis/analysis.json').write_text('{}')
''')
    launcher.codex.chmod(0o755)
    result = launcher.launch()
    assert result['returncode'] == 0, (phase / 'agent.stderr').read_text()
    if role == 'analyzer':
        assert (phase / 'reports/analysis.json').is_file()


def test_named_profilers_resume_drift_and_tampering_block_dispatch(tmp_path):
    skill = tmp_path / 'cannbot-profiler'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('pinned')
    calls = []
    campaign = make_campaign(tmp_path, lambda *args: calls.append(1) or {'exit_code': 0},
                             rounds=1, profiling_skills=[skill])
    campaign.run()
    assert campaign.run()['state'] == 'completed'
    assert len(calls) == 1
    (skill / 'SKILL.md').write_text('source changed')
    replacement = Campaign(campaign.root, campaign.bundle, campaign.model, campaign.tool_command,
                           campaign.instruction, rounds=1, profiling_skills=[skill])
    with pytest.raises(Exception, match='configuration changed'):
        replacement.run()
    (campaign.root / 'profiling-skills/cannbot-profiler/SKILL.md').write_text('tampered')
    with pytest.raises(Exception, match='profiling snapshot changed'):
        campaign.run()
    assert len(calls) == 1


@pytest.mark.parametrize('conflict', ['duplicate', 'coding', 'legacy', 'symlink'])
def test_named_profilers_reject_conflicting_or_linked_inputs(tmp_path, conflict):
    skill = tmp_path / ('ascend-profiling' if conflict == 'legacy' else 'profiler')
    skill.mkdir()
    (skill / 'SKILL.md').write_text('guidance')
    options = {'profiling_skills': [skill]}
    if conflict == 'duplicate': options['profiling_skills'].append(skill)
    if conflict == 'coding': options['coding_skills'] = [skill]
    if conflict == 'legacy': options['skill'] = skill
    if conflict == 'symlink': (skill / 'alias').symlink_to(skill / 'SKILL.md')
    with pytest.raises(ValueError):
        make_campaign(tmp_path, None, **options)
    assert not (tmp_path / 'run').exists()


def test_profiling_snapshot_frozen_across_roles_and_resume_source_change_rejected(tmp_path):
    skill = tmp_path / 'profiling-source'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('original profiling guidance')
    seen = []
    def runner(campaign, phase, role, budget, commit):
        seen.append((campaign.skill / 'SKILL.md').read_text())
        (skill / 'SKILL.md').write_text('replaced source guidance')
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=2, skill=skill)
    result = campaign.run()
    assert seen == ['original profiling guidance'] * 4
    assert campaign.run() == result
    replacement = Campaign(campaign.root, campaign.bundle, campaign.model, campaign.tool_command,
                           campaign.instruction, rounds=2, skill=skill, runner=runner)
    with pytest.raises(Exception, match='resume configuration changed'):
        replacement.run()


@pytest.mark.parametrize('stream_verified', [False, True])
def test_single_launch_tasks_require_profile_stream_for_qualification(tmp_path, stream_verified):
    roles = []
    def runner(campaign, phase, role, budget, commit):
        roles.append(role)
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                'source_sha256': digest(campaign.workspace / 'candidate.py'),
                'launch_contract': {'runner_checked': True, 'profile_stream_verified': stream_verified}})
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    manifest = campaign.bundle / 'task.json'
    spec = json.loads(manifest.read_text())
    spec['single_kernel'] = True
    manifest.write_text(json.dumps(spec))
    # Construct against the final task specification, as an ordinary CLI does.
    campaign = Campaign(campaign.root, campaign.bundle, campaign.model, campaign.tool_command,
                        campaign.instruction, rounds=1, runner=runner)
    result = campaign.run()
    assert ('analyzer' in roles) == stream_verified
    assert bool(result['verified']) == stream_verified


@pytest.mark.parametrize('symlink', [False, True])
def test_analyzer_evidence_file_conflict_is_archived_without_stalling(tmp_path, symlink):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            (phase / 'candidate-snapshot/evidence/call').mkdir(parents=True)
            (phase / 'candidate-snapshot/evidence/call/summary.json').write_text('{}')
            (phase / 'reports').mkdir()
            if symlink:
                (phase / 'reports/evidence').symlink_to('mistaken/missing')
            else:
                (phase / 'reports/evidence').write_text('mistaken file instead of a directory')
            (phase / 'reports/analysis.json').write_text('{incomplete')
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    assert campaign.run()['state'] == 'completed'
    assert campaign.run()['state'] == 'completed'
    assert 'mistaken' in campaign.git('show', 'HEAD:experiments/round-1-analyzer/analysis/evidence')


@pytest.mark.parametrize('changed', ['bundle', 'coding'])
def test_mutated_retained_inputs_block_next_role(tmp_path, changed):
    skill = tmp_path / 'coding'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('original')
    roles = []
    def runner(campaign, phase, role, budget, commit):
        roles.append(role)
        campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                'source_sha256': digest(campaign.workspace / 'candidate.py')})
        target = campaign.bundle / 'private.py' if changed == 'bundle' else \
            campaign.root / 'coding-skills/coding/SKILL.md'
        target.write_text('changed')
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1, coding_skills=[skill])
    with pytest.raises(Exception, match='(bundle|coding).*changed'):
        campaign.run()
    assert roles == ['actor']


def test_changed_coding_copy_is_rejected_immediately(tmp_path, monkeypatch):
    import actor_analyzer_controller as controller
    skill = tmp_path / 'coding'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('original')
    campaign = make_campaign(tmp_path, None, coding_skills=[skill])
    original = controller.shutil.copytree
    def changed_copy(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        if Path(destination).parent.name == 'coding-skills':
            (Path(destination) / 'SKILL.md').write_text('changed during copy')
        return result
    monkeypatch.setattr(controller.shutil, 'copytree', changed_copy)
    with pytest.raises(Exception, match='coding snapshot changed'):
        campaign.prepare()


def test_valid_profile_uses_controller_owned_evidence_despite_report_path_conflict(tmp_path):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            relative = 'evidence/call/results/summary.json'
            summary = phase / 'candidate-snapshot' / relative
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'state': 'completed', 'correctness': True,
                'source_sha256': digest(campaign.workspace / 'candidate.py'),
                'selector': 'task_kernel', 'cases': ['1'],
                'captures': [{'pipe_rows': [{'vector_ratio': '40'}]}]}))
            (phase / 'reports').mkdir()
            (phase / 'reports/evidence').write_text('unrelated mistaken file')
            report = {'candidate_commit': commit, 'profiling': {'status': 'acquired'},
                      'evidence': [{'path': relative, 'kind': 'profile', 'kernels': ['task_kernel'], 'cases': ['1']}],
                      'observations': [], 'bottlenecks': [], 'ceilings': [], 'unknowns': []}
            (phase / 'reports/analysis.json').write_text(json.dumps(report))
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    result = campaign.run()
    assert result['phases']['round-1-analyzer']['analysis_valid']
    assert 'vector_ratio' in campaign.git('show',
        'HEAD:experiments/round-1-analyzer/trusted-evidence/call/results/summary.json')


def pending_tool(tmp_path):
    helper = tmp_path / 'trusted-pending-tool.py'
    helper.write_text('''import hashlib,json,sys
from pathlib import Path
workspace = Path(sys.argv[sys.argv.index('--workspace') + 1])
operation = sys.argv[sys.argv.index('--workspace') + 2]
receipt = workspace.parent / 'tool-state/baseline/receipt.json'
if operation == 'timing':
    receipt.parent.mkdir(parents=True)
    payload = {'call_id':'baseline','state':'running','handle':'retained-original',
               'source_sha256':hashlib.sha256((workspace/'candidate.py').read_bytes()).hexdigest()}
else:
    payload = json.loads(receipt.read_text())
    payload.update(state='completed',correctness=True,
        launch_contract={'runner_checked':True,'profile_stream_verified':sys.argv[1]=='True'})
receipt.write_text(json.dumps(payload))
print(json.dumps(payload))
''')
    return helper


@pytest.mark.parametrize('restore,stream_verified,expected', [(True, True, True),
    (False, True, False), (True, False, False)])
def test_pending_baseline_result_qualifies_restored_final_bytes_only(tmp_path, restore, stream_verified, expected):
    helper = pending_tool(tmp_path)
    seen = []
    def runner(campaign, phase, role, budget, commit):
        seen.append(role)
        if role == 'actor':
            source = campaign.workspace / 'candidate.py'
            baseline = source.read_bytes()
            config = phase / 'tool-config.json'
            config.write_text(json.dumps({'campaign': str(campaign.root), 'workspace': str(campaign.workspace),
                'command': [sys.executable, str(helper), str(stream_verified)]}))
            assert bridge(config, ['timing']) == 0
            source.write_text('value = 2\n')
            assert bridge(config, ['observe', 'baseline']) == 0
            # Public evidence is deliberately forged; qualification must use only
            # the private bridge result journal, never this writable directory.
            public = campaign.workspace / 'evidence/forged.json'
            public.parent.mkdir()
            public.write_text(json.dumps({'state':'completed','correctness':True,
                'source_sha256':digest(source),'launch_contract':{'profile_stream_verified':True}}))
            if restore:
                source.write_bytes(baseline)
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    manifest = campaign.bundle / 'task.json'
    spec = json.loads(manifest.read_text())
    spec['single_kernel'] = True
    manifest.write_text(json.dumps(spec))
    campaign = Campaign(campaign.root, campaign.bundle, campaign.model, campaign.tool_command,
                        campaign.instruction, rounds=1, runner=runner)
    result = campaign.run()
    assert ('analyzer' in seen) == expected
    assert bool(result['verified']) == expected
    retained = [json.loads(line) for line in
                (campaign.root / 'round-1-actor/qualifications.jsonl').read_text().splitlines()]
    assert len(retained) == 1
    assert retained[0]['source_sha256'] == hashlib.sha256(b'value = 0\n').hexdigest()
    assert retained[0]['handle'] == 'retained-original'


@pytest.mark.parametrize('restore', [False, True])
def test_prior_phase_observation_keeps_original_source_identity(tmp_path, restore):
    helper = pending_tool(tmp_path)
    analyzed = []
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            config = phase / 'tool-config.json'
            config.write_text(json.dumps({'campaign': str(campaign.root), 'workspace': str(campaign.workspace),
                'command': [sys.executable, str(helper), 'True']}))
            source = campaign.workspace / 'candidate.py'
            if phase.name == 'round-1-actor':
                assert bridge(config, ['timing']) == 0
                source.write_text('value = 2\n')
            assert bridge(config, ['observe', 'baseline']) == 0
            if phase.name == 'round-2-actor' and restore:
                source.write_text('value = 0\n')
        else:
            analyzed.append(commit)
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=2)
    result = campaign.run()
    assert len(analyzed) == int(restore)
    assert bool(result['verified']) == restore
    record = json.loads((campaign.root / 'round-2-actor/qualifications.jsonl').read_text())
    assert record['source_sha256'] == hashlib.sha256(b'value = 0\n').hexdigest()
    assert record['call_id'] == 'baseline' and record['handle'] == 'retained-original'


@pytest.mark.parametrize('aliases', [False, True])
def test_unavailable_pipe_analysis_handoff_uses_original_timing_compiler_paths(tmp_path, aliases):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, {'state': 'completed', 'correctness': True,
                                    'source_sha256': digest(campaign.workspace / 'candidate.py')})
        else:
            native = {
                'timing': 'evidence/retained-timing/results/summary.json',
                'compiler': 'evidence/retained-ir/compiler-evidence/candidate/ttir.txt'}
            retained = {'timing': json.dumps({'state': 'completed', 'selector': 'task_kernel',
                'correctness': [{'correctness': True, 'check': 0}],
                'timing': {'median_us': 3245.815}, 'source_sha256': digest(campaign.workspace / 'candidate.py')}),
                'compiler': 'tt.func @task_kernel() { return }'}
            (phase / 'reports').mkdir()
            evidence = []
            for kind, relative in native.items():
                original = phase / 'candidate-snapshot' / relative
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_text(retained[kind])
                if kind == 'compiler':
                    summary = original.parents[2] / 'results/summary.json'
                    summary.parent.mkdir()
                    summary.write_text(json.dumps(dict(state='completed', implementation='candidate',
                        source_sha256=digest(campaign.workspace / 'candidate.py'),
                        correctness=[dict(correctness=True, check=0, selector='task_kernel')],
                        compiler_artifacts={'ttir.txt': digest(original)})))
                if aliases:
                    relative = 'evidence/' + kind + '.json'
                    alias = phase / 'reports' / relative
                    alias.parent.mkdir(parents=True, exist_ok=True)
                    alias.write_text(retained[kind])
                evidence.append({'path': relative, 'kind': kind, 'kernels': ['task_kernel'], 'cases': ['0']})
            report = {'candidate_commit': commit, 'profiling': {'status': 'unavailable',
                      'reason': 'Retained pipe handle remained observation-unavailable'},
                      'evidence': evidence, 'observations': [{'finding': 'Timing retained; pipe bottleneck unknown',
                      'evidence': [evidence[0]['path']], 'confidence': 'high'}],
                      'bottlenecks': [], 'ceilings': [], 'unknowns': ['Pipe metrics unavailable']}
            (phase / 'reports/analysis.json').write_text(json.dumps(report))
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    result = campaign.run()
    assert result['phases']['round-1-analyzer']['analysis_valid'] == (not aliases)
    assert (campaign.workspace / 'analysis.json').exists() == (not aliases)
    if not aliases:
        handoff = json.loads((campaign.workspace / 'analysis.json').read_text())
        assert handoff['profiling']['status'] == 'unavailable'
        assert handoff['evidence'][0]['path'] == 'evidence/retained-timing/results/summary.json'
        assert 'tt.func @task_kernel' in campaign.git('show',
            'HEAD:experiments/round-1-analyzer/trusted-evidence/retained-ir/compiler-evidence/candidate/ttir.txt')


@pytest.mark.parametrize('kind,defect', [(kind, defect)
    for kind in ('timing', 'compiler', 'profile')
    for defect in ('stale-source', 'baseline', 'wrong-kernel', 'wrong-case',
        'missing-summary', 'failed-call', 'missing-identities', 'artifact-mismatch', 'none')
    if defect != 'artifact-mismatch' or kind == 'compiler'])
def test_analysis_binds_all_citations_to_authoritative_candidate_call(tmp_path, kind, defect):
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, dict(state='completed', correctness=True,
                source_sha256=digest(campaign.workspace / 'candidate.py')))
        else:
            call = phase / 'candidate-snapshot/evidence/call'
            (call / 'results').mkdir(parents=True)
            summary = dict(state='completed', implementation='candidate',
                source_sha256=digest(campaign.workspace / 'candidate.py'),
                correctness=[dict(correctness=True, check=0, selector='task_kernel')],
                selector='task_kernel', cases=['case-0'], timing={'median_us': 10},
                captures=[dict(selector='task_kernel', pipe_rows=[{'vector_ratio': 40}])])
            relative = 'evidence/call/results/summary.json'
            if kind == 'compiler':
                artifact = call / 'compiler-evidence/candidate/ttir.txt'
                artifact.parent.mkdir(parents=True)
                artifact.write_text('tt.func @task_kernel() { return }')
                summary['compiler_artifacts'] = {'ttir.txt': digest(artifact)}
                relative = 'evidence/call/compiler-evidence/candidate/ttir.txt'
                if defect == 'artifact-mismatch': artifact.write_text('different artifact')
            if defect == 'stale-source': summary['source_sha256'] = 'a' * 64
            if defect == 'baseline': summary['implementation'] = 'baseline'
            if defect == 'failed-call': summary['state'] = 'failed'
            if defect == 'missing-identities':
                for field in ('selector', 'cases'): summary.pop(field)
                summary['correctness'] = True
                summary['captures'][0].pop('selector')
            if defect != 'missing-summary':
                (call / 'results/summary.json').write_text(json.dumps(summary))
            report = dict(candidate_commit=commit,
                profiling={'status': 'acquired'} if kind == 'profile' else {'status': 'unavailable', 'reason': 'not requested'},
                evidence=[dict(path=relative, kind=kind,
                    kernels=['invented' if defect == 'wrong-kernel' else 'task_kernel'],
                    cases=['invented' if defect == 'wrong-case' else 'case-0'])],
                observations=[], bottlenecks=[], ceilings=[], unknowns=[])
            (phase / 'analysis.json').write_text(json.dumps(report))
        return {'exit_code': 0}
    campaign = make_campaign(tmp_path, runner, rounds=1)
    state = campaign.run()
    assert state['phases']['round-1-analyzer']['analysis_valid'] == (defect == 'none')


def test_checkpoint_resumes_original_four_rounds(tmp_path):
    seen = []
    def runner(campaign, phase, role, budget, commit):
        seen.append((phase.name, budget))
        if role == 'actor':
            (campaign.workspace / 'candidate.py').write_text('value = ' + str(len(seen)))
            campaign.qualify(phase, dict(state='completed', correctness=True,
                source_sha256=digest(campaign.workspace / 'candidate.py')))
        else:
            (phase / 'analysis.json').write_text(json.dumps(dict(candidate_commit=commit,
                profiling=dict(status='unavailable', reason='fixture'), evidence=[],
                observations=[], bottlenecks=[], ceilings=[], unknowns=[])))
        return dict(exit_code=0)
    campaign = make_campaign(tmp_path, runner)
    assert campaign.run(stop_after_round=1)['state'] == 'checkpoint'
    assert len(seen) == 2 and campaign.rounds == 4
    assert campaign.run()['state'] == 'completed'
    assert len(seen) == 8
    assert len({name for name, budget in seen}) == 8


@pytest.mark.parametrize('role,returncode,timed_out,expected', [
    ('analyzer', 0, False, 0), ('actor', 0, False, 3),
    ('analyzer', 2, False, 2), ('analyzer', 0, True, 124)])
def test_role_launcher_analyzer_completion_requires_process_success_not_commits(
        tmp_path, monkeypatch, role, returncode, timed_out, expected):
    import actor_analyzer_controller as controller
    from blind_optimization_launcher import completion_exit_code
    campaign = make_campaign(tmp_path, None, rounds=1)
    campaign.prepare()
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    campaign.auth_home = auth
    phase = campaign.root / ('round-1-' + role)
    launcher = RoleLauncher(campaign, phase, role, 60, None)
    def launch(self, preflight_only=False):
        self.root.mkdir()
        result = dict(returncode=returncode, timed_out=timed_out,
                      preflight_only=False, candidate_commits=0)
        result['exit_code'] = completion_exit_code(result)
        controller.save(self.root / 'receipt.json', result)
        return result
    monkeypatch.setattr(controller.BlindLauncher, 'launch', launch)
    monkeypatch.setattr(controller, 'cleanup_phase', lambda path: dict(state='terminal'))
    result = launcher.launch()
    assert result['exit_code'] == expected
    assert json.loads((phase / 'receipt.json').read_text()) == result
    if role == 'analyzer' and expected == 0:
        from campaign_batch import reconcile, CELLS
        observed = dict(state='checkpoint', phases={
            'round-1-actor': dict(state='terminal', candidate_commit='abc', candidate_verified=True),
            'round-1-analyzer': dict(state='terminal', candidate_commit='abc',
                                    analysis_valid=True, result=result)})
        assert not reconcile(None, {CELLS[0]: observed}, 1)['stopped']
