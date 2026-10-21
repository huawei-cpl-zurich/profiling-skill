from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from actor_analyzer_controller import Campaign, RoleLauncher, bridge, digest


def make_campaign(tmp_path, runner, **options):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('value = 0\n')
    (bundle / 'API.md').write_text('Public API')
    (bundle / 'private.py').write_text('secret')
    (bundle / 'task.json').write_text(json.dumps({'public_files': ['candidate.py', 'API.md']}))
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


def test_actor_only_has_no_analyzer_cost(tmp_path):
    roles = []
    campaign = make_campaign(tmp_path, lambda c, p, r, b, v: roles.append(r) or {'exit_code': 0},
                             actor_only=True, rounds=2)
    result = campaign.run()
    assert roles == ['actor', 'actor']
    assert result['configured_analyzer_seconds'] == 0


def test_bridge_rejects_stale_source_and_routes_pending_observation(tmp_path, capsys):
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
