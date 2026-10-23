"""Strict report repair over real-shaped original evidence, without device work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from actor_analyzer_controller import Campaign, bridge, digest, save
import analyzer_history
from analyzer_report_repair import permitted_citations, repair
from test_analyzer_history import report as report_shape


def fixture(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('kernel = 1\n')
    (bundle / 'task.json').write_text(json.dumps(dict(public_files=['candidate.py'])))
    campaign = Campaign(tmp_path / 'campaign', bundle, dict(name='fake', reasoning_effort='low'), ['fake-remote'], 'task')
    state = campaign.prepare()
    commit = campaign.git('rev-parse', 'HEAD').strip()
    phase = campaign.root / 'round-1-analyzer'
    workspace = phase / 'candidate-snapshot'
    workspace.mkdir(parents=True)
    (workspace / 'candidate.py').write_bytes((campaign.workspace / 'candidate.py').read_bytes())
    source = digest(workspace / 'candidate.py')
    ctx = analyzer_history.context(campaign.history, (workspace / 'candidate.py').read_bytes(), commit, 1)
    save(campaign.history / 'context.json', ctx)
    evidence = []
    for call, kind in [('profile', 'profile'), ('timing', 'timing'), ('check', 'compiler')]:
        summary = dict(state='completed', correctness=True, source_sha256=source)
        if call != 'check': summary.update(cases=['case'], selector='kernel')
        if call == 'profile': summary['captures'] = [dict(selector='kernel', pipe_rows=[dict(mac_ratio=1)])]
        if call == 'timing': summary['timing'] = dict(median_us=10)
        path = f'evidence/{call}/results/summary.json'
        original = workspace / path
        original.parent.mkdir(parents=True)
        save(original, summary)
        private = phase / 'tool-state' / call / 'receipt.json'
        private.parent.mkdir(parents=True)
        save(private, dict(call_id=call, target='gz-a3', runtime='native',
            request=dict(operation='pipes' if call == 'profile' else 'timing', source_sha256=source)))
        evidence.append(dict(path=path, kind=kind, cases=['case' if call != 'check' else 'matmul'], kernels=['kernel']))
    value = report_shape(ctx)
    value.update(profiling=dict(status='acquired'), evidence=evidence,
        bottlenecks=[dict(finding='Possible memory bottleneck', confidence='low', evidence=[evidence[0]['path']])],
        utilization=[dict(status='measured', description='MAC activity', cases=['case'], kernels=['kernel'], evidence=[evidence[0]['path']])])
    reports = phase / 'reports'
    reports.mkdir()
    save(reports / 'analysis.json', value)
    save(phase / 'required-profile.json', dict(state='completed', source_sha256=source, evidence=evidence[0]['path']))
    frozen = phase / 'frozen'
    frozen.mkdir()
    (frozen / 'prompt.md').write_text('Original report contract')
    save(phase / 'tool-config.json', dict(command=['fake-remote'], workspace=str(workspace), campaign=str(campaign.root)))
    save(phase / 'role-execution.json', dict(utc_start='2026-10-22T00:00:00+00:00', utc_deadline='2026-10-22T00:10:00+00:00', wall_budget_seconds=600))
    launcher = SimpleNamespace(root=phase, workspace=workspace, frozen=frozen, reports=reports,
        commit=commit, campaign=campaign, deadline_monotonic=600, model=campaign.model, base_command=lambda: ['fake-model'])
    return launcher, value, state


def success():
    return dict(returncode=0, timed_out=False, stdout='', stderr='')


def test_invalid_optional_compiler_citation_is_repaired_with_original_metadata(tmp_path, monkeypatch, capsys):
    launcher, raw, state = fixture(tmp_path)
    with pytest.raises(ValueError, match='kernel/case'): launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    commands = []
    def run(command, cwd, allowance, outputs):
        commands.append(command)
        assert 0 < allowance <= 84
        feedback = json.loads((cwd / 'report-repair/feedback.json').read_text())
        assert 'kernel/case' in feedback['error']
        allowed = {item['path']: item for item in feedback['allowed_citations']}
        assert allowed[raw['evidence'][0]['path']]['kinds'] == ['profile']
        assert allowed[raw['evidence'][1]['path']]['kinds'] == ['timing']
        assert allowed[raw['evidence'][2]['path']]['kinds'] == []
        # Every remote operation is blocked at the actual trusted bridge.
        monkeypatch.setattr('actor_analyzer_controller.subprocess.run', lambda *a, **k: pytest.fail('remote command dispatched'))
        for op in ('pipes', 'timing', 'check', 'observe', 'help'):
            assert bridge(cwd / 'tool-config.json', [op]) == 2
        with pytest.raises(ValueError): bridge(cwd / 'tool-config.json', ['stop'])
        monkeypatch.undo()
        corrected = copy.deepcopy(raw)
        corrected['evidence'].pop()
        save(launcher.reports / 'analysis.json', corrected)
        return success()
    result = repair(launcher, success(), run, save, clock=lambda: 0)
    assert result['report_repair']['valid'] and len(commands) == 1
    assert json.loads((launcher.root / 'report-repair/initial-report.json').read_text()) == raw
    assert digest(launcher.workspace / 'candidate.py') == raw['source_sha256']
    launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    entry = dict(candidate_commit=launcher.commit)
    launcher.campaign.finish_phase(launcher.root, 'analyzer', entry, state)
    assert entry['analysis_valid']
    record = json.loads((launcher.campaign.history / 'round-1/record.json').read_text())
    assert record['report_repair']['status'] == 'valid'
    assert (launcher.campaign.workspace / 'experiments/round-1-analyzer/report-repair/initial-report.json').is_file()
    repair(launcher, success(), lambda *a: pytest.fail('second repair launched'), save, clock=lambda: 0)


@pytest.mark.parametrize('failure', ['invalid-again', 'timeout', 'candidate-change', 'lost-result'])
def test_repair_failure_remains_strict_and_never_retries(tmp_path, failure):
    launcher, raw, state = fixture(tmp_path)
    calls = []
    def run(*args):
        calls.append(1)
        if failure == 'lost-result': raise KeyboardInterrupt('repair observation lost')
        if failure in ('timeout', 'candidate-change'):
            corrected = copy.deepcopy(raw);corrected['evidence'].pop()
            save(launcher.reports / 'analysis.json', corrected)
        if failure == 'candidate-change': (launcher.workspace / 'candidate.py').write_text('changed')
        return dict(success(), returncode=124, timed_out=True) if failure == 'timeout' else success()
    if failure == 'lost-result':
        with pytest.raises(KeyboardInterrupt): repair(launcher, success(), run, save, clock=lambda: 0)
    else:
        repair(launcher, success(), run, save, clock=lambda: 0)
    with pytest.raises(ValueError): launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    repair(launcher, success(), lambda *a: pytest.fail('duplicate repair'), save, clock=lambda: 0)
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['already-valid', 'nonzero', 'initial-timeout', 'no-budget'])
def test_no_repair_when_not_needed_or_not_permitted(tmp_path, mode):
    launcher, raw, _ = fixture(tmp_path)
    result = success()
    if mode == 'already-valid':
        raw['evidence'].pop();save(launcher.reports / 'analysis.json', raw)
    if mode == 'nonzero': result['returncode'] = 1
    if mode == 'initial-timeout': result.update(returncode=124, timed_out=True)
    clock = (lambda: 599) if mode == 'no-budget' else (lambda: 0)
    assert repair(launcher, result, lambda *a: pytest.fail('repair not permitted'), save, clock=clock) == result
    assert not (launcher.root / 'report-repair/dispatch.json').exists()


def test_malformed_report_and_unreadable_optional_metadata_can_be_repaired(tmp_path):
    launcher, raw, _ = fixture(tmp_path)
    (launcher.reports / 'analysis.json').write_text('{broken')
    (launcher.workspace / 'evidence/check/results/summary.json').write_text('{broken')
    assert permitted_citations(launcher)[0]['kinds'] == []
    def run(*args):
        corrected = copy.deepcopy(raw);corrected['evidence'].pop()
        save(launcher.reports / 'analysis.json', corrected)
        return success()
    assert repair(launcher, success(), run, save, clock=lambda: 0)['report_repair']['valid']


def test_real_repair_mount_allows_only_existing_report_write(tmp_path):
    import shutil
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('Bubblewrap and Codex runtime required')
    from test_actor_analyzer_controller import make_campaign
    from actor_analyzer_controller import RoleLauncher
    from blind_optimization_launcher import execute
    root = tmp_path / 'sandbox'
    root.mkdir()
    campaign = make_campaign(root, None, rounds=1, analysis_contract=2)
    campaign.prepare()
    auth = root / 'auth';auth.mkdir();(auth / 'auth.json').write_text('{}')
    campaign.auth_home = auth
    commit = campaign.git('rev-parse', 'HEAD').strip()
    save(campaign.history / 'context.json', analyzer_history.context(campaign.history,
        (campaign.workspace / 'candidate.py').read_bytes(), commit, 1))
    launcher = RoleLauncher(campaign, campaign.root / 'round-1-analyzer', 'analyzer', 600, commit)
    launcher.prepare()
    directory = launcher.root / 'report-repair';directory.mkdir()
    save(directory / 'feedback.json', {})
    save(launcher.reports / 'analysis.json', {})
    launcher.report_repair_only = True
    code = '''from pathlib import Path
Path('/analysis/analysis.json').write_text('{"fixed": true}')
for path in ('/analysis/extra.json', '/workspace/candidate.py', '/report-repair/feedback.json'):
    try: Path(path).write_text('forbidden')
    except OSError: pass
    else: raise AssertionError(path + ' writable')
'''
    try:
        result = execute([*launcher.base_command(), 'python3', '-c', code], launcher.root, 30)
        assert result['returncode'] == 0, result
        assert json.loads((launcher.reports / 'analysis.json').read_text()) == dict(fixed=True)
    finally:
        launcher.socket_temporary.cleanup()


def test_simultaneous_optional_citation_and_unsupported_diff_errors_are_both_reported(tmp_path):
    launcher, raw, _ = fixture(tmp_path)
    raw['optimizations'] = [dict(status='observed', description='Tile geometry changed', diff_lines=['+BLOCK_M=64'])]
    save(launcher.reports / 'analysis.json', raw)
    with pytest.raises(ValueError) as errors:
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    assert 'kernel/case' in str(errors.value) and 'diff_lines' in str(errors.value)
    partial = copy.deepcopy(raw);partial['evidence'].pop()
    save(launcher.reports / 'analysis.json', partial)
    with pytest.raises(ValueError, match='diff_lines'):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    save(launcher.reports / 'analysis.json', raw)
    def run(*args):
        feedback = json.loads((launcher.root / 'report-repair/feedback.json').read_text())
        assert 'kernel/case' in feedback['error'] and 'diff_lines' in feedback['error']
        assert not feedback['source_diff_available']
        assert feedback['trusted_context']['status'] == 'first'
        assert 'use none' in feedback['optimization_attribution']
        corrected = copy.deepcopy(raw);corrected['evidence'].pop()
        corrected['optimizations'] = [dict(status='none', description='No provided historical source diff')]
        save(launcher.reports / 'analysis.json', corrected)
        return success()
    assert repair(launcher, success(), run, save, clock=lambda: 0)['report_repair']['valid']
    assert json.loads((launcher.root / 'report-repair/initial-report.json').read_text()) == raw


def test_publication_requires_retained_artifact_but_precheck_accepts_original(tmp_path):
    import shutil
    launcher, raw, state = fixture(tmp_path)
    raw['evidence'].pop();save(launcher.reports / 'analysis.json', raw)
    # Pre-cleanup validation uses the actual original captures.
    launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    with pytest.raises(ValueError, match='missing retained evidence'):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit, require_retained=True)
    shutil.copytree(launcher.workspace / 'evidence', launcher.root / 'trusted-evidence',
                    ignore=shutil.ignore_patterns('*.tar.gz'))
    launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit, require_retained=True)
    entry = dict(candidate_commit=launcher.commit)
    launcher.campaign.finish_phase(launcher.root, 'analyzer', entry, state)
    assert entry['analysis_valid']
    for item in raw['evidence']:
        retained = launcher.campaign.history / 'round-1' / item['path']
        assert retained.read_bytes() == (launcher.workspace / item['path']).read_bytes()


def test_excluded_archive_citation_is_rejected_before_repair_and_publication(tmp_path):
    from actor_analyzer_controller import validate_citation
    launcher, raw, state = fixture(tmp_path)
    archive = launcher.workspace / 'evidence/check/compiler-evidence/candidate/bundle.tar.gz'
    archive.parent.mkdir(parents=True);archive.write_bytes(b'compiler artifact archive')
    summary_path = launcher.workspace / 'evidence/check/results/summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(selector='kernel', cases=['case'], compiler_artifacts={archive.name: digest(archive)})
    save(summary_path, summary)
    item = dict(path=str(archive.relative_to(launcher.workspace)), kind='compiler', kernels=['kernel'], cases=['case'])
    # Attribution/inventory is genuine; retention policy still excludes the artifact.
    validate_citation(item, summary, raw['source_sha256'], archive)
    raw['evidence'][-1] = item;save(launcher.reports / 'analysis.json', raw)
    with pytest.raises(ValueError, match='archive excluded'):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    entry = dict(candidate_commit=launcher.commit)
    launcher.campaign.finish_phase(launcher.root, 'analyzer', entry, state)
    assert not entry['analysis_valid'] and 'archive excluded' in entry['analysis_error']
    assert not (launcher.campaign.history / 'round-1' / item['path']).exists()


def test_retained_compiler_artifact_also_requires_its_retained_summary(tmp_path):
    import shutil
    launcher, raw, _ = fixture(tmp_path)
    artifact = launcher.workspace / 'evidence/check/compiler-evidence/candidate/kernel.ll'
    artifact.parent.mkdir(parents=True);artifact.write_text('compiler IR')
    summary_path = launcher.workspace / 'evidence/check/results/summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(selector='kernel', cases=['case'], compiler_artifacts={artifact.name: digest(artifact)})
    save(summary_path, summary)
    raw['evidence'][-1] = dict(path=str(artifact.relative_to(launcher.workspace)), kind='compiler', kernels=['kernel'], cases=['case'])
    save(launcher.reports / 'analysis.json', raw)
    shutil.copytree(launcher.workspace / 'evidence', launcher.root / 'trusted-evidence')
    launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit, require_retained=True)
    (launcher.root / 'trusted-evidence/check/results/summary.json').unlink()
    launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
    with pytest.raises(ValueError, match='authoritative call summary'):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit, require_retained=True)


@pytest.mark.parametrize('valid_host_report', [False, True])
def test_symlink_report_is_rejected_before_validation_copy_or_model_dispatch(tmp_path, valid_host_report):
    launcher, raw, _ = fixture(tmp_path)
    host_file = tmp_path / 'host-only-report'
    raw['evidence'].pop()
    payload = json.dumps(raw) if valid_host_report else 'host-only confidential contents'
    host_file.write_text(payload)
    report = launcher.reports / 'analysis.json'
    report.unlink();report.symlink_to(host_file)
    launcher.base_command = lambda: pytest.fail('unsafe report was mounted')
    result = repair(launcher, success(), lambda *a: pytest.fail('repair model dispatched'), save, clock=lambda: 0)
    assert result['report_repair'] == dict(attempts=0, valid=False)
    directory = launcher.root / 'report-repair'
    assert not (directory / 'initial-report.json').exists()
    assert not (directory / 'dispatch.json').exists()
    assert all(payload.encode() not in path.read_bytes() for path in directory.iterdir() if path.is_file())
    assert host_file.read_text() == payload
    with pytest.raises((OSError, ValueError)):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)


def test_repaired_symlink_is_not_followed_into_retained_report(tmp_path):
    launcher, _, _ = fixture(tmp_path)
    host_file = tmp_path / 'host-only-report'
    host_file.write_text('host-only confidential contents')
    def run(*args):
        report = launcher.reports / 'analysis.json'
        report.unlink();report.symlink_to(host_file)
        return success()
    result = repair(launcher, success(), run, save, clock=lambda: 0)
    assert result['report_repair'] == dict(attempts=1, valid=False)
    directory = launcher.root / 'report-repair'
    assert not (directory / 'repaired-report.json').exists()
    assert all(host_file.read_bytes() not in path.read_bytes() for path in directory.iterdir() if path.is_file())
    with pytest.raises((OSError, ValueError)):
        launcher.campaign.validate_phase_analysis(launcher.root, launcher.commit)
