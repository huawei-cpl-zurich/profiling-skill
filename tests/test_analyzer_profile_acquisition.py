"""Required profiling is controller-driven and consumes the original allowance."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyzer_profile_acquisition import acquire
from actor_analyzer_controller import RoleLauncher, save


def setup(tmp_path, status='new'):
    workspace, history = tmp_path / 'snapshot', tmp_path / 'history'
    workspace.mkdir()
    history.mkdir()
    context = dict(status=status, source_sha256='a'*64)
    save(history / 'context.json', context)
    execution = dict(utc_start='2026-10-22T00:00:00+00:00',
        utc_deadline='2026-10-22T00:10:00+00:00', wall_budget_seconds=600, finish_reserve_seconds=120)
    save(workspace / 'execution.json', execution)
    launcher = SimpleNamespace(root=tmp_path, workspace=workspace,
        campaign=SimpleNamespace(history=history), deadline_monotonic=600,
        finish_reserve_seconds=120, tool_command=['managed-bridge'], journal=tmp_path / 'commands.jsonl')
    return launcher, execution


def result(**payload):
    return dict(returncode=0, timed_out=False, stdout=json.dumps(payload), stderr='')


def test_profile_observes_same_call_and_restores_original_deadline(tmp_path):
    launcher, original = setup(tmp_path)
    now, calls = [0.], []
    def run(command, cwd, timeout):
        calls.append((command[-2:], timeout))
        bounded = json.loads((cwd / 'execution.json').read_text())
        assert datetime.fromisoformat(bounded['utc_deadline']).minute == 8
        assert bounded['original_utc_deadline'] == original['utc_deadline']
        assert json.loads((tmp_path / 'required-profile.json').read_text())['dispatch_intent']
        now[0] += 100
        receipt = tmp_path / 'tool-state/call/receipt.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        save(receipt, dict(request=dict(operation='pipes', source_sha256='a'*64),
                           handle='remote:gz-a3:job:original', state='running'))
        if len(calls) == 1:
            return result(state='reconnecting', call_id='call', handle='remote:gz-a3:job:original')
        summary = cwd / 'evidence/call/results/summary.json'
        summary.parent.mkdir(parents=True)
        save(summary, dict(state='completed', source_sha256='a'*64))
        return result(state='completed', call_id='call')
    acquire(launcher, run, save, clock=lambda: now[0], pause=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert calls[0][0][-1] == 'pipes'
    assert calls[1][0] == ['observe', 'call']
    assert calls[0][1] == 474 and calls[1][1] == 373
    assert json.loads((launcher.workspace / 'execution.json').read_text()) == original
    required = json.loads((launcher.campaign.history / 'context.json').read_text())['required_profile']
    assert required['evidence'] == 'evidence/call/results/summary.json'
    assert required['handle'] == 'remote:gz-a3:job:original'
    assert required['state'] == 'completed'
    acquire(launcher, lambda *args: pytest.fail('terminal acquisition redispatched'), save, clock=lambda: now[0])


def test_interrupted_intent_without_call_never_redispatches(tmp_path):
    launcher, original = setup(tmp_path)
    save(tmp_path / 'required-profile.json', dict(source_sha256='a'*64, state='running', dispatch_intent=True))
    acquire(launcher, lambda *args: pytest.fail('unknown intent redispatched'), save, clock=lambda: 0)
    required = json.loads((tmp_path / 'required-profile.json').read_text())
    assert required['state'] == 'observation-required'
    assert not required.get('command_dispatched')
    assert json.loads((launcher.workspace / 'execution.json').read_text()) == original


def test_existing_receipt_resumes_observation_and_timeout_preserves_reserve(tmp_path):
    launcher, original = setup(tmp_path)
    save(tmp_path / 'required-profile.json', dict(source_sha256='a'*64, state='running', dispatch_intent=True))
    receipt = tmp_path / 'tool-state/original/receipt.json'
    receipt.parent.mkdir(parents=True)
    save(receipt, dict(request=dict(verb='pipes', source_sha256='a'*64), handle='retained'))
    now = [0.]
    def run(command, cwd, timeout):
        assert command[-2:] == ['observe', 'original']
        assert timeout == 474
        now[0] += timeout
        return dict(returncode=-15, timed_out=True, stdout='', stderr='interrupted observer')
    acquire(launcher, run, save, clock=lambda: now[0])
    state = json.loads((tmp_path / 'required-profile.json').read_text())
    assert state['state'] == 'observation-required'
    assert state['call_id'] == 'original' and state['handle'] == 'retained'
    assert launcher.deadline_monotonic - now[0] >= 120
    assert json.loads((launcher.workspace / 'execution.json').read_text()) == original


def test_unchanged_and_nonanalyzer_roles_do_not_acquire(tmp_path, monkeypatch):
    launcher, _ = setup(tmp_path, 'unchanged')
    acquire(launcher, lambda *args: pytest.fail('unchanged profile dispatched'), save)
    assert not (tmp_path / 'required-profile.json').exists()
    monkeypatch.setattr('analyzer_profile_acquisition.acquire', lambda *args: pytest.fail('skipped role profiled'))
    for role, contract in (('actor', 2), ('analyzer', 1)):
        RoleLauncher.before_agent(SimpleNamespace(role=role, campaign=SimpleNamespace(analysis_contract=contract)))


def test_terminal_tool_failure_is_available_to_analyzer_without_retry(tmp_path):
    launcher, _ = setup(tmp_path)
    acquire(launcher, lambda *args: result(state='failed', error='device unavailable'), save, clock=lambda: 0)
    state = json.loads((launcher.campaign.history / 'context.json').read_text())['required_profile']
    assert state['state'] == 'failed' and state['error'] == 'device unavailable'
    assert state['command_dispatched']


@pytest.mark.parametrize('interruption', ['during-profile', 'after-profile', 'model-intent', 'expired', 'before-context'])
def test_campaign_run_resumes_pre_model_interruption_without_new_clock(tmp_path, monkeypatch, interruption):
    import actor_analyzer_controller as controller
    import blind_optimization_launcher as launcher_module
    from test_analyzer_history import report as make_report
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('kernel = 1\n')
    (bundle / 'task.json').write_text(json.dumps(dict(public_files=['candidate.py'])))
    calls, models = [], []
    original_launch = controller.Campaign.launch_role
    def role(self, campaign, phase, kind, budget, commit):
        if kind == 'actor':
            phase.mkdir()
            campaign.qualify(phase, dict(state='completed', correctness=True,
                source_sha256=controller.digest(campaign.workspace / 'candidate.py')))
            return dict(exit_code=0)
        return original_launch(campaign, campaign, phase, kind, budget, commit)
    monkeypatch.setattr(controller.Campaign, 'launch_role', role)
    monkeypatch.setattr(controller.RoleLauncher, 'preflight', lambda self: {})
    monkeypatch.setattr(controller.RoleLauncher, 'base_command', lambda self: ['fake-model'])
    monkeypatch.setattr(controller, 'cleanup_phase', lambda phase: dict(state='terminal', calls=[]))
    campaign = controller.Campaign(tmp_path / 'campaign', bundle, dict(name='fake', reasoning_effort='low'),
                                   ['fake-tool'], 'task', rounds=1)
    phase = campaign.root / 'round-1-analyzer'
    fail = [True]
    prepare = controller.RoleLauncher.prepare
    def interrupted_prepare(self):
        prepare(self)
        if fail[0] and interruption == 'before-context':
            fail[0] = False
            raise KeyboardInterrupt('preparation finished but clock not persisted')
    monkeypatch.setattr(controller.RoleLauncher, 'prepare', interrupted_prepare)
    def tool(command, cwd, timeout):
        calls.append(command[-2:])
        ctx = json.loads((campaign.history / 'context.json').read_text())
        private = phase / 'tool-state/original/receipt.json'
        private.parent.mkdir(parents=True, exist_ok=True)
        save(private, dict(request=dict(operation='pipes', source_sha256=ctx['source_sha256']),
                           handle='remote:gz-a3:job:original', state='running'))
        if fail[0] and interruption in ('during-profile', 'model-intent', 'expired'):
            fail[0] = False
            raise KeyboardInterrupt('controller disappeared after dispatch')
        summary = cwd / 'evidence/original/results/summary.json'
        summary.parent.mkdir(parents=True, exist_ok=True)
        save(summary, dict(state='completed', correctness=True, source_sha256=ctx['source_sha256'],
            cases=['case'], captures=[dict(selector='kernel', pipe_rows=[dict(mac_ratio=1)])]))
        return result(state='completed', call_id='original', handle='remote:gz-a3:job:original')
    monkeypatch.setattr(controller, 'execute', tool)
    before = controller.RoleLauncher.before_agent
    def interrupted_after_profile(self):
        before(self)
        if fail[0] and interruption == 'after-profile':
            fail[0] = False
            raise KeyboardInterrupt('before model intent')
    monkeypatch.setattr(controller.RoleLauncher, 'before_agent', interrupted_after_profile)
    def model(command, cwd, timeout, output_files=None):
        models.append(timeout)
        assert (phase / 'model-dispatch.json').exists()
        ctx = json.loads((campaign.history / 'context.json').read_text())
        value = make_report(ctx)
        path = 'evidence/original/results/summary.json'
        value.update(profiling=dict(status='acquired'),
            evidence=[dict(path=path, kind='profile', kernels=['kernel'], cases=['case'])],
            bottlenecks=[dict(finding='Possible memory limit', evidence=[path], confidence='low')],
            utilization=[dict(status='measured', description='MAC', cases=['case'], kernels=['kernel'], evidence=[path])])
        save(phase / 'reports/analysis.json', value)
        return result()
    monkeypatch.setattr(launcher_module, 'execute', model)
    with pytest.raises(KeyboardInterrupt): campaign.run()
    if interruption == 'before-context':
        assert campaign.run()['state'] == 'observation-required'
        assert not (phase / 'role-execution.json').exists()
        assert not calls and not models
        return
    original = json.loads((phase / 'role-execution.json').read_text())
    if interruption == 'model-intent':
        save(phase / 'model-dispatch.json', dict(dispatch_intent=True))
        assert campaign.run()['state'] == 'observation-required'
        assert len(calls) == 1 and not models
        return
    if interruption == 'expired':
        original.update(utc_start='2000-01-01T00:00:00+00:00', utc_deadline='2000-01-01T00:10:00+00:00')
        save(phase / 'role-execution.json', original)
    state = campaign.run()
    assert state['state'] == 'completed'
    assert json.loads((phase / 'candidate-snapshot/execution.json').read_text()) == original
    assert calls[0][-1] == 'pipes'
    if interruption == 'during-profile': assert calls[1] == ['observe', 'original']
    else: assert len(calls) == 1
    if interruption == 'expired':
        assert not models
        assert state['phases']['round-1-analyzer']['result']['timed_out']
    else:
        assert len(models) == 1 and 0 < models[0] < 600
        assert state['phases']['round-1-analyzer']['analysis_valid']
        assert campaign.run() == state
