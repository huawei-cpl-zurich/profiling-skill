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
