"""Functional gates before any device work, without synchronized host clocks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/phase_deadline.py'
SPEC = importlib.util.spec_from_file_location('phase_deadline', SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize('activation', ['absent', 'late', 'valid'])
def test_worker_cannot_touch_device_without_timely_activation(tmp_path, activation):
    script = ('import sys; from pathlib import Path; '
              f'sys.path.insert(0,{str(SCRIPT.parent)!r}); '
              'from phase_deadline import wait_for_activation; '
              f'wait_for_activation(Path({str(tmp_path / "activation.json")!r}), .2); '
              f'Path({str(tmp_path / "device-touched")!r}).touch()')
    process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    ready = json.loads(process.stdout.readline().split('=', 1)[1])
    if activation != 'absent':
        cutoff = ready['monotonic'] + (.15 if activation == 'valid' else -.01)
        (tmp_path / 'activation.json').write_text(json.dumps(dict(cutoff_monotonic=cutoff)))
    stdout, stderr = process.communicate(timeout=3)
    assert (process.returncode == 0) == (activation == 'valid'), stdout + stderr
    assert (tmp_path / 'device-touched').exists() == (activation == 'valid')


def receipt():
    return dict(handle='remote:t:job:original', target='t', remote_folder='/original',
                request=dict(activation_required=True,
                             utc_deadline=(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()))


def test_host_gate_preserves_original_cutoff_after_interrupted_upload(tmp_path):
    item = receipt()
    calls = []
    def remote(*args):
        calls.append(args)
        if args[0] == 'logs':
            return dict(content='PHASE_READY=' + json.dumps(dict(monotonic=100)))
        return dict(state='running', handle='remote:t:transfer:original')
    tool = SimpleNamespace(remote=remote, transfer=lambda r, k: dict(state='running'))
    assert gate.activate(tool, tmp_path, item) is False
    cutoff = json.loads((tmp_path / 'activation.json').read_text())['cutoff_monotonic']
    assert 150 < cutoff <= 160  # remote and host clocks need not agree
    time.sleep(.01)
    tool.transfer = lambda r, k: dict(state='completed')
    assert gate.activate(tool, tmp_path, item) is True
    assert len([x for x in calls if x[0] == 'upload']) == 1
    assert json.loads((tmp_path / 'activation.json').read_text())['cutoff_monotonic'] == cutoff
    assert item['activation_state'] == 'activated'


def test_deadline_expiry_never_uploads_activation(tmp_path):
    item = receipt()
    item['request']['utc_deadline'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    tool = SimpleNamespace(remote=lambda *a: pytest.fail('expired gate must not call transport'))
    with pytest.raises(TimeoutError):
        gate.activate(tool, tmp_path, item)


def test_no_ready_anchor_never_activates(tmp_path):
    tool = SimpleNamespace(remote=lambda *a: dict(content='starting'))
    assert gate.activate(tool, tmp_path, receipt()) is False
    assert not (tmp_path / 'activation.json').exists()


@pytest.mark.parametrize('state', ['queued', 'dispatching', 'running', 'reconnecting', 'observation-unavailable'])
def test_not_yet_created_log_waits_for_same_retained_worker(tmp_path, state):
    item, calls = receipt(), []
    ready = [False]
    def remote(*args):
        calls.append(args)
        if args[0] == 'logs':
            if not ready[0]:
                raise RuntimeError('logs/stdout HTTP404 log not found')
            return dict(content='PHASE_READY=' + json.dumps(dict(monotonic=100)))
        if args[0] == 'observe':
            return dict(state=state, handle=item['handle'])
        assert args[0] == 'upload'
        return dict(state='completed')
    tool = SimpleNamespace(remote=remote, transfer=lambda r, k: dict(state='completed'))
    assert gate.activate(tool, tmp_path, item) is False
    assert calls == [('logs', item['handle'], '--tail', '100'), ('observe', item['handle'])]
    assert not (tmp_path / 'activation.json').exists()
    ready[0] = True
    assert gate.activate(tool, tmp_path, item) is True
    assert len([c for c in calls if c[0] == 'upload']) == 1
    assert item['handle'] == 'remote:t:job:original'


def test_log_error_on_terminal_worker_is_not_hidden(tmp_path):
    def remote(*args):
        if args[0] == 'logs':
            raise RuntimeError('missing terminal logs')
        return dict(state='failed')
    with pytest.raises(RuntimeError, match='missing terminal logs'):
        gate.activate(SimpleNamespace(remote=remote), tmp_path, receipt())


def test_legacy_deadline_job_without_handshake_remains_observable(tmp_path):
    item = receipt()
    item['request'].pop('activation_required')
    tool = SimpleNamespace(remote=lambda *a: pytest.fail('legacy job must not wait for READY'))
    assert gate.activate(tool, tmp_path, item) is True
