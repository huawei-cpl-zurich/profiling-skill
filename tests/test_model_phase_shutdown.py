import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_phase_shutdown import shutdown
from model_campaign_warm import shutdown_phase


@pytest.mark.parametrize('hung', [False, True])
def test_shutdown_closes_owned_process_but_not_unrelated_sibling(tmp_path, hung):
    root = tmp_path / 'phase'
    root.mkdir()
    (root / 'measurement').mkdir()
    script = root / 'execute-phase.py'
    script.write_text('''import os,json,signal,time,subprocess,sys
from pathlib import Path
root=Path.cwd()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],cwd=root/'measurement')
(root/'child.json').write_text(json.dumps(dict(pid=child.pid)))
(root/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
''' + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\nwhile True:time.sleep(.05)\n" if hung else
       "while not (root/'stop').exists():time.sleep(.05)\n"))
    owned = subprocess.Popen([sys.executable, str(script)], cwd=root)
    peer = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=root)
    try:
        for _ in range(100):
            if (root / 'worker.json').exists(): break
            time.sleep(.01)
        result = shutdown(root, grace=.15)
        owned.wait(timeout=2)
        assert result['state'] == 'terminal' and result['worker_closed']
        assert owned.pid in result['owned_pids']
        assert json.loads((root/'child.json').read_text())['pid'] in result['owned_pids']
        assert peer.poll() is None
    finally:
        if owned.poll() is None: owned.kill()
        peer.terminate()
        owned.wait()
        peer.wait()


def test_shutdown_reuses_control_handle_after_outer_job_already_cancelled(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'phase.json').write_text(json.dumps(dict(remote_folder='/owned/phase',
        handle='remote:fake:job:outer', state='cancelled')))
    calls = []
    def remote(verb, *args):
        calls.append((verb, args))
        if verb == 'run': return dict(state='running', handle='remote:fake:job:stop')
        assert args[0] == 'remote:fake:job:stop'
        if verb == 'logs': return dict(content=json.dumps(dict(state='terminal', worker_closed=True)))
        return dict(state='running' if len(calls) == 2 else 'completed', exit=0)
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', cwd=str(tmp_path), remote=remote)
    assert shutdown_phase(tools)['worker_closed'] is False
    result = shutdown_phase(tools)
    assert result['worker_closed'] is True
    assert [verb for verb, args in calls] == ['run', 'observe', 'observe', 'logs']


def test_shutdown_refuses_reused_or_unowned_identity(tmp_path):
    from model_phase_shutdown import identity
    root = tmp_path / 'phase'
    root.mkdir()
    peer = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=tmp_path)
    try:
        current = identity(peer.pid)
        (root / 'worker.json').write_text(json.dumps(dict(pid=peer.pid, start=current['start'])))
        with pytest.raises(RuntimeError, match='identity changed'):
            shutdown(root, grace=.01)
        assert peer.poll() is None
    finally:
        peer.terminate()
        peer.wait()


def test_shutdown_includes_child_created_during_cooperative_grace(tmp_path):
    import os
    import signal
    from model_phase_shutdown import identity
    root = tmp_path / 'phase'
    root.mkdir()
    script = root / 'execute-phase.py'
    script.write_text('''import os,json,time,subprocess,sys
from pathlib import Path
root=Path.cwd()
(root/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
while not (root/'stop').exists():time.sleep(.005)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
(root/'late-child.json').write_text(json.dumps(dict(pid=child.pid)))
while True:time.sleep(.05)
''')
    owned = subprocess.Popen([sys.executable, str(script)], cwd=root)
    late = None
    try:
        for _ in range(100):
            if (root / 'worker.json').exists(): break
            time.sleep(.01)
        result = shutdown(root, grace=.2)
        late = json.loads((root / 'late-child.json').read_text())['pid']
        assert result['worker_closed'] is True
        assert late in result['owned_pids']
        assert identity(late) is None or identity(late)['state'] == 'Z'
    finally:
        if late is None and (root / 'late-child.json').exists():
            late = json.loads((root / 'late-child.json').read_text())['pid']
        for pid in (owned.pid, late):
            if pid and identity(pid):
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
        owned.wait(timeout=2)


def test_uncertain_shutdown_dispatch_never_duplicates(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'phase.json').write_text(json.dumps(dict(remote_folder='/owned/phase',
        handle='remote:fake:job:outer')))
    calls = []
    def remote(*args):
        calls.append(args)
        raise TimeoutError('lost acknowledgement')
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', cwd=str(tmp_path), remote=remote)
    assert shutdown_phase(tools)['worker_closed'] is False
    assert shutdown_phase(tools)['worker_closed'] is False
    assert len(calls) == 1


def test_shutdown_marks_queued_phase_before_worker_admission(tmp_path, monkeypatch):
    import model_campaign_worker as worker
    root = tmp_path / 'not-created-yet'
    assert shutdown(root, grace=.01)['worker_closed'] is True
    monkeypatch.setenv('CAMPAIGN_REMAINING_SECONDS', '10')
    worker.phase(root, SimpleNamespace(check=lambda *args: pytest.fail('late workload admitted')))
    assert (root / 'stop').is_file()


def test_terminal_outer_phase_is_not_replaced_until_actual_worker_closed(tmp_path, monkeypatch):
    import model_campaign_warm as warm
    from datetime import datetime, timedelta, timezone
    state, public, bundle = tmp_path / 'state', tmp_path / 'public', tmp_path / 'bundle'
    state.mkdir(); public.mkdir(); bundle.mkdir()
    (state / 'oldcall').mkdir()
    (public / 'candidate.py').write_text('class Model: pass\n')
    (public / 'execution.json').write_text(json.dumps(dict(utc_deadline=(datetime.now(timezone.utc)
        + timedelta(seconds=60)).isoformat())))
    old = dict(remote_folder='/owned/oldphase', handle='remote:fake:job:old',
               first_call='oldcall', deadline='old-deadline', devices=None)
    (state / 'phase.json').write_text(json.dumps(old))
    calls = []
    def remote(verb, *args):
        calls.append(verb)
        return dict(state='running' if verb == 'upload' else
                    'available' if verb == 'capabilities' else 'completed')
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', cwd='/owned', remote=remote)
    service = SimpleNamespace(tools=tools, public=public, bundle=bundle, remaining=lambda value: 60,
        observe=lambda ident: dict(state='running', call_id=ident))
    closed = [False]
    monkeypatch.setattr(warm, 'shutdown_phase', lambda tools: dict(
        state='terminal' if closed[0] else 'observation-required', worker_closed=closed[0]))
    result = warm.start(service, 'check')
    assert result['state'] == 'phase-cleanup-required'
    assert json.loads((state / 'phase.json').read_text()) == old
    assert not (state / 'oldcall/phase.json').exists()
    assert calls == ['observe']
    closed[0] = True
    result = warm.start(service, 'check')
    assert result['state'] == 'running'
    assert (state / 'oldcall/phase.json').exists()
    assert json.loads((state / 'phase.json').read_text())['first_call'] != 'oldcall'
    assert calls.count('upload') == 1


def test_completed_shutdown_control_requires_worker_closed_acknowledgment(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'phase.json').write_text(json.dumps(dict(remote_folder='/owned/phase',
        handle='remote:fake:job:outer')))
    calls, acknowledged = [], [False]
    def remote(verb, *args):
        calls.append(verb)
        if verb == 'logs':
            return dict(content=json.dumps(dict(state='terminal', worker_closed=True, owned_pids=[123]))
                        if acknowledged[0] else '')
        return dict(state='completed', exit=0, handle='remote:fake:job:control')
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', cwd=str(tmp_path), remote=remote)
    assert shutdown_phase(tools)['worker_closed'] is False
    acknowledged[0] = True
    result = shutdown_phase(tools)
    assert result['worker_closed'] is True and result['acknowledgment']['owned_pids'] == [123]
    assert calls == ['run', 'logs', 'observe', 'logs']
    assert json.loads((state/'phase.json').read_text())['shutdown']['acknowledgment']['worker_closed']
