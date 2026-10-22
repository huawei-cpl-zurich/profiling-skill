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
    script = root / 'execute-phase.py'
    script.write_text('''import os,json,signal,time
from pathlib import Path
root=Path.cwd()
(root/'worker.json').write_text(json.dumps(dict(pid=os.getpid(),start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])))
''' + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\nwhile True:time.sleep(.05)\n" if hung else
       "while not (root/'stop').exists():time.sleep(.05)\n"))
    owned = subprocess.Popen([sys.executable, str(script)], cwd=root)
    peer = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=tmp_path)
    try:
        for _ in range(100):
            if (root / 'worker.json').exists(): break
            time.sleep(.01)
        result = shutdown(root, grace=.15)
        owned.wait(timeout=2)
        assert result['state'] == 'terminal' and result['worker_closed']
        assert owned.pid in result['owned_pids']
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
        return dict(state='running' if len(calls) == 2 else 'completed', exit=0)
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', remote=remote)
    assert shutdown_phase(tools)['worker_closed'] is False
    result = shutdown_phase(tools)
    assert result['worker_closed'] is True
    assert [verb for verb, args in calls] == ['run', 'observe', 'observe']


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


def test_uncertain_shutdown_dispatch_never_duplicates(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'phase.json').write_text(json.dumps(dict(remote_folder='/owned/phase',
        handle='remote:fake:job:outer')))
    calls = []
    def remote(*args):
        calls.append(args)
        raise TimeoutError('lost acknowledgement')
    tools = SimpleNamespace(state=state, target='fake', runtime='runtime', remote=remote)
    assert shutdown_phase(tools)['worker_closed'] is False
    assert shutdown_phase(tools)['worker_closed'] is False
    assert len(calls) == 1
