import json
from pathlib import Path
import sys
from types import SimpleNamespace
import shutil
import tarfile
import os
import hashlib
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_phase_reuses_runtime_and_recovers_candidate_failures(tmp_path, monkeypatch):
    import model_campaign_worker as worker
    requests = []
    for index in range(3):
        root = tmp_path / 'calls' / str(index)
        root.mkdir(parents=True)
        (root / 'request.json').write_text(json.dumps(dict(operation='check',
            host_remaining_seconds=30, source_sha256=str(index))))
        requests.append(root)
    imports, leases = [], []
    monkeypatch.setattr(worker.importlib, 'import_module', lambda name:
        imports.append(name) or SimpleNamespace(npu=SimpleNamespace(set_device=lambda _: None,
            synchronize=lambda: None), ones=lambda *a, **k: None))
    def acquire(probe):
        env = dict(ASCEND_RT_VISIBLE_DEVICES='2')
        assert probe(env)[0]
        leases.append(True)
        return 2, env, SimpleNamespace(close=lambda: leases.append(False))
    adapter = SimpleNamespace(check=lambda root, env, remaining:
        dict(passed=root.name != '1', variants=[0, 1]))
    runtime = {}
    results = [worker.execute(root, adapter, acquire=acquire, runtime=runtime) for root in requests]
    assert [value['state'] for value in results] == ['completed', 'failed', 'completed']
    assert imports == ['torch', 'torch_npu']
    assert leases == [True]
    runtime['lease'].close()
    assert leases == [True, False]


def test_public_calls_share_worker_and_only_upload_new_requests(tmp_path):
    from model_campaign_service import CampaignService, CampaignTools
    from model_campaign_tools import export
    public, bundle = tmp_path / 'public', tmp_path / 'bundle'
    public.mkdir()
    bundle.mkdir()
    (bundle / 'private-provider').write_text('not transferred repeatedly')
    (public / 'execution.json').write_text(json.dumps(dict(utc_deadline=
        (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat())))
    tools = CampaignTools.__new__(CampaignTools)
    tools.state, tools.workspace = tmp_path / 'state', tmp_path / 'private'
    tools.state.mkdir()
    tools.cwd, tools.target, tools.runtime = str(tmp_path / 'remote'), 'test', 'runtime'
    calls, submitted = [], []
    worker_handle = 'remote:test:job:worker'
    worker_state = ['running']
    pending = [True]
    def remote(*args):
        calls.append(args)
        if args[0] in ('targets', 'capabilities', 'preflight'):
            return dict(state='completed')
        if args[0] == 'upload':
            source, target = Path(args[2]), Path(args[3])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if source.name == 'phase.tar.gz':
                folder = Path(str(target)[:-7])
                folder.mkdir()
                with tarfile.open(source) as stream:
                    stream.extractall(folder, filter='data')
            return dict(state='completed')
        if args[0] == 'run':
            return dict(state='running', handle=worker_handle)
        if args[0] == 'observe':
            return dict(state=worker_state[0], handle=worker_handle)
        if args[0] == 'logs':
            return dict(state='running', content='PHASE_READY=' + json.dumps(dict(monotonic=time.monotonic())))
        if args[0] == 'download':
            if pending[0]:
                return dict(state='failed')
            target = Path(args[2])
            phase = target.parents[2]
            ident = target.parent.name
            with tarfile.open(phase / 'inbox' / (ident + '.tar.gz')) as stream:
                request = json.loads(stream.extractfile('request.json').read())
                submitted.append(stream.extractfile('candidate.py').read())
            result = tmp_path / ('result-' + ident)
            (result / 'results').mkdir(parents=True, exist_ok=True)
            (result / 'results/summary.json').write_text(json.dumps(dict(state='completed',
                source_sha256=request['source_sha256'], worker_pid=42,
                check=dict(passed=True, variants=[0, 1]))))
            with tarfile.open(args[3], 'w:gz') as stream:
                stream.add(result / 'results/summary.json', arcname='results/summary.json')
            return dict(state='completed')
        raise AssertionError(args)
    tools.remote = remote
    service = CampaignService(tools, bundle, public, export)
    (public / 'candidate.py').write_text('version one')
    first = service.start('check')
    assert first['state'] == 'running'
    # Lost observations and edits do not create another request or worker.
    (public / 'candidate.py').write_text('not submitted')
    assert service.start('check')['call_id'] == first['call_id']
    pending[0] = False
    assert service.observe(first['call_id'])['worker_pid'] == 42
    for source in ('version two', 'version three'):
        (public / 'candidate.py').write_text(source)
        assert service.start('check')['worker_pid'] == 42
    assert submitted == [b'version one', b'version two', b'version three']
    assert len([call for call in calls if call[0] == 'run']) == 1
    uploads = [Path(call[2]).name for call in calls if call[0] == 'upload']
    assert uploads == ['phase.tar.gz', 'activation.json', 'request.tar.gz', 'request.tar.gz']
    (public / 'execution.json').write_text(json.dumps(dict(utc_deadline=
        (datetime.now(timezone.utc) + timedelta(seconds=1200)).isoformat())))
    assert service.start('check')['state'] == 'phase-cleanup-required'
    assert len([call for call in calls if call[0] == 'run']) == 1
    worker_state[0] = 'completed'
    assert service.start('check')['worker_pid'] == 42
    assert len([call for call in calls if call[0] == 'run']) == 2
    assert (tools.state / first['call_id'] / 'phase.json').exists()


def test_phase_loop_loads_revisions_and_releases_one_lease(tmp_path, monkeypatch):
    import model_campaign_worker as worker
    import model_device
    inbox = tmp_path / 'inbox'
    inbox.mkdir()
    for index, value in enumerate((1, -1, 3)):
        source = ('VALUE = ' + str(value)).encode()
        request = dict(operation='check', host_remaining_seconds=10,
                       source_sha256=hashlib.sha256(source).hexdigest())
        folder = tmp_path / ('request-' + str(index))
        folder.mkdir()
        (folder / 'candidate.py').write_bytes(source)
        (folder / 'request.json').write_text(json.dumps(request))
        with tarfile.open(inbox / (str(index) + '.tar.gz'), 'w:gz') as stream:
            for name in ('candidate.py', 'request.json'):
                stream.add(folder / name, arcname=name)
    closed, seen = [], []
    fake_torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda _: None,
        synchronize=lambda: None), ones=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    def acquire(probe, remaining):
        env = dict(ASCEND_RT_VISIBLE_DEVICES='2')
        assert probe(env)[0]
        return 2, env, SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(model_device, 'acquire', acquire)
    monkeypatch.setenv('CAMPAIGN_REMAINING_SECONDS', '10')
    def check(root, env, remaining):
        import candidate
        seen.append(candidate.VALUE)
        if candidate.VALUE == 3:
            (tmp_path / 'stop').touch()
        return dict(passed=candidate.VALUE > 0, variants=[0, 1])
    worker.phase(tmp_path, SimpleNamespace(check=check))
    assert seen == [1, -1, 3]
    assert closed == [True]
    results = [json.loads((tmp_path / 'calls' / str(i) / 'results/summary.json').read_text())
               for i in range(3)]
    assert [value['state'] for value in results] == ['completed', 'failed', 'completed']
    assert {value['worker_pid'] for value in results} == {os.getpid()}
    assert {value['runtime_initializations'] for value in results} == {1}


def test_failed_bootstrap_upload_is_retryable_without_a_worker(tmp_path):
    from model_campaign_service import CampaignService, CampaignTools
    from model_campaign_tools import export
    public, bundle = tmp_path / 'public', tmp_path / 'bundle'
    public.mkdir()
    bundle.mkdir()
    (public / 'candidate.py').write_text('version one')
    (public / 'execution.json').write_text(json.dumps(dict(utc_deadline=
        (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat())))
    tools = CampaignTools.__new__(CampaignTools)
    tools.state, tools.workspace = tmp_path / 'state', tmp_path / 'private'
    tools.state.mkdir()
    tools.cwd, tools.target, tools.runtime = str(tmp_path / 'remote'), 'test', 'runtime'
    uploads, runs = [], []
    def remote(*args):
        if args[0] in ('targets', 'capabilities', 'preflight'):
            return dict(state='completed')
        if args[0] == 'upload':
            uploads.append(args)
            return dict(state='failed' if len(uploads) == 1 else 'completed')
        if args[0] == 'run':
            runs.append(args)
            return dict(state='running', handle='remote:test:job:worker')
        if args[0] == 'logs':
            return dict(state='running', content='')
        raise AssertionError(args)
    tools.remote = remote
    service = CampaignService(tools, bundle, public, export)
    failed = service.start('check')
    assert failed['state'] == 'failed' and failed['failure_stage'] == 'infrastructure'
    assert not runs
    pending = service.start('check')
    assert pending['state'] == 'running' and len(runs) == 1
    assert len(uploads) == 2
