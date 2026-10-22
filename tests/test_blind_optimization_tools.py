import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import tarfile
import time
from datetime import datetime, timedelta, timezone

import pytest


SPEC = importlib.util.spec_from_file_location('blind_tools', Path(__file__).parents[1] / 'scripts/blind_optimization_tools.py')
tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tools)


def instance(tmp_path):
    obj = object.__new__(tools.Tools)
    obj.bundle, obj.workspace = tmp_path / 'bundle', tmp_path / 'workspace'
    obj.state = tmp_path / 'state'
    for folder in (obj.bundle, obj.workspace, obj.state): folder.mkdir()
    obj.cwd, obj.target, obj.runtime = '/configured-workspace', 'configured-target', 'named-runtime'
    (obj.bundle / 'task.json').write_text(json.dumps({'language': 'triton', 'selector': 'task_kernel'}))
    for name in ('runner.py', 'native_host.asc'): (obj.bundle / name).write_text('TRUSTED')
    (obj.bundle / 'baseline.py').write_text('ORIGINAL')
    (obj.workspace / 'candidate.py').write_text('EDITABLE')
    return obj


def test_stages_trusted_oracle_and_candidate_snapshot(tmp_path):
    obj = instance(tmp_path)
    (obj.workspace / 'runner.py').write_text('CHANGED ORACLE')
    (obj.workspace / 'task.json').write_text('CHANGED SPEC')
    calls = []
    def remote(*args):
        calls.append(args)
        if args[0] == 'run': return dict(state='reconnecting', handle='remote:t:job:existing')
        return dict(state='completed')
    obj.remote = remote
    obj.observe = lambda ident: json.loads((obj.state / ident / 'receipt.json').read_text())
    receipt = obj.start('check', 'candidate', 'racecheck')
    folder = obj.state / receipt['call_id']
    assert (folder / 'runner.py').read_text() == 'TRUSTED'
    assert json.loads((folder / 'task.json').read_text())['selector'] == 'task_kernel'
    assert (folder / 'candidate.py').read_text() == 'EDITABLE'
    assert calls[-1][0:6] == ('run', obj.target, '--runtime', obj.runtime, '--cwd', obj.cwd)
    assert calls[-1][-2] == '--file'
    assert receipt['handle'] == 'remote:t:job:existing'


def test_baseline_ignores_agent_edit_and_symlink_candidate_rejected(tmp_path):
    obj = instance(tmp_path)
    (obj.workspace / 'baseline.py').write_text('TAMPERED')
    obj.remote = lambda *args: dict(state='completed', handle='remote:t:job:x')
    obj.observe = lambda ident: ident
    ident = obj.start('check', 'baseline', 'racecheck')
    assert (obj.state / ident / 'baseline.py').read_text() == 'ORIGINAL'
    (obj.workspace / 'candidate.py').unlink()
    (obj.workspace / 'candidate.py').symlink_to(obj.bundle / 'baseline.py')
    with pytest.raises(ValueError, match='non-symlink'): obj.start('check', 'candidate', 'racecheck')


def test_reconnecting_observation_never_redispatches(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(handle='remote:t:job:same'))
    calls = []
    def remote(*args):
        calls.append(args)
        return dict(state='observation-unavailable')
    obj.remote = remote
    result = obj.observe('abc123')
    assert result['next'] == 'experiment-tool observe abc123'
    assert calls == [('observe', 'remote:t:job:same', '--wait', '--timeout', '120')]


def test_transport_timeout_keeps_call_id_and_same_handle(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(call_id='abc123', handle='remote:t:job:same'))
    def remote(*args): raise subprocess.TimeoutExpired(args, 240)
    obj.remote = remote
    result = obj.observe('abc123')
    assert result['state'] == 'observation-unavailable'
    assert result['call_id'] == 'abc123'
    assert result['handle'] == 'remote:t:job:same'
    assert result['next'] == 'experiment-tool observe abc123'


@pytest.mark.parametrize('failure', ['lost-response', 'missing-handle'])
def test_dispatch_intent_prevents_duplicate_after_unknown_submission(tmp_path, failure):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    receipt = dict(call_id='abc123', target='saved-target', runtime='saved-runtime', cwd='/saved-cwd',
                   upload=dict(state='completed'))
    calls = []
    def remote(*args):
        calls.append(args)
        # The durable intent must already exist when the remote accepts the run.
        assert json.loads((folder / 'receipt.json').read_text())['dispatch_intent'] is True
        if failure == 'lost-response': raise RuntimeError('accepted remotely, JSON response lost')
        return dict(state='reconnecting')
    obj.remote = remote
    submitted = obj.dispatch(folder, receipt)
    assert submitted['state'] == 'dispatch-unconfirmed'
    assert submitted['call_id'] == 'abc123'
    assert 'DO NOT REDISPATCH' in submitted['recovery']
    observed = obj.observe('abc123')
    assert observed['state'] == 'dispatch-unconfirmed'
    assert len(calls) == 1 and calls[0][0] == 'run'


@pytest.mark.parametrize('legacy_deadline', [False, True])
def test_terminal_retrieval_materializes_compact_result(tmp_path, legacy_deadline):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    receipt = dict(target='original-target', handle='remote:t:job:same', remote_folder='/remote/call')
    if legacy_deadline:
        receipt['request'] = dict(utc_deadline='2000-01-01T00:00:00+00:00')
    tools.save(folder / 'receipt.json', receipt)
    payload = tmp_path / 'summary.json'
    tools.save(payload, dict(state='failed', failure_stage='candidate', error='compiler diagnostic'))
    def remote(*args):
        if args[0] == 'observe': return dict(state='failed')
        assert args[0:2] == ('download', 'original-target')
        with tarfile.open(args[-1], 'w:gz') as archive: archive.add(payload, arcname='results/summary.json')
        return dict(state='completed')
    obj.remote = remote
    result = obj.observe('abc123')
    assert result['failure_stage'] == 'candidate'
    assert result['error'] == 'compiler diagnostic'
    assert (obj.workspace / result['evidence'] / 'results/summary.json').is_file()


def test_remote_calls_global_cli_json_boundary_and_keeps_diagnostics(tmp_path):
    obj = instance(tmp_path)
    obj.client = tmp_path / 'cpl-remote-test-double'
    obj.client.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({"state":"running","handle":"remote:t:job:retained","arguments":sys.argv[1:]}))\n')
    obj.client.chmod(0o700)
    result = obj.remote('observe', 'remote:t:job:retained')
    assert result['arguments'] == ['--json', 'observe', 'remote:t:job:retained']
    journal = json.loads((obj.state / 'transport.jsonl').read_text())
    assert journal['command'][0] == str(obj.client)
    assert journal['code'] == 0


@pytest.mark.parametrize('deadline_expired', [True, False])
def test_transport_timeout_distinguishes_original_deadline_from_client_timeout(tmp_path, monkeypatch, deadline_expired):
    obj = instance(tmp_path)
    obj.client = tmp_path / 'managed-client'
    obj._staging_deadline = (datetime.now(timezone.utc) + timedelta(seconds=.05 if deadline_expired else 600)).isoformat()
    def timed_out(command, **kwargs):
        if deadline_expired:
            assert 0 < kwargs['timeout'] <= .05
            time.sleep(.06)
        else:
            assert kwargs['timeout'] == 240
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(tools.subprocess, 'run', timed_out)
    expected = TimeoutError if deadline_expired else subprocess.TimeoutExpired
    with pytest.raises(expected):
        obj.remote('preflight', obj.target)


def test_neutral_help_runs_without_remote_configuration(capsys):
    assert tools.main(['--bundle', '/absent', '--workspace', '/absent', 'help']) == 0
    assert 'observe CALL_ID' in capsys.readouterr().out


def test_help_reports_effective_capture_configuration_used_by_dispatch(monkeypatch, capsys):
    prefix = ['--bundle', '/unused', '--workspace', '/unused', '--timing-captures', '3', '--pipe-captures', '1']
    assert tools.main([*prefix, 'help']) == 0
    effective = json.loads(capsys.readouterr().out.splitlines()[-1])['capture_configuration']
    calls = []
    class FakeTools:
        def __init__(self, *args):
            pass
        def start(self, verb, implementation, sanitizer, **configuration):
            calls.append((verb, configuration))
            return {'state': 'completed', 'capture_configuration': configuration}
    monkeypatch.setattr(tools, 'Tools', FakeTools)
    assert tools.main([*prefix, 'pipes']) == 0
    actual = json.loads(capsys.readouterr().out)['capture_configuration']
    assert effective == actual == {'timing_captures': 3, 'pipe_captures': 1}
    assert calls == [('pipes', effective)]


@pytest.mark.parametrize('verb', ['ir', 'sanitize'])
def test_native_optional_diagnostics_are_capability_results_without_remote_calls(tmp_path, verb):
    obj = instance(tmp_path)
    tools.save(obj.bundle / 'task.json', dict(language='ascendc'))
    obj.remote = lambda *args: pytest.fail('unsupported operation must not dispatch')
    result = obj.start(verb, 'candidate', 'racecheck')
    assert result['state'] == 'unsupported'
    assert result['failure_stage'] == 'capability'


@pytest.mark.parametrize('state', ['running', 'completed', 'failed', 'cancelled'])
def test_upload_resume_observes_same_transfer_then_dispatches_once(tmp_path, state):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(call_id='abc123', target='saved-target',
               upload=dict(state='running', handle='remote:t:transfer:upload')))
    calls = []
    def remote(*args):
        calls.append(args)
        return dict(state=state, handle='remote:t:transfer:upload')
    obj.remote = remote
    obj.dispatch = lambda folder,receipt: dict(dispatched_to=receipt['target'])
    result = obj.observe('abc123')
    assert calls == [('observe', 'remote:t:transfer:upload', '--wait', '--timeout', '120')]
    if state == 'completed': assert result == dict(dispatched_to='saved-target')
    elif state == 'running': assert result['next'] == 'experiment-tool observe abc123'
    else:
        assert result['state'] == 'failed'
        assert result['failure_stage'] == 'infrastructure'
        saved = json.loads((folder / 'receipt.json').read_text())
        assert saved['state'] == 'failed'


def test_failed_upload_does_not_block_fresh_candidate_retry(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(call_id='abc123', state='prepared',
        upload=dict(state='failed', handle='remote:t:transfer:upload')))
    result = obj.observe('abc123')
    assert result['state'] == 'failed'
    (obj.workspace / 'candidate.py').write_text('REPAIRED')
    calls = []
    def remote(*args):
        calls.append(args[0])
        return dict(state='running', handle='remote:t:job:new') if args[0] == 'run' else dict(state='completed')
    obj.remote = remote
    obj.observe = lambda ident: dict(call_id=ident)
    result = obj.start('check', 'candidate', 'racecheck')
    assert result['call_id'] != 'abc123'
    assert calls[-1] == 'run'


def test_download_resume_observes_then_fetches_without_new_transfer(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(call_id='abc123', handle='remote:t:job:same',
               download=dict(state='running',handle='remote:t:transfer:download')))
    payload = tmp_path / 'summary.json'
    tools.save(payload, dict(state='completed'))
    calls = []
    def remote(*args):
        calls.append(args)
        if args[0] == 'fetch':
            with tarfile.open(args[-1], 'w:gz') as archive: archive.add(payload, arcname='results/summary.json')
        return dict(state='completed',handle=args[1])
    obj.remote = remote
    assert obj.observe('abc123')['state'] == 'completed'
    assert [c[0] for c in calls] == ['observe', 'observe', 'fetch']
    assert calls[1][1] == calls[2][1] == 'remote:t:transfer:download'


def exercise_remote_program(tmp_path, verb, count, extra, event_error=None, settings=None):
    shutil.copyfile(Path(tools.__file__).with_name('phase_deadline.py'), tmp_path / 'phase_deadline.py')
    (tmp_path / 'execute.py').write_text(tools.REMOTE_PROGRAM)
    request = dict(verb=verb, implementation='candidate', source_sha256='a' * 64, **(settings or {}))
    if verb == 'events':
        request['variant'] = 'long'
    tools.save(tmp_path / 'request.json', request)
    tools.save(tmp_path / 'task.json', dict(selector='task_kernel', language='triton'))
    if verb == 'events':
        tools.save(tmp_path / 'task.json', dict(language='ascendc', native_wait_kind=3))
    (tmp_path / 'profile_device.py').write_text('''
class Lease:
    def close(self): pass
def wait_for_device(probe, **kwargs): return 7, {}, Lease()
def bounded_run(*args, **kwargs): raise AssertionError('unused probe')
''')
    (tmp_path / 'a3_memory_batch.py').write_text('''
import json
from pathlib import Path
from types import SimpleNamespace
def execute(argv, folder, env, **kwargs):
    folder.mkdir(parents=True)
    if '--checks=4' in argv:
        assert '--variant' not in argv
    if '--event-timing' in argv:
        assert argv[argv.index('--variant')+1] == 'long'
    if argv[0] == 'msprof':
        report = folder / 'report'
        report.mkdir()
        assert '--launch-count=5000' in argv
        (report / 'OpBasicInfo.csv').write_text('Kernel Name,Duration(us)\\nrenamed_kernel_compiled_123,2\\n')
        return SimpleNamespace(returncode=0,stdout='profiler completed',stderr='')
    artifact = Path(__file__).parent / 'compiler-evidence/candidate'
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / 'lowered.ll').write_text('lowered IR')
    (artifact / 'lowered.bc').write_bytes(b'compiler bitcode')
    build = Path(__file__).parent / 'build'
    build.mkdir(exist_ok=True)
    (build / 'compile.stderr').write_text('compiler diagnostics')
    if '--event-timing' in argv:
        return SimpleNamespace(returncode=0,stdout='{"correctness":true,"warmup":3,"samples_us":[1,2,3,4,5,6,7]}',stderr='')
    return SimpleNamespace(returncode=0,stdout='{"correctness":true}',stderr='')
def validate_capture(code,text): assert code == 0
def capture(command,folder,env,metric,selector):
    assert selector == 'renamed_kernel_compiled_123'
    folder.mkdir()
    return {'metric':metric,'selector':selector,'basic_rows':[{'Duration(us)':'2'}]}
def duration(row): return float(row['Duration(us)'])
def summarize(samples): return {'samples_us':samples,'median_us':2}
''')
    if extra and verb in ('timing', 'pipes'):
        helper = tmp_path / 'a3_memory_batch.py'
        helper.write_text(helper.read_text().replace('renamed_kernel_compiled_123,2\\n', 'renamed_kernel_compiled_123,2\\nsecond_compute,3\\n'))
    if verb == 'events':
        helper = tmp_path / 'a3_memory_batch.py'
        content = helper.read_text().replace("if argv[0] == 'msprof':", "if argv[0] == 'msprof':\n        raise AssertionError('msprof unavailable: events must not dispatch it')")
        if event_error == 'bad-samples':
            content = content.replace('[1,2,3,4,5,6,7]', '[0,2,3,4,5,6,7]')
        if event_error == 'runtime':
            content = content.replace("return SimpleNamespace(returncode=0,stdout='{\"correctness\":true,\"warmup\":3", "return SimpleNamespace(returncode=1,stdout='{\"correctness\":true,\"warmup\":3")
        if event_error == 'correctness':
            content = content.replace('"correctness":true,"warmup":3', '"correctness":false,"warmup":3')
        content = content.replace("def summarize(samples): return {'samples_us':samples,'median_us':2}",
                                  "def summarize(samples):\n    if any(x<=0 for x in samples): raise ValueError('nonpositive event sample')\n    return {'samples_us':samples,'median_us':2}")
        helper.write_text(content)
    result = subprocess.run(['python', str(tmp_path / 'execute.py')], capture_output=True, text=True)
    summary = json.loads((tmp_path / 'results/summary.json').read_text())
    assert summary['source_sha256'] == 'a' * 64
    if verb == 'events':
        if event_error:
            assert result.returncode == 1
            assert summary['failure_stage'] == 'event-timing'
            assert summary['state'] == 'failed'
            return
        assert result.returncode == 0, result.stdout + result.stderr
        assert summary['event_timing']['samples_us'] == [1, 2, 3, 4, 5, 6, 7]
        assert summary['state'] == 'completed'
        assert summary['kernel_contract']['evidence_kind'] == 'trusted_host_launch_structure'
        assert summary['kernel_contract']['exported_selector_available'] is False
        assert 'selector' not in summary
        assert not (tmp_path / 'results/discovery').exists()
        return
    if extra and verb in ('timing', 'pipes'):
        assert result.returncode == 1
        assert summary['failure_stage'] == 'candidate-contract'
        assert 'exactly one compute launch' in summary['error']
        return
    assert result.returncode == 0, result.stderr + result.stdout
    if verb in ('timing', 'pipes'):
        assert summary['selector'] == 'renamed_kernel_compiled_123'
    else:
        assert not (tmp_path / 'results/discovery').exists()
    assert len(summary.get('captures', [])) == count
    if count:
        assert summary['capture_count'] == count
    assert summary['physical_device'] == 7
    assert summary['launch_contract'] == dict(runner_checked=True,
        profile_stream_verified=verb in ('timing', 'pipes'))
    assert (tmp_path / 'evidence.tar.gz').is_file()
    with tarfile.open(tmp_path / 'evidence.tar.gz') as archive:
        assert 'build/compile.stderr' in archive.getnames()
        assert 'compiler-evidence/candidate/lowered.ll' in archive.getnames()
        assert 'compiler-evidence/candidate/lowered.bc' in archive.getnames()


@pytest.mark.parametrize('verb,count,extra,settings', [('timing', 5, False, {}), ('pipes', 3, False, {}), ('ir', 0, False, {}), ('check', 0, False, {}), ('timing', 5, True, {}), ('timing', 3, False, {'timing_captures': 3}), ('pipes', 1, False, {'pipe_captures': 1}), ('events', 0, False, {})])
def test_remote_program_discovers_exported_name_before_exact_captures(tmp_path, verb, count, extra, settings):
    exercise_remote_program(tmp_path, verb, count, extra, settings=settings)


@pytest.mark.parametrize('error', ['bad-samples', 'runtime', 'correctness'])
def test_independent_events_preserve_measurement_failures(tmp_path, error):
    exercise_remote_program(tmp_path, 'events', 0, False, event_error=error)
def test_expired_role_never_uploads_or_dispatches(tmp_path):
    obj = instance(tmp_path)
    tools.save(obj.workspace / 'execution.json', dict(utc_deadline=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()))
    obj.remote = lambda *args: pytest.fail('expired role must not call transport')
    result = obj.start('check', 'candidate', 'racecheck')
    assert result['failure_stage'] == 'budget'
    assert result['source_sha256'] == tools.hashlib.sha256(b'EDITABLE').hexdigest()


def test_pending_request_is_observed_even_after_candidate_changes(tmp_path):
    obj = instance(tmp_path)
    pending = obj.state / 'abc123'
    pending.mkdir()
    tools.save(pending / 'receipt.json', dict(call_id='abc123', state='running', handle='remote:t:job:same'))
    (obj.workspace / 'candidate.py').write_text('NEW CANDIDATE')
    obj.observe = lambda ident: dict(observed=ident)
    obj.remote = lambda *args: pytest.fail('must observe pending request')
    assert obj.start('check', 'candidate', 'racecheck') == dict(observed='abc123')


def test_observation_activates_only_existing_job_and_keeps_its_original_deadline(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'original'
    folder.mkdir()
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    tools.save(folder / 'receipt.json', dict(call_id='original', state='running', target='t',
        remote_folder='/original', handle='remote:t:job:original',
        request=dict(utc_deadline=deadline, activation_required=True)))
    calls = []
    def remote(*args):
        calls.append(args)
        if args[0] == 'logs':
            return dict(content='PHASE_READY=' + json.dumps(dict(monotonic=100)))
        return dict(state='running') if args[0] == 'observe' else dict(state='completed')
    obj.remote = remote
    result = obj.observe('original')
    assert result['state'] == 'running'
    assert result['activation_state'] == 'activated'
    assert result['request']['utc_deadline'] == deadline
    assert [x[0] for x in calls] == ['logs', 'upload', 'observe']
    assert calls[-1][1] == 'remote:t:job:original'
    assert 150 < json.loads((folder / 'activation.json').read_text())['cutoff_monotonic'] <= 160


@pytest.mark.parametrize('settings,counts', [({}, (5, 3)), ({'timing_captures': 3, 'pipe_captures': 1}, (3, 1))])
def test_request_deadline_survives_staging_and_dispatch(tmp_path, settings, counts):
    obj = instance(tmp_path)
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
    tools.save(obj.workspace / 'execution.json', dict(utc_deadline=deadline))
    def remote(*args):
        return dict(state='running', handle='remote:t:job:same') if args[0] == 'run' else dict(state='completed')
    obj.remote = remote
    obj.observe = lambda ident: json.loads((obj.state / ident / 'receipt.json').read_text())
    receipt = obj.start('check', 'candidate', 'racecheck', **settings)
    request = json.loads((obj.state / receipt['call_id'] / 'request.json').read_text())
    assert request['utc_deadline'] == deadline
    assert 0 < request['host_remaining_seconds'] <= 600
    assert receipt['request']['utc_deadline'] == deadline
    assert receipt['dispatch_remaining_seconds'] <= request['host_remaining_seconds']
    assert (request['timing_captures'], request['pipe_captures']) == counts


@pytest.mark.parametrize('option', ['--timing-captures', '--pipe-captures'])
def test_invalid_capture_count_rejected_before_transport(tmp_path, monkeypatch, option):
    monkeypatch.setattr(tools, 'Tools', lambda *a, **kw: pytest.fail('must reject before constructing transport'))
    with pytest.raises(SystemExit) as error:
        tools.main(['--bundle', str(tmp_path), '--workspace', str(tmp_path), 'pipes', option, '0'])
    assert error.value.code == 2


def test_upload_time_is_not_a_new_remote_budget(tmp_path):
    obj = instance(tmp_path)
    tools.save(obj.workspace / 'execution.json', dict(utc_deadline=(datetime.now(timezone.utc) + timedelta(seconds=.02)).isoformat()))
    calls = []
    def remote(*args):
        calls.append(args[0])
        if args[0] == 'upload': time.sleep(.04)
        return dict(state='completed')
    obj.remote = remote
    result = obj.start('check', 'candidate', 'racecheck')
    assert result['failure_stage'] == 'budget'
    assert result['source_sha256'] == tools.hashlib.sha256(b'EDITABLE').hexdigest()
    assert 'run' not in calls


def test_pre_dispatch_failure_is_terminal_and_preserves_source_identity(tmp_path):
    obj = instance(tmp_path)
    def remote(*args): raise RuntimeError('preflight unreachable')
    obj.remote = remote
    result = obj.start('check', 'candidate', 'racecheck')
    assert result['failure_stage'] == 'infrastructure'
    assert result['source_sha256'] == tools.hashlib.sha256(b'EDITABLE').hexdigest()
    saved = json.loads((obj.state / result['call_id'] / 'receipt.json').read_text())
    assert saved['state'] == 'failed'
    assert 'dispatch_intent' not in saved


@pytest.mark.parametrize('budget,delay,stage', [(300, 0, 'environment'), (.2, .3, 'budget')])
def test_worker_bounds_inventory_probe_and_retains_failure_identity(tmp_path, budget, delay, stage):
    shutil.copyfile(Path(tools.__file__).with_name('phase_deadline.py'), tmp_path / 'phase_deadline.py')
    tools.save(tmp_path / 'activation.json', dict(cutoff_monotonic=time.monotonic() + budget))
    (tmp_path / 'execute.py').write_text(tools.REMOTE_PROGRAM)
    tools.save(tmp_path / 'request.json', dict(verb='check', implementation='candidate',
        source_sha256='b' * 64, host_remaining_seconds=budget,
        utc_deadline='2000-01-01T00:00:00+00:00'))
    (tmp_path / 'profile_device.py').write_text('''
import json,time
from pathlib import Path
from types import SimpleNamespace
def bounded_run(argv, **kwargs):
    with Path('timeouts.jsonl').open('a') as stream:
        stream.write(json.dumps({'argv':argv,'timeout':kwargs['timeout']})+'\\n')
    return SimpleNamespace(returncode=0,stdout='',stderr='')
def wait_for_device(probe,query):
    query('-m'); probe({})
    time.sleep(DELAY)
    raise RuntimeError('inventory unavailable')
'''.replace('DELAY', repr(delay)))
    (tmp_path / 'a3_memory_batch.py').write_text('''
def execute(*a,**kw): raise AssertionError('no runtime')
capture=duration=summarize=validate_capture=execute
''')
    result = subprocess.run(['python', 'execute.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    summary = json.loads((tmp_path / 'results/summary.json').read_text())
    assert summary['source_sha256'] == 'b' * 64
    assert summary['failure_stage'] == stage
    inv, probe = map(json.loads, (tmp_path / 'timeouts.jsonl').read_text().splitlines())
    assert 0 < inv['timeout'] <= min(120, budget)
    assert 0 < probe['timeout'] <= min(180, budget)
    assert (tmp_path / 'evidence.tar.gz').is_file()
