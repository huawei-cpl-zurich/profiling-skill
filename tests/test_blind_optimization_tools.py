import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

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


def test_terminal_retrieval_materializes_compact_result(tmp_path):
    obj = instance(tmp_path)
    folder = obj.state / 'abc123'
    folder.mkdir()
    tools.save(folder / 'receipt.json', dict(handle='remote:t:job:same', remote_folder='/remote/call'))
    payload = tmp_path / 'summary.json'
    tools.save(payload, dict(state='failed', failure_stage='candidate', error='compiler diagnostic'))
    def remote(*args):
        if args[0] == 'observe': return dict(state='failed')
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


def test_neutral_help_runs_without_remote_configuration(capsys):
    assert tools.main(['--bundle', '/absent', '--workspace', '/absent', 'help']) == 0
    assert 'observe CALL_ID' in capsys.readouterr().out


@pytest.mark.parametrize('verb,count', [('timing', 5), ('pipes', 3)])
def test_remote_program_discovers_exported_name_before_exact_captures(tmp_path, verb, count):
    (tmp_path / 'execute.py').write_text(tools.REMOTE_PROGRAM)
    tools.save(tmp_path / 'request.json', dict(verb=verb, implementation='candidate'))
    tools.save(tmp_path / 'task.json', dict(selector='task_kernel', language='triton'))
    (tmp_path / 'profile_device.py').write_text('''
class Lease:
    def close(self): pass
def wait_for_device(probe): return 7, {}, Lease()
def bounded_run(*args, **kwargs): raise AssertionError('unused probe')
''')
    (tmp_path / 'a3_memory_batch.py').write_text('''
import json
from types import SimpleNamespace
def execute(argv, folder, env, **kwargs):
    folder.mkdir(parents=True)
    if argv[0] == 'msprof':
        report = folder / 'report'
        report.mkdir()
        (report / 'OpBasicInfo.csv').write_text('Kernel Name,Duration(us)\\ntorch_fill,1\\ntask_kernel_compiled_123,2\\n')
        return SimpleNamespace(returncode=0,stdout='profiler completed',stderr='')
    return SimpleNamespace(returncode=0,stdout='{"correctness":true}',stderr='')
def validate_capture(code,text): assert code == 0
def capture(command,folder,env,metric,selector):
    assert selector == 'task_kernel_compiled_123'
    folder.mkdir()
    return {'metric':metric,'selector':selector,'basic_rows':[{'Duration(us)':'2'}]}
def duration(row): return float(row['Duration(us)'])
def summarize(samples): return {'samples_us':samples,'median_us':2}
''')
    result = subprocess.run(['python', str(tmp_path / 'execute.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((tmp_path / 'results/summary.json').read_text())
    assert summary['selector'] == 'task_kernel_compiled_123'
    assert len(summary['captures']) == count
    assert summary['physical_device'] == 7
    assert (tmp_path / 'evidence.tar.gz').is_file()
