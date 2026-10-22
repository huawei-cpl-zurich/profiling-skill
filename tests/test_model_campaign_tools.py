import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_campaign_tools as tools
import model_campaign_native as native


def test_prepare_private_bundle_and_public_api(tmp_path):
    result = tools.prepare(tmp_path / 'bundle')
    public, private = Path(result['public']), Path(result['private'])
    assert json.loads((public / 'task.json').read_text())['single_kernel'] is False
    assert {p.name for p in public.iterdir()} == {'candidate.py', 'task.json', 'API.md'}
    assert (private / 'execute.py').is_file()
    assert (private / 'scripts/private_campaign_cases.py').is_file()
    compile((private / 'execute.py').read_text(), str(private / 'execute.py'), 'exec')
    with pytest.raises(FileExistsError):
        tools.prepare(tmp_path / 'bundle')


def test_export_preserves_trusted_identity_and_pipe_rows_not_private_text(tmp_path):
    private = tmp_path / 'private'
    (private / 'results').mkdir(parents=True)
    value = dict(state='completed', source_sha256=hashlib.sha256(b'candidate').hexdigest(),
        check=dict(passed=True, variants=[0, 1], diagnostics=[dict(state='passed')]),
        error='SECRET ORACLE', traceback='PRIVATE INPUT', result=dict(
            busy_time=dict(samples_us=[11, 12, 13], active_union_samples_us=[9, 10, 11]), captures=[
            dict(selector='candidate_kernel', pipe_rows=[{'Name': 'candidate_kernel',
                'vec_ratio': '14.3', 'private': 'SECRET ORACLE'}])]))
    (private / 'results/summary.json').write_text(json.dumps(value))
    tools.export(private, tmp_path / 'public')
    report = json.loads((tmp_path / 'public/results/summary.json').read_text())
    assert report['correctness'] is True
    assert report['source_sha256'] == value['source_sha256']
    assert report['busy_time']['active_union_samples_us'] == [9, 10, 11]
    assert report['captures'][0]['pipe_rows'] == [{'Name': 'candidate_kernel', 'vec_ratio': 14.3}]
    assert 'SECRET' not in json.dumps(report)


def test_cli_uses_owned_tool_state_and_operation_mapping(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / 'candidate-snapshot'
    workspace.mkdir()
    called = []
    class FakeTools:
        def __init__(self, bundle, workspace, target, runtime):
            self.workspace = workspace
    class FakeService:
        def __init__(self, client, bundle, public, exporter):
            assert client.state == public.parent / 'tool-state'
        def start(self, operation):
            called.append(operation)
            return dict(state='completed', correctness=True)
        def observe(self, call_id):
            called.append(call_id)
            return dict(state='running', call_id=call_id)
    monkeypatch.setattr(tools, 'CampaignTools', FakeTools)
    monkeypatch.setattr(tools, 'CampaignService', FakeService)
    prefix = ['--bundle', str(tmp_path / 'bundle'), '--workspace', str(workspace)]
    assert tools.main(prefix + ['timing']) == 0
    assert json.loads(capsys.readouterr().out)['correctness'] is True
    assert tools.main(prefix + ['observe', 'abc123']) == 0
    assert json.loads(capsys.readouterr().out)['call_id'] == 'abc123'
    assert called == ['benchmark', 'abc123']


def test_pipe_discovery_follows_renamed_multiple_kernels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(native, '_prepare', lambda root: (object(), {}))
    class Acquisition:
        def profile(self, prepared, **kwargs):
            return dict(tasks=[dict(name=name, kind='device') for name in
                               ('renamed_stage_a', 'renamed_stage_b')])
    monkeypatch.setitem(sys.modules, 'profile_capture', SimpleNamespace(Acquisition=Acquisition))
    def fake_execute(command, folder, env, timeout):
        calls.append(command)
        report = folder / 'report'
        report.mkdir(parents=True)
        selector = next((arg.split('=', 1)[1] for arg in command if arg.startswith('--kernel-name=')), None)
        names = [selector] if selector else ['renamed_stage_a', 'renamed_stage_b']
        (report / 'OpBasicInfo.csv').write_text('Kernel Name,Duration(us)\n' +
                                               ''.join(name + ',11\n' for name in names))
        if selector:
            (report / 'PipeUtilization.csv').write_text('Kernel Name,vec_ratio\n' + selector + ',44\n')
        return SimpleNamespace(returncode=0, stdout='Profiling running finished. All task success.', stderr='')
    monkeypatch.setattr(native, 'execute', fake_execute)
    result = native.pipes(tmp_path, {}, lambda: 100)
    assert result['discovered_names'] == ['renamed_stage_a', 'renamed_stage_b']
    assert len(result['captures']) == 2
    assert [arg for command in calls for arg in command if arg.startswith('--kernel-name=')] == [
        '--kernel-name=renamed_stage_a', '--kernel-name=renamed_stage_b']


def test_export_failed_check_never_qualifies(tmp_path):
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/summary.json').write_text(json.dumps(dict(state='failed',
        check=dict(passed=True, variants=[0]), failure_stage='candidate')))
    tools.export(tmp_path, tmp_path / 'public')
    result = json.loads((tmp_path / 'public/results/summary.json').read_text())
    assert result['correctness'] is False


def test_compiled_graph_rejects_pytorch_compute_but_allows_multiple_triton(monkeypatch):
    class GraphCompiler:
        def __init__(self, **kwargs):
            pass
        def backend_wrapper(self, graph, inputs):
            return 'compiled'
    monkeypatch.setitem(sys.modules, 'model_execution', SimpleNamespace(GraphCompiler=GraphCompiler))
    compiler = native.audited_compiler()
    def graph(*targets):
        return SimpleNamespace(graph=SimpleNamespace(nodes=[
            SimpleNamespace(op='call_function', target=target) for target in targets]))
    assert compiler.backend_wrapper(graph('torch.empty', 'triton_kernel_wrapper_mutation',
                                           'triton_kernel_wrapper_mutation'), []) == 'compiled'
    with pytest.raises(RuntimeError, match='non-Triton'):
        compiler.backend_wrapper(graph('torch.matmul', 'triton_kernel_wrapper_mutation'), [])
    with pytest.raises(RuntimeError, match='no Triton'):
        compiler.backend_wrapper(graph('torch.empty'), [])


def test_every_operation_checks_all_shapes_and_variants_with_fresh_capture(tmp_path, monkeypatch):
    resets, checks = [], []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        _dynamo=SimpleNamespace(reset=lambda: resets.append(True)),
        npu=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'candidate', SimpleNamespace(Model=object))
    monkeypatch.setitem(sys.modules, 'private_campaign_cases', SimpleNamespace(
        make_case=lambda task, index, variant: dict(args=(index, variant), expected=0),
        validate=lambda *args: None))
    def probe(factory, provider, validator, **kwargs):
        checks.extend(list(provider()))
        return dict(state='passed')
    monkeypatch.setitem(sys.modules, 'candidate_probe', SimpleNamespace(probe_candidate=probe))
    monkeypatch.setattr(native, 'audited_compiler', lambda: object())
    for operation in ('check', 'benchmark', 'pipes'):
        (tmp_path / 'request.json').write_text(json.dumps(dict(operation=operation)))
        result = native.check(tmp_path, {}, lambda: 100)
        assert result['passed'] and result['variants'] == [0, 1]
    assert len(resets) == len(checks) == 48
    assert {args for args, kwargs in checks} == {(index, variant) for index in range(8) for variant in (0, 1)}


def test_candidate_import_syntax_error_is_observable_without_private_trace(tmp_path, monkeypatch):
    (tmp_path / 'candidate.py').write_text('def broken(:\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'candidate', raising=False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'candidate_probe', SimpleNamespace(probe_candidate=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'private_campaign_cases', SimpleNamespace(make_case=None, validate=None))
    result = native.check(tmp_path, {}, lambda: 100)
    assert not result['passed']
    detail = result['diagnostics'][0]['diagnostic']
    assert detail['file'] == 'candidate.py' and detail['line'] == 1
    assert detail['error_type'] == 'SyntaxError' and detail['message']
    assert str(tmp_path) not in json.dumps(result)


def test_terminal_evidence_failure_releases_call_but_pending_transfer_does_not(tmp_path):
    from model_campaign_service import CampaignService
    from blind_optimization_tools import save
    public = tmp_path / 'public'
    public.mkdir()
    save(public / 'execution.json', dict(utc_deadline='2000-01-01T00:00:00+00:00'))
    client = SimpleNamespace(workspace=tmp_path / 'private/workspace', state=tmp_path / 'tool-state')
    receipt = dict(call_id='abc123', state='failed', handle='remote:test:job:one',
                   request=dict(source_sha256='a' * 64), download=dict(state='running'))
    (client.state / 'abc123').mkdir(parents=True)
    save(client.state / 'abc123/receipt.json', receipt)
    service = tools.CampaignService(client, tmp_path / 'bundle', public, tools.export)
    pending = service.publish(receipt)
    assert pending['next'] and not (public / 'evidence/abc123/results/summary.json').exists()
    receipt['download']['state'] = 'failed'
    save(client.state / 'abc123/receipt.json', receipt)
    final = service.publish(receipt)
    assert final['state'] == 'failed' and final['failure_stage'] == 'infrastructure'
    assert 'next' not in final and final['diagnostic']['kind'] == 'terminal-evidence-unavailable'
    assert service.start('check')['failure_stage'] == 'budget'


def test_worker_probe_bound_and_remote_clock_skew(tmp_path, monkeypatch):
    import model_campaign_worker as worker
    (tmp_path / 'request.json').write_text(json.dumps(dict(operation='check',
        utc_deadline='2000-01-01T00:00:00+00:00', host_remaining_seconds=240,
        source_sha256='a' * 64)))
    import os
    seen, alarms = [], []
    def load(name):
        assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '3'
        seen.append(name)
        return SimpleNamespace(npu=SimpleNamespace(set_device=lambda device: None,
            synchronize=lambda: None), ones=lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.importlib, 'import_module', load)
    monkeypatch.setattr(worker, 'bounded_run', lambda *args, **kwargs: pytest.fail('no child runtime initialization'))
    monkeypatch.setattr(worker.signal, 'setitimer', lambda kind, value: alarms.append(value))
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', 'previous')
    def acquire(probe):
        env = dict(ASCEND_RT_VISIBLE_DEVICES='3')
        assert probe(env)[0] is True
        return 3, env, SimpleNamespace(close=lambda: None)
    adapter = SimpleNamespace(check=lambda *args: dict(passed=True, variants=[0, 1]))
    result = worker.execute(tmp_path, adapter, acquire=acquire, monotonic=lambda: 0)
    assert result['state'] == 'completed' and result['physical_device'] == 3
    assert seen == ['torch', 'torch_npu']
    assert alarms == [240, 180, 240, 0]
