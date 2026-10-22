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
                'vec_ratio': '14.3', 'sub_block_id': 'vector0', 'private': 'SECRET ORACLE',
                'unsupported_ratio': 'NaN', 'unsupported_time': 'inf'},
                {'sub_block_id': 'SECRET ORACLE'}])]))
    (private / 'results/summary.json').write_text(json.dumps(value))
    tools.export(private, tmp_path / 'public')
    report = json.loads((tmp_path / 'public/results/summary.json').read_text(),
                        parse_constant=lambda value: pytest.fail('nonfinite JSON number: ' + value))
    assert report['correctness'] is True
    assert report['source_sha256'] == value['source_sha256']
    assert report['busy_time']['active_union_samples_us'] == [9, 10, 11]
    assert report['captures'][0]['pipe_rows'] == [
        {'Name': 'candidate_kernel', 'vec_ratio': 14.3, 'sub_block_id': 'vector0'}, {}]
    assert 'SECRET' not in json.dumps(report)


def test_warm_timing_exports_only_names_from_actual_device_capture(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setattr(native, '_prepare', lambda root:
        (lambda: calls.append('forward'), dict(id='private-case')))
    class Acquisition:
        def profile(self, prepared, **kwargs):
            assert kwargs['samples'] == 3
            return dict(iterations=[{}] * 3, tasks=[
                dict(name='kernel_b', kind='device'), dict(name='kernel_a', kind='device'),
                dict(name='kernel_a', kind='device'), dict(name='setup_cpu', kind='aicpu')])
    monkeypatch.setitem(sys.modules, 'profile_capture', SimpleNamespace(Acquisition=Acquisition))
    monkeypatch.setitem(sys.modules, 'profile_accounting', SimpleNamespace(account_profile=
        lambda *args, **kwargs: dict(metric='busy-time', iterations=[
            dict(device_busy_us=4, device_active_union_us=3)] * 3)))
    result = native.benchmark(tmp_path, {}, lambda: 100)
    assert result['discovered_names'] == ['kernel_a', 'kernel_b']
    assert len(calls) == 80
    private = tmp_path / 'private'
    (private / 'results').mkdir(parents=True)
    (private / 'results/summary.json').write_text(json.dumps(dict(state='completed',
        check=dict(passed=True, variants=[0, 1]), result=result)))
    tools.export(private, tmp_path / 'public')
    published = json.loads((tmp_path / 'public/results/summary.json').read_text())
    assert published['discovered_names'] == ['kernel_a', 'kernel_b']


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


@pytest.mark.parametrize('problem', ['valid', 'incomplete', 'duplicates', 'blank', 'na',
                                    'nan', 'infinite', 'partial', 'multi-launch'])
def test_application_pmu_lifecycle_exports_five_tasks_per_kernel(tmp_path, monkeypatch, problem):
    from contextlib import nullcontext
    from types import ModuleType
    calls, steps, synchronized = [], [], []
    names = ('renamed_stage_a', 'renamed_stage_b')
    compiler = SimpleNamespace(monitor_compilation=nullcontext, snapshot=lambda: 0)
    class Prepared:
        def __init__(self):
            self.compiler = compiler
        def __call__(self):
            calls.append('forward')
    monkeypatch.setattr(native, '_prepare', lambda root: (Prepared(), {'id': 'operator-case'}))
    class Acquisition:
        def profile(self, prepared, **kwargs):
            assert kwargs['samples'] == 1
            return dict(tasks=[dict(name=name, kind='device') for name in names])
    monkeypatch.setitem(sys.modules, 'profile_capture', SimpleNamespace(Acquisition=Acquisition))
    def handler(folder):
        def export():
            output = Path(folder)
            output.mkdir(parents=True)
            count = 4 if problem == 'incomplete' else 10 if problem == 'multi-launch' else 5
            rows = []
            for name in names:
                for task in range(count):
                    identity = 0 if problem == 'duplicates' else task
                    ratio = {'blank': '', 'na': 'N/A', 'nan': 'NaN', 'infinite': 'inf'}.get(problem, '44')
                    if problem == 'partial' and task:
                        ratio = 'N/A'
                    rows.append(f'{name},0,47,{identity},{ratio},N/A\n')
            (output / 'kernel_details.csv').write_text(
                'Name,Device_id,Stream ID,Task ID,aiv_mte2_ratio,aic_mac_ratio\n' + ''.join(rows))
        return export
    def profile(**kwargs):
        assert kwargs['activities'] == ['cpu', 'npu']
        assert kwargs['schedule'] == dict(wait=0, warmup=0, active=5, repeat=1)
        assert kwargs['experimental_config']['aic_metrics'] == 'pipes'
        class Capture:
            def __enter__(self):
                return self
            def step(self):
                steps.append(len(calls))
            def __exit__(self, kind, error, tb):
                assert kind is None
                assert len(steps) == 5
                kwargs['on_trace_ready']()
        return Capture()
    profiler = SimpleNamespace(profile=profile, schedule=lambda **kw: kw,
        _ExperimentalConfig=lambda **kw: kw, tensorboard_trace_handler=handler,
        ProfilerActivity=SimpleNamespace(CPU='cpu', NPU='npu'),
        ProfilerLevel=SimpleNamespace(Level1='level1'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext,
        npu=SimpleNamespace(synchronize=lambda: synchronized.append(True))))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(profiler=profiler))
    config = ModuleType('torch_npu.profiler.experimental_config')
    config.AiCMetrics = SimpleNamespace(PipeUtilization='pipes')
    monkeypatch.setitem(sys.modules, 'torch_npu.profiler.experimental_config', config)
    if problem not in ('valid', 'multi-launch'):
        with pytest.raises(RuntimeError, match='export incomplete'):
            native.pipes(tmp_path, {}, lambda: 100)
        return
    result = native.pipes(tmp_path, {}, lambda: 100)
    assert len(calls) == 25
    assert steps == [21, 22, 23, 24, 25]
    assert len(synchronized) == 6
    assert result['collector'] == 'Torch-NPU application PMU'
    assert result['granularity'] == 'per-task'
    assert result['discovered_names'] == list(names)
    count = 10 if problem == 'multi-launch' else 5
    assert [len(capture['pipe_rows']) for capture in result['captures']] == [count, count]
    assert all(row['Name'] == capture['selector'] for capture in result['captures']
               for row in capture['pipe_rows'])
    private = tmp_path / 'private'
    (private / 'results').mkdir(parents=True)
    (private / 'results/summary.json').write_text(json.dumps(dict(state='completed',
        check=dict(passed=True, variants=[0, 1]), result=result)))
    tools.export(private, tmp_path / 'public')
    public = json.loads((tmp_path / 'public/results/summary.json').read_text())
    assert public['collector'] == result['collector']
    assert public['granularity'] == result['granularity']
    assert public['captures'][0]['pipe_rows'][0]['aiv_mte2_ratio'] == 44
    assert public['captures'][0]['pipe_rows'][0]['aic_mac_ratio'] == 'N/A'


@pytest.mark.parametrize('finder_name', ['DependenciesFinder', 'DependencyFinder'])
def test_native_triton_dependency_attribute_error_is_owned_and_observable(tmp_path, monkeypatch, finder_name):
    import ast
    import importlib.util
    from types import ModuleType
    helper = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
    monkeypatch.syspath_prepend(str(helper))
    import candidate_probe
    language = ModuleType('triton.language')
    jit = ModuleType('triton.runtime.jit')
    jit.language = language
    exec('class ' + finder_name + ':\n    def visit_Attribute(self, node):\n        return getattr(language, node.attr)\n', jit.__dict__)
    source = 'import triton.language as tl\ndef kernel():\n    return tl.missing_operation()\n'
    candidate = tmp_path / 'candidate.py'
    candidate.write_text(source)
    node = next(part for part in ast.walk(ast.parse(source)) if isinstance(part, ast.Attribute))
    real_import = candidate_probe.importlib.import_module
    monkeypatch.setattr(candidate_probe.importlib, 'import_module', lambda name:
        {'triton.runtime.jit': jit, 'triton.language': language}.get(name) or real_import(name))
    try:
        getattr(jit, finder_name)().visit_Attribute(node)
    except AttributeError as error:
        report = candidate_probe._frontend_diagnostic(error, candidate)
        assert report == dict(kind='triton-frontend', file='candidate.py', line=3,
            source_line='    return tl.missing_operation()',
            message="module 'triton.language' has no attribute 'missing_operation'")
        candidate.write_text(source.replace('missing_operation', 'different_operation'))
        assert candidate_probe._frontend_diagnostic(error, candidate) is None
        delattr(jit, finder_name)
        assert candidate_probe._frontend_diagnostic(error, candidate) is None
    else:
        pytest.fail('expected actual missing-module-member error')


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
    (tmp_path / 'diagnostics').mkdir()
    (tmp_path / 'diagnostics/traceback.txt').write_text('PRIVATE COMPILER TRACE')
    result = worker.execute(tmp_path, adapter, acquire=acquire, monotonic=lambda: 0)
    assert result['state'] == 'completed' and result['physical_device'] == 3
    assert seen == ['torch', 'torch_npu']
    assert alarms == [240, 180, 240, 0]
    import tarfile
    with tarfile.open(tmp_path / 'evidence.tar.gz') as archive:
        assert archive.extractfile('diagnostics/traceback.txt').read() == b'PRIVATE COMPILER TRACE'
    tools.export(tmp_path, tmp_path / 'public')
    assert 'PRIVATE COMPILER TRACE' not in (tmp_path / 'public/results/summary.json').read_text()


@pytest.mark.parametrize('failure', ['mutation', 'eager-timeout', 'compiled-timeout',
    'wrapped-compiled-timeout', 'validator-timeout', 'compiled-validator-timeout'])
def test_probe_checks_compiled_input_integrity_and_preserves_deadlines(tmp_path, monkeypatch, failure):
    from contextlib import nullcontext
    helpers = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
    monkeypatch.syspath_prepend(str(helpers))
    import candidate_probe
    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def clone(self): return Tensor(self.value)
    def assert_close(left, right, **kwargs):
        assert left.value == right.value, 'input mutated'
    tensor = Tensor(1)
    torch = SimpleNamespace(Tensor=Tensor, inference_mode=nullcontext,
        utils=SimpleNamespace(_pytree=SimpleNamespace(tree_flatten=lambda value: ([tensor], None))),
        testing=SimpleNamespace(assert_close=assert_close))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    class Model:
        def __call__(self, value):
            if failure == 'eager-timeout': raise TimeoutError('original deadline')
            return value
    def prepare(factory, args, kwargs, validate, **options):
        if failure == 'compiled-timeout': raise TimeoutError('original deadline')
        if failure == 'wrapped-compiled-timeout':
            try:
                raise TimeoutError('original deadline')
            except TimeoutError as error:
                raise RuntimeError('backend wrapper') from error
        if failure == 'mutation': tensor.value = 2
        validate(tensor)
        return SimpleNamespace(evidence={'compile_count': 1, 'compiled_executions': 1})
    validations = []
    def validate(*args):
        validations.append(True)
        if failure == 'validator-timeout': raise TimeoutError('original deadline')
        if failure == 'compiled-validator-timeout' and len(validations) == 2:
            raise TimeoutError('original deadline')
    monkeypatch.setattr(candidate_probe, 'prepare_model', prepare)
    options = dict(candidate_file=tmp_path / 'candidate.py', evidence_dir=tmp_path / 'diagnostics',
        backend='npu-inductor', compiler=SimpleNamespace(backend_name='npu-inductor', torch=torch))
    if failure == 'mutation':
        result = candidate_probe.probe_candidate(Model, lambda: [((tensor,), {})], validate, **options)
        assert result['state'] == 'failed' and result['phase'] == 'candidate-input-mutation'
    else:
        with pytest.raises(TimeoutError, match='original deadline'):
            candidate_probe.probe_candidate(Model, lambda: [((tensor,), {})], validate, **options)


def test_candidate_import_deadline_is_not_a_compile_failure(tmp_path, monkeypatch):
    (tmp_path / 'candidate.py').write_text('raise TimeoutError("original deadline")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, 'candidate', raising=False)
    for name in ('torch', 'torch_npu', 'candidate_probe', 'private_campaign_cases'):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(probe_candidate=None, make_case=None, validate=None))
    with pytest.raises(TimeoutError, match='original deadline'):
        native.check(tmp_path, {}, lambda: 100)


def test_prepared_worker_rejects_expired_activation_without_device_access(tmp_path):
    import subprocess
    import time
    prepared = tools.prepare(tmp_path / 'bundle')
    root = Path(prepared['private'])
    (root / 'request.json').write_text(json.dumps(dict(operation='check', activation_required=True,
        host_remaining_seconds=60, source_sha256='a' * 64)))
    (root / 'activation.json').write_text(json.dumps(dict(cutoff_monotonic=time.monotonic() - 1)))
    result = subprocess.run([sys.executable, str(root / 'execute.py')], cwd=root,
        capture_output=True, text=True, timeout=5)
    report = json.loads((root / 'results/summary.json').read_text())
    assert result.returncode == 1 and report['failure_stage'] == 'budget'
    assert 'PHASE_READY=' in result.stdout and 'physical_device' not in report


def test_worker_activation_cutoff_is_applied_before_acquiring_device(tmp_path, monkeypatch):
    import model_campaign_worker as worker
    import phase_deadline
    (tmp_path / 'request.json').write_text(json.dumps(dict(operation='check', activation_required=True,
        host_remaining_seconds=60, source_sha256='a' * 64)))
    events = []
    def activate(path, remaining):
        assert path == tmp_path / 'activation.json' and remaining == 60
        events.append('activated')
        return 110
    def acquire(probe):
        assert events == ['activated']
        events.append('device')
        return 3, {}, SimpleNamespace(close=lambda: None)
    def check(root, env, remaining):
        assert events == ['activated', 'device'] and remaining() == 10
        return dict(passed=True, variants=[0, 1])
    monkeypatch.setattr(phase_deadline, 'wait_for_activation', activate)
    result = worker.execute(tmp_path, SimpleNamespace(check=check), acquire=acquire, monotonic=lambda: 100)
    assert result['state'] == 'completed'


def test_model_service_requests_activation_before_native_dispatch(tmp_path):
    from datetime import datetime, timedelta, timezone
    from model_campaign_service import CampaignService
    public, bundle = tmp_path / 'public', tmp_path / 'bundle'
    public.mkdir()
    bundle.mkdir()
    (public / 'candidate.py').write_text('candidate = 1\n')
    (public / 'execution.json').write_text(json.dumps(dict(
        utc_deadline=(datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat())))
    state = tmp_path / 'state'
    state.mkdir()
    client = SimpleNamespace(state=state, workspace=tmp_path / 'private', cwd='/remote',
        target='test', runtime='test', remote=lambda *args: dict(state='running' if args[0] == 'upload' else 'completed'))
    client.observe = lambda ident: dict(call_id=ident, state='running')
    service = CampaignService(client, bundle, public, tools.export)
    result = service.start('check')
    request = json.loads((state / result['call_id'] / 'request.json').read_text())
    assert request['activation_required'] is True


def test_task_router_selects_private_cases_and_variants(tmp_path, monkeypatch):
    import model_campaign_task as router
    config = dict(task='gdn', profile_index=3, check_indices=[0, 3])
    (tmp_path / 'private-task.json').write_text(json.dumps(config))
    calls = []
    validator = object()
    def make(root, index, variant=0):
        calls.append((root, index, variant))
        return dict(id=f'gdn-{index}-v{variant}')
    monkeypatch.setitem(sys.modules, 'comparison_gdn', SimpleNamespace(make_case=make, validate=validator))
    case, validate = router.case(tmp_path, variant=1)
    assert case['id'] == 'gdn-3-v1' and validate is validator
    assert calls == [(tmp_path / 'reference', 3, 1)]
    assert router.check_indices(tmp_path) == [0, 3]


def test_prepare_comparison_releases_no_private_material(tmp_path, monkeypatch):
    def prepare(private, root):
        private.mkdir()
        (private / 'secret').write_text('PRIVATE ORACLE')
        return dict(public=dict(arguments=['x'], outputs=['y'], starter='class Model: pass\n'))
    monkeypatch.setitem(sys.modules, 'comparison_gdn', SimpleNamespace(prepare=prepare,
        specifications=lambda root: [None] * 7))
    # Copy sources through the actual preparation boundary without changing source tree.
    original = tools.shutil.copyfile
    def copy(source, dest):
        if Path(source).name in ('comparison_gdn.py', 'comparison_gdn_starter.py'):
            Path(dest).write_text('# private adapter placeholder\n')
        else:
            original(source, dest)
    monkeypatch.setattr(tools.shutil, 'copyfile', copy)
    result = tools.prepare(tmp_path / 'bundle', 'gdn', tmp_path / 'references')
    public, private = Path(result['public']), Path(result['private'])
    assert 'PRIVATE ORACLE' not in ''.join(p.read_text() for p in public.iterdir())
    assert (private / 'reference/secret').read_text() == 'PRIVATE ORACLE'
    assert json.loads((private / 'private-task.json').read_text())['check_indices'] == [0, 6]
    assert json.loads((public / 'task.json').read_text())['task'] == 'gdn'


def test_prepare_wraps_bsa_function_without_changing_public_call(tmp_path, monkeypatch):
    def prepare(private, root):
        private.mkdir()
        return dict(api='forward(value)', starter='import torch\ndef forward(value): return value\n')
    monkeypatch.setitem(sys.modules, 'comparison_bsa', SimpleNamespace(prepare=prepare,
        specifications=lambda root: [None] * 50))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(nn=SimpleNamespace(Module=object)))
    original = tools.shutil.copyfile
    def copy(source, dest):
        if Path(source).name in ('comparison_bsa.py', 'comparison_bsa_starter.py'):
            Path(dest).write_text('# private adapter\n')
        else:
            original(source, dest)
    monkeypatch.setattr(tools.shutil, 'copyfile', copy)
    bundle = tools.prepare(tmp_path / 'bundle', 'bsa', tmp_path / 'references')
    namespace = {}
    exec((Path(bundle['public']) / 'candidate.py').read_text(), namespace)
    marker = object()
    assert namespace['Model']().forward(marker) is marker
