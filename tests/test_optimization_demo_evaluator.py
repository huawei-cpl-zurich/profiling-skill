"""Functional evaluator contracts; real tensor checks and trace recount, no NPU."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import optimization_demo_worker as worker
import optimization_demo_evaluator as host
from test_optimization_pair import Adapter


class Jit:
    calls = 0
    def __init__(self, fn):
        self.fn = fn
        self.compiled = SimpleNamespace(name='kernel', hash='stable', asm={'binary': b'ELF'})
    def run(self, *args, **kwargs):
        Jit.calls += 1
        return self.compiled


def fixture_request(tmp_path, monkeypatch, *, wrong=False, import_launch=False):
    module = ModuleType('demo_fake_runtime'); module.Jit = Jit
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path/'candidate.py'
    source.write_text('from demo_fake_runtime import Jit\n'
        'def kernel(): pass\nlaunch=Jit(kernel)\nKERNEL_NAMES=("dishonest",)\n' +
        ('launch.run()\n' if import_launch else '') +
        'def run(inputs):\n launch.run()\n return inputs["x"]+'+('2' if wrong else '1')+'\n')
    provider = tmp_path/'provider.py'
    provider.write_text('import torch\ndef validate_actual(actual,fixture,operation_id,inputs):\n'
        ' return {"passed":torch.equal(actual,fixture["expected"]) and torch.equal(inputs["x"],fixture["inputs"]["x"])}\n')
    payload = tmp_path/'fixture.pt'; value=torch.arange(6)
    torch.save(dict(inputs={'x':value},expected=value+1),payload)
    return dict(candidate_path=str(source),candidate_sha256=worker.sha(source),
        provider_path=str(provider),provider_sha256=worker.sha(provider),operation_id='test',
        physical_die=0,cases=[dict(id=name,fixture_path=str(payload),fixture_sha256=worker.sha(payload),
            baseline_samples_us=[20.]*10) for name in ('first','second')])


def test_one_check_then_ten_trace_proven_calls_per_case(tmp_path, monkeypatch):
    request=fixture_request(tmp_path,monkeypatch); Jit.calls=0
    captured=Adapter()
    def capture(run,path,**kwargs): return captured.capture(run,None,path)
    result=worker.evaluate_cases(request,tmp_path/'out',torch=torch,jit_class=Jit,
        capture=capture,synchronize=lambda:None,device='cpu')
    assert result['outcome']=='success'
    assert Jit.calls==22
    for row in result['cases']:
        assert row['correctness_calls']==1 and row['profiled_calls']==10
        assert row['identity_verified'] and row['candidate_samples_us']==[10.]*10
        assert row['compiled'][0]['name']=='kernel'


@pytest.mark.parametrize('failure',['wrong','import_launch','provider_digest','source_change','binary_change'])
def test_failure_stops_all_later_device_calls(tmp_path,monkeypatch,failure):
    request=fixture_request(tmp_path,monkeypatch,wrong=failure=='wrong',import_launch=failure=='import_launch')
    if failure=='provider_digest':
        Path(request['provider_path']).write_text('from pathlib import Path\nPath('+repr(str(tmp_path/'executed'))+').touch()\n')
        request['provider_sha256']='bad'
    def capture(run,path,**kwargs):
        result=Adapter().capture(run,None,path)
        if failure=='source_change': Path(request['candidate_path']).write_text('# changed\n')
        if failure=='binary_change':
            sys.modules['submitted_' + request['candidate_sha256']].launch.compiled.asm['binary'] = b'changed'
        return result
    if failure=='provider_digest':
        with pytest.raises(worker.InfrastructureError):
            worker.evaluate_cases(request,tmp_path/'out',torch=torch,jit_class=Jit,
                capture=capture,synchronize=lambda:None,device='cpu')
        assert not (tmp_path/'executed').exists()
    else:
        result=worker.evaluate_cases(request,tmp_path/'out',torch=torch,jit_class=Jit,
            capture=capture,synchronize=lambda:None,device='cpu')
        assert result['outcome']=='candidate_failure'
        assert len(result['cases'])==2 and result['cases'][1]['correctness_calls']==0
        assert result['cases'][1]['status']=='not-run'


class Remote:
    def __init__(self, *, lost=False, recovery=None):
        self.submits=0;self.uploads=0;self.lost=lost;self.recovery=recovery;self.interrupted=False
    def preflight(self, request): return {}
    def _call(self,args):
        if args[0]=='upload':
            self.uploads+=1
            return {'state':'running','handle':'remote:test:transfer:up'} if self.recovery=='transfer' else {'state':'completed'}
        if self.recovery and not self.interrupted:
            self.interrupted=True
            raise host.ObservationLost('observer connection lost')
        return dict(target='test',handle=args[1],state='completed',exit=0)
    def submit(self,*args):
        self.submits+=1
        if self.lost: raise host.ObservationLost('dispatch outcome unknown')
        return 'remote:test:job:one'
    def _identity(self,*args): pass
    def _logs(self,*args):return 'DEMO_RESULT '+json.dumps(self.result)


@pytest.mark.parametrize('recovery',[None,'transfer','job'])
def test_cached_result_and_uncertain_dispatch_never_duplicate(tmp_path,recovery):
    source=tmp_path/'candidate.py';source.write_text('def run(inputs): return inputs\n')
    config=dict(target='test',runtime='runtime',remote_workspace='workspace',remote_root='runs',
        cache_directory=str(tmp_path/'cache'),provider_path='provider',provider_sha256='providerhash',
        operation_id='operation',physical_die=0,device_query={'card':0,'chip':0},lease_directory='leases',
        cases={'development':[dict(id='case',fixture_path='fixture',fixture_sha256='fixturehash',baseline_samples_us=[20.]*10)]})
    request=dict(candidate_path=str(source),candidate_sha256=host.sha(source),phase='development')
    remote=Remote(recovery=recovery);remote.result=dict(schema='optimization-coding-evaluation-v1',measurement_kind='profiled_device_operation_span_us',
        candidate_sha256=request['candidate_sha256'],outcome='success',cases=[dict(id='case',correct=True,identity_verified=True,
        baseline_samples_us=[20.]*10,candidate_samples_us=[10.]*10)])
    first=host.evaluate(config,request,remote=remote)
    if recovery:
        assert first['outcome']=='infrastructure_failure'
        first=host.evaluate(config,request,remote=remote)
    assert first['outcome']=='success' and host.evaluate(config,request,remote=remote)==first
    assert (remote.submits,remote.uploads)==(1,1)
    config['cache_directory']=str(tmp_path/'uncertain'); lost=Remote(lost=True)
    assert host.evaluate(config,request,remote=lost)['outcome']=='infrastructure_failure'
    assert host.evaluate(config,request,remote=lost)['outcome']=='infrastructure_failure'
    assert (lost.submits,lost.uploads)==(1,1)


@pytest.mark.parametrize('change', [None, 'kernel', 'compiled'])
def test_measured_hook_uses_only_frozen_object_identities(tmp_path, monkeypatch, change):
    request = fixture_request(tmp_path, monkeypatch)
    def capture(run, path, **kwargs):
        module = sys.modules['submitted_' + request['candidate_sha256']]
        if change == 'kernel': module.launch = Jit(module.kernel)
        if change == 'compiled':
            module.launch.compiled = SimpleNamespace(name='kernel', hash='stable', asm={'binary': b'ELF'})
        def forbidden(*args, **kwargs):
            raise AssertionError('measured launch inspected source or hashed binary')
        # Limit sentinels to forward execution; the exporter and post-capture
        # evidence validation legitimately inspect files and compute digests.
        def measured():
            with monkeypatch.context() as patch:
                patch.setattr(worker, 'sha', forbidden)
                patch.setattr(worker.hashlib, 'sha256', forbidden)
                patch.setattr(Path, 'resolve', forbidden)
                patch.setattr(Path, 'read_bytes', forbidden)
                patch.setattr(Jit, 'fn', property(forbidden), raising=False)
                return run()
        return Adapter().capture(measured, None, path)
    result = worker.evaluate_cases(request, tmp_path/'out', torch=torch, jit_class=Jit,
        capture=capture, synchronize=lambda:None, device='cpu')
    assert result['measurement_version'] == 2
    if change is None:
        assert result['outcome'] == 'success'
        assert all(row['profiled_calls'] == 10 for row in result['cases'])
    else:
        assert result['outcome'] == 'candidate_failure'
        assert result['cases'][0]['profiled_calls'] == 1
        assert result['cases'][1]['status'] == 'not-run'
        assert ('unqualified kernel' if change == 'kernel' else 'specialization changed') in result['cases'][0]['error']


@pytest.mark.parametrize('fault', [None, 'stride', 'dtype', 'values'])
def test_trusted_nd_preparation_preserves_input_contract_before_capture(fault):
    source = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    untouched = torch.arange(4, dtype=torch.int32)
    formats = {id(source): 0, id(untouched): 2}
    casts = []
    def cast(value, target):
        casts.append((value, target))
        result = value.clone()
        if fault == 'stride': result = result.transpose(-1, -2).contiguous().transpose(-1, -2)
        if fault == 'dtype': result = result.half()
        if fault == 'values': result = result + 1
        formats[id(result)] = target
        return result
    backend = SimpleNamespace(get_npu_format=lambda value: formats[id(value)], npu_format_cast=cast)
    inputs = dict(initial=source, offsets=untouched)
    if fault:
        with pytest.raises(worker.InfrastructureError, match='trusted ND preparation failed'):
            worker.prepare_nd_inputs(inputs, inputs, backend, torch)
    else:
        prepared, converted = worker.prepare_nd_inputs(inputs, inputs, backend, torch)
        assert converted == ['initial'] and len(casts) == 1
        assert prepared['offsets'] is untouched
        assert prepared['initial'].dtype == source.dtype and prepared['initial'].stride() == source.stride()
        assert torch.equal(prepared['initial'], source) and backend.get_npu_format(prepared['initial']) == 2


def test_correct_values_in_strided_output_fail_before_any_capture(tmp_path, monkeypatch):
    request = fixture_request(tmp_path, monkeypatch)
    source = Path(request['candidate_path'])
    source.write_text(source.read_text().replace('return inputs["x"]+1',
        'return (inputs["x"]+1).repeat_interleave(2)[::2]'))
    request['candidate_sha256'] = worker.sha(source)
    def unexpected_capture(*args, **kwargs):
        pytest.fail('a rejected output must never enter capture')
    Jit.calls = 0
    report = worker.evaluate_cases(request, tmp_path/'out', torch=torch, jit_class=Jit,
        capture=unexpected_capture, synchronize=lambda:None, device='cpu')
    assert report['outcome'] == 'candidate_failure' and Jit.calls == 1
    assert report['cases'][0]['correctness_calls'] == 1 and report['cases'][0]['profiled_calls'] == 0
    assert report['cases'][0]['output_metadata'] == [dict(shape=[6],strides=[2],dtype='torch.int64',contiguous=False)]
    assert report['cases'][1]['status'] == 'not-run'


@pytest.mark.parametrize('stdin_action', [False, True])
def test_cli_help_never_reads_private_config_or_candidate(tmp_path, stdin_action):
    import subprocess
    secret_config = tmp_path/'private-task-secret.json'
    secret_config.write_text('invalid private JSON: help must not read this')
    command = [sys.executable, str(Path(host.__file__)), '--config', str(secret_config)]
    if not stdin_action: command += ['--workspace', str(tmp_path/'absent-workspace'), 'help']
    result = subprocess.run(command, input=json.dumps(dict(action='help')) if stdin_action else '',
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stderr == ''
    assert 'ten profiled forwards' in result.stdout and 'cached evidence' in result.stdout
    assert 'private-task-secret' not in result.stdout and str(tmp_path) not in result.stdout


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'digest', 'transfer'])
def test_trusted_fixture_failures_are_infrastructure_before_candidate(tmp_path, monkeypatch, failure):
    request = fixture_request(tmp_path, monkeypatch)
    payload = Path(request['cases'][0]['fixture_path'])
    if failure == 'missing': payload.unlink()
    elif failure == 'corrupt':
        payload.write_bytes(b'not a torch payload')
        request['cases'][0]['fixture_sha256'] = worker.sha(payload)
    elif failure == 'digest': request['cases'][0]['fixture_sha256'] = 'wrong'
    else:
        def failed_transfer(*args, **kwargs): raise RuntimeError('device transfer unavailable')
        monkeypatch.setattr(torch.Tensor, 'to', failed_transfer)
    def unexpected_capture(*args, **kwargs): pytest.fail('trusted preparation must precede capture')
    Jit.calls = 0
    report = worker.evaluate_cases(request, tmp_path/'out', torch=torch, jit_class=Jit,
        capture=unexpected_capture, synchronize=lambda:None, device='cpu')
    assert report['outcome'] == 'infrastructure_failure' and Jit.calls == 0
    assert report['classification_version'] == 2
    assert report['cases'][0]['error_stage'] == 'fixture_preparation'
    assert report['cases'][0]['correctness_calls'] == report['cases'][0]['profiled_calls'] == 0
    assert report['cases'][1]['status'] == 'not-run'


@pytest.mark.parametrize('failed_forward,exit_error', [(1,False), (3,False), (1,True), (None,False)])
def test_real_capture_swallowed_candidate_error_stays_candidate_failure(tmp_path, monkeypatch,
                                                                       failed_forward, exit_error):
    from contextlib import nullcontext
    request = fixture_request(tmp_path, monkeypatch)
    source = Path(request['candidate_path'])
    source.write_text(source.read_text().replace('def run(inputs):\n launch.run()',
        '_runs=0\ndef run(inputs):\n global _runs\n _runs+=1\n' +
        (f' if _runs=={failed_forward+1}: raise ValueError("candidate profile failure")\n' if failed_forward else '') +
        ' launch.run()'))
    request['candidate_sha256'] = worker.sha(source)
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): pass
        def elapsed_time(self, end): return .5
    class Profile:
        def __enter__(self): return self
        def __exit__(self, *args):
            if exit_error: raise RuntimeError('export masked candidate failure')
        def step(self): pass
    fake_torch = SimpleNamespace(inference_mode=nullcontext,
        npu=SimpleNamespace(Event=Event, synchronize=lambda:None),
        autograd=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name:nullcontext())))
    fake_npu = SimpleNamespace(profiler=SimpleNamespace(profile=lambda **kwargs:Profile(),
        schedule=lambda **kwargs:None, ProfilerActivity=SimpleNamespace(CPU=1,NPU=2),
        _ExperimentalConfig=lambda **kwargs:kwargs, ProfilerLevel=SimpleNamespace(Level1=1),
        AiCMetrics=SimpleNamespace(PipeUtilization='pipe'), tensorboard_trace_handler=lambda *args,**kwargs:None))
    def failed_export(*args, **kwargs): raise RuntimeError('export unavailable')
    monkeypatch.setitem(sys.modules, 'torch_npu.profiler.profiler', SimpleNamespace(analyse=failed_export))
    captures = []
    def capture(run, path, **kwargs):
        result = worker.capture_ten(run, path, torch_module=fake_torch, npu_module=fake_npu, **kwargs)
        captures.append(result)
        return result
    Jit.calls = 0
    report = worker.evaluate_cases(request, tmp_path/'out', torch=torch, jit_class=Jit,
        capture=capture, synchronize=lambda:None, device='cpu')
    assert len(captures) == 1 and captures[0]['profile']['status'] == 'failed'
    assert report['cases'][1]['status'] == 'not-run'
    assert report['cases'][0]['correctness_calls'] == 1
    if failed_forward:
        assert report['outcome'] == 'candidate_failure'
        assert report['cases'][0]['profiled_calls'] == failed_forward
        assert report['cases'][0]['cause_type'] == 'ValueError'
        assert report['cases'][0]['error'] == 'candidate profile failure'
        assert Jit.calls == failed_forward
    else:
        assert report['outcome'] == 'infrastructure_failure'
        assert report['cases'][0]['profiled_calls'] == 10 and Jit.calls == 11


def host_request(tmp_path):
    source = tmp_path/'candidate.py'; source.write_text('def run(inputs): return inputs\n')
    config = dict(task_id='private-task',target='test',runtime='runtime',remote_workspace='workspace',
        remote_root='runs',cache_directory=str(tmp_path/'cache'),provider_path='private-provider',
        provider_sha256='providerhash',operation_id='operation',physical_die=0,
        device_query=dict(card=0,chip=0),lease_directory='leases',
        cases={'development':[dict(id='private-case',fixture_path='private-fixture',fixture_sha256='hash',
                                   baseline_samples_us=[20.]*10)]})
    request = dict(candidate_path=str(source),candidate_sha256=host.sha(source),phase='development')
    report = dict(schema='optimization-coding-evaluation-v1',measurement_kind='profiled_device_operation_span_us',
        candidate_sha256=request['candidate_sha256'],outcome='success',cases=[dict(id='private-case',correct=True,
        identity_verified=True,baseline_samples_us=[20.]*10,candidate_samples_us=[10.]*10)])
    return config, request, report


def test_readonly_preflight_retries_without_duplicate_upload_or_dispatch(tmp_path):
    config, request, report = host_request(tmp_path)
    class RetryRemote(Remote):
        preflights = 0
        def preflight(self, request):
            self.preflights += 1
            if self.preflights == 1: raise host.ObservationLost('read-only preflight unavailable')
    remote = RetryRemote(); remote.result = report
    assert host.evaluate(config,request,remote=remote)['outcome'] == 'infrastructure_failure'
    assert (remote.preflights,remote.uploads,remote.submits) == (1,0,0)
    assert host.evaluate(config,request,remote=remote)['outcome'] == 'success'
    assert host.evaluate(config,request,remote=remote)['outcome'] == 'success'
    assert (remote.preflights,remote.uploads,remote.submits) == (2,1,1)
    receipt = json.loads(next(Path(config['cache_directory']).glob('*/receipt.json')).read_text())
    assert receipt['preflight_attempts'] == 2
    assert receipt['preflight_errors'][0]['error'] == 'read-only preflight unavailable'


@pytest.mark.parametrize('state', ['unknown-upload', 'legacy-diagnostic', 'ambiguous-preflight'])
def test_uncertain_intents_never_become_retryable_preflight(tmp_path, state):
    config, request, report = host_request(tmp_path)
    class CountingRemote(Remote):
        preflights = 0
        def preflight(self, request): self.preflights += 1
        def _call(self, args):
            if args[0] == 'upload':
                self.uploads += 1
                raise host.ObservationLost('unknown upload outcome')
            return super()._call(args)
    remote = CountingRemote(); remote.result = report
    if state != 'unknown-upload':
        cache = Path(config['cache_directory'])/host.cache_key(config,'development',request['candidate_sha256'])
        cache.mkdir(parents=True)
        receipt = dict(error_type='ObservationLost',error='no affirmative no-intent proof')
        if state == 'ambiguous-preflight':receipt.update(state='preflight',remote_intent_created=False,transfer={})
        (cache/'receipt.json').write_text(json.dumps(receipt))
    for _ in range(2): assert host.evaluate(config,request,remote=remote)['outcome'] == 'infrastructure_failure'
    assert remote.submits == 0
    assert (remote.preflights,remote.uploads) == ((1,1) if state == 'unknown-upload' else (0,0))


def test_actor_can_follow_public_help_and_cannot_override_managed_arguments(tmp_path):
    import subprocess
    config, request, report = host_request(tmp_path)
    cache = Path(config['cache_directory'])/host.cache_key(config,'development',request['candidate_sha256'])
    cache.mkdir(parents=True); (cache/'report.json').write_text(json.dumps(report))
    private_config = tmp_path/'managed.json';private_config.write_text(json.dumps(config))
    prefix = [sys.executable,str(Path(host.__file__)),'--config',str(private_config),'--workspace',str(tmp_path)]
    for flag in ('help','--help'):
        result = subprocess.run(prefix+[flag],capture_output=True,text=True,timeout=10)
        assert result.returncode == 0 and result.stderr == ''
        assert '--config' not in result.stdout and '--workspace' not in result.stdout
    result = subprocess.run(prefix+['evaluate'],capture_output=True,text=True,timeout=10)
    assert result.returncode == 0 and json.loads(result.stdout) == report
    assert not (cache/'receipt.json').exists()  # Cached public evaluate needs no transport.
    for flag in ('--config','--workspace'):
        result = subprocess.run(prefix+[flag,str(tmp_path/'override'),'evaluate'],
                                capture_output=True,text=True,timeout=10)
        assert result.returncode == 2 and 'cannot be overridden' in result.stderr
        assert result.stdout == '' and not (cache/'receipt.json').exists()
