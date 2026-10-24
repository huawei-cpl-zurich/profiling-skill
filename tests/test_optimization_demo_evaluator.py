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
