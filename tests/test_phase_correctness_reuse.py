import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_campaign_worker as worker
import model_campaign_tools as tools


def fixture(tmp_path):
    (tmp_path/'private-task.json').write_text(json.dumps(dict(task='matmul')))
    runtime = dict(physical=2, env={}, runtime_initializations=1, torch_module_id=123)
    checks, measurements = [], []
    def check(root, env, remaining):
        checks.append(root.name)
        return dict(passed=True, variants=[0,1], diagnostics=[dict(compile_count=1)])
    def measurement(root, env, remaining):
        measurements.append(root.name)
        return dict(state='complete')
    adapter = SimpleNamespace(check=check, benchmark=measurement, pipes=measurement)
    def run(name, operation, source=b'kernel'):
        call = tmp_path/'calls'/name
        call.mkdir(parents=True)
        (call/'candidate.py').write_bytes(source)
        (call/'request.json').write_text(json.dumps(dict(operation=operation,
            host_remaining_seconds=10, source_sha256=hashlib.sha256(source).hexdigest())))
        return worker.execute(call, adapter, runtime=runtime)
    return run, runtime, adapter, checks, measurements


def test_measurements_reuse_only_passed_source_and_publish_original_provenance(tmp_path):
    run, runtime, adapter, checks, measurements = fixture(tmp_path)
    assert run('first','check')['state']=='completed'
    timing=run('timing','benchmark')
    assert run('pipes','pipes')['state']=='completed'
    assert run('explicit','check')['state']=='completed'
    assert checks==['first','explicit'] and measurements==['timing','pipes']
    assert timing['check']['diagnostics']==[]
    assert timing['correctness_provenance']==dict(kind='phase-local-reuse',original_call='first',
        source_sha256=hashlib.sha256(b'kernel').hexdigest())
    tools.export(tmp_path/'calls/timing',tmp_path/'public')
    published=json.loads((tmp_path/'public/results/summary.json').read_text())
    assert published['correctness_provenance']==timing['correctness_provenance']
    assert published['correctness'] is True and published['diagnostics']==[]


@pytest.mark.parametrize('changed', ['source','device','runtime','bundle','adapter','context'])
def test_changed_identity_forces_full_measurement_check(tmp_path,changed):
    run,runtime,adapter,checks,measurements=fixture(tmp_path)
    run('first','check')
    source=b'kernel'
    if changed=='source':source=b'changed'
    if changed=='device':runtime['physical']=5
    if changed=='runtime':runtime['torch_module_id']=456
    if changed=='bundle':(tmp_path/'private-task.json').write_text(json.dumps(dict(task='bsa')))
    if changed=='adapter':
        original=adapter.check
        adapter.check=lambda *args:original(*args)
    if changed=='context':runtime['env']['MODE']='changed'
    result=run('measure','benchmark',source)
    assert result['state']=='completed' and checks==['first','measure']
    assert result['correctness_provenance']['kind']=='fresh'


def test_failed_check_is_never_reused(tmp_path,monkeypatch):
    run,runtime,adapter,checks,measurements=fixture(tmp_path)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(ones=lambda *a,**k:None,
        npu=SimpleNamespace(synchronize=lambda:None)))
    good=adapter.check
    passed=[False]
    def check(*args):
        result=good(*args)
        result['passed']=passed[0]
        return result
    adapter.check=check
    assert run('failed','check')['state']=='failed'
    passed[0]=True
    assert run('timing','benchmark')['state']=='completed'
    assert checks==['failed','timing']


def test_measurement_fresh_profile_validation_still_runs_and_failure_invalidates(tmp_path):
    run,runtime,adapter,checks,measurements=fixture(tmp_path)
    run('first','check')
    def invalid(*args):
        raise AssertionError('fresh profile-case validation failed')
    adapter.benchmark=invalid
    result=run('timing','benchmark')
    assert result['state']=='failed' and result['failure_stage']=='measurement'
    assert result['correctness_provenance']['kind']=='phase-local-reuse'
    assert checks==['first']
    adapter.benchmark=adapter.pipes
    assert run('retry','benchmark')['state']=='completed'
    assert checks==['first','retry']
