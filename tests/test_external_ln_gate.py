"""Observable external-gate semantics, numerical contract, and cache boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from external_ln_contract import checked_input, diagnostics, make_case, parameters, reference
from external_ln_gate import Adapter
from optimization_pair import fixture


def spec(**patch):
    return dict(case_id='cpu-external-ln', role='guard', seed=17,
                parameters=dict(A=3,R=17,**patch))


def test_gate_is_after_affine_not_inside_statistics():
    s=spec(distribution='zero-negative-nonuniform-U');f=make_case(s)
    expected=reference(f['x'],f['U'],f['gamma'],f['beta'])
    assert torch.equal(expected[f['U']==0],torch.zeros_like(expected[f['U']==0]))
    wrong=reference(f['x']*f['U'],torch.ones_like(f['U']),f['gamma'],f['beta'])
    assert not torch.allclose(expected,wrong)
    assert diagnostics(expected.float(),f)['passed']


def test_constant_row_preserves_affine_beta_before_gate():
    x=torch.ones((2,9),dtype=torch.float32);u=torch.arange(18,dtype=torch.float32).reshape(2,9)-4
    gamma=torch.ones(9,dtype=torch.float32);beta=torch.full((9,),.125,dtype=torch.float32)
    assert torch.equal(reference(x,u,gamma,beta),u.double()*.125)


def test_fixture_cache_and_ambient_float64(tmp_path):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a,ia=fixture(Adapter(),spec(),tmp_path,torch)
        b,ib=fixture(Adapter(),spec(),tmp_path,torch)
        assert ia['oracle_calls']==1 and ib['oracle_calls']==0
        assert all(a[k].dtype==torch.float32 for k in ('x','U','gamma','beta'))
        assert torch.equal(a['expected'],b['expected'])
    finally:torch.set_default_dtype(old)


def test_reject_shape_dtype_layout_and_nonfinite():
    s=spec();f=make_case(s)
    for key,bad in [('U',f['U'][:1]),('x',f['x'].double()),('x',f['x'].t()),('x',f['x']*float('nan'))]:
        with pytest.raises(ValueError):checked_input(s,dict(f,**{key:bad}))


def test_malformed_guard_before_any_device_work(tmp_path):
    s=spec(malformed_U_shape=True)
    with pytest.raises(ValueError,match='U shape'):make_case(s)
    for arm in ('before','after'):
        with pytest.raises(ValueError,match='U shape'):Adapter().prepare(arm,s,{},tmp_path)


@pytest.mark.parametrize('patch',[dict(A=0,R=8),dict(A=True,R=8),dict(A=1,R=0)])
def test_bad_shape_parameters(patch):
    s=spec();s['parameters'].update(patch)
    with pytest.raises(ValueError):parameters(s)


def test_public_fixture_cli(tmp_path):
    case=tmp_path/'case.json';case.write_text(json.dumps(spec()))
    output=tmp_path/'fixture.pt'
    result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/external_ln_contract.py'),
                           '--case',str(case),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert torch.load(output,weights_only=False)['expected'].shape==(3,17)


def test_public_runner_host_rejection_has_no_cache_or_calls(tmp_path):
    case=tmp_path/'case.json';case.write_text(json.dumps(spec(malformed_U_shape=True)))
    provenance=tmp_path/'provenance.json';provenance.write_text('{}')
    output,cache=tmp_path/'output',tmp_path/'cache'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),
        '--adapter','external_ln_gate:Adapter','--case',str(case),'--output',str(output),
        '--cache',str(cache),'--provenance',str(provenance)],
        env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),capture_output=True,text=True)
    assert result.returncode==1,result.stderr
    report=json.loads((output/'result.json').read_text())
    assert report['error']['type']=='ValueError' and 'U shape' in report['error']['message']
    assert report['counts']==dict(profiled_calls=0,capture_calls=0,correctness_calls=0)
    assert not list(cache.rglob('fixture.pt')) and not list(cache.rglob('manifest.json'))
