"""Functional token mixing, fixture, rejection and public runner checks."""
import json, os, subprocess, sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
from token_mixing_contract import make_case, parameters, reference, check_inputs
from token_mixing import Adapter
from optimization_pair import fixture


def spec(**patch):
    return dict(case_id='cpu-token',role='guard',seed=73,parameters=dict(B=2,S=3,**patch))


def test_reference_matches_scalar():
    f=make_case(spec());x=f['x'].double();g=f['gamma'].double();beta=f['beta'].double()
    for b in range(2):
        for i in range(3):
            row=[float(x[b,i,j]+x[b,j,i]) for j in range(3)]
            mean=sum(row)/3;var=sum((v-mean)**2 for v in row)/3
            expected=torch.tensor([(v-mean)/(var+1e-5)**.5*float(g[j])+float(beta[j]) for j,v in enumerate(row)],dtype=torch.float64)
            assert torch.allclose(f['expected'][b,i],expected,atol=1e-12,rtol=1e-12)


def test_constant_symmetric_nonzero_beta():
    f=make_case(spec(distribution='constant-and-symmetric'))
    assert torch.equal(f['x'][1],f['x'][1].T)
    assert torch.equal(f['expected'][0],f['beta'].double().expand(3,3))


def test_cache_under_ambient_float64(tmp_path):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a,ai=fixture(Adapter(),spec(),tmp_path,torch);b,bi=fixture(Adapter(),spec(),tmp_path,torch)
        assert a['x'].dtype==torch.float32 and a['x'].device.type=='cpu'
        assert ai['oracle_calls']==1 and bi['oracle_calls']==0
        assert torch.equal(a['expected'],b['expected'])
    finally:torch.set_default_dtype(old)


@pytest.mark.parametrize('patch',[{'B':0},{'S':True},{'distribution':'unknown'}])
def test_bad_parameters(patch):
    s=spec();s['parameters'].update(patch)
    with pytest.raises(ValueError):parameters(s)


def test_bad_layout_dtype():
    f=make_case(spec())
    for bad in (f['x'].double(),f['x'].transpose(-1,-2),f['x']*float('nan'), f['x']+5):
        with pytest.raises(ValueError):check_inputs(dict(f,x=bad),2,3)


def test_public_fixture_cli(tmp_path):
    path=tmp_path/'case.json';path.write_text(json.dumps(spec()))
    result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/token_mixing_contract.py'),'--case',str(path),'--output',str(tmp_path/'fixture.pt')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert torch.load(tmp_path/'fixture.pt',weights_only=False)['expected'].shape==(2,3,3)


def test_nonsquare_public_runner_no_poison(tmp_path):
    s=spec(malformed_nonsquare=True)
    for arm in ('before','after'):
        with pytest.raises(ValueError,match='square'):Adapter().prepare(arm,s,{},tmp_path)
    case=tmp_path/'case.json';case.write_text(json.dumps(s))
    provenance=tmp_path/'prov.json';provenance.write_text('{}')
    output,cache=tmp_path/'output',tmp_path/'cache'
    p=subprocess.run([sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','token_mixing:Adapter','--case',str(case),'--output',str(output),'--cache',str(cache),'--provenance',str(provenance)],env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),capture_output=True,text=True)
    assert p.returncode==1,p.stderr
    report=json.loads((output/'result.json').read_text())
    assert report['error']['type']=='ValueError' and 'square' in report['error']['message']
    assert report['counts']==dict(profiled_calls=0,capture_calls=0,correctness_calls=0)
    assert not list(cache.rglob('fixture.pt')) and not list(cache.rglob('manifest.json'))
