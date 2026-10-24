"""Functional CPU coverage of F11 generation, reconstruction and gates."""
import importlib.util
from pathlib import Path
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton/two_component_contract.py'
spec = importlib.util.spec_from_file_location('msd_contract', PATH)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def case(**kw):
    return dict(seed=11102, parameters=dict(M=2,N=8,K=128,amax=1.,weight_max=8,scale=2**-10), **kw)


def test_frozen_generator_exact_codes_and_dense_gate():
    f = c.make_case(case())
    assert f['rounding_margin'] >= 1e-5
    assert torch.equal(f['inputs']['x'][:, 0], torch.ones(2).half())
    assert c.validate_actual(f['expected'], f, f['inputs'], 'after', (f['a1'],f['a2']))['passed']
    assert c.validate_actual(f['dense'].half(), f, f['inputs'], 'before')['passed']
    bad = f['a2'].clone();bad[0,0] = 1
    assert not c.validate_actual(f['expected'], f, f['inputs'], 'after', (f['a1'],bad))['passed']
    assert not c.validate_actual(f['expected']+1, f, f['inputs'], 'before')['passed']


def test_zero_row_and_input_mutation():
    f = c.make_case(case(distribution='zero-row'))
    assert not f['a1'][0].any() and not f['a2'][0].any() and not f['expected'][0].any()
    changed = {k:v.clone() for k,v in f['inputs'].items()};changed['x'][0,0]=1
    with pytest.raises(ValueError, match='input mutated'):
        c.validate_actual(f['expected'], f, changed, 'after')


def test_scope_and_reconstruction_rounding():
    s=case();s['parameters']['K']=262144;s['parameters']['weight_max']=127
    assert not c.eligible(s)
    amax=torch.tensor([[1.]])
    c1=torch.tensor([[100,-100]],dtype=torch.int64);c2=torch.tensor([[127,-127]],dtype=torch.int64)
    expected=((c1.float()+c2.float()*torch.tensor(1/254,dtype=torch.float32))*(amax/127)*2**-10).half()
    assert torch.equal(c.reconstruct(c1,c2,amax,torch.full((2,),2**-10)),expected)
    s['parameters']['weight_max']=128
    with pytest.raises(ValueError):c.check_spec(s)
