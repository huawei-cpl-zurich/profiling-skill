"""Functional CPU contract and scheduling qualification tests (no device launches)."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[1]/'examples/optimization-library/catlass'
sys.path.insert(0,str(EXAMPLE))
import tail_streamk_contract as contract


def spec(**kw):
    return dict(m=32,n=48,k=256,seed=6100,**kw)


@pytest.mark.parametrize('shape,expected',[
    ((128,10496,4096),(41,40,1,16)),
    ((384,6656,512),(78,60,18,36)),
    ((128,15616,6144),(61,60,1,24)),
    ((384,6656,1024),(78,60,18,72)),
    ((128,5120,256),(20,20,0,0))])
def test_geometry(shape,expected):
    g=contract.geometry(dict(zip(('m','n','k'),shape),seed=6101))
    assert tuple(g[k] for k in ('output_tiles','normal_tiles','tail_tiles','tail_k_blocks'))==expected
    # Each K block in the tail is assigned exactly once, with <=1 load imbalance.
    loads=[g['tail_k_blocks_per_worker']+(i<g['tail_extra_workers']) for i in range(20)]
    assert sum(loads)==g['tail_k_blocks']
    assert max(loads)-min(loads)<=1


def test_fixture_oracle_and_tail_overwrite():
    fixture=contract.make_case(spec())
    assert all(t.dtype==torch.float16 and t.abs().max()<=.125 for t in fixture['inputs'].values())
    fp64=fixture['inputs']['a'].double()@fixture['inputs']['b'].double()
    assert torch.allclose(fixture['expected'].double(),fp64,atol=1e-7,rtol=1e-5)
    assert contract.validate(fixture['expected'].half(),fixture)['passed']
    bad=fixture['expected'].half();bad[-1,-1]=float('nan')
    assert not contract.validate(bad,fixture)['passed']
    bad=fixture['expected'].half();bad[0,0]+=1
    assert not contract.validate(bad,fixture)['passed']


@pytest.mark.parametrize('change',[dict(m=17),dict(n=31),dict(k=128),dict(m=0),dict(seed=True)])
def test_invalid_shape_rejects(change):
    with pytest.raises(ValueError):contract.make_case(dict(spec(),**change))


def test_wrong_core_count_and_input_mutation():
    with pytest.raises(ValueError):contract.geometry(spec(),19)
    fixture=contract.make_case(spec())
    mutated={k:v.clone() for k,v in fixture['inputs'].items()};mutated['a'][0,0]+=1
    with pytest.raises(ValueError,match='input mutation'):
        contract.validate(fixture['expected'].half(),fixture,mutated)
