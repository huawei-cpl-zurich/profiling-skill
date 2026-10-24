"""Functional FP32 crossing, tied support and mandatory fallback checks."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/optimization-library/ascendc'))
from top_p_prefix_contract import reference,make_case,diagnostics,dimensions


def test_strict_crossing_and_margin_fallback():
    scores=torch.zeros(1,16,dtype=torch.float16)
    exact=reference(scores,.5,16)[0]
    assert exact['count']==9 and exact['tag']==0
    away=reference(scores,.51,16)[0]
    assert away['count']==9 and away['tag']==1
    assert reference(scores,.51,8)[0]['tag']==0


def test_normalization_uses_suffix():
    assert reference(torch.zeros(1,16,dtype=torch.float16),.9,4)[0]['tag']==0
    r=reference(torch.tensor([[8.,8.,-8.,-8.]]).half(),.6,2)[0]
    assert r['count']==2 and r['tag']==1


def test_diagnostics_checks_count_gather_tags_and_unique_support():
    spec=dict(rows=1,V=16,p=.51,prefix=12,seed=1,distribution='uniform',role='guard')
    f=make_case(spec);ids=torch.arange(16).reshape(1,-1).int()
    count=torch.tensor([9],dtype=torch.int32);tag=torch.tensor([1],dtype=torch.int32)
    assert diagnostics((ids,count,tag),f,'after')['passed']
    assert not diagnostics((ids,count,tag),f,'before')['passed']
    repeated=ids.clone();repeated[0,1]=0
    assert not diagnostics((repeated,count,tag),f,'after')['passed']
    assert not diagnostics((ids,count-1,tag),f,'after')['passed']


@pytest.mark.parametrize('p',[0,1,float('nan'),float('inf')])
def test_admission_guard(p):
    with pytest.raises(ValueError):dimensions(dict(rows=1,V=16,prefix=4,seed=1,p=p))
