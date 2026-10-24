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


@pytest.mark.parametrize('order',[('before','after'),('after','before')])
def test_fallback_agreement_is_order_independent(order):
    from top_p_prefix_adapter import FallbackAgreement
    agreement=FallbackAgreement();spec=dict(case_id='near-p',near_boundary=True)
    actual=(torch.tensor([[1,2,0]],dtype=torch.int32),torch.tensor([2]),torch.tensor([0]))
    close=[]
    for i,arm in enumerate(order):
        close.append(agreement.prepare(arm,spec))
        assert agreement.validate(arm,spec,actual) is (None if i==0 else True)
    for retire in close:retire()
    assert not agreement.pending


def test_fallback_agreement_isolates_cases_and_repeated_runs():
    from top_p_prefix_adapter import FallbackAgreement
    agreement=FallbackAgreement();a=dict(case_id='a');b=dict(case_id='b')
    one=(torch.tensor([1]),);two=(torch.tensor([2]),)
    old_close=agreement.prepare('after',a)
    assert agreement.validate('after',a,one) is None
    old_close()  # An interrupted pair must not leave a baseline for the next run.
    close_a=agreement.prepare('before',a)
    assert agreement.validate('before',a,two) is None
    close_b=agreement.prepare('after',b)
    assert agreement.validate('after',b,one) is None
    agreement.prepare('after',a)
    assert agreement.validate('after',a,two) is True
    agreement.prepare('before',b)
    assert agreement.validate('before',b,two) is False  # Mismatch is actionable in reverse order.
    # A completed pair starts a new generation; old close callbacks cannot retire it.
    agreement.prepare('after',a)
    assert agreement.validate('after',a,one) is None
    close_a();old_close()
    agreement.prepare('before',a)
    assert agreement.validate('before',a,one) is True
    close_b()


def test_fallback_agreement_replaces_repeated_arm_and_snapshots_outputs():
    from top_p_prefix_adapter import FallbackAgreement
    agreement=FallbackAgreement();spec=dict(case_id='same')
    actual=(torch.tensor([1]),)
    agreement.prepare('before',spec)
    assert agreement.validate('before',spec,actual) is None
    actual[0].fill_(2)
    agreement.prepare('before',spec)
    assert agreement.validate('before',spec,actual) is None
    agreement.prepare('after',spec)
    assert agreement.validate('after',spec,(torch.tensor([1]),)) is False


def test_public_validator_reverse_order_and_reuse(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from top_p_prefix_adapter import Adapter,FallbackAgreement
    monkeypatch.setattr(torch,'npu',SimpleNamespace(synchronize=lambda:None),raising=False)
    adapter=Adapter.__new__(Adapter);adapter.fallback_agreement=FallbackAgreement()
    scores=torch.zeros(1,4,dtype=torch.float16)
    fixture=dict(scores=scores,expected=[dict(count=3,tag=0,values=torch.zeros(3))])
    adapter.prepared={arm:SimpleNamespace(inputs=scores,metadata=dict(diagnostic_output=str(tmp_path))) for arm in ('before','after')}
    spec=dict(case_id='near',near_boundary=True)
    def check_pair(after_ids,before_ids):
        results=[]
        for arm,ids in (('after',after_ids),('before',before_ids)):
            adapter.fallback_agreement.prepare(arm,spec)
            actual=(torch.tensor([ids],dtype=torch.int32),torch.tensor([3]),torch.tensor([0]))
            results.append(adapter.validate(arm,actual,fixture,spec))
        return results
    first=check_pair([0,1,2,3],[0,1,2,3])
    assert first[0]['passed'] and first[1]['exact_fallback_agreement']
    reused=check_pair([3,2,1,0],[3,2,1,0])
    assert reused[0]['passed'] and reused[1]['passed']
    mismatch=check_pair([3,2,1,0],[0,1,2,3])
    assert mismatch[0]['passed'] and not mismatch[1]['passed']
    assert mismatch[1]['exact_fallback_agreement'] is False
