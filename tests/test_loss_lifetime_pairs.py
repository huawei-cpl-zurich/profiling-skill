"""Functional contract and real interpreter coverage for the complete E11 pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'examples/optimization-library/triton'
sys.path.insert(0,str(SOURCE))
import loss_lifetime_contract as contract
import loss_lifetime_pairs as pair


def spec(t=7,d=19,v=37,c=4):
    return dict(family='E11',tokens=t,hidden=d,vocab=v,chunk=c,seed=2111)


def test_workspace_is_bound_by_chunk_with_common_loss_storage():
    s=spec(t=129,d=32,v=257,c=16)
    before,after=(pair.configuration(a,s) for a in ('before','after'))
    assert before['chunks']==1 and after['chunks']==9
    for c in (before,after):
        outputs,workspace=pair.allocate(c,'cpu')
        assert sum(x.numel()*x.element_size() for x in workspace)==c['logical_workspace_bytes']
        assert tuple(x.shape for x in outputs)==((1,),(129,32),(257,32))
    assert after['logical_workspace_bytes']<before['logical_workspace_bytes']/7


@pytest.mark.parametrize('change',[dict(chunk=0),dict(chunk=8),dict(vocab=16),dict(tokens=True)])
def test_invalid_domain_is_rejected_before_allocation(change):
    s=spec();s.update(change)
    with pytest.raises(ValueError):pair.configuration('after',s)


def test_cpu_oracle_and_diagnostics_cover_complete_contract():
    fixture=contract.make_case(spec())
    assert contract.diagnostics(fixture['expected'],fixture)['passed']
    assert not contract.diagnostics(fixture['expected'][:2],fixture)['passed']
    wrong=list(fixture['expected']);wrong[2]=wrong[2]+1
    assert not contract.diagnostics(wrong,fixture)['passed']
    fixture['inputs']['target'][0]=37
    with pytest.raises(ValueError,match='target'):contract.validate_inputs(fixture['inputs'],spec())


def test_actual_interpreter_pipeline_resets_outputs_and_workspace():
    code = '''
import torch
import loss_lifetime_contract as c
import loss_lifetime_pairs as p
torch.set_num_threads(2)
count=0
for t,d,v,chunk in [(1,16,17,1),(7,19,37,4),(17,32,65,8)]:
 spec=dict(family='E11',tokens=t,hidden=d,vocab=v,chunk=chunk,seed=2111+t)
 fixture=c.make_case(spec)
 original={k:v.clone() for k,v in fixture['inputs'].items()}
 for arm in ('before','after'):
  config=p.configuration(arm,spec)
  outputs,workspace=p.allocate(config,'cpu')
  for repeat in range(2):
   for tensor in (*outputs,*workspace):tensor.fill_(float('nan'))
   actual=p.execute(fixture['inputs'],outputs,workspace,config,{})
   result=c.diagnostics(actual,fixture)
   assert result['passed'], (arm,spec,result)
   assert all(torch.equal(v,original[k]) for k,v in fixture['inputs'].items())
   count+=1
print(count)
'''
    result=subprocess.run([sys.executable,'-c',code],env={**os.environ,
        'TRITON_INTERPRET':'1','PYTHONPATH':str(SOURCE)},capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=='12'
