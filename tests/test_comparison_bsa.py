"""Functional tests of the private Rain adapter; no comparison sources shipped."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('comparison_bsa', Path(__file__).parents[1] / 'scripts/comparison_bsa.py')
bsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsa)

ORACLE = '''import torch
class Model(torch.nn.Module):
 def forward(self,q,k,v,ids,counts,blocks,qlens,klens,scale,*rest):
  out=torch.zeros_like(q,dtype=torch.float32); qo=ko=bo=0
  sx,sy=blocks
  for nq,nk in zip(qlens,klens):
   for r in range(nq):
    for h in range(q.shape[1]):
     b=bo+r//sx
     selected=ids[b,h,:int(counts[b,h])].tolist()
     tokens=[j for block in selected if 0<=block<(nk+sy-1)//sy
             for j in range(block*sy,min((block+1)*sy,nk))]
     if tokens:
      kh=h//(q.shape[1]//k.shape[1])
      scores=k[ko+torch.tensor(tokens),kh].float()@q[qo+r,h].float()*scale
      out[qo+r,h]=scores.softmax(0)@v[ko+torch.tensor(tokens),kh].float()
   qo+=nq;ko+=nk;bo+=(nq+sx-1)//sx
  return out
'''


@pytest.fixture
def root(tmp_path):
    inputs = []
    for name, shape in [('query',[5,4,3]),('key',[7,2,3]),('value',[7,2,3]),
                        ('select_idx',[3,4,3]),('select_num_idx',[3,4])]:
        item = dict(name=name,type='tensor',shape=shape,
                    dtype='int64' if name.startswith('select') else 'float16')
        if name == 'select_num_idx':
            item['value'] = [[2]*4,[2]*4,[1]*4]
        inputs.append(item)
    attrs = dict(blockshape=[2,2],actual_seq_qlen=[3,2],actual_seq_kvlen=[5,2],
                 scale=0.5,q_input_layout='TND',kv_input_layout='TND',inner_precise=0)
    inputs += [dict(name=k,type='attr',value=v) for k,v in attrs.items()]
    (tmp_path / bsa.REFERENCE).write_text(ORACLE)
    (tmp_path / bsa.CASES).write_text(json.dumps(dict(id=0,inputs=inputs))+'\n')
    return tmp_path


@pytest.mark.parametrize('variant',range(5))
def test_case_validation_and_runtime_variants(root,variant):
    case = bsa.make_case(root,0,variant,device='cpu')
    assert bsa.validate(case,'reference',case['expected'])['passed']
    if variant >= 3:
        assert torch.count_nonzero(case['expected']) == 0
    if variant:
        original = bsa.make_case(root,0,device='cpu')
        assert not torch.equal(original['args'][0],case['args'][0])


def test_rejects_errors_and_mutation(root):
    case = bsa.make_case(root,0,device='cpu')
    for bad in (case['expected']+1,case['expected'].half(),case['expected'][:1],
                torch.full_like(case['expected'],float('nan'))):
        with pytest.raises(AssertionError):
            bsa.validate(case,'candidate',bad)
    case['args'][3].fill_(0)
    with pytest.raises(AssertionError,match='mutation'):
        bsa.validate(case,'candidate',case['expected'])


def test_prepare_preserves_evidence_and_isolates_public_material(root,tmp_path):
    dest=tmp_path/'private'
    public=bsa.prepare(dest,root)
    case=bsa.make_case(dest,0,device='cpu')
    assert public['case_count']==1
    assert bsa.validate(case,'copied-reference',case['expected'])['passed']
    (root/bsa.REFERENCE).write_text('changed')
    with pytest.raises(ValueError,match='replace'):
        bsa.prepare(dest,root)


def test_repeatability_and_attribute_mutation(root):
    a=bsa.make_case(root,0,1,device='cpu'); b=bsa.make_case(root,0,1,device='cpu')
    assert torch.equal(a['expected'],b['expected'])
    a['args'][5][0]=99
    with pytest.raises(AssertionError,match='mutation'):
        bsa.validate(a,'candidate',a['expected'])
