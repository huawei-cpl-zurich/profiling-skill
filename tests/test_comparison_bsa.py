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


@pytest.mark.parametrize('broken_variant', [None, 3, 4])
def test_native_qualification_rejects_empty_or_invalid_selector_candidate(root, monkeypatch, broken_variant):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    import model_campaign_native as native
    import model_campaign_task as task
    (root / 'private-task.json').write_text(json.dumps(dict(task='bsa', check_indices=[0])))
    reference = bsa.load_reference(root)
    class Candidate:
        def __call__(self, *args):
            result = reference(*args)
            ids, counts = args[3:5]
            empty = not bool(counts.any())
            invalid = bool(counts.any()) and not bool(((ids >= 0) & (ids < 3)).any())
            broken = empty if broken_variant == 3 else invalid if broken_variant == 4 else False
            return result + 1 if broken else result
    seen = []
    def probe(factory, provider, validate, **kwargs):
        args, named = next(iter(provider()))
        seen.append(int(kwargs['evidence_dir'].parent.name))
        try:
            validate(factory()(*args, **named), args, named)
        except AssertionError:
            return dict(state='failed')
        return dict(state='passed')
    monkeypatch.setattr(task, 'case', lambda bundle, index, variant: (
        bsa.make_case(root, index, variant, device='cpu'), bsa.validate))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: None), raising=False)
    monkeypatch.setattr(torch._dynamo, 'reset', lambda: None)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'candidate', SimpleNamespace(Model=Candidate))
    monkeypatch.setitem(sys.modules, 'candidate_probe', SimpleNamespace(probe_candidate=probe))
    monkeypatch.setattr(native, 'audited_compiler', lambda: object())
    result = native.check(root, {}, lambda: 100)
    assert result['passed'] is (broken_variant is None)
    assert seen == list(range(5 if broken_variant is None else broken_variant + 1))
    assert result['variants'] == list(range(5 if broken_variant is None else broken_variant))


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


def test_half_scale_is_static_fullgraph_metadata():
    pytest.importorskip('triton')
    path = Path(__file__).parents[1] / 'benchmarks/comparison_bsa_starter.py'
    spec = importlib.util.spec_from_file_location('bsa_starter', path)
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    for scale in (0.1234567, -0.3333333, 0., 1e-8, 65504., 1e10):
        expected = torch.tensor(scale, dtype=torch.float16).item()
        assert starter._half_scale(scale) == expected
    def apply(x):
        return x * starter._half_scale(0.1234567)
    compiled = torch.compile(apply, fullgraph=True, backend='eager')
    x = torch.arange(7, dtype=torch.float32)
    torch.testing.assert_close(compiled(x), apply(x))
