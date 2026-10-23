import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('comparison_gdn', Path(__file__).parents[1] / 'scripts/comparison_gdn.py')
gdn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdn)


@pytest.fixture
def reference(tmp_path):
    root = tmp_path / 'reference/GDN_vllm/GDN_vllm'
    root.mkdir(parents=True)
    # A tiny private oracle exercises the same state/output contract without supplied files.
    (root / gdn.REFERENCE).write_text('import torch\nclass Model(torch.nn.Module):\n def forward(self,q,k,v,g,beta,state,cu,scale,norm):\n  return v.clone(),state.clone()\n')
    inputs = []
    for name, shape, dtype in [('q',[1,8,1,2],'bfloat16'),('k',[1,8,1,2],'bfloat16'),('v',[1,8,2,2],'bfloat16'),('g',[1,8,2],'float32'),('beta',[1,8,2],'bfloat16'),('initial_state',[2,2,2,2],'float32')]:
        inputs.append(dict(name=name,shape=shape,dtype=dtype))
    inputs += [dict(name='cu_seqlens',value=[0,1,8]),dict(name='scale',value=0.5),dict(name='use_qk_l2norm_in_kernel',value=1)]
    (root / gdn.CASES).write_text(json.dumps(dict(inputs=inputs))+'\n')
    private = tmp_path / 'private'
    metadata = gdn.prepare(private, tmp_path / 'reference')
    return private, metadata


def test_prepare_pins_sources(reference, tmp_path):
    private, metadata = reference
    assert metadata['private']['case_count'] == 1
    assert len(metadata['private']['source_sha256']) == 2
    assert not any('source_sha256' in key for key in metadata['public'])
    (private / gdn.REFERENCE).write_text('changed')
    with pytest.raises(ValueError, match='refusing'):
        gdn.prepare(private, tmp_path / 'reference')


def test_runtime_variants_and_repeatability(reference):
    root, _ = reference
    a = gdn.make_case(root, 0, device='cpu')
    b = gdn.make_case(root, 0, device='cpu')
    c = gdn.make_case(root, 0, variant=1, device='cpu')
    assert all(torch.equal(x,y) for x,y in zip(a['args'][:7], b['args'][:7]))
    assert not torch.equal(a['args'][0], c['args'][0])
    assert c['metadata']['initial_state_nonzero']
    assert c['metadata']['boundaries'][0] == 0
    assert c['metadata']['boundaries'][-1] == 8
    assert gdn.validate(a, 'candidate', a['expected'])['passed']


def test_outputs_and_nonmutation(reference):
    root, _ = reference
    case = gdn.make_case(root, 0, device='cpu')
    output, state = case['expected']
    with pytest.raises(AssertionError, match='final_state'):
        gdn.validate(case, 'candidate', (output, state + 1))
    with pytest.raises(AssertionError, match='expected'):
        gdn.validate(case, 'candidate', output)
    with pytest.raises(AssertionError, match='dtype'):
        gdn.validate(case, 'candidate', (output.float(), state))
    case['args'][0].zero_()
    with pytest.raises(AssertionError, match='input mutated: q'):
        gdn.validate(case, 'candidate', (output, state))
