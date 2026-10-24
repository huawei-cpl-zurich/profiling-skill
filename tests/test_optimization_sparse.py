import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples/optimization-library/triton'))
import sparse_contract as contract
import sparse_pairs as pairs


def spec(**updates):
    return dict(dict(family='E05', hq=4, hkv=2, q=17, k=65, d=16,
                     slots=5, selected_count=3, seed=1501), **updates)


def test_oracle_gqa_uniform_scores_and_empty_partial_tiles():
    case = contract.make_case(spec())
    inputs = case['inputs']
    inputs['q'].zero_()
    inputs['v'][0].fill_(2)
    inputs['v'][1].fill_(5)
    inputs['selected'].fill_(-1)
    # All keys selected out of order with interior holes; final block has one key.
    inputs['selected'][1:,:,0] = 2
    inputs['selected'][1:,:,2] = 0
    inputs['selected'][1:,:,4] = 1
    actual = contract.reference(inputs)
    assert torch.count_nonzero(actual[0]) == 0
    torch.testing.assert_close(actual[1],torch.full_like(actual[1],2))
    torch.testing.assert_close(actual[2:],torch.full_like(actual[2:],5))
    inputs['selected'][1,0] = torch.tensor([-1,2,-1,-1,-1])
    inputs['v'][0,-1] = torch.arange(16).half()
    torch.testing.assert_close(contract.reference(inputs)[1,:16],
                               torch.arange(16).float().expand(16,16))


def test_deterministic_fixture_and_selection_permutation_equivalence():
    first, second = contract.make_case(spec()), contract.make_case(spec())
    assert all(torch.equal(first['inputs'][k],second['inputs'][k]) for k in first['inputs'])
    contract.validate_inputs(spec(),first['inputs'])
    first['inputs']['selected'] = first['inputs']['selected'].flip(-1).contiguous()
    torch.testing.assert_close(contract.reference(first['inputs']),second['expected'])
    assert contract.diagnostics(first['expected'], first)['passed']


def test_invalid_envelope_and_invalid_index_lists_reject_before_device_use():
    for updates in [dict(hq=3,hkv=2),dict(q=0),dict(k=8193),dict(d=48),
                    dict(slots=0),dict(selected_count=4),dict(seed=-1),dict(seed=2**64)]:
        with pytest.raises(ValueError):contract.make_case(spec(**updates))
    for values in ([0,0,-1,-1,-1],[3,-1,-1,-1,-1],[-2,-1,-1,-1,-1]):
        case=contract.make_case(spec());case['inputs']['selected'][0,0]=torch.tensor(values)
        with pytest.raises(ValueError,match='selection'):contract.validate_inputs(spec(),case['inputs'])
    case=contract.make_case(spec());case['inputs']['q'][0,0,0]=float('nan')
    with pytest.raises(ValueError):pairs.Adapter().prepare('before',spec(),case,Path('unused'))


def test_diagnostics_reject_dtype_shape_nonfinite_errors_and_input_mutation():
    case=contract.make_case(spec())
    for actual in (case['expected'].half(),case['expected'][:,:-1],
                   case['expected']+1,torch.full_like(case['expected'],float('nan'))):
        assert not contract.diagnostics(actual,case)['passed']
    inputs={key:value.clone() for key,value in case['inputs'].items()}
    inputs['v'][0,0,0]+=1
    assert not contract.diagnostics(case['expected'],case,inputs)['passed']


@pytest.mark.skipif(importlib.util.find_spec('triton') is None,reason='Triton unavailable')
def test_actual_tiled_kernel_math_in_cpu_interpreter():
    script=r'''
import torch,triton
import sparse_contract as c
import sparse_pairs as p
for nq,nk,count in ((1,1,0),(1,1,1),(17,65,3)):
 s=dict(family='E05',hq=4,hkv=2,q=nq,k=nk,d=16,slots=5,selected_count=count,seed=1502)
 f=c.make_case(s)
 if count==3:f['inputs']['selected'][1,0]=torch.tensor([2,-1,0,-1,1],dtype=torch.int32)
 f['expected']=c.reference(f['inputs'])
 original={k:v.clone() for k,v in f['inputs'].items()}
 for sparse in (False,True):
  out=torch.empty_like(f['expected'])
  p.kernels()[((nq+15)//16,4)](*[f['inputs'][k] for k in ('q','k','v','selected')],
   out,nq,nk,16,4,2,5,8,sparse,16,32)
  assert c.diagnostics(out,f)['passed'],(nq,nk,sparse,c.diagnostics(out,f))
  assert all(torch.equal(v,original[k]) for k,v in f['inputs'].items())
'''
    env=dict(os.environ,TRITON_INTERPRET='1',PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
    result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None,reason='Triton unavailable')
def test_lazy_frontend_globals_resolve_without_device_execution():
    kernel=pairs.kernels()
    assert isinstance(kernel.cache_key,str) and kernel.cache_key
    assert kernel.fn.__closure__ is None
    assert pairs._online_update.fn.__closure__ is None


def test_preparation_has_no_launches_then_one_complete_operation_with_identity():
    calls=[]
    class Kernel:
        fn=SimpleNamespace(__name__='candidate_sparse')
        def __getitem__(self,grid):
            def run(*args):
                calls.append((grid,args));return SimpleNamespace(name='actual',hash='binary',asm={'o':b'code'})
            return run
    case=contract.make_case(spec())
    def transfer(value,*args,**kwargs):return value
    with patch.dict(sys.modules,torch_npu=SimpleNamespace(get_npu_format=lambda x:2)),\
            patch.object(torch.Tensor,'to',transfer),\
            patch.object(torch,'empty_like',side_effect=lambda x,**kw:torch.empty(x.shape,dtype=x.dtype)),\
            patch.object(pairs,'kernels',return_value=Kernel()):
        adapter=pairs.Adapter()
        for arm in ('before','after'):
            prepared=adapter.prepare(arm,spec(),case,Path('unused'))
            assert not calls and prepared.compilation_state()==[]
            prepared.run()
            assert len(calls)==1 and calls[0][1][-3] is (arm=='after')
            assert prepared.compilation_state()[0]['hash']=='binary'
            assert prepared.metadata['query_tile']==16 and prepared.metadata['key_tile']==32
            calls.clear()
