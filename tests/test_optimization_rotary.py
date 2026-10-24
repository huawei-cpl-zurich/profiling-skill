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
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
import rotary_contract as contract
import rotary_pairs as pairs


def spec(**updates):
    return dict(dict(family='E09', batch=2, tokens=3, q_heads=3, k_heads=1,
                     width=32, seed=1900, scale=2.0), **updates)


def test_cpu_oracle_both_halves_and_both_outputs_follow_coeff_rounding():
    fixture = contract.make_case(spec())
    x = fixture['inputs']
    for name, result in zip(('q', 'k'), fixture['expected']):
        angle = x['positions'][0, 0].float()/2*x['freq'][3]
        cos, sin = angle.cos().half().float(), angle.sin().half().float()
        lo, hi = x[name][0, 0, 0, 3].float(), x[name][0, 0, 0, 19].float()
        assert result[0, 0, 0, 3] == (lo*cos-hi*sin).half()
        assert result[0, 0, 0, 19] == (hi*cos+lo*sin).half()
    assert contract.diagnostics(fixture['expected'], fixture)['passed']
    repeated = contract.make_case(spec())
    assert all(torch.equal(value, repeated['inputs'][name]) for name,value in x.items())


def test_bounded_contract_and_complete_output_validation():
    for updates in (dict(family='bad'), dict(batch=0), dict(tokens=8193), dict(q_heads=65),
                    dict(k_heads=0), dict(width=48), dict(seed=True), dict(seed=-1),
                    dict(scale=float('nan')), dict(scale=0), dict(scale=1e-300), dict(scale=129), dict(scale=True),
                    dict(batch=8, tokens=8192, q_heads=64, k_heads=16, width=256)):
        with pytest.raises(ValueError): contract.make_case(spec(**updates))
    with pytest.raises(ValueError): pairs.configuration('bad', spec())
    fixture = contract.make_case(spec());q,k = fixture['expected']
    for value in ((q,), (q.double(),k), (q,k[..., :-1]), (q,k+1), (q,torch.full_like(k,float('nan')))):
        assert not contract.diagnostics(value,fixture)['passed']


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_actual_cpu_interpreter_both_kernels_asymmetric_heads_and_token_tails():
    script = r'''
import torch
import rotary_contract as c
import rotary_pairs as p
parallel,shared=p.kernels()
for b,t,hq,hk,d,scale in ((1,1,1,1,32,1.),(2,3,3,1,64,2.),(1,9,2,3,128,.5),(1,2,1,2,256,4.)):
 s=dict(family='E09',batch=b,tokens=t,q_heads=hq,k_heads=hk,width=d,seed=1909,scale=scale)
 fixture=c.make_case(s);x=fixture['inputs'];saved={n:v.clone() for n,v in x.items()}
 outputs=[]
 for arm,kernel in (('before',parallel),('after',shared)):
  cfg=p.configuration(arm,s);oq,ok=torch.full_like(x['q'],99),torch.full_like(x['k'],99)
  grid=(b,(t+7)//8,hq+hk) if arm=='before' else (b,(t+7)//8)
  kernel[grid](x['q'],x['k'],x['positions'],x['freq'],oq,ok,t,hq,hk,d,scale,8,d//2,enable_fp_fusion=False)
  result=c.diagnostics((oq,ok),fixture)
  assert result['passed'],(s,arm,result)
  assert all(torch.equal(v,saved[n]) for n,v in x.items())
  outputs.append((oq,ok))
 assert all(torch.equal(a,z) for a,z in zip(*outputs))
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT/'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable,'-c',script], env=env,
                            capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_frontend_cache_keys_resolve_without_closure_captures():
    for kernel in pairs.kernels():
        assert isinstance(kernel.cache_key,str) and kernel.cache_key
        assert kernel.fn.__closure__ is None


def test_adapter_prepare_does_not_launch_and_both_results_measured_together():
    calls=[]
    class Kernel:
        def __init__(self,name): self.fn=SimpleNamespace(__name__=name)
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                calls.append((self.fn.__name__,grid,args))
                return SimpleNamespace(name=self.fn.__name__,hash='key',asm={'npubin':b'code'})
            return run
    with patch.dict(sys.modules,torch_npu=SimpleNamespace(get_npu_format=lambda x:2)), \
            patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True), \
            patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None),create=True), \
            patch.object(pairs,'kernels',return_value=(Kernel('parallel'),Kernel('shared'))):
        for arm in ('before','after'):
            calls.clear();s=spec();fixture=contract.make_case(s);adapter=pairs.Adapter()
            prepared=adapter.prepare(arm,s,fixture,None)
            assert not calls and not prepared.compilation_state()
            assert prepared.metadata['workspace_bytes']==0
            out=prepared.run();assert len(out)==2 and len(calls)==1
            assert calls[0][0]==prepared.expected_kernels[0]
            assert calls[0][1]==((2,1,4) if arm=='before' else (2,1))
            assert len(prepared.compilation_state())==1
            assert adapter.validate(arm,fixture['expected'],fixture,s)['passed']
            adapter.inputs[arm]['positions'][0,0]+=1
            with pytest.raises(ValueError,match='input mutated: positions'):
                adapter.validate(arm,fixture['expected'],fixture,s)


def test_adapter_rejects_wrong_format_dtype_and_physical_strides():
    with patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True):
        fixture=contract.make_case(spec())
        with patch.dict(sys.modules,torch_npu=SimpleNamespace(get_npu_format=lambda x:29,
                npu_format_cast=lambda value,fmt:value)):
            with pytest.raises(ValueError,match='trusted ND preparation'): pairs.Adapter().prepare('after',spec(),fixture,None)
        with patch.dict(sys.modules,torch_npu=SimpleNamespace(get_npu_format=lambda x:2)):
            fixture['inputs']['q']=fixture['inputs']['q'].float()
            with pytest.raises(ValueError,match='ND inputs'): pairs.Adapter().prepare('after',spec(),fixture,None)
            fixture=contract.make_case(spec())
            fixture['inputs']['q']=fixture['inputs']['q'].transpose(2,3).contiguous().transpose(2,3)
            with pytest.raises(ValueError,match='ND inputs'): pairs.Adapter().prepare('after',spec(),fixture,None)


def test_trusted_nd_preparation_preserves_values_and_rejects_silent_conversion_changes():
    converted=set()
    def normalize(value,fmt):
        assert fmt==2
        new=value.clone();converted.add(id(new));return new
    backend=SimpleNamespace(get_npu_format=lambda value:2 if id(value) in converted else 0,
                            npu_format_cast=normalize)
    with patch.object(torch.Tensor,'npu',lambda self:self.clone(),create=True), \
            patch.dict(sys.modules,torch_npu=backend):
        fixture=contract.make_case(spec());adapter=pairs.Adapter()
        prepared=adapter.prepare('after',spec(),fixture,None)
        assert set(prepared.metadata['trusted_nd_preparation'])==set(fixture['inputs'])
        assert all(torch.equal(value,fixture['inputs'][name]) for name,value in adapter.inputs['after'].items())
        def corrupt(value,fmt):
            new=normalize(value,fmt);new.reshape(-1)[0]+=1;return new
        backend.npu_format_cast=corrupt
        with pytest.raises(ValueError,match='trusted ND preparation changed input ABI or values'):
            pairs.Adapter().prepare('after',spec(),fixture,None)
