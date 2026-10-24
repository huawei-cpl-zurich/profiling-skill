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
import conv_contract as contract
import conv_pairs as pairs


def spec(**updates):
    return dict(dict(family='E06', batch=2, tokens=2, width=17, kernel=5, seed=1600), **updates)


def test_oracle_prefix_and_final_state_have_independent_known_values():
    s = spec(batch=1, width=1)
    case = contract.make_case(s)
    assert torch.count_nonzero(case['inputs']['initial']) == 4
    expected = case['expected']
    stream = torch.cat((case['inputs']['initial'][0, 0], case['inputs']['x'][0, :, 0]))
    for token in range(2):
        dot = (stream[token:token+5].float() * case['inputs']['w'][0].float()).sum().half()
        torch.testing.assert_close(expected[0][0, token, 0], dot)
    assert torch.equal(expected[1][0, 0], stream[-4:])
    assert contract.diagnostics(expected, case)['passed']
    again = contract.make_case(s)
    assert all(torch.equal(v, again['inputs'][name]) for name, v in case['inputs'].items())


def test_contract_rejects_bad_domain_and_checks_final_state_exactly():
    for update in [dict(family='other'), dict(batch=0), dict(tokens=129),
                   dict(width=4097), dict(kernel=1), dict(kernel=9),
                   dict(seed=-1), dict(seed=True), dict(seed=2**64),
                   dict(batch=32, tokens=128, width=4096)]:
        with pytest.raises(ValueError):
            contract.make_case(spec(**update))
    with pytest.raises(ValueError):
        pairs.configuration('bad', spec())
    fixture = contract.make_case(spec())
    y, state = fixture['expected']
    wrong_state = state.clone();wrong_state[0, 0, 0] += .001
    for value in [(y,), (y.double(), state), (y[:, :, :-1], state),
                  (torch.full_like(y, float('nan')), state), (y+1, state), (y, wrong_state)]:
        assert not contract.diagnostics(value, fixture)['passed']


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_actual_before_after_kernels_in_cpu_interpreter():
    script = r'''
import torch
import conv_contract as c
import conv_pairs as p
init,one,windows=p.kernels()
for b,t,d,k in ((1,1,1,2),(2,1,33,5),(2,2,17,5),(2,7,257,3),(1,4,65,8)):
 s=dict(family='E06',batch=b,tokens=t,width=d,kernel=k,seed=1605)
 fixture=c.make_case(s);x,w,i=(fixture['inputs'][n] for n in ('x','w','initial'))
 saved=[v.clone() for v in (x,w,i)]
 for arm in ('before','after'):
  cfg=p.configuration(arm,s);y=torch.empty_like(x);state=torch.full_like(i,-99)
  for repeat in range(2):
   if arm=='before':
    init[((b*d*(k-1)+255)//256,)](i,state,b*d*(k-1),256)
    for token in range(t):
     one[(b,(d+255)//256)](x,w,state,y,t,d,k,token,256,cfg['taps'],enable_fp_fusion=False)
   else:
    windows[(b,t,(d+255)//256)](x,w,i,y,state,t,d,k,256,cfg['taps'],enable_fp_fusion=False)
   result=c.diagnostics((y,state),fixture)
   assert result['passed'],(s,arm,result)
  assert all(torch.equal(a,z) for a,z in zip((x,w,i),saved))
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT / 'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_lazy_frontend_cache_keys_are_resolvable():
    for kernel in pairs.kernels():
        assert isinstance(kernel.cache_key, str) and kernel.cache_key
        assert kernel.fn.__closure__ is None


def test_prepare_allocates_only_run_includes_reset_and_all_tokens():
    calls = []
    class Kernel:
        def __init__(self, name):
            self.fn = SimpleNamespace(__name__=name)
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append((self.fn.__name__, grid, args))
                return SimpleNamespace(name=self.fn.__name__, hash='key', asm={'npubin': b'code'})
            return run
    real_empty = torch.empty
    def empty(*args, **kwargs):
        return real_empty(*args, **dict(kwargs, device='cpu'))
    with patch.dict(sys.modules, torch_npu=SimpleNamespace(get_npu_format=lambda v: 2)), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True), \
            patch.object(torch, 'empty', empty), \
            patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True), \
            patch.object(pairs, 'kernels', return_value=tuple(Kernel(n) for n in ('init', 'one', 'windows'))):
        for arm in ('before', 'after'):
            calls.clear();s=spec();fixture=contract.make_case(s);adapter=pairs.Adapter()
            prepared=adapter.prepare(arm,s,fixture,None)
            assert calls == [] and prepared.compilation_state() == []
            assert prepared.metadata['workspace_bytes'] == 0
            for repeat in range(2):
                calls.clear();prepared.run()
                expected=['init','one','one'] if arm=='before' else ['windows']
                assert [c[0] for c in calls] == expected
                assert set(expected) == set(prepared.expected_kernels)
                assert len(calls) == prepared.metadata['configuration']['launches']
            assert all(row['binaries'] for row in prepared.compilation_state())
            assert adapter.validate(arm,fixture['expected'],fixture,s)['passed']
            adapter.inputs[arm]['initial'][0,0,0] += 1
            with pytest.raises(ValueError, match='input mutated: initial'):
                adapter.validate(arm,fixture['expected'],fixture,s)


def test_physical_format_and_strides_rejected_before_launch():
    fixture = contract.make_case(spec())
    with patch.dict(sys.modules, torch_npu=SimpleNamespace(get_npu_format=lambda v: 29)), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True):
        with pytest.raises(ValueError, match='ND inputs'):
            pairs.Adapter().prepare('after',spec(),fixture,None)
    fixture['inputs']['initial'] = fixture['inputs']['initial'].transpose(1,2).contiguous().transpose(1,2)
    with patch.dict(sys.modules, torch_npu=SimpleNamespace(get_npu_format=lambda v: 2)), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True):
        with pytest.raises(ValueError, match='ND inputs'):
            pairs.Adapter().prepare('after',spec(),fixture,None)
