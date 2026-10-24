import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
import crossentropy_contract as contract
import crossentropy_pairs as pairs


def spec(**updates):
    return dict(dict(family='E10', rows=7, width=33, seed=2010), **updates)


def test_oracle_complete_loss_gradient_extremes_and_reproducibility():
    fixture = contract.make_case(spec())
    again = contract.make_case(spec())
    assert all(torch.equal(a, b) for a, b in zip(fixture['expected'], again['expected']))
    loss, gradient = fixture['expected']
    assert contract.diagnostics((loss, gradient), fixture)['passed']
    assert loss[0] >= 159
    torch.testing.assert_close(gradient.sum(-1), torch.zeros(7), atol=1e-6, rtol=0)
    for result in ((loss + 1, gradient), (loss, gradient + 1), (loss,)):
        assert not contract.diagnostics(result, fixture)['passed']
    single = contract.make_case(spec(rows=1, width=1))
    assert all(torch.count_nonzero(value) == 0 for value in single['expected'])


def test_invalid_contract_and_malformed_outputs_rejected():
    for update in (dict(family='other'), dict(rows=0), dict(width=0), dict(width=8193),
                   dict(rows=4097), dict(rows=4096, width=8192), dict(seed=-1),
                   dict(seed=True), dict(width=True), dict(seed=2**64)):
        with pytest.raises(ValueError):
            contract.make_case(spec(**update))
    fixture = contract.make_case(spec())
    loss, dx = fixture['expected']
    for result in ((loss.double(), dx), (loss, dx[:, :-1]), (loss, dx.T.contiguous().T),
                   (loss, torch.full_like(dx, float('nan'))), ('bad', dx)):
        assert not contract.diagnostics(result, fixture)['passed']
    with pytest.raises(ValueError):
        pairs.configuration('bad', spec())


def test_workspace_capacity_tracks_only_the_removed_materialization():
    before, after = (pairs.configuration(arm, spec()) for arm in ('before', 'after'))
    assert before['workspace_bytes'] == 7 * 33 * 4
    assert after['workspace_bytes'] == 7 * 4
    assert before['block'] == after['block'] == 64


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_actual_interpreter_two_complete_pipelines_and_no_input_mutation():
    script = r'''
import torch
import crossentropy_contract as c
import crossentropy_pairs as p
statistics, gradient = p.kernels()
for rows,width in ((1,1),(3,33),(5,129),(2,513),(1,8192)):
 s=dict(family='E10',rows=rows,width=width,seed=2011)
 fixture=c.make_case(s);x,y=fixture['inputs']['logits'],fixture['inputs']['targets']
 old_x,old_y=x.clone(),y.clone()
 for arm in ('before','after'):
  cfg=p.configuration(arm,s)
  loss=torch.empty(rows);dx=torch.empty_like(x)
  scratch=torch.empty(rows) if cfg['direct'] else torch.empty_like(x)
  for repetition in range(2):
   loss.fill_(float('nan'));dx.fill_(float('nan'));scratch.fill_(float('nan'))
   statistics[(rows,)](x,y,scratch,loss,width,cfg['block'],cfg['direct'])
   assert torch.isfinite(scratch).all(),(rows,width,arm,'unwritten scratch')
   gradient[(rows,)](x,y,scratch,loss,dx,width,cfg['block'],cfg['direct'])
   result=c.diagnostics((loss,dx),fixture)
   assert result['passed'],(rows,width,arm,repetition,result)
   assert torch.equal(x,old_x) and torch.equal(y,old_y)
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT / 'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_lazy_frontend_has_resolvable_globals_without_launching():
    for kernel in pairs.kernels():
        assert kernel.cache_key and kernel.fn.__closure__ is None


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_int64_public_target_boundaries_remain_exact_after_local_narrowing():
    script = r'''
import torch
import crossentropy_contract as c
import crossentropy_pairs as p
statistics,gradient=p.kernels()
rows,width=2,8192
x=torch.linspace(-4,4,width).repeat(rows,1)
y=torch.tensor([0,width-1],dtype=torch.int64)
original_x,original_y=x.clone(),y.clone()
expected_loss=-torch.log_softmax(x.double(),-1).gather(1,y[:,None]).squeeze(1)
expected_dx=torch.softmax(x.double(),-1)
expected_dx[torch.arange(rows),y]-=1
fixture=dict(expected=(expected_loss.float(),expected_dx.float()))
for arm in ('before','after'):
 cfg=p.configuration(arm,dict(family='E10',rows=rows,width=width,seed=0))
 loss=torch.empty(rows);dx=torch.empty_like(x)
 scratch=torch.empty(rows) if cfg['direct'] else torch.empty_like(x)
 statistics[(rows,)](x,y,scratch,loss,width,cfg['block'],cfg['direct'])
 gradient[(rows,)](x,y,scratch,loss,dx,width,cfg['block'],cfg['direct'])
 assert c.diagnostics((loss,dx),fixture)['passed']
 assert y.dtype==torch.int64 and torch.equal(y,original_y) and torch.equal(x,original_x)
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT / 'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_adapter_preparation_is_work_free_and_each_forward_allocates_full_pipeline():
    calls, allocations = [], []
    class Kernel:
        def __init__(self, name):
            self.fn = SimpleNamespace(__name__=name)
        def __getitem__(self, grid):
            def launch(*args):
                calls.append((self.fn.__name__, grid, args))
                return SimpleNamespace(name=self.fn.__name__, hash='compiled',
                                       asm={'npubin': self.fn.__name__.encode()})
            return launch
    original = torch.empty
    def empty(*args, **kwargs):
        allocations.append(args)
        return original(*args, **dict(kwargs, device='cpu'))
    with patch.dict(sys.modules, torch_npu=SimpleNamespace(get_npu_format=lambda value: 2)), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True), \
            patch.object(torch, 'empty', empty), \
            patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True), \
            patch.object(pairs, 'kernels', return_value=tuple(Kernel(name) for name in (
                'ce_statistics', 'ce_gradient'))):
        for arm in ('before', 'after'):
            fixture = contract.make_case(spec()); calls.clear(); allocations.clear()
            adapter = pairs.Adapter(); item = adapter.prepare(arm, spec(), fixture, None)
            assert calls == [] and allocations == [] and item.compilation_state() == []
            for _ in range(2):
                loss, dx = item.run()
                assert loss.shape == (7,) and dx.shape == (7, 33)
            assert len(allocations) == 6
            assert [row[0] for row in calls] == item.expected_kernels * 2
            assert all(row['binaries'] for row in item.compilation_state())
            assert adapter.validate(arm, fixture['expected'], fixture, spec())['passed']
            adapter.inputs[arm]['targets'][0] = 1
            with pytest.raises(ValueError, match='input mutated'):
                adapter.validate(arm, fixture['expected'], fixture, spec())


def test_bad_physical_layout_rejected_before_launch():
    with patch.dict(sys.modules, torch_npu=SimpleNamespace(get_npu_format=lambda value: 29)), \
            patch.object(torch.Tensor, 'npu', lambda self: self, create=True):
        with pytest.raises(ValueError, match='dense ND'):
            pairs.Adapter().prepare('before', spec(), contract.make_case(spec()), None)
