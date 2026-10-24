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
import gradient_contract as contract
import gradient_pairs as pairs


def spec(**updates):
    return dict(dict(family='E08', rows=7, width=33, seed=1801, eps=1e-5), **updates)


def test_oracle_checks_input_and_weight_vjp_and_zero_rows():
    case = contract.make_case(spec())
    again = contract.make_case(spec())
    assert all(torch.equal(a, b) for a, b in zip(case['expected'], again['expected']))
    assert contract.diagnostics(case['expected'], case)['passed']
    dx, dw = case['expected']
    torch.testing.assert_close(dx[0], case['inputs']['dy'][0] * case['inputs']['w'] / 1e-5**.5)
    assert not contract.diagnostics((dx, dw + 1), case)['passed']
    assert not contract.diagnostics((dx + 1, dw), case)['passed']


def test_invalid_spec_and_missing_or_malformed_gradient_rejected():
    for update in [dict(family='other'), dict(rows=0), dict(rows=16385), dict(width=4097),
                   dict(rows=16384, width=4096), dict(seed=-1), dict(seed=2**64),
                   dict(seed=True), dict(eps=True), dict(eps=0), dict(eps=float('nan'))]:
        with pytest.raises(ValueError):
            contract.make_case(spec(**update))
    case = contract.make_case(spec())
    dx, dw = case['expected']
    for actual in [(dx,), (dx.double(), dw), (dx, dw[:-1]),
                   (dx, torch.full_like(dw, float('nan'))), ('bad', dw)]:
        assert not contract.diagnostics(actual, case)['passed']
    with pytest.raises(ValueError):
        pairs.configuration('unknown', spec())


def test_strided_private_programs_cover_rows_once_including_empty_programs():
    for rows in (1, 3, 32, 33, 69, 8192):
        for programs in (rows, 32):
            visited = [row for pid in range(programs) for row in pairs.program_rows(pid, rows, programs)]
            assert sorted(visited) == list(range(rows))
    assert list(pairs.program_rows(31, 3, 32)) == []
    for args in [(0, 0, 32), (32, 3, 32), (0, 3, 0), (True, 3, 32)]:
        with pytest.raises(ValueError):
            pairs.program_rows(*args)


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_actual_cpu_interpreter_full_pipeline_resets_every_call():
    script = r'''
import torch
import gradient_contract as c
import gradient_pairs as p
zero, backward, reduce = p.kernels()
for rows,width in ((1,1),(2,33),(35,65),(67,129)):
 s=dict(family='E08',rows=rows,width=width,seed=1802,eps=1e-5)
 fixture=c.make_case(s);x,w,dy=(fixture['inputs'][key] for key in ('x','w','dy'))
 original={key:value.clone() for key,value in fixture['inputs'].items()}
 for arm in ('before','after'):
  cfg=p.configuration(arm,s);dx=torch.empty_like(x);dw=torch.empty_like(w)
  partial=torch.empty(cfg['partials'],width) if cfg['private'] else dw
  for repetition in range(2):
   dx.fill_(float('nan'));dw.fill_(float('nan'));partial.fill_(float('nan'))
   if not cfg['private']:zero[((width+255)//256,)](dw,width,256)
   programs=cfg['partials'] if cfg['private'] else rows
   backward[(programs,)](x,w,dy,dx,partial,rows,width,cfg['eps'],cfg['block'],programs,cfg['private'])
   if cfg['private']:
    assert torch.isfinite(partial).all(),(rows,width,'unwritten private partial')
    if rows<programs:assert torch.count_nonzero(partial[rows:])==0
    reduce[((width+63)//64,)](partial,dw,width,programs,64)
   result=c.diagnostics((dx,dw),fixture)
   assert result['passed'],(rows,width,arm,repetition,result)
   assert all(torch.equal(value,original[key]) for key,value in fixture['inputs'].items())
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT / 'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_lazy_frontend_has_resolvable_globals_without_device_launch():
    for kernel in pairs.kernels():
        assert kernel.cache_key and kernel.fn.__closure__ is None


def test_preparation_is_device_work_free_and_complete_pipeline_is_audited():
    calls = []
    class Kernel:
        def __init__(self, name):
            self.fn = SimpleNamespace(__name__=name)
        def __getitem__(self, grid):
            def launch(*args):
                calls.append((self.fn.__name__, grid, args))
                return SimpleNamespace(name=self.fn.__name__, hash='compiled',
                                       asm={'npubin': self.fn.__name__.encode()})
            return launch
    original_empty = torch.empty
    def empty(*args, **kwargs):
        return original_empty(*args, **dict(kwargs, device='cpu'))
    fake_npu = SimpleNamespace(get_npu_format=lambda value: 2)
    with patch.dict(sys.modules, torch_npu=fake_npu), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True), \
            patch.object(torch, 'empty', empty), \
            patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True), \
            patch.object(pairs, 'kernels', return_value=tuple(Kernel(name) for name in (
                'gradient_zero', 'weighted_vjp', 'gradient_reduce'))):
        for arm in ('before', 'after'):
            calls.clear(); s = spec(); fixture = contract.make_case(s); adapter = pairs.Adapter()
            item = adapter.prepare(arm, s, fixture, None)
            assert calls == [] and item.compilation_state() == []
            assert item.metadata['workspace_bytes'] == (32 * 33 * 4 if arm == 'after' else 0)
            for repetition in range(2):
                result = item.run()
                assert len(result) == 2
            assert [row[0] for row in calls] == item.expected_kernels * 2
            assert all(row['binaries'] for row in item.compilation_state())
            assert adapter.validate(arm, fixture['expected'], fixture, s)['passed']
            adapter.inputs[arm]['w'][0] += 1
            with pytest.raises(ValueError, match='input mutated'):
                adapter.validate(arm, fixture['expected'], fixture, s)


def test_bad_physical_layout_fails_before_any_kernel_launch():
    fake_npu = SimpleNamespace(get_npu_format=lambda value: 29)
    with patch.dict(sys.modules, torch_npu=fake_npu), \
            patch.object(torch.Tensor, 'npu', lambda self: self, create=True):
        with pytest.raises(ValueError, match='dense FP32 ND'):
            pairs.Adapter().prepare('before', spec(), contract.make_case(spec()), None)
