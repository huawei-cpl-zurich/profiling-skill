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
import norm_contract as contract
import norm_pairs as pairs


def spec(family='E01', **updates):
    return dict(dict(family=family, rows=7, width=33, seed=1401, eps=1e-6), **updates)


def test_cpu_oracle_deterministic_zero_rows_and_full_input_vjp():
    for family in ('E01', 'E07'):
        case = contract.make_case(spec(family))
        again = contract.make_case(spec(family))
        assert all(torch.equal(a, b) for a, b in zip(case['expected'], again['expected']))
        assert contract.diagnostics(case['expected'], case)['passed']
        assert torch.count_nonzero(case['expected'][0][0]) == 0
        if family == 'E07':
            torch.testing.assert_close(case['expected'][1][0], case['inputs']['dy'][0] * 1000)
        else:
            torch.testing.assert_close(case['expected'][1][0], torch.tensor(1000.))


def test_contract_rejects_invalid_or_resource_unbounded_specs():
    for updates in [dict(family='other'), dict(rows=0), dict(width=2049),
                    dict(rows=65536, width=2048), dict(seed=-1), dict(seed=2**64),
                    dict(seed=True), dict(eps=0), dict(eps=float('nan')),
                    dict(eps=float('inf')), dict(eps=True)]:
        with pytest.raises(ValueError):
            contract.make_case(spec(**updates))


def test_every_required_output_is_checked_and_bad_precision_rejected():
    case = contract.make_case(spec('E07'))
    y, dx = case['expected']
    assert not contract.diagnostics((y, dx + 1), case)['passed']
    for actual in [(y,), (y.double(), dx), (y[:, :-1], dx),
                   (y, torch.full_like(dx, float('nan'))), ('bad', dx)]:
        assert not contract.diagnostics(actual, case)['passed']


def test_row_grid_covers_nonmultiples_exactly_once():
    for rows in (1, 3, 4, 5, 17, 4099):
        for batch in (1, 4):
            visited = [row for pid in range((rows + batch - 1) // batch)
                       for row in pairs.program_rows(pid, rows, batch)]
            assert visited == list(range(rows))
    for args in [(0, 0, 4), (0, 7, 0), (-1, 7, 4), (2, 7, 4), (True, 7, 4)]:
        with pytest.raises(ValueError):
            pairs.program_rows(*args)


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_actual_kernel_math_in_cpu_interpreter():
    script = r'''
import torch
import norm_contract as c
import norm_pairs as p
forward, backward = p.kernels()
for family in ('E01', 'E07'):
 for arm in ('before', 'after'):
  for rows,width in ((1,1),(5,17),(7,33)):
   s=dict(family=family,rows=rows,width=width,seed=1402,eps=1e-6)
   fixture=c.make_case(s);cfg=p.configuration(arm,s)
   x=fixture['inputs']['x'];y=torch.empty_like(x);stat=torch.empty(rows)
   untouched={k:v.clone() for k,v in fixture['inputs'].items()}
   forward[((rows+cfg['batch']-1)//cfg['batch'],)](
    x,y,stat,rows,width,cfg['eps'],cfg['block'],cfg['batch'],cfg['rms'],cfg['saved'])
   if cfg['rms']:
    dx=torch.empty_like(x)
    backward[(rows,)](x,fixture['inputs']['dy'],stat,dx,rows,width,
                     cfg['eps'],cfg['block'],cfg['saved'])
    actual=(y,dx)
   else:actual=(y,stat)
   assert c.diagnostics(actual,fixture)['passed'],(s,arm)
   assert all(torch.equal(v,untouched[k]) for k,v in fixture['inputs'].items())
'''
    env = dict(os.environ, TRITON_INTERPRET='1',
               PYTHONPATH=str(ROOT / 'examples/optimization-library/triton'))
    result = subprocess.run([sys.executable, '-c', script], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec('triton') is None, reason='Triton unavailable')
def test_lazy_jit_frontend_resolves_globals_without_device_execution():
    for kernel in pairs.kernels():
        assert isinstance(kernel.cache_key, str) and kernel.cache_key
        assert kernel.fn.__closure__ is None


def test_preparation_launches_nothing_and_records_complete_pipeline():
    calls = []
    class Kernel:
        def __init__(self, name):
            self.fn = SimpleNamespace(__name__=name)
        def __getitem__(self, grid):
            def run(*args):
                calls.append((self.fn.__name__, grid, args))
                return SimpleNamespace(name=self.fn.__name__, hash='binary-key',
                                       asm={'npubin': self.fn.__name__.encode()})
            return run
    original_empty = torch.empty
    def empty(*args, **kwargs):
        return original_empty(*args, **dict(kwargs, device='cpu'))
    fake_npu = SimpleNamespace(get_npu_format=lambda tensor: 2)
    with patch.dict(sys.modules, torch_npu=fake_npu), \
            patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True), \
            patch.object(torch, 'empty', empty), \
            patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True), \
            patch.object(pairs, 'kernels', return_value=(Kernel('norm_forward'), Kernel('rms_backward'))):
        for family in ('E01', 'E07'):
            for arm in ('before', 'after'):
                calls.clear();s=spec(family);fixture=contract.make_case(s);adapter=pairs.Adapter()
                item=adapter.prepare(arm,s,fixture,None)
                assert calls == [] and item.compilation_state() == []
                assert item.metadata['workspace_bytes'] == (7*4 if family=='E07' and arm=='after' else 0)
                result=item.run()
                assert len(result)==2 and len(calls)==(2 if family=='E07' else 1)
                assert [c[0] for c in calls] == item.expected_kernels
                assert all(row['binaries'] for row in item.compilation_state())
                assert adapter.validate(arm,fixture['expected'],fixture,s)['passed']
                adapter.inputs[arm]['x'][0,0]=1
                with pytest.raises(ValueError,match='input mutated'):
                    adapter.validate(arm,fixture['expected'],fixture,s)
