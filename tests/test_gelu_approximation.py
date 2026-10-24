"""Functional numerical budget, mode permission and nested fixture contracts."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from gelu_contract import parameters, make_case, reference, approximation, grid_check, diagnostics
from gelu_approximation import Adapter
from optimization_pair import fixture


def spec(**kwargs):
    return dict(case_id='cpu-gelu', role='guard', seed=29,
                parameters=dict(elements=kwargs.pop('elements', 17), **kwargs))


def test_reference_scalar_and_budget_grid():
    result = grid_check()
    assert result['passed'] and result['maximum_absolute_error'] < 6e-4
    x = torch.tensor([-8.,-3.,-.01,0.,.01,3.,8.], dtype=torch.float32)
    expected = torch.tensor([.5*float(v)*(1+math.erf(float(v)/math.sqrt(2))) for v in x], dtype=torch.float64)
    torch.testing.assert_close(reference(x), expected, atol=1e-14, rtol=1e-14)


def test_exact_permission_uses_tight_gate_and_rejects_bad_results():
    f = make_case(spec(elements=16, distribution='endpoints-nearzero'))
    approx = approximation(f['x']).float()
    assert diagnostics(approx, f, 'after', True)['passed']
    assert diagnostics(f['expected'].float(), f, 'after', False)['dispatch'] == 'erf'
    assert not diagnostics(f['expected'].float()+.001, f, 'after', False)['passed']
    assert not diagnostics(f['expected'].float()+.001, f, 'after', True)['passed']
    assert not diagnostics(f['expected'], f, 'after', True)['passed']
    assert not diagnostics(approx*float('nan'), f, 'after', True)['passed']


def test_ambient_dtype_cache_and_endpoint_shape(tmp_path):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first, ident = fixture(Adapter(), spec(), tmp_path, torch)
        cached, receipt = fixture(Adapter(), spec(), tmp_path, torch)
    finally:
        torch.set_default_dtype(old)
    assert first['x'].dtype == torch.float32 and first['x'].device.type == 'cpu'
    assert ident['oracle_calls'] == 1 and receipt['oracle_calls'] == 0
    assert torch.equal(first['x'], cached['x'])
    endpoint = make_case(spec(elements=16, distribution='endpoints-nearzero'))['x']
    assert endpoint.shape == (16,) and float(endpoint.min()) == -8 and float(endpoint.max()) == 8


def test_nested_json_cli(tmp_path):
    p = tmp_path/'case.json'
    p.write_text(json.dumps(spec(allow_approx=False)))
    code = "import json,sys;from gelu_contract import make_case,parameters;s=json.load(open(sys.argv[1]));print(make_case(s)['x'].numel(),parameters(s)[2])"
    result = subprocess.run([sys.executable,'-c',code,str(p)],cwd=ROOT/'examples/optimization-library/triton',capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '17 False'


@pytest.mark.parametrize('patch', [dict(elements=0),dict(elements=True),dict(low=-9),dict(high=float('inf')),dict(allow_approx=1),dict(distribution='unknown')])
def test_reject_contract(patch):
    s = spec()
    s['parameters'].update(patch)
    with pytest.raises(ValueError):
        parameters(s)
