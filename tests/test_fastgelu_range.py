"""Functional FP32 fixtures, range diagnostics and public nested-case inputs."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/triton'), str(ROOT/'scripts')]
from fastgelu_contract import A, HALF, parameters, make_case, reference, diagnostics
from fastgelu_range import Adapter
from optimization_pair import fixture


def spec(**kwargs):
    return dict(case_id='cpu-fastgelu', role='guard', seed=19, parameters=dict(elements=kwargs.pop('elements', 17), **kwargs))


def test_reference_against_scalar_math_and_coefficient():
    f = make_case(spec(elements=16, distribution='finite-range-boundaries'))
    assert A == 2*HALF
    expected = torch.tensor([float(x)/(1+math.exp(-A*float(x))) for x in f['x']], dtype=torch.float64)
    torch.testing.assert_close(f['expected'], expected, atol=1e-14, rtol=1e-14)
    x = f['x'].double()
    stable = (x/(1+torch.exp(-A*x.abs()))) * torch.exp(HALF*(x-x.abs()))
    torch.testing.assert_close(stable, expected, atol=1e-14, rtol=1e-14)


def test_tiny_tail_diagnostic_exposes_hidden_zeros():
    f = make_case(spec(elements=16, distribution='finite-range-boundaries'))
    actual = f['expected'].float()
    actual[actual.abs() < 1e-30] = -0.
    result = diagnostics(actual, f)
    assert result['passed']
    assert result['tiny_tail']['elements'] > 0
    assert result['tiny_tail']['actual_nonzero'] == 0
    assert result['tiny_tail']['max_absolute_error'] > 0
    assert result['expected_intermediate_ranges']['before_exp_FP32_overflow_arguments'] > 0
    assert result['expected_intermediate_ranges']['after_first_exp_argument_max'] <= 0


def test_signed_zero_and_wrong_outputs():
    f = make_case(spec(elements=8, distribution='signedzero'))
    result = diagnostics(f['expected'].float(), f)
    assert result['passed'] and result['zeros'] == dict(negative=1, positive=1)
    assert not diagnostics(f['expected'].float()+1, f)['passed']
    assert not diagnostics(f['expected'], f)['passed']
    assert not diagnostics(f['expected'].float()*float('nan'), f)['passed']


def test_ambient_dtype_and_fixture_cache(tmp_path):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first, ident = fixture(Adapter(), spec(), tmp_path, torch)
        second, cached = fixture(Adapter(), spec(), tmp_path, torch)
    finally:
        torch.set_default_dtype(old)
    assert first['x'].dtype == torch.float32 and first['x'].device.type == 'cpu'
    assert ident['oracle_calls'] == 1 and cached['oracle_calls'] == 0
    assert torch.equal(first['x'], second['x'])


def test_nested_json_cli(tmp_path):
    p = tmp_path/'case.json'
    p.write_text(json.dumps(spec()))
    code = "import json,sys;from fastgelu_contract import make_case;f=make_case(json.load(open(sys.argv[1])));print(f['x'].shape[0])"
    r = subprocess.run([sys.executable,'-c',code,str(p)],cwd=ROOT/'examples/optimization-library/triton',capture_output=True,text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '17'


@pytest.mark.parametrize('patch', [dict(elements=0),dict(elements=True),dict(low=-81),dict(high=float('inf')),dict(distribution='unknown')])
def test_reject_contract(patch):
    s = spec()
    s['parameters'].update(patch)
    with pytest.raises(ValueError):
        parameters(s)
