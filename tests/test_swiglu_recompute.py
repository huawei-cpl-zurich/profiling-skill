"""Functional formula, fixture/cache and validation checks; no NPU forwards."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'examples/optimization-library/triton'), str(ROOT / 'scripts')]
from swiglu_contract import make_case, reference, diagnostics, dimensions
from swiglu_recompute import Adapter
from optimization_pair import fixture


def spec(**kwargs):
    return dict(case_id='cpu-check', seed=17, role='guard', parameters=dict(R=2, D=13, **kwargs))


def test_derivatives_and_zero_symmetry():
    f = make_case(spec(distribution='zero-and-symmetric-a'))
    a = f['a'].double().requires_grad_()
    b = f['b'].double().requires_grad_()
    y = (a * torch.sigmoid(a)) * b
    da, db = torch.autograd.grad(y, (a, b), f['dc'].double())
    assert torch.equal(y, f['expected'][0])
    torch.testing.assert_close(da, f['expected'][1], atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(db, f['expected'][2], atol=1e-14, rtol=1e-14)
    zero = f['a'] == 0
    torch.testing.assert_close(f['expected'][1][zero], (f['dc'].double() * f['b'].double() * .5)[zero])


def test_aliased_inputs_remain_formal_partials_and_cache(tmp_path):
    s = spec(share_ab_storage=True)
    f = make_case(s)
    assert f['a'].data_ptr() == f['b'].data_ptr()
    a = f['a'].double().requires_grad_()
    (gradient,) = torch.autograd.grad(((a * torch.sigmoid(a)) * a).sum(), (a,))
    da, db = reference(f['a'], f['a'], torch.ones_like(f['dc']))[1:]
    torch.testing.assert_close(gradient, da + db, atol=1e-14, rtol=1e-14)
    first, info = fixture(Adapter(), s, tmp_path, torch)
    second, cached = fixture(Adapter(), s, tmp_path, torch)
    assert info['oracle_calls'] == 1 and cached['oracle_calls'] == 0
    assert torch.equal(first['a'], second['b'])


def test_default_dtype_isolation_and_bad_output():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        f = make_case(spec())
    finally:
        torch.set_default_dtype(old)
    assert f['a'].dtype == torch.float32
    out = tuple(x.float() for x in f['expected'])
    assert diagnostics(out, f)['passed']
    assert not diagnostics((out[0] + 1, *out[1:]), f)['passed']
    assert not diagnostics(tuple(x.double() for x in out), f)['passed']
    assert not diagnostics((out[0] * float('nan'), *out[1:]), f)['passed']


def test_nested_cli_fixture(tmp_path):
    s = spec()
    case = tmp_path / 'case.json'
    case.write_text(json.dumps(s))
    code = "import json,sys; from swiglu_contract import make_case; f=make_case(json.load(open(sys.argv[1]))); print(list(f['a'].shape))"
    result = subprocess.run([sys.executable, '-c', code, str(case)], cwd=ROOT / 'examples/optimization-library/triton', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[2, 13]'


@pytest.mark.parametrize('key,value', [('R',0),('D',-1),('R',True),('share_ab_storage',1)])
def test_invalid_contract(key, value):
    s = spec()
    s['parameters'][key] = value
    with pytest.raises(ValueError):
        dimensions(s)


def test_adapter_checks_aliasing_and_immutable_inputs():
    from types import SimpleNamespace
    s = spec(share_ab_storage=True)
    f = make_case(s)
    adapter = Adapter()
    a, dc = f['a'].clone(), f['dc'].clone()
    adapter.prepared['after'] = SimpleNamespace(inputs=(a, a, dc))
    outputs = tuple(x.float() for x in f['expected'])
    assert adapter.validate('after', outputs, f, s)['passed']
    adapter.prepared['after'] = SimpleNamespace(inputs=(a, a.clone(), dc))
    assert not adapter.validate('after', outputs, f, s)['passed']
    adapter.prepared['after'] = SimpleNamespace(inputs=(a, a, dc))
    a.add_(1)
    assert not adapter.validate('after', outputs, f, s)['passed']
