"""Functional row-mean API, cache, reproducibility and error-path tests."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'examples/optimization-library/triton'), str(ROOT / 'scripts')]
from batch_mean_contract import parameters, make_case, diagnostics
from batch_mean import Adapter
from optimization_pair import fixture


def spec(**kwargs):
    return dict(case_id='cpu-small', role='guard', seed=17, parameters=dict(R=3, N=13, **kwargs))


def test_last_dim_and_keepdim():
    for dim in (-1, 1):
        s = spec(dim=dim, keepdim=True)
        f = make_case(s)
        out = (f['xs'][0].mean(dim=dim, keepdim=True),)
        assert out[0].shape == (3, 1) and diagnostics(out, f, s, 'after')['passed']
        assert torch.equal(f['expected'][0].reshape(-1), f['xs'][0].double().sum(1) / 13)


def test_cross_batch_guard_detects_bit_changes():
    s = spec(distribution='bounded-cancellation', context_rows=[1, 3, 129], context_positions=[0, 1, 128])
    s['parameters']['N'] = 1024
    f = make_case(s)
    outputs = tuple(x.float() for x in f['expected'])
    assert diagnostics(outputs, f, s, 'after')['designated_row_bitwise_equal']
    bad = tuple(x.clone() for x in outputs)
    bad[1][1] = torch.nextafter(bad[1][1], torch.tensor(float('inf')))
    assert diagnostics(bad, f, s, 'before')['passed']
    assert not diagnostics(bad, f, s, 'after')['passed']
    assert torch.equal(f['xs'][0][0], f['xs'][2][128])


def test_cache_nested_spec_ambient_dtype_and_bad_output(tmp_path):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first, info = fixture(Adapter(), spec(), tmp_path, torch)
        second, cached = fixture(Adapter(), spec(), tmp_path, torch)
    finally:
        torch.set_default_dtype(old)
    assert info['oracle_calls'] == 1 and cached['oracle_calls'] == 0
    assert first['xs'][0].dtype == torch.float32 and torch.equal(first['xs'][0], second['xs'][0])
    outputs = (first['expected'][0].float(),)
    assert diagnostics(outputs, first, spec(), 'after')['passed']
    assert not diagnostics((outputs[0] + 1,), first, spec(), 'after')['passed']
    assert not diagnostics((outputs[0].double(),), first, spec(), 'after')['passed']
    assert not diagnostics((outputs[0] * float('nan'),), first, spec(), 'after')['passed']


def test_validator_rejects_mutation_and_alias():
    s = spec(); f = make_case(s); a = Adapter(); x = f['xs'][0].clone()
    a.prepared['after'] = SimpleNamespace(inputs=(x,))
    out = (f['expected'][0].float(),)
    assert a.validate('after', out, f, s)['passed']
    x.add_(1)
    assert not a.validate('after', out, f, s)['passed']


def test_public_cli_rejects_invalid_axis(tmp_path):
    s = spec(dim=0); case = tmp_path / 'case.json'; case.write_text(json.dumps(s))
    provenance = tmp_path/'provenance.json'; provenance.write_text('{}')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/optimization_pair.py'),
        '--adapter', 'batch_mean:Adapter', '--case', str(case), '--output', str(tmp_path/'out'),
        '--cache', str(tmp_path/'cache'), '--provenance', str(provenance), '--correctness-only'],
        env={**__import__('os').environ, 'PYTHONPATH':str(ROOT/'examples/optimization-library/triton')},
        capture_output=True, text=True)
    report = json.loads((tmp_path/'out/result.json').read_text())
    assert proc.returncode == 1 and report['status'] != 'complete'
    assert report['error']['type'] == 'ValueError' and 'dim=-1 or1' in report['error']['message']
    assert report['counts']['profiled_calls'] == report['counts']['correctness_calls'] == 0


@pytest.mark.parametrize('key,value', [('R',0), ('N',-1), ('R',True), ('dim',True), ('keepdim',1)])
def test_invalid_parameters(key, value):
    s = spec(); s['parameters'][key] = value
    with pytest.raises(ValueError):
        parameters(s)
