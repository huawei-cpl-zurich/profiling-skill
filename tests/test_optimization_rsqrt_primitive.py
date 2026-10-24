import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
import rsqrt_contract as contract
from rsqrt_primitive import Adapter, compiled_identity


def spec(**kw):
    return dict(N=31, seed=91234, distribution='loguniform', exp_low=-7, exp_high=7, **kw)


def test_oracle_uses_represented_fp32_and_fixed_accuracy():
    fixture = contract.make_case(spec())
    x = fixture['x']
    assert fixture['expected'].dtype == torch.float64
    assert torch.equal(fixture['expected'], 1 / torch.sqrt(x.double()))
    assert contract.diagnostics(fixture['expected'].float(), fixture)['passed']
    bad = fixture['expected'].float().clone()
    bad[0] += .01
    assert not contract.diagnostics(bad, fixture)['passed']
    bad[0] = float('nan')
    assert not contract.diagnostics(bad, fixture)['passed']


@pytest.mark.parametrize('distribution', ['powers-of-four', 'powers-endpoints-and-neighbors'])
def test_boundary_fixture(distribution):
    fixture = contract.make_case(dict(N=37, seed=991, distribution=distribution))
    x = fixture['x']
    assert x.shape == (37,)
    assert x.min() == contract.LOW and x.max() == contract.HIGH
    assert contract.diagnostics(fixture['expected'].float(), fixture)['passed']
    if distribution.endswith('neighbors'):
        assert bool((x == torch.nextafter(torch.tensor(1.), torch.tensor(0.))).any())


@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan'), 2**-9, 2**9])
@pytest.mark.parametrize('arm', ['before', 'after'])
def test_actual_prepare_rejects_invalid_before_backend_and_cache(tmp_path, monkeypatch, value, arm):
    fixture = contract.make_case(spec())
    fixture['x'][7] = value
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ('torch_npu', 'triton'):
            pytest.fail('backend imported for invalid host input')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='finite'):
        Adapter().prepare(arm, spec(), fixture, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('x', [torch.ones(31, dtype=torch.float64), torch.ones(62)[::2], torch.ones(30)])
def test_public_shape_dtype_and_layout_gate(tmp_path, x):
    fixture = contract.make_case(spec())
    fixture['x'] = x
    with pytest.raises(ValueError):
        Adapter().prepare('before', spec(), fixture, tmp_path)


@pytest.mark.parametrize('changes', [dict(N=0), dict(seed=-1), dict(exp_low=-9), dict(distribution='invalid')])
def test_invalid_spec_cannot_generate_oracle(changes):
    case = spec(); case.update(changes)
    with pytest.raises(ValueError):
        contract.make_case(case)


def test_compilation_state_uses_binary_list():
    from types import SimpleNamespace
    artifact = SimpleNamespace(hash='abc', asm=dict(elf=b'ELF', ir='text'))
    state = compiled_identity(artifact)
    assert isinstance(state, list) and state[0]['hash'] == 'abc'
    assert len(state[0]['binaries']['elf']) == 64
    assert 'ir' not in state[0]['binaries']


def test_public_cli_rejects_invalid_case_without_cache(tmp_path):
    import json
    case = tmp_path / 'case.json'; case.write_text(json.dumps(dict(N=0, seed=9, distribution='powers-of-four')))
    provenance = tmp_path / 'provenance.json'; provenance.write_text('{}')
    cache = tmp_path / 'cache'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/optimization_pair.py'),
        '--adapter', 'rsqrt_primitive:factory', '--case', str(case), '--output', str(tmp_path/'result'),
        '--cache', str(cache), '--provenance', str(provenance), '--correctness-only'],
        env=dict(__import__('os').environ, PYTHONPATH=str(ROOT/'examples/optimization-library/triton')),
        capture_output=True, text=True)
    assert result.returncode != 0
    assert not cache.exists()


def test_oracle_cache_reuses_payload_and_detects_mutation(tmp_path):
    sys.path.insert(0, str(ROOT/'scripts'))
    from optimization_pair import fixture
    adapter = Adapter()
    case = spec()
    first, identity = fixture(adapter, case, tmp_path, torch)
    second, reused = fixture(adapter, case, tmp_path, torch)
    assert identity['oracle_calls'] == 1 and reused['oracle_calls'] == 0
    assert torch.equal(first['expected'], second['expected'])
    payload = Path(identity['cache'])/'fixture.pt'
    payload.write_bytes(b'changed')
    with pytest.raises(ValueError, match='payload changed'):
        fixture(adapter, case, tmp_path, torch)
