"""Functional CPU checks for the bounded integer operation and guards."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
from integer_cube_contract import integer_eligible, make_case, validate_actual
from integer_cube import identity


def spec(**parameters):
    return dict(seed=10101, parameters=dict(M=3, N=5, K=7, code_max=16, output='fp32', **parameters))


def test_oracle_and_representations():
    s = spec()
    f = make_case(s)
    x = f['inputs']
    independent = x['sb'].T * (x['sa'] * (x['a'].double() @ x['b'].double()).float())
    assert validate_actual(independent, f, x)['passed']
    bad = independent.clone()
    bad.view(torch.int32)[0, 0] ^= 1
    assert not validate_actual(bad, f, x)['passed']


def test_seed_and_mutation():
    f = make_case(spec())
    again = make_case(spec())
    assert torch.equal(f['expected'], again['expected'])
    inputs = {k: v.clone() for k, v in f['inputs'].items()}
    inputs['a'][0, 0] += 1
    with pytest.raises(ValueError, match='mutated'):
        validate_actual(f['expected'], f, inputs)


def test_bound_and_tail_dispatch():
    s = spec()
    s['parameters'].update(K=2048, code_max=127)
    assert not integer_eligible(s)
    s['distribution'] = 'positive-bound'
    f = make_case(s)
    x = f['inputs']
    safe_float = x['sb'].T * (x['sa'] * (x['a'].double() @ x['b'].double()).float())
    assert validate_actual(safe_float, f, x)['passed']
    s['parameters'].update(K=133, code_max=16)
    assert integer_eligible(s)


def test_compiled_binary_identity():
    binary = b'compiled-object'
    item = SimpleNamespace(name='integer_cube_scaled', hash='a' * 64,
                           asm={'npubin': binary, 'ttir': 'text IR'})
    assert identity(item) == [dict(name=item.name, hash=item.hash,
        binaries={'npubin': hashlib.sha256(binary).hexdigest()})]
    assert identity(None) == []


@pytest.mark.parametrize('key,value', [('M', 0), ('K', -1), ('code_max', 128), ('output', 'fp16')])
def test_rejected_contract(key, value):
    s = spec()
    s['parameters'][key] = value
    with pytest.raises(ValueError):
        make_case(s)
