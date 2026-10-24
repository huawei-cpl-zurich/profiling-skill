"""Functional statistics, weighted-tail and contract regressions on CPU."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

MODULE = Path(__file__).resolve().parents[1]/'examples/optimization-library/triton'
sys.path.insert(0, str(MODULE))
import welford_contract as contract
from welford import Adapter


def spec(**kwargs):
    return dict(N=2, C=3, H=3, W=47, seed=8101, distribution='bounded-iid', **kwargs)


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('distribution', ['bounded-iid', 'constant-plus-small-noise', 'constant'])
def test_statistics_match_independent_fp64(streaming, distribution):
    case = spec()
    case['distribution'] = distribution
    fixture = contract.make_case(case)
    actual = contract.host_statistics(fixture['x'], streaming)
    assert contract.diagnostics(actual, fixture)['passed']


def test_tail_lane_counts_are_weighted():
    # First lanes have three samples, others only two. Equal lane weights fail.
    x = torch.arange(513, dtype=torch.float32).reshape(1, 1, 1, 513)/513
    mean, variance = contract.host_statistics(x, True)
    expected = contract.reference(x)
    torch.testing.assert_close(mean.double(), expected[0], atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(variance.double(), expected[1], atol=1e-7, rtol=2e-3)


@pytest.mark.parametrize('streaming', [False, True])
def test_repeated_calls_overwrite_prior_statistics(streaming):
    x = torch.rand(2, 3, 2, 17)
    contract.host_statistics(x, streaming)
    mean, variance = contract.host_statistics(torch.full_like(x, 16), streaming)
    assert torch.equal(mean, torch.full((3,), 16.))
    assert torch.equal(variance, torch.zeros(3))


def test_gate_rejects_stale_or_nonfinite_outputs():
    fixture = contract.make_case(spec())
    good = tuple(v.float() for v in fixture['expected'])
    assert contract.diagnostics(good, fixture)['passed']
    assert not contract.diagnostics((good[0]+1, good[1]), fixture)['passed']
    assert not contract.diagnostics((good[0], good[1]*float('nan')), fixture)['passed']
    assert not contract.diagnostics((good[0],), fixture)['passed']


@pytest.mark.parametrize('field,value', [('N', 0), ('H', 0), ('W', 0), ('C', 0), ('N', True), ('seed', -1)])
def test_empty_and_invalid_dimensions_reject_before_device_import(field, value):
    case = spec()
    case[field] = value
    with pytest.raises(ValueError):
        Adapter().prepare('before', case, {'x':torch.empty(0)}, None)
    with pytest.raises(ValueError):
        Adapter().prepare('after', case, {'x':torch.empty(0)}, None)


def test_input_contract_rejects_nonfinite_dtype_and_stride():
    case = spec()
    fixture = contract.make_case(case)
    for x in (fixture['x'].double(), fixture['x'].transpose(2, 3), fixture['x']*float('inf')):
        with pytest.raises(ValueError):
            contract.validate_inputs(x, case)


def test_constant_gate_requires_exact_zero_variance():
    case = spec()
    case['distribution'] = 'constant'
    fixture = contract.make_case(case)
    actual = (fixture['expected'][0].float(), torch.full((3,), 1e-9))
    assert not contract.diagnostics(actual, fixture)['passed']
