"""Functional finite-code-value, gather and strict-threshold contracts."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/optimization-library/ascendc'))
from threshold_topk_contract import make_case, running_reference, diagnostics, dimensions


@pytest.mark.parametrize('distribution', ['equal', 'first-tile-dominant-fp16', 'ascending-tiles', 'ascending-quantized-levels'])
@pytest.mark.parametrize('k', [1, 32, 1024])
def test_running_topk(distribution, k):
    fixture = make_case(dict(rows=2, V=16401, K=k, seed=3101, distribution=distribution))
    actual = running_reference(fixture['scores'], k)
    assert diagnostics(actual, fixture)['passed']


def test_ties_allow_any_unique_indices():
    fixture = make_case(dict(rows=1, V=40, K=3, seed=3105, distribution='equal'))
    assert diagnostics((torch.ones(1, 3).half(), torch.tensor([[35, 1, 17]])), fixture)['passed']
    assert not diagnostics((torch.ones(1, 3).half(), torch.tensor([[1, 1, 17]])), fixture)['passed']


def test_reject_wrong_order_or_gather():
    fixture = dict(scores=torch.tensor([[1, 3, 2]]).half(), expected=torch.tensor([[3, 2]]).half())
    assert not diagnostics((torch.tensor([[2, 3]]).half(), torch.tensor([[2, 1]])), fixture)['passed']
    assert not diagnostics((fixture['expected'], torch.tensor([[0, 2]])), fixture)['passed']


@pytest.mark.parametrize('k', [0, 1025, 40000, True])
def test_k_guard(k):
    with pytest.raises(ValueError):
        dimensions(dict(rows=1, V=16384, K=k, seed=3106))
