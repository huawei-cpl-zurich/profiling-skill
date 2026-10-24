"""Functional tests of the CPU probability and tie contract."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton'))
from normalize_argmax_contract import make_case, reference, diagnostics, dimensions


def spec(width=128, mode='unique-winner', seed=4101):
    return dict(seed=seed, parameters=dict(rows=2, V=width, distribution=mode))


@pytest.mark.parametrize('width', [64, 128, 256, 8192, 16384])
def test_probability_recipe_and_exact_order(width):
    f = make_case(spec(width))
    target, draft, q = (f[k] for k in ('target', 'draft', 'q'))
    assert torch.equal(target.sum(-1), torch.ones(2))
    assert torch.equal(draft.sum(-1), torch.ones(2))
    residual = (target - draft).clamp_min(0)
    assert torch.equal(residual.sum(-1), torch.full((2,), .5))
    assert torch.equal(f['expected'], q.argmin(-1).int())
    assert diagnostics(reference(target, draft, q, False), f)['passed']


def test_zero_and_tie_lowest_original_index():
    zero = make_case(spec(mode='target-equals-draft'))
    assert zero['expected'].tolist() == [0, 0]
    tie = make_case(spec(mode='dyadic-residual-q-all-one'))
    active = tie['draft'] == 0
    assert torch.equal(tie['expected'], active.int().argmax(-1).int())


def test_wrong_output_rejected():
    fixture = make_case(spec())
    assert not diagnostics(fixture['expected'].long(), fixture)['passed']
    assert not diagnostics(fixture['expected'] + 1, fixture)['passed']


def test_seed_reproducible_and_changes_inputs():
    a, b, c = make_case(spec()), make_case(spec()), make_case(spec(seed=4102))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a['q'], c['q'])


@pytest.mark.parametrize('value', [0, 3, 32768])
def test_unsupported_width(value):
    with pytest.raises(ValueError):
        dimensions(spec(value))
