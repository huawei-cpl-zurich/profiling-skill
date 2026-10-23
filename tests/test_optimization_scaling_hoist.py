import importlib.util
from pathlib import Path

import pytest
import torch

FILE = Path(__file__).resolve().parents[1] / 'examples/optimization-library/triton/scaling_contract.py'
SPEC = importlib.util.spec_from_file_location('scaling_contract', FILE)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def test_oracle_has_row_scaling_strict_triangle_and_sign():
    k = torch.tensor([[[1, 2], [-1, 1], [2, 0]]], dtype=torch.bfloat16)
    beta = torch.tensor([[0.5, 0.25, 1.0]])
    expected = torch.tensor([[[0, 0, 0], [0.25, 0, 0], [2, -2, 0]]], dtype=torch.float32)
    assert torch.equal(contract.reference(k, beta), expected)


@pytest.mark.parametrize('bt,dk', [(64, 256), (64, 32), (32, 128), (32, 16)])
def test_fixed_contract_accepts_both_conversion_orders_and_stress_inputs(bt, dk):
    fixture = contract.make_case(dict(groups=2, bt=bt, dk=dk, seed=701 + dk))
    k, beta = fixture['k'], fixture['beta']
    assert k.dtype == torch.bfloat16 and beta.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(k.float(), dim=-1), torch.ones(2, bt), atol=0.003)
    assert (beta[:, 0] == 0).all() and (beta[:, 1] == 1).all()
    scaled = (k.float() * beta[..., None]).to(torch.bfloat16).float()
    before = torch.tril(torch.bmm(scaled, k.float().transpose(1, 2)), diagonal=-1)
    after = torch.tril(torch.bmm(k.float(), k.float().transpose(1, 2)) * beta[..., None], diagonal=-1)
    for actual in (before, after):
        result = contract.diagnostics(actual, fixture)
        assert result['passed'] and result['max_tolerance_fraction'] <= 1
    assert torch.equal(fixture['k'], contract.make_case(dict(groups=2, bt=bt, dk=dk, seed=701 + dk))['k'])


def test_diagnostics_reject_wrong_semantics_nonfinite_shape_and_dtype():
    fixture = contract.make_case(dict(groups=1, bt=32, dk=16, seed=7))
    good = fixture['expected']
    assert contract.diagnostics(good, fixture)['passed']
    for bad in (good + 0.1, torch.full_like(good, float('nan')), good[..., :-1], good.half()):
        assert not contract.diagnostics(bad, fixture)['passed']


@pytest.mark.parametrize('change', [dict(bt=17), dict(dk=65), dict(groups=0), dict(seed=-1), dict(groups=True)])
def test_unsupported_envelope_rejected_before_fixture_generation(change):
    spec = dict(groups=1, bt=32, dk=16, seed=0)
    spec.update(change)
    with pytest.raises(ValueError):
        contract.make_case(spec)
