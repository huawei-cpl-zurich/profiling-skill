"""Functional CPU contract checks; no replay of the scientific matrix."""
import importlib.util
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('csr_contract', ROOT / 'examples/optimization-library/triton/csr_balanced_contract.py')
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def spec(**changes):
    return dict(dict(rows=8, cols=16, nnz=32, max_row=16, seed=17, distribution='long-tail'), **changes)


def test_repeated_columns_and_empty_rows_have_fp64_oracle():
    fixture = dict(indptr=torch.tensor([0, 2, 2, 3], dtype=torch.int32),
                   columns=torch.tensor([1, 1, 0], dtype=torch.int32),
                   values=torch.tensor([.25, -.125, .125]), x=torch.tensor([.25, -.25]))
    expected = contract.reference(**fixture)
    torch.testing.assert_close(expected, torch.tensor([-.03125, 0., .03125], dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize('distribution', ['long-tail', 'long-tail-with-empty-rows', 'uniform', 'empty', 'long-row-guard'])
def test_generator_respects_exact_shape_and_bounds(distribution):
    s = spec(distribution=distribution)
    if distribution == 'uniform':
        s.update(max_row=4)
    if distribution == 'empty':
        s.update(nnz=0, max_row=0)
    fixture = contract.make_case(s)
    assert contract.validate_inputs(fixture, s) <= s['max_row']
    assert fixture['indptr'][-1] == s['nnz']
    assert fixture['expected'].dtype == torch.float64
    assert torch.isnan(fixture['initial_y']).all()
    assert contract.diagnostics(fixture['expected'].float(), fixture)['passed']


def test_empty_rows_zero_and_nan_output_rejected():
    fixture = contract.make_case(spec(nnz=0, max_row=0, distribution='empty'))
    assert contract.diagnostics(torch.zeros(8), fixture)['empty_rows_zero']
    assert not contract.diagnostics(torch.full((8,), float('nan')), fixture)['passed']
    assert not contract.diagnostics(torch.ones(8), fixture)['passed']


@pytest.mark.parametrize('mutation', ['ptr', 'column', 'value', 'dtype'])
def test_invalid_csr_is_rejected(mutation):
    s = spec()
    fixture = contract.make_case(s)
    if mutation == 'ptr':
        fixture['indptr'][0] = 1
    if mutation == 'column':
        fixture['columns'][0] = s['cols']
    if mutation == 'value':
        fixture['values'][0] = .5
    if mutation == 'dtype':
        fixture['columns'] = fixture['columns'].long()
    with pytest.raises(ValueError):
        contract.validate_inputs(fixture, s)


def test_nonuniform_claim_rejected():
    with pytest.raises(ValueError):
        contract.make_case(spec(distribution='uniform', max_row=5))
