import importlib.util
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
from resident_gemm_contract import check_spec, check_prefix, make_case, validate_actual


def test_prefix_zero_groups_keep_descriptor_positions():
    assert check_prefix([0, 0, 3, 3, 20, 20], 5, 20) == [0, 3, 0, 17, 0]
    with pytest.raises(ValueError, match='monotonically'):
        check_prefix([0, 3, 2, 6], 3, 6)


def test_interleaved_groups_oracle_and_validator():
    spec = dict(seed=1, parameters=dict(Ms=[0, 3, 0, 17, 0, 2, 0], N=65, K=64))
    fixture = make_case(spec)
    a, b = fixture['inputs']['a'], fixture['inputs']['b']
    expected = torch.cat([(a[:3].double() @ b[1].double()).float(),
                          (a[3:20].double() @ b[3].double()).float(),
                          (a[20:].double() @ b[5].double()).float()])
    assert torch.equal(expected, fixture['expected'])
    assert validate_actual(expected, fixture, fixture['inputs'])['passed']
    wrong = expected.clone(); wrong[4, 2] += 1
    assert not validate_actual(wrong, fixture, fixture['inputs'])['passed']
    incomplete = expected.clone(); incomplete[4, 2] = float('nan')
    assert not validate_actual(incomplete, fixture, fixture['inputs'])['passed']
    assert max(float(a.abs().max()), float(b.abs().max())) <= .125


@pytest.mark.parametrize('ms,n,k', [([-1, 2], 64, 64), ([0], 64, 64), ([3], 0, 64), ([3], 64, 0)])
def test_reject_unsupported(ms, n, k):
    with pytest.raises(ValueError):
        check_spec(dict(parameters=dict(Ms=ms, N=n, K=k)))
