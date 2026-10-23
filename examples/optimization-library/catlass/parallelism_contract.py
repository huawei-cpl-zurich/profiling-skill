"""Shared GEMM oracle with stable identity for native families 06 and 07."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton'))
import gemm_contract
CASE_SOURCES = [Path(__file__), Path(gemm_contract.__file__)]


def check_spec(spec):
    if spec.get('family') not in ('06', '07'):
        raise ValueError('family must be 06 or 07')
    gemm_contract.check_spec(dict(spec, family='05'))
    if spec.get('b_layout', 'row') != 'row' or any(spec[k] % 16 for k in ('m','n','k')):
        raise ValueError('native parallelism examples require aligned row-major inputs')


def make_case(spec):
    check_spec(spec)
    return gemm_contract.make_case(dict(spec, family='05'))
