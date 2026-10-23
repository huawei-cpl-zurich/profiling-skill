"""Padding-only oracle wrapper; epilogue uses the unchanged cross-language provider."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'triton'))
import gemm_contract
CASE_SOURCES=[Path(__file__),Path(gemm_contract.__file__)]


def check_spec(spec,family):
    if spec.get('family')!=family or family not in ('02','10'):
        raise ValueError('case does not match native mixed family')
    gemm_contract.check_spec(dict(spec,family='05'))
    if spec.get('b_layout','row')!='row':
        raise ValueError('mixed examples require row-major operands')
    if family=='02' and any(spec[key]%16 for key in ('m','n','k')):
        raise ValueError('epilogue requires dimensions aligned to 16 elements')


def make_padding_case(spec):
    check_spec(spec,'10')
    return gemm_contract.make_case(dict(spec,family='05'))
