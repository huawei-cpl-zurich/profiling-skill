"""Stable shared FP16 GEMM oracle for preload, residency and ShuffleK."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'triton'))
import gemm_contract
CASE_SOURCES = [Path(__file__),Path(gemm_contract.__file__)]


def check_spec(spec):
    if spec.get('family') not in ('08','09','11'):
        raise ValueError('family must be 08, 09 or 11')
    gemm_contract.check_spec(dict(spec,family='05'))
    if spec.get('b_layout','row')!='row' or any(spec[k]%16 for k in ('m','n','k')):
        raise ValueError('native preload cases require aligned row-major operands')
    if spec['family']=='09' and (spec['m']>128 or 128*spec['k']*2+256*256*2*2>512*1024):
        raise ValueError('full A must fit the fixed L1 partition with M<=128')


def make_case(spec):
    check_spec(spec)
    return gemm_contract.make_case(dict(spec,family='05'))
