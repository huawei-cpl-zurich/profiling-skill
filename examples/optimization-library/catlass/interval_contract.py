"""Restricted unpadded FP16 ND GEMM contract; small M is intentionally allowed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'triton'))
import gemm_contract

CASE_SOURCES=[Path(__file__),Path(gemm_contract.__file__)]


def check_spec(spec):
    if spec.get('family')!='E12':
        raise ValueError('family must be E12')
    gemm_contract.check_spec(dict(spec,family='05'))
    if spec.get('b_layout','row')!='row' or spec['n']%16 or spec['k']%16:
        raise ValueError('interval DMA requires row-major operands and N/K multiples of16')


def make_case(spec):
    check_spec(spec)
    return gemm_contract.make_case(dict(spec,family='05'))


def copy_plan(spec):
    """Logical source-call model, not measured DMA instructions or HBM traffic."""
    check_spec(spec)
    m,n,k=(spec[key] for key in ('m','n','k'))
    ntiles,ktiles=(n+255)//256,(k+255)//256
    return dict(l1_tile=[128,256,256],l0_tile=[128,256,64],l1_reserved_bytes=393216,
        workspace_bytes=0,nd2nz_requests=((m+127)//128)*ntiles*ktiles,
        interval_row_requests=m*ntiles*ktiles,source_model_only=True)
