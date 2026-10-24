"""Functional coordinate oracle, contract rejection and nested fixture-cache tests."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'examples/optimization-library/triton'),str(ROOT/'scripts')]
from bilinear_contract import make_case, parameters, reference, checked_input
from bilinear_reuse import Adapter
from optimization_pair import fixture


def spec(**patch):
    return dict(case_id='cpu-bilinear',role='guard',seed=71,
                parameters=dict(N=1,H=2,W=3,C=2,OH=3,OW=5,**patch))


def test_scalar_four_corner_reference():
    s=spec();f=make_case(s);x=f['x'];p=parameters(s)
    expected=torch.empty((1,3,5,2),dtype=torch.float64)
    for y in range(3):
        for z in range(5):
            sy,sx=y*2/3,z*3/5
            y0,x0=math.floor(sy),math.floor(sx)
            y1,x1=min(y0+1,1),min(x0+1,2)
            dy,dx=sy-y0,sx-x0
            for c in range(2):
                a=(float(x[0,y0,x0,c])*(1-dx)+float(x[0,y0,x1,c])*dx)
                b=(float(x[0,y1,x0,c])*(1-dx)+float(x[0,y1,x1,c])*dx)
                expected[0,y,z,c]=a*(1-dy)+b*dy
    torch.testing.assert_close(f['expected'],expected,atol=1e-15,rtol=1e-15)


def test_one_pixel_axis_and_right_border():
    p=dict(N=1,H=1,W=2,C=1,OH=4,OW=3)
    x=torch.tensor([[[[-1.],[1.]]]],dtype=torch.float32)
    y=reference(x,p)
    expected=torch.tensor([-1.,1/3,1.],dtype=torch.float64).reshape(1,1,3,1).expand(1,4,3,1)
    torch.testing.assert_close(y,expected,atol=1e-15,rtol=1e-15)


def test_ambient_dtype_and_cache(tmp_path):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        first,a=fixture(Adapter(),spec(),tmp_path,torch)
        second,b=fixture(Adapter(),spec(),tmp_path,torch)
    finally:
        torch.set_default_dtype(old)
    assert first['x'].dtype==torch.float32 and first['x'].device.type=='cpu'
    assert a['oracle_calls']==1 and b['oracle_calls']==0
    assert torch.equal(first['x'],second['x'])


def test_public_nested_fixture_cli(tmp_path):
    case=tmp_path/'case.json';case.write_text(json.dumps(spec()))
    output=tmp_path/'fixture.pt'
    result=subprocess.run([sys.executable,str(ROOT/'examples/optimization-library/triton/bilinear_contract.py'),
                           '--case',str(case),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert torch.load(output,weights_only=False)['expected'].shape==(1,3,5,2)


def test_unsupported_mode_rejected_before_device_work(tmp_path):
    s=spec();s['parameters']['half_pixel_centers']=True
    f=make_case(s)
    assert f=={'expected_host_rejection':True}
    for arm in ('before','after'):
        with pytest.raises(ValueError,match='asymmetric'):
            Adapter().prepare(arm,s,f,tmp_path)


@pytest.mark.parametrize('patch',[dict(H=0),dict(C=True),dict(align_corners=True),dict(antialias=True)])
def test_reject_parameters(patch):
    s=spec();s['parameters'].update(patch)
    with pytest.raises(ValueError):parameters(s)


def test_reject_layout_dtype_range():
    p=parameters(spec());x=make_case(spec())['x']
    for invalid in (x.double(),x.transpose(1,2),x+3,x*float('nan')):
        with pytest.raises(ValueError):checked_input(invalid,p)
