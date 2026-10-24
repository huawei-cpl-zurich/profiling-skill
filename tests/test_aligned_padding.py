import importlib.util
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/ascendc'))
from aligned_padding import Adapter, dimensions, empty_result, make_case, vector_admitted
from optimization_pair import fixture


def spec(rows=5, width=24, pitch=48):
    return dict(case_id='unit-padding', role='guard', seed=105, parameters=dict(rows=rows, width=width, pitch=pitch))


@pytest.mark.parametrize('rows,width,pitch', [(5,24,48),(4,25,32),(7,16,16),(0,24,32)])
def test_bitwise_oracle_and_admission(rows,width,pitch):
    value=make_case(spec(rows,width,pitch))
    assert torch.equal(value['expected'][:,:width].view(torch.int32),value['x'][:,:width].view(torch.int32))
    assert (value['expected'][:,width:].view(torch.int32)==0).all()
    assert (value['x'][:,width:]==73.25).all()
    assert vector_admitted(rows,width,pitch)==(rows>0 and width<pitch and width%8==0)
    if rows:
        assert (value['expected'][:,0].view(torch.int32)==-2147483648).all()


@pytest.mark.parametrize('args', [(-1,8,16),(1,17,16),(1,8,17),(1,True,16),(100,512,1024)])
def test_invalid_before_native(args,tmp_path):
    with pytest.raises(ValueError):
        Adapter(tmp_path).prepare('after',spec(*args),{},tmp_path)


def test_empty_new_output_and_rejection():
    x=torch.empty((0,32),dtype=torch.float32)
    y=empty_result(x,24)
    assert y is not x and y.shape==x.shape and y.dtype==x.dtype
    with pytest.raises(ValueError):
        empty_result(torch.empty((1,32)),24)


def test_once_only_cache(tmp_path):
    adapter=Adapter()
    value,first=fixture(adapter,spec(),tmp_path,torch)
    adapter.make_case=lambda _: (_ for _ in ()).throw(AssertionError('regenerated'))
    cached,second=fixture(adapter,spec(),tmp_path,torch)
    assert first['oracle_calls']==1 and second['oracle_calls']==0
    assert torch.equal(value['expected'].view(torch.int32),cached['expected'].view(torch.int32))
