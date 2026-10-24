"""Public host, numerical/cache, canary and executable native ABI contracts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples/optimization-library/ascendc'), str(ROOT/'scripts')]
from int8_widening import Adapter, CompositeArm, check_input, empty_result, make_case, verify_binary
from optimization_pair import fixture
from optimization_native_bridge import NativeArm


def test_nonfrozen_oracles_cover_signed_values_and_odd_tail():
    for n in (263, 4103, 13, 0):
        value = make_case(dict(seed=12345, parameters=dict(N=n, also_empty=True)))
        x = value['x']
        assert x.dtype == torch.int8 and x.is_contiguous()
        assert value['expected'][0].tolist() == [float(int(v)) for v in x.tolist()]
        assert value['expected'][1].shape == (0,)
        if n >= 256:
            assert sorted(x[:256].tolist()) == list(range(-128, 128))


def test_cache_once_and_empty_no_native_launch(tmp_path):
    spec = dict(seed=22222, parameters=dict(N=263))
    value, first = fixture(Adapter(tmp_path), spec, tmp_path/'cache', torch)
    again, second = fixture(Adapter(tmp_path), spec, tmp_path/'cache', torch)
    assert (first['oracle_calls'], second['oracle_calls']) == (1, 0)
    assert first['sha256'] == second['sha256']
    assert torch.equal(value['expected'].view(torch.int32), again['expected'].view(torch.int32))
    x = torch.empty((0,), dtype=torch.int8, device='cpu')
    y = empty_result(x)
    assert y is not x and y.dtype == torch.float32 and y.shape == (0,)
    native = SimpleNamespace(sources=[], expected_kernels=['stub'], metadata={}, run=lambda: 'native')
    composite = CompositeArm(native, y)
    assert composite.run() == ('native', y)
    assert composite.metadata['component_operations']['empty_native_launches'] == 0


def test_actual_signed_dtype_layout_rejected_before_prepare_resources(tmp_path):
    spec = dict(seed=2, parameters=dict(N=9))
    for x in (torch.zeros(9, dtype=torch.uint8), torch.zeros(18, dtype=torch.int8)[::2], torch.zeros(3,3,dtype=torch.int8)):
        with pytest.raises(ValueError, match='signed int8'):
            Adapter(tmp_path).prepare('before', spec, dict(make_case(spec), x=x), tmp_path/'out')
    for p in (dict(N=9, dtype='uint8'), dict(N=9, layout='strided')):
        with pytest.raises(ValueError, match='signed int8'):
            fixture(Adapter(tmp_path), dict(seed=2, parameters=p), tmp_path/'cache', torch)
        assert not (tmp_path/'cache').exists()


def test_public_nested_cli_rejects_before_cache_device(tmp_path):
    spec = tmp_path/'case.json'; spec.write_text(json.dumps(dict(seed=2, parameters=dict(N=9,dtype='uint8'))))
    provenance=tmp_path/'provenance.json'; provenance.write_text('{}')
    process = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'),
        '--adapter','int8_widening:factory','--case',str(spec),'--output',str(tmp_path/'out'),
        '--cache',str(tmp_path/'cache'),'--provenance',str(provenance),'--correctness-only'],
        env=dict(os.environ,PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc')),capture_output=True,text=True)
    assert process.returncode == 1
    report=json.loads((tmp_path/'out/result.json').read_text())
    assert report['error']['type']=='ValueError'
    assert report['counts']==dict(profiled_calls=0,capture_calls=0,correctness_calls=0)
    assert not (tmp_path/'cache').exists()


def test_validator_detects_input_output_canary_and_bits(tmp_path):
    value=make_case(dict(seed=3,parameters=dict(N=13)))
    inp=torch.full((77,),-93,dtype=torch.int8); inp[32:-32]=value['x']
    out=torch.full((77,),-8191.,dtype=torch.float32); result=out[32:-32]; result.copy_(value['expected'])
    adapter=Adapter(tmp_path);adapter.prepared['before']=(None,inp,out,torch.empty(0,dtype=torch.int8))
    with patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None),create=True):
        assert adapter.validate('before',result,value,{})['passed']
        for storage in (inp,out):
            old=storage[0].item();storage[0]=0
            assert not adapter.validate('before',result,value,{})['gm_canaries']
            storage[0]=old
        inp[32]=0
        assert not adapter.validate('before',result,value,{})['immutable_input']
        inp[32:-32]=value['x'];result[0]=0
        assert not adapter.validate('before',result,value,{})['exact_bitwise']


@pytest.mark.skipif(not shutil.which('c++'),reason='host compiler required')
def test_native_abi_dimensions_arm_and_tail_pointer(tmp_path):
    source=tmp_path/'stub.cpp'
    source.write_text('''#include <stdint.h>
extern "C" uint32_t int8_widening_n(){return 13;}
extern "C" uint32_t int8_widening_after(){return 1;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"stub\\"]";}
extern "C" int catlass_bridge_launch(void**p,uint64_t c,void*){
 if(c!=2)return 1;for(int i=0;i<13;i++)((float*)p[1])[i]=(float)((int8_t*)p[0])[i];return 0;}
''')
    binary=tmp_path/'stub.so';subprocess.run(['c++','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
    verify_binary(binary,13,'after')
    with pytest.raises(ValueError,match='contract'):verify_binary(binary,13,'before')
    value=make_case(dict(seed=2,parameters=dict(N=13)))
    storage=torch.full((77,),-8191.,dtype=torch.float32);result=storage[32:-32]
    arm=NativeArm(binary,[value['x']],result,sources=[source],source_revision='host-test',stream_provider=lambda:0)
    try:
        arm.run();assert torch.equal(result.view(torch.int32),value['expected'].view(torch.int32))
        assert bool((storage[:32]==-8191).all() and (storage[-32:]==-8191).all())
    finally:arm.close()
