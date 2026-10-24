"""Executable lifetime, numerical, canary, cache and ABI checks."""
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
from scratch_reuse import Adapter, dimensions, make_case, verify_binary
from optimization_pair import fixture
from optimization_native_bridge import NativeArm


def test_oracle_stages_and_late_lifetime():
    spec = dict(seed=123, parameters=dict(rows=2, width=64))
    normal = make_case(spec)
    late = make_case(dict(seed=123, parameters=dict(rows=2, width=64, needs_first_late=True)))
    assert torch.equal(normal['x'], late['x'])
    a = (normal['x'].double() ** 2).float()
    y = (a.double() + normal['x'].double()).float()
    b = (y.double()/2).float()
    assert torch.equal(normal['expected'].view(torch.int32), (b.double()-.25).float().view(torch.int32))
    assert torch.equal(late['expected'].view(torch.int32), (b.double()+a.double()).float().view(torch.int32))
    assert not torch.equal(normal['expected'], late['expected'])
    assert dimensions(dict(rows=6, width=1536)) == (6, 1536, 9216, False)
    assert dimensions(dict(rows=3, width=33, tile_elements=64)) == (3, 33, 64, False)


def test_fixture_once_and_rejection_before_cache(tmp_path):
    case = dict(seed=1234, parameters=dict(rows=3, width=33, tile_elements=64))
    one, initial = fixture(Adapter(tmp_path), case, tmp_path/'cache', torch)
    two, reused = fixture(Adapter(tmp_path), case, tmp_path/'cache', torch)
    assert initial['oracle_calls'] == 1 and reused['oracle_calls'] == 0
    assert initial['sha256'] == reused['sha256']
    assert torch.equal(one['expected'], two['expected'])
    for parameters in (dict(rows=3, width=0), dict(rows=3, width=33, tile_elements=63), dict(rows=16, width=1024)):
        with pytest.raises(ValueError):
            fixture(Adapter(tmp_path), dict(seed=12, parameters=parameters), tmp_path/'invalid', torch)
    assert not (tmp_path/'invalid').exists()


def test_nested_public_cli_rejects_without_device(tmp_path):
    case=tmp_path/'case.json'; case.write_text(json.dumps(dict(seed=1, parameters=dict(rows=1, width=0))))
    provenance=tmp_path/'provenance.json'; provenance.write_text('{}')
    env=dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc'))
    result=subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--adapter', 'scratch_reuse:factory',
        '--case', str(case), '--output', str(tmp_path/'out'), '--cache', str(tmp_path/'cache'),
        '--provenance', str(provenance), '--correctness-only'], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    report=json.loads((tmp_path/'out/result.json').read_text())
    assert report['error']['type'] == 'ValueError'
    assert report['counts'] == dict(profiled_calls=0, capture_calls=0, correctness_calls=0)
    assert not (tmp_path/'cache').exists()


def test_validator_checks_both_canaries_and_immutable_source(tmp_path):
    value=make_case(dict(rows=3,width=33,tile_elements=64,seed=3))
    source=torch.full((163,), -8191., dtype=torch.float32)
    source[32:-32].copy_(value['x'].flatten())
    target=source.clone(); actual=target[32:-32].view(3,33); actual.copy_(value['expected'])
    adapter=Adapter(tmp_path)
    adapter.prepared['after']=(SimpleNamespace(inputs=[source[32:-32].view(3,33)]),target,source)
    with patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None),create=True):
        assert adapter.validate('after',actual,value,{})['passed']
        source[0]=0
        assert not adapter.validate('after',actual,value,{})['gm_canaries']
        source[0]=-8191; target[-1]=0
        assert not adapter.validate('after',actual,value,{})['gm_canaries']
        target[-1]=-8191; source[35]=999
        assert not adapter.validate('after',actual,value,{})['immutable_input']
        source[32:-32].copy_(value['x'].flatten()); actual[-1,-1]=999
        assert not adapter.validate('after',actual,value,{})['exact_bitwise']


@pytest.mark.skipif(not shutil.which('c++'), reason='host C++ compiler required')
def test_real_abi_late_fallback_and_repeated_tail(tmp_path):
    # Host executable exercises the same public ABI with a staged float oracle.
    for late in (0,1):
        source=tmp_path/f'stub{late}.cpp'
        source.write_text('''#include <stdint.h>
extern "C" uint32_t scratch_reuse_rows(){return 3;}
extern "C" uint32_t scratch_reuse_width(){return 33;}
extern "C" uint32_t scratch_reuse_tile(){return 64;}
extern "C" uint32_t scratch_reuse_after(){return 1;}
extern "C" uint32_t scratch_reuse_late(){return LATE;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"host_scratch\\"]";}
extern "C" int catlass_bridge_launch(void**p,uint64_t count,void*){
 if(count!=2)return 1;
 for(int off=0;off<99;off+=64) for(int i=off;i<off+64 && i<99;i++){
 float x=((float*)p[0])[i]; volatile float a=x*x; volatile float y=a+x;
 volatile float b=.5f*y; ((float*)p[1])[i]=LATE ? b+a : b-.25f;}return 0;}
''')
        binary=tmp_path/f'stub{late}.so'
        subprocess.run(['c++','-shared','-fPIC',f'-DLATE={late}',str(source),'-o',str(binary)],check=True)
        spec=dict(rows=3,width=33,tile_elements=64,needs_first_late=bool(late),seed=9)
        verify_binary(binary,spec,'after')
        with pytest.raises(ValueError,match='contract'):verify_binary(binary,spec,'before')
        with pytest.raises(ValueError,match='contract'):verify_binary(binary,dict(spec,tile_elements=128),'after')
        value=make_case(spec); backing=torch.full((163,),-8191.,dtype=torch.float32)
        result=backing[32:-32].view(3,33)
        arm=NativeArm(binary,[value['x']],result,sources=[source],source_revision='host',stream_provider=lambda:0)
        try:
            arm.run()
            assert torch.equal(result.view(torch.int32),value['expected'].view(torch.int32))
            assert bool((backing[:32]==-8191).all() and (backing[-32:]==-8191).all())
        finally:arm.close()
