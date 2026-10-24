"""Functional bit-exact lookup and invalid-input/cache/native ABI boundaries."""
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
from ub_gather import Adapter, dimensions, make_case, validate_indices, verify_binary
from optimization_pair import fixture
from optimization_native_bridge import NativeArm


def test_oracle_patterns_and_signedzero():
    for pattern in ('uniform', 'repeat-one', 'alternating-two', 'endpoints-and-signedzero'):
        data = make_case(dict(seed=19001, parameters=dict(N=33, K=23, pattern=pattern)))
        assert torch.equal(data['expected'].view(torch.int32), data['table'][data['indices'].long()].view(torch.int32))
    assert data['expected'].view(torch.int32)[:4].tolist() == [0, -2147483648, 0, -2147483648]
    for n, k in ((4099, 31), (1, 1)):
        assert make_case(dict(N=n, K=k, seed=19002))['expected'].shape == (n,)


@pytest.mark.parametrize('bad', [-1, 31])
def test_invalid_index_rejected_before_cache_and_prepare(tmp_path, bad):
    spec = dict(seed=19003, parameters=dict(N=3, K=31, indices=[0, bad, 30]))
    adapter = Adapter(tmp_path)
    with pytest.raises(ValueError, match=r'\[0,K\)'):
        fixture(adapter, spec, tmp_path/'cache', torch)
    assert not (tmp_path/'cache').exists()
    valid = make_case(dict(seed=19004, N=3, K=31))
    valid['indices'][1] = bad
    for arm in ('before', 'after'):
        with pytest.raises(ValueError, match=r'\[0,K\)'):
            adapter.prepare(arm, spec, valid, tmp_path/arm)


def test_public_nested_cli_rejects_before_cache(tmp_path):
    spec = tmp_path/'case.json'; spec.write_text(json.dumps(dict(seed=19005, parameters=dict(N=2, K=11, indices=[0,11]))))
    provenance = tmp_path/'provenance.json'; provenance.write_text('{}')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc'))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--adapter', 'ub_gather:factory',
        '--case', str(spec), '--output', str(tmp_path/'out'), '--cache', str(tmp_path/'cache'),
        '--provenance', str(provenance), '--correctness-only'], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads((tmp_path/'out/result.json').read_text())
    assert report['error']['type'] == 'ValueError'
    assert report['counts'] == dict(profiled_calls=0, capture_calls=0, correctness_calls=0)
    assert not (tmp_path/'cache').exists()


def test_cached_fixture_reused_and_layout_rejected(tmp_path):
    adapter = Adapter(tmp_path); spec = dict(seed=19006, N=7, K=13)
    a, first = fixture(adapter, spec, tmp_path/'cache', torch)
    b, second = fixture(adapter, spec, tmp_path/'cache', torch)
    assert first['oracle_calls'] == 1 and second['oracle_calls'] == 0
    assert first['sha256'] == second['sha256']
    assert torch.equal(a['expected'].view(torch.int32), b['expected'].view(torch.int32))
    for invalid in (torch.tensor([0.]), torch.arange(8, dtype=torch.int32)[::2]):
        with pytest.raises(ValueError, match='contiguous CPU int32'):
            validate_indices(invalid, 13)
    with pytest.raises(ValueError): dimensions(dict(N=True, K=13))


def test_validation_detects_bit_changes_mutation_and_canaries(tmp_path):
    adapter = Adapter(tmp_path)
    value = make_case(dict(seed=19007, N=5, K=11, pattern='endpoints-and-signedzero'))
    backing = torch.full((69,), -8191., dtype=torch.float32)
    result = backing[32:-32]; result.copy_(value['expected'])
    item = SimpleNamespace(inputs=[value['table'].clone(), value['indices'].clone()])
    adapter.prepared['before'] = item, backing
    with patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True):
        assert adapter.validate('before', result, value, {})['passed']
        result[1] = +0.
        assert not adapter.validate('before', result, value, {})['exact_bitwise']
        result.copy_(value['expected']); backing[-1] = 0
        assert not adapter.validate('before', result, value, {})['gm_canaries']
        backing[-1] = -8191; item.inputs[1][0] = 1
        assert not adapter.validate('before', result, value, {})['immutable_input']


@pytest.mark.skipif(not shutil.which('c++'), reason='host C++ compiler required')
def test_native_abi_byte_offsets_and_output_span(tmp_path):
    source = tmp_path/'stub.cpp'
    source.write_text('''#include <stdint.h>
extern "C" uint32_t ub_gather_n(){return 5;}
extern "C" uint32_t ub_gather_k(){return 11;}
extern "C" uint32_t ub_gather_after(){return 1;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"host_stub\\"]";}
extern "C" int catlass_bridge_launch(void**p,uint64_t count,void*){
 if(count!=3)return 1;
 for(int i=0;i<5;i++){uint32_t offset=((int32_t*)p[1])[i]*4;
 ((float*)p[2])[i]=*(float*)((uint8_t*)p[0]+offset);}return 0;}
''')
    binary = tmp_path/'stub.so'
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(binary)], check=True)
    verify_binary(binary, 5, 11, 'after')
    with pytest.raises(ValueError, match='contract'): verify_binary(binary, 5, 11, 'before')
    data = make_case(dict(seed=19008, N=5, K=11, pattern='endpoints-and-signedzero'))
    storage = torch.full((69,), -8191., dtype=torch.float32); result = storage[32:-32]
    arm = NativeArm(binary, [data['table'], data['indices']], result, sources=[source], source_revision='host-stub', stream_provider=lambda: 0)
    try:
        arm.run()
        assert torch.equal(result.view(torch.int32), data['expected'].view(torch.int32))
        assert bool((storage[:32] == -8191).all() and (storage[-32:] == -8191).all())
        assert len(arm.compilation_state()['binary_sha256']) == 64
    finally: arm.close()
