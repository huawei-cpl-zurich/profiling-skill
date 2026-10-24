"""Functional host rejection, cache boundary, oracle and real native ABI tests."""
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
from row_dma import Adapter, dimensions, make_case, verify_binary
from optimization_pair import fixture
from optimization_native_bridge import NativeArm


def test_nested_oracles_and_conservative_fallback():
    # Independent test inputs avoid regenerating frozen scientific payloads.
    cases = [dict(seed=10000+i, parameters=dict(rows=r, columns=c))
             for i, (r, c) in enumerate(((9, 9), (1, 17), (3, 31), (129, 7)))]
    for case in cases:
        value = make_case(case)
        rows, columns, pitch = dimensions(case)
        assert value['x'].shape == (rows, columns)
        assert pitch % 8 == 0
        assert value['expected'].dtype == torch.float32
        assert torch.equal(value['expected'].view(torch.int32),
                           (value['x'].double() + 1).float().view(torch.int32))
    assert dimensions(cases[-1])[0] == 129  # Supported fallback, not a hardware maximum.


def test_fixture_generated_once_then_reused(tmp_path):
    case = dict(seed=2222, parameters=dict(rows=3, columns=9))
    first, initial = fixture(Adapter(tmp_path), case, tmp_path/'cache', torch)
    second, reused = fixture(Adapter(tmp_path), case, tmp_path/'cache', torch)
    assert initial['oracle_calls'] == 1 and reused['oracle_calls'] == 0
    assert initial['sha256'] == reused['sha256']
    assert torch.equal(first['expected'].view(torch.int32), second['expected'].view(torch.int32))

def test_host_pitch_rejected_before_fixture_cache_and_prepare(tmp_path):
    invalid = dict(parameters=dict(rows=3, columns=9, destination_pitch_elements=17), seed=7492)
    adapter = Adapter(tmp_path)
    for arm in ('before', 'after'):
        with pytest.raises(ValueError, match='pitch'):
            adapter.prepare(arm, invalid, {}, tmp_path/arm)
    with pytest.raises(ValueError, match='pitch'):
        fixture(adapter, invalid, tmp_path/'cache', torch)
    assert not (tmp_path/'cache').exists()
    with pytest.raises(ValueError, match='65536'):
        dimensions(dict(rows=129, columns=128))


def test_public_nested_cli_rejects_before_cache_and_device(tmp_path):
    case = tmp_path/'case.json'
    case.write_text(json.dumps(dict(seed=7492, parameters=dict(rows=3, columns=9, destination_pitch_elements=17))))
    provenance = tmp_path/'provenance.json'; provenance.write_text('{}')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc'))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'),
        '--adapter', 'row_dma:factory', '--case', str(case), '--output', str(tmp_path/'out'),
        '--cache', str(tmp_path/'cache'), '--provenance', str(provenance), '--correctness-only'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads((tmp_path/'out/result.json').read_text())
    assert report['error']['type'] == 'ValueError' and 'pitch' in report['error']['message']
    assert report['counts'] == dict(profiled_calls=0, capture_calls=0, correctness_calls=0)
    assert not (tmp_path/'cache').exists()


def test_validation_detects_canaries_input_mutation_and_wrong_output(tmp_path):
    adapter = Adapter(tmp_path)
    value = make_case(dict(rows=2, columns=9, seed=1))
    backing = torch.full((82,), -8191., dtype=torch.float32)
    result = backing[32:-32].view(2, 9); result.copy_(value['expected'])
    item = SimpleNamespace(inputs=[value['x'].clone()])
    adapter.prepared['before'] = item, backing
    with patch.object(torch, 'npu', SimpleNamespace(synchronize=lambda: None), create=True):
        assert adapter.validate('before', result, value, {})['passed']
        backing[0] = 0
        assert not adapter.validate('before', result, value, {})['gm_canaries']
        backing[0] = -8191
        item.inputs[0][0, 0] = 0
        assert not adapter.validate('before', result, value, {})['immutable_input']
        item.inputs[0].copy_(value['x']); result[0, 0] = 9
        assert not adapter.validate('before', result, value, {})['exact_bitwise']


@pytest.mark.skipif(not shutil.which('c++'), reason='host C++ compiler required')
def test_native_abi_shape_arm_and_pointer_span(tmp_path):
    source = tmp_path/'stub.cpp'
    source.write_text('''#include <stdint.h>
extern "C" uint32_t row_dma_rows(){return 2;}
extern "C" uint32_t row_dma_columns(){return 9;}
extern "C" uint32_t row_dma_after(){return 1;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
extern "C" const char* catlass_bridge_kernel_names(){return "[\\"host_stub\\"]";}
extern "C" int catlass_bridge_launch(void**p,uint64_t count,void*){
 if(count!=2)return 1;
 for(int i=0;i<18;i++)((float*)p[1])[i]=((float*)p[0])[i]+1;return 0;}
''')
    binary = tmp_path/'stub.so'
    subprocess.run(['c++', '-shared', '-fPIC', str(source), '-o', str(binary)], check=True)
    verify_binary(binary, 2, 9, 'after')
    with pytest.raises(ValueError, match='contract'):
        verify_binary(binary, 2, 9, 'before')
    value = make_case(dict(rows=2, columns=9, seed=1))
    backing = torch.full((82,), -8191., dtype=torch.float32)
    result = backing[32:-32].view(2, 9)
    arm = NativeArm(binary, [value['x']], result, sources=[source], source_revision='host-stub', stream_provider=lambda: 0)
    try:
        arm.run()
        assert torch.equal(result.view(torch.int32), value['expected'].view(torch.int32))
        assert bool((backing[:32] == -8191).all() and (backing[-32:] == -8191).all())
    finally:
        arm.close()
