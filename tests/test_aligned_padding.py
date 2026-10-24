import json
import os
import subprocess
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


@pytest.mark.parametrize('with_empty', [False, True])
def test_public_cli_flat_nested_equivalence(tmp_path, with_empty):
    # A CPU preparation boundary exercises the real CLI and fixture provider
    # without compiling, loading native libraries or touching an NPU.
    provider = tmp_path/'cpu_padding.py'
    provider.write_text("""
from pathlib import Path
from aligned_padding import Adapter, dimensions
class CPUAdapter(Adapter):
    def prepare(self, arm, spec, value, output):
        width = dimensions(spec)[1]
        class Item:
            sources = [Path(__file__)]
            expected_kernels = ['cpu_interface_test']
            metadata = {'backend': 'cpu-interface-test'}
            def run(self):
                result = value['x'].clone()
                result[:, width:] = 0.0
                return result
        return Item()
    def validate(self, arm, actual, value, spec):
        return {'passed': actual.equal(value['expected'])}
def factory():
    return CPUAdapter()
""")
    nested = spec()
    nested['parameters']['also_empty_rows'] = with_empty
    flat = {k: v for k, v in nested.items() if k != 'parameters'}
    flat.update(nested['parameters'])
    provenance = tmp_path/'provenance.json'
    provenance.write_text('{}')
    payloads = []
    for label, case in [('nested', nested), ('flat', flat)]:
        case_path = tmp_path/(label+'.json')
        case_path.write_text(json.dumps(case))
        cache, output = tmp_path/(label+'-cache'), tmp_path/(label+'-out')
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join([
            str(tmp_path), str(ROOT/'examples/optimization-library/ascendc')]))
        run = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'),
            '--adapter', 'cpu_padding:factory', '--case', str(case_path),
            '--output', str(output), '--cache', str(cache), '--provenance', str(provenance),
            '--correctness-only'], cwd=tmp_path, env=environment, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        report = json.loads((output/'result.json').read_text())
        assert report['status'] == 'complete'
        assert report['counts'] == dict(profiled_calls=0, capture_calls=0, correctness_calls=2)
        payloads.append(torch.load(next(cache.glob('*/fixture.pt')), weights_only=True))
    assert payloads[0].keys() == payloads[1].keys()
    for key in payloads[0]:
        assert torch.equal(payloads[0][key].view(torch.int32), payloads[1][key].view(torch.int32))
    assert ('empty' in payloads[0]) == with_empty
