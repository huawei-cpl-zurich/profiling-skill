"""Host contracts and actual frozen-case CLI routing; device kernels gate on A3."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT/'examples/optimization-library/triton'
sys.path.insert(0,str(EXAMPLES))
from rfft_contract import diagnostics, dimensions, make_case, validate_inputs


def test_reconstruction_math_and_order():
    # Independent host algebra check, not a claim that device kernels ran.
    x = torch.tensor([[.125, -.125, .0625, 0., -.0625, .125, 0., -.125]], dtype=torch.float64)
    z = torch.fft.fft(torch.complex(x[:,::2], x[:,1::2]))
    k = torch.arange(5)
    a, b = z[:,k%4], z[:,(-k)%4].conj()
    reconstructed = (a+b)/2 + torch.exp(-2j*torch.pi*k.double()/8)*(a-b)/(2j)
    torch.testing.assert_close(reconstructed, torch.fft.rfft(x), atol=1e-14,rtol=1e-14)


def test_dc_nyquist_and_exact_zero_gate():
    f = make_case(dict(batch=2,N=64,seed=4391,distribution='dc-constant-and-alternating-nyquist'))
    assert f['expected'][0,0] == f['expected'][1,-1] == 8
    real, imag = f['expected'].real.float(), f['expected'].imag.float()
    assert diagnostics((real,imag),f)['passed']
    imag[0,0] = 1e-20
    assert not diagnostics((real,imag),f)['passed']
    assert not diagnostics((real.flip(1), imag.zero_()),f)['passed']


@pytest.mark.parametrize('n',[1,32,63,64,512,1024])
def test_oracle_and_inputs(n):
    f = make_case(dict(batch=1,N=n,seed=1))
    validate_inputs(f['inputs'],dict(batch=1,N=n))
    assert f['expected'].dtype == torch.complex128
    assert diagnostics((f['expected'].real.float(),f['expected'].imag.float()),f)['passed']
    f['inputs']['real'][0,0] = .126
    with pytest.raises(ValueError): validate_inputs(f['inputs'],dict(batch=1,N=n))


@pytest.mark.parametrize('spec',[dict(batch=0,N=64),dict(batch=1,N=0),dict(batch=True,N=64),dict(batch=1,N=1025),dict(batch=1,N=32,common_complex_algorithm='two-stage')])
def test_rejections(spec):
    with pytest.raises(ValueError): dimensions(spec)


def test_frozen_nested_cases_through_cli(tmp_path):
    # Exercise CLI -> frozen nested schema -> fixture cache -> both validations.
    # CPU adapter explicitly substitutes only execution, with no scientific claims.
    module = tmp_path/'cpu_rfft.py'
    module.write_text('''from types import SimpleNamespace
from pathlib import Path
from rfft_contract import CASE_SOURCES, make_case, diagnostics, validate_inputs
class Adapter:
 case_sources=CASE_SOURCES
 make_case=staticmethod(make_case)
 def prepare(self,arm,spec,fixture,output):
  validate_inputs(fixture['inputs'],spec)
  return SimpleNamespace(run=lambda:(fixture['expected'].real.float(),fixture['expected'].imag.float()),sources=[Path(__file__),*CASE_SOURCES],expected_kernels=['cpu-contract'],metadata={'execution':'CPU contract'})
 def validate(self,arm,actual,fixture,spec):return diagnostics(actual,fixture)
def factory():return Adapter()
''')
    matrix = json.loads((EXAMPLES/'real_fft_cases.json').read_text())
    cases = matrix['cases'] + matrix['contract']['guards']
    assert [dimensions(x) for x in cases] == [(32,1024),(1,32),(24,512),(1,64),(2,64),(1,63)]
    provenance = tmp_path/'provenance.json'
    provenance.write_text('{"device":"CPU contract only"}')
    env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(tmp_path),str(EXAMPLES)]))
    for spec in cases:
        case = tmp_path/'case.json';case.write_text(json.dumps(spec))
        output = tmp_path/spec['case_id']
        command = [sys.executable,str(ROOT/'scripts/optimization_pair.py'),'--adapter','cpu_rfft:factory','--case',str(case),'--output',str(output),'--cache',str(tmp_path/'cache'),'--provenance',str(provenance),'--correctness-only']
        completed = subprocess.run(command,env=env,capture_output=True,text=True)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        report = json.loads((output/'result.json').read_text())
        assert report['fixture']['oracle_calls'] == 1
        assert report['counts'] == dict(profiled_calls=0,capture_calls=0,correctness_calls=2)
