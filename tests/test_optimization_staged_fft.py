"""Functional contract checks; actual device kernels are verified on A3."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples/optimization-library/triton'))
from fft_contract import coefficients, diagnostics, dimensions, factors, make_case, validate_inputs
from staged_fft import ComplexFFT


def test_ordering_oracle():
    fixture = make_case(dict(batch=2, N=64, seed=4291, distribution='ordering'))
    expected = fixture['expected']
    k = torch.arange(64, dtype=torch.float64)
    analytic = .125 * torch.exp(-2j * torch.pi * 3 * k / 64)
    torch.testing.assert_close(expected[0], analytic, atol=1e-14, rtol=1e-14)
    assert expected[1].abs().argmax().item() == 7
    assert abs(expected[1, 7].real - 8) < 1e-6
    assert diagnostics((expected.real.float(), expected.imag.float()), fixture)['passed']
    assert not diagnostics((expected.real.flip(1).float(), expected.imag.flip(1).float()), fixture)['passed']


@pytest.mark.parametrize('n', [17, 32, 64, 256, 512])
def test_fixture_and_fixed_error_gate(n):
    spec = dict(batch=1, N=n, seed=4201)
    f = make_case(spec)
    validate_inputs(f['inputs'], spec)
    assert f['expected'].dtype == torch.complex128
    real, imag = f['expected'].real.float(), f['expected'].imag.float()
    assert diagnostics((real, imag), f)['passed']
    assert not diagnostics((real + 1., imag), f)['passed']
    assert not diagnostics((real.double(), imag), f)['passed']
    assert not diagnostics((real.fill_(float('nan')), imag), f)['passed']


def test_tables_and_dispatch():
    c, s = coefficients(64)
    assert c.dtype == s.dtype == torch.float32
    assert c[0] == 1 and s[0] == 0
    assert abs(c[16]) < 1e-7 and s[16] == -1
    assert factors(32) == (4, 8) and factors(512) == (16, 32)
    assert factors(17) is None and factors(16) is None


@pytest.mark.parametrize('b,n', [(0,64),(1,0),(1,513),(True,64),(1,3.5)])
def test_shape_rejections(b, n):
    with pytest.raises(ValueError): dimensions(dict(batch=b, N=n))


def test_input_rejections():
    spec = dict(batch=2, N=64, seed=4204)
    f = make_case(spec)
    f['inputs']['real'][0,0] = .126
    with pytest.raises(ValueError): validate_inputs(f['inputs'], spec)
    with pytest.raises(ValueError): ComplexFFT(torch.zeros(1,64), torch.zeros(1,64), True)


def test_compile_only_and_triton_identity():
    launches = []
    binary = b'compiled-npubin'
    artifact = SimpleNamespace(hash='a'*64, asm={'npubin': binary})
    class Compiler:
        def warmup(self, *args, **options):
            launches.append((args, options))
            return artifact
        def __getitem__(self, grid):
            raise AssertionError('compile must not launch')
    op = ComplexFFT.__new__(ComplexFFT)
    op.calls = [(Compiler(), (2,4), (1,2))]
    state = op.compile()
    assert len(launches) == 1 and launches[0][1]['enable_fp_fusion'] is False
    assert state == [dict(hash='a'*64, binaries={'npubin': hashlib.sha256(binary).hexdigest()})]
