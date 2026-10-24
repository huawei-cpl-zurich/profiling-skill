"""Functional host/ABI checks use seeds and shapes outside the frozen study."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'examples/optimization-library/ascendc')]
import twoprod as dot


def case(rows=2, width=7, seed=192, distribution='uniform'):
    return dict(seed=seed, parameters=dict(rows=rows, width=width, distribution=distribution))


@pytest.mark.parametrize('rows,width', [(0, 8), (True, 8), (65, 8), (2, 0), (1, 4097)])
def test_invalid_dimensions_reject_before_fixture_or_binary(rows, width):
    with pytest.raises(ValueError):
        dot.make_case(case(rows, width))
    with pytest.raises(ValueError):
        dot.Adapter('/nonexistent').prepare('before', case(rows, width), {}, None)


def test_oracle_dtypes_and_reproducibility():
    spec = case()
    first, second = dot.make_case(spec), dot.make_case(spec)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert first['expected'].dtype == torch.float64
    assert first['a'].device.type == 'cpu'
    assert torch.equal(first['expected'], torch.stack([torch.dot(a.double(), b.double()) for a,b in zip(first['a'], first['b'])]))
    assert torch.equal(dot.expected_flags(first['a'], first['b']), torch.ones(2))


def test_certificate_zero_and_overflow_domains():
    # Includes zero x huge and conservative product-bound rejection.
    a = torch.tensor([[0., 0.], [1., -1.], [1e20, 0.], [1., 0.]], dtype=torch.float32)
    b = torch.tensor([[1e35, 0.], [0., 0.], [1e20, 0.], [2., 0.]], dtype=torch.float32)
    assert dot.expected_flags(a, b).tolist() == [0., 1., 0., 1.]


def test_guard_numerics_and_fallback():
    fixture = dot.make_case(case(3, 9, 193, 'mixed-safe-unsafe-cancellation'))
    assert dot.expected_flags(fixture['a'], fixture['b']).tolist() == [1., 0., 1.]
    assert fixture['expected'][0] == 1+2**-10
    assert fixture['expected'][2] == 2**-19
    assert torch.isfinite(fixture['expected']).all()
    dot.validate_fixture(fixture, case(3, 9, 193, 'mixed-safe-unsafe-cancellation'))


@pytest.mark.parametrize('corruption', ['shape', 'dtype', 'noncontiguous', 'nan', 'domain'])
def test_fixture_contract_rejects(corruption):
    fixture = dot.make_case(case())
    if corruption == 'shape': fixture['a'] = fixture['a'][:1]
    if corruption == 'dtype': fixture['a'] = fixture['a'].double()
    if corruption == 'noncontiguous': fixture['a'] = fixture['a'].t().contiguous().t()
    if corruption == 'nan': fixture['a'][0, 0] = float('nan')
    if corruption == 'domain': fixture['a'][0, 0] = 5
    with pytest.raises(ValueError): dot.validate_fixture(fixture, case())


def test_public_native_shape_abi(tmp_path):
    compiler = shutil.which('cc')
    if not compiler: pytest.skip('C compiler unavailable')
    source = tmp_path/'abi.c'; library = tmp_path/'abi.so'
    source.write_text('unsigned twoprod_rows(){return 2;} unsigned twoprod_width(){return 7;} unsigned twoprod_after(){return 1;}')
    subprocess.run([compiler, '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    dot.verify_binary(library, 2, 7, 'after')
    for rows, width, arm in [(1, 7, 'after'), (2, 8, 'after'), (2, 7, 'before')]:
        with pytest.raises(ValueError): dot.verify_binary(library, rows, width, arm)


def test_validation_observes_flags_canaries_and_immutable_input(monkeypatch):
    spec = case(); fixture = dot.make_case(spec)
    adapter = dot.Adapter(); storage = torch.full((68,), -8191.)
    output = storage[32:-32].view(2, 2)
    output[:, 0] = fixture['expected'].float(); output[:, 1] = 1
    item = type('Item', (), dict(inputs=[fixture['a'].clone(), fixture['b'].clone()]))()
    adapter.prepared['after'] = (item, storage)
    monkeypatch.setattr(torch, 'npu', type('Npu', (), dict(synchronize=lambda: None)), raising=False)
    good = adapter.validate('after', output, fixture, spec)
    assert good['passed'] and len(good['fp32_output_sha256']) == 64
    output[0, 1] = 0
    assert not adapter.validate('after', output, fixture, spec)['passed']
    output[0, 1] = 1; storage[0] = 0
    assert not adapter.validate('after', output, fixture, spec)['passed']
    storage[0] = -8191; item.inputs[0][0, 0] += 1
    assert not adapter.validate('after', output, fixture, spec)['passed']


def test_fixture_cache_reuses_oracle_and_detects_payload_mutation(tmp_path):
    from optimization_pair import fixture
    adapter = dot.Adapter(); spec = case(1, 11, 194)
    first, identity = fixture(adapter, spec, tmp_path, torch)
    assert identity['oracle_calls'] == 1
    adapter.make_case = lambda spec: pytest.fail('cache regenerated the CPU oracle')
    second, reused = fixture(adapter, spec, tmp_path, torch)
    assert reused['oracle_calls'] == 0 and torch.equal(first['expected'], second['expected'])
    (Path(identity['cache'])/'fixture.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='payload changed'):
        fixture(adapter, spec, tmp_path, torch)


def test_public_cli_rejects_bad_shape_before_native_launch(tmp_path):
    import os
    spec = tmp_path/'case.json'; spec.write_text(json.dumps(case(0, 7)))
    provenance = tmp_path/'provenance.json'; provenance.write_text('{}')
    output = tmp_path/'output'; cache = tmp_path/'cache'
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc'))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/optimization_pair.py'),
                             '--adapter', 'twoprod:factory', '--case', str(spec), '--output', str(output),
                             '--cache', str(cache), '--provenance', str(provenance), '--correctness-only'],
                            capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode != 0
    report = json.loads((output/'result.json').read_text())
    assert report['status'] == 'failed' and report['arms'] == {}
    assert not cache.exists()


def test_unqualified_capture_never_enters_forward_or_creates_output(tmp_path):
    def forbidden(): pytest.fail('unqualified capture entered a forward')
    with pytest.raises(RuntimeError, match='paired FP32 preflight is unqualified'):
        dot.Adapter.capture(forbidden, None, tmp_path/'capture')
    assert not (tmp_path/'capture').exists()


def test_public_cli_default_blocks_profile_and_correctness_only_remains(tmp_path):
    import os
    module = tmp_path/'cpu_twoprod.py'
    module.write_text('''from pathlib import Path
import torch
from twoprod import Adapter
class Prepared:
    expected_kernels = ['cpu_dot']
    metadata = {}
    sources = [__file__]
    def __init__(self, fixture): self.fixture = fixture
    def run(self): return (self.fixture['a'].double()*self.fixture['b'].double()).sum(1).float()
    def compilation_state(self): return {'cpu_test': True}
class CPU(Adapter):
    def prepare(self, arm, spec, fixture, output): return Prepared(fixture)
    def validate(self, arm, actual, fixture, spec):
        return {'passed': bool(torch.allclose(actual.double(), fixture['expected'], atol=1e-6, rtol=1e-6))}
def factory(): return CPU()
''')
    spec = tmp_path/'case.json'; spec.write_text(json.dumps(case(1, 13, 195)))
    provenance = tmp_path/'provenance.json'; provenance.write_text('{}')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT/'examples/optimization-library/ascendc')]))
    for mode in ('default', 'correctness-only'):
        output = tmp_path/mode
        command = [sys.executable, str(ROOT/'scripts/optimization_pair.py'), '--adapter', 'cpu_twoprod:factory',
                   '--case', str(spec), '--output', str(output), '--cache', str(tmp_path/'cache'), '--provenance', str(provenance)]
        if mode == 'correctness-only': command.append('--correctness-only')
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
        report = json.loads((output/'result.json').read_text())
        assert all(row['profiled_calls'] == 0 for row in report['arms'].values())
        assert not list(output.glob('*/capture'))
        if mode == 'default':
            assert result.returncode != 0 and report['status'] == 'failed'
            assert all('paired FP32 preflight is unqualified' in row['error']['message'] for row in report['arms'].values())
        else:
            assert result.returncode == 0 and report['status'] == 'complete'
            assert report['fixture']['oracle_calls'] == 0


def test_standalone_import_from_outside_repository(tmp_path):
    import os
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'examples/optimization-library/ascendc'))
    result = subprocess.run(
        [sys.executable, '-c',
         "import twoprod; assert callable(twoprod.compile_all); "
         "assert callable(twoprod.audit_placement); "
         "assert twoprod.dimensions({'rows': 1, 'width': 3}) == (1, 3)"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
