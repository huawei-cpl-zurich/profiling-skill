import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def load(folder, name):
    path = Path(__file__).parents[1] / folder / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_counts_span_and_uneven_tail():
    m = load('benchmarks', 'a3_controlled_memory')
    matrix = m.cases(4097)
    assert len(matrix) == len({c['id'] for c in matrix}) == 17
    for c in matrix:
        addresses = list(m.indices(c))
        f = m.footprint(c)
        assert len(addresses) == c['n'] == len(set(addresses))
        assert max(addresses) < c['allocation_elements']
        assert f['accessed_span_bytes'] == (max(addresses) + 1) * 4
        assert f['grid'] == [(len(addresses) + 255) // 256]
        if c['id'].startswith('span/'):
            assert c['allocation_elements'] == 4097
            assert max(addresses) < 4097 <= max(addresses) + c['stride']
        else:
            assert c['n'] == 4097 and c['allocation_elements'] == 4097 * 64
    assert len({m.footprint(c)['allocation_bytes'] for c in matrix[:7]}) == 1
    with pytest.raises(ValueError):
        m.cases(0)


def test_runtime_indices_are_repeatable_permutations():
    m = load('benchmarks', 'a3_controlled_memory')
    for c in m.cases(70)[14:]:
        values = m.indices(c)
        assert values == m.indices(c)
        assert sorted(values) == list(range(70))
        assert m.footprint(c)['index_bytes'] == 280
        if c['pattern'] == 'local32':
            assert all(i // 32 == v // 32 for i, v in enumerate(values))


def test_compiler_target_identity_and_explicit_unavailable():
    m = load('benchmarks', 'a3_controlled_memory')
    triton = SimpleNamespace(__file__='/runtime/triton/__init__.py')
    direct = m.compiler_identity(SimpleNamespace(metadata={'target': 'actual-target'}), triton)
    assert direct['target'] == 'actual-target'
    assert direct['target_origin'] == 'compiled.metadata.target'
    triton.runtime = SimpleNamespace(driver=SimpleNamespace(
        active=SimpleNamespace(get_current_target=lambda: 'driver-target')))
    assert m.compiler_identity(SimpleNamespace(), triton)['target'] == 'driver-target'
    triton.runtime = None
    missing = m.compiler_identity(SimpleNamespace(), triton)
    assert missing['target'] is None and 'AttributeError' in missing['target_error']


def fake_backend(batch, monkeypatch, calls):
    def execute(command, folder, env, **kw):
        folder.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, '{"correctness":true}', '')
    def capture(command, folder, env, metric, selector=None):
        folder.mkdir(parents=True)
        calls.append((command[command.index('--case') + 1], metric, selector))
        return dict(metric=metric, basic_rows=[{'Kernel Name': 'a3_memory_probe_exact',
                                               'Duration(us)': '10'}])
    monkeypatch.setattr(batch.base, 'execute', execute)
    monkeypatch.setattr(batch.base, 'capture', capture)
    monkeypatch.setattr(batch, 'memory_capture',
                        lambda command, folder, env, selector: capture(command, folder, env, 'Memory', selector))


def test_interleaved_rounds_and_completed_resume(tmp_path, monkeypatch):
    b = load('scripts', 'a3_controlled_memory_batch')
    calls = []
    fake_backend(b, monkeypatch, calls)
    selected = b.cases(70)[:2]
    out = tmp_path / 'batch'
    result = b.interleaved(selected, out, {}, physical=7)
    assert len(calls) == 24  # Two discoveries plus 22 measured captures.
    assert calls[2:6] == [('count/1/70', 'BasicInfo', 'a3_memory_probe_exact'),
                          ('count/2/70', 'BasicInfo', 'a3_memory_probe_exact'),
                          ('count/2/70', 'BasicInfo', 'a3_memory_probe_exact'),
                          ('count/1/70', 'BasicInfo', 'a3_memory_probe_exact')]
    assert all(c['timing']['samples_us'] == [10.] * 5 for c in result['cells'].values())
    assert result['cells']['count/1/70']['useful_input_GB_per_s'] == .028
    assert len(result['capture_order']) == 22
    resumed = b.interleaved(selected, out, {}, physical=8, resume=True)
    assert resumed == result and len(calls) == 24
    assert all(c['physical_device'] == 7 for c in resumed['capture_order'])
    with pytest.raises(RuntimeError, match='configuration'):
        b.interleaved(selected[:1], out, {}, physical=8, resume=True)


def test_checkpoint_rejects_partial_and_changed_evidence(tmp_path):
    b = load('scripts', 'a3_controlled_memory_batch')
    folder = tmp_path / 'point'
    folder.mkdir()
    with pytest.raises(RuntimeError, match='incomplete'):
        b.checkpoint(folder, {}, lambda: {})
    fresh = tmp_path / 'fresh'
    def action():
        fresh.mkdir()
        (fresh / 'stdout.log').write_text('evidence')
        return {'value': 1}
    assert b.checkpoint(fresh, {'a': 1}, action) == {'value': 1}
    with pytest.raises(RuntimeError, match='request changed'):
        b.checkpoint(fresh, {'a': 2}, action)
    (fresh / 'stdout.log').write_text('changed')
    with pytest.raises(RuntimeError, match='evidence changed'):
        b.checkpoint(fresh, {'a': 1}, action)


@pytest.mark.parametrize('change', ['mutate', 'delete'])
def test_checkpoint_validates_sibling_compiler_artifacts(tmp_path, change):
    b = load('scripts', 'a3_controlled_memory_batch')
    artifact = tmp_path / 'compiler-evidence/adapter.txt'
    artifact.parent.mkdir()
    artifact.write_text('actual compiler IR')
    value = {'compiler': [{'path': str(artifact),
                            'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    phase = tmp_path / 'correctness'
    assert b.checkpoint(phase, {}, lambda: value) == value
    if change == 'mutate':
        artifact.write_text('different IR')
    else:
        artifact.unlink()
    with pytest.raises(RuntimeError, match='referenced compiler evidence'):
        b.checkpoint(phase, {}, lambda: value)


def test_checkpoint_rejects_invalid_compiler_pointer_before_completion(tmp_path):
    b = load('scripts', 'a3_controlled_memory_batch')
    phase = tmp_path / 'correctness'
    with pytest.raises(RuntimeError, match='referenced compiler evidence'):
        b.checkpoint(phase, {}, lambda: {'compiler': [{'path': str(tmp_path / 'missing'),
                                                       'sha256': '0' * 64}]})
    assert not (phase / 'completed.json').exists()


def test_memory_rows_raw_units_and_selector(tmp_path, monkeypatch):
    b = load('scripts', 'a3_controlled_memory_batch')
    folder = tmp_path / 'capture'
    report = folder / 'report'
    report.mkdir(parents=True)
    csv = report / 'Memory.csv'
    csv.write_text('Kernel Name,UnknownBandwidth\nexact,99\n')
    monkeypatch.setattr(b.base, 'capture', lambda *a: {'metric': 'Memory'})
    result = b.memory_capture([], folder, {}, 'exact')
    assert result['availability'] == 'available'
    assert result['exports'][0]['rows'][0]['UnknownBandwidth'] == '99'
    csv.write_text('Kernel Name,UnknownBandwidth\nother,99\n')
    with pytest.raises(RuntimeError, match='different selector'):
        b.memory_capture([], folder, {}, 'exact')


@pytest.mark.parametrize('message,unavailable', [('Memory is not supported', True),
                                               ('unsupported metric: Memory', True),
                                               ('Invalid value for --aic-metrics: Memory', True),
                                               ('invalid device memory', False),
                                               ('Memory allocation failed: invalid device', False),
                                               ('compile error', False),
                                               ('data collection failed', False)])
def test_memory_capability_not_runtime_failure(tmp_path, monkeypatch, message, unavailable):
    b = load('scripts', 'a3_controlled_memory_batch')
    (tmp_path / 'stderr.log').write_text(message)
    def fail(*args):
        raise RuntimeError('capture failure')
    monkeypatch.setattr(b.base, 'capture', fail)
    if unavailable:
        assert b.memory_capture([], tmp_path, {}, 'exact')['availability'] == 'unsupported'
    else:
        with pytest.raises(RuntimeError):
            b.memory_capture([], tmp_path, {}, 'exact')


def test_correctness_requires_explicit_success():
    b = load('scripts', 'a3_controlled_memory_batch')
    with pytest.raises(RuntimeError):
        b.checked_output(subprocess.CompletedProcess([], 0, '{}', ''))
    with pytest.raises(RuntimeError, match='process failed'):
        b.checked_output(subprocess.CompletedProcess([], 7, '{"correctness":true}', 'compile error'))


def test_runtime_probe_uses_real_subprocess_interface():
    b = load('scripts', 'a3_controlled_memory_batch')
    assert b.probe(None, expression='print("probe works")') == (True, '')
    ok, diagnostic = b.probe(None, expression='import sys; print("runtime broken", '
                             'file=sys.stderr); sys.exit(3)')
    assert not ok and diagnostic.strip() == 'runtime broken'
