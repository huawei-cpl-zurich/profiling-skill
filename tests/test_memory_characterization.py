import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def load(folder, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / folder / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_and_footprints():
    memory = load('benchmarks', 'a3_profile_memory')
    cases = memory.cases()
    assert len(cases) == len({case['id'] for case in cases}) == 22
    for case in cases:
        addresses = memory.indices(case)
        footprint = memory.footprint(case, addresses)
        assert len(addresses) == case['n']
        assert footprint['distinct_input_bytes'] == len(set(addresses)) * 4
        assert footprint['accessed_span_bytes'] == (max(addresses) - min(addresses) + 1) * 4
        assert footprint['input_allocation_bytes'] >= footprint['accessed_span_bytes']
        assert footprint['index_read_bytes'] == (case['n'] * 4 if case['kind'] == 'irregular' else 0)
    hot = next(case for case in cases if case['id'] == 'irregular/hotset64/4096')
    assert memory.footprint(hot, memory.indices(hot))['distinct_input_bytes'] == 256


def test_indices_reproducible_and_local():
    memory = load('benchmarks', 'a3_profile_memory')
    case = {'kind': 'irregular', 'pattern': 'local32', 'n': 4096}
    addresses = memory.indices(case)
    assert addresses == memory.indices(case)
    assert sorted(addresses) == list(range(4096))
    assert all(i // 32 == value // 32 for i, value in enumerate(addresses))


def test_sample_summary_all_samples():
    batch = load('scripts', 'a3_memory_batch')
    summary = batch.summarize([1, 2, 3, 4, 100])
    assert summary['median_us'] == 3
    assert summary['samples_us'] == [1, 2, 3, 4, 100]
    assert summary['low_confidence'] is True
    assert batch.summarize([10] * 5)['low_confidence'] is False
    with pytest.raises(ValueError):
        batch.summarize([0, 2])


def test_parser_selector_and_partial(tmp_path):
    batch = load('scripts', 'a3_memory_batch')
    report = tmp_path / 'OpBasicInfo.csv'
    report.write_text('Kernel Name,Duration(us)\na3_memory_probe_x,2.5\nother,90\n')
    assert batch.basic_rows(tmp_path, 'a3_memory_probe_x')[0]['Duration(us)'] == '2.5'
    with pytest.raises(RuntimeError, match='selector'):
        batch.basic_rows(tmp_path, 'absent')
    with pytest.raises(RuntimeError, match='success'):
        batch.validate_capture(0, 'partial data collection failed')


def test_execute_keeps_command_and_error_logs(tmp_path):
    batch = load('scripts', 'a3_memory_batch')
    command = [sys.executable, '-c', 'import sys; print("compile failure",file=sys.stderr); sys.exit(7)']
    result = batch.execute(command, tmp_path / 'run', None)
    assert result.returncode == 7
    assert json.loads((tmp_path / 'run/command.json').read_text()) == command
    assert 'compile failure' in (tmp_path / 'run/stderr.log').read_text()
    assert json.loads((tmp_path / 'run/exit.json').read_text())['code'] == 7


def test_run_case_keeps_all_captures_and_exact_selector(tmp_path, monkeypatch):
    batch = load('scripts', 'a3_memory_batch')
    monkeypatch.setattr(batch, 'execute', lambda *a, **kw: subprocess.CompletedProcess(
        a, 0, '{"correctness":true}', ''))
    calls = []
    def capture(command, folder, env, metric, selector=None):
        calls.append((metric, selector))
        return {'basic_rows': [{'Kernel Name': 'native_exact', 'Duration(us)': str(len(calls))}],
                'pipe_rows': [{'block_id': '0', 'vec_ratio': '75'}]}
    monkeypatch.setattr(batch, 'capture', capture)
    result = batch.run_case(['/bin/native-test'], tmp_path / 'case', {}, 'native_exact')
    assert calls == [('BasicInfo', 'native_exact')] * 6 + [('PipeUtilization', 'native_exact')] * 3
    assert result['timing']['samples_us'] == [2, 3, 4, 5, 6]
    assert len(result['pipe_captures']) == 3
    assert json.loads((tmp_path / 'case/result.json').read_text())['selector'] == 'native_exact'


def test_partial_pipe_is_not_success(tmp_path, monkeypatch):
    batch = load('scripts', 'a3_memory_batch')
    def execute(command, folder, env):
        (folder / 'report').mkdir(parents=True)
        (folder / 'report/OpBasicInfo.csv').write_text('Kernel Name,Duration(us)\nexact,2\n')
        return subprocess.CompletedProcess(command, 0, batch.SUCCESS, '')
    monkeypatch.setattr(batch, 'execute', execute)
    with pytest.raises(RuntimeError, match='missing pipe rows'):
        batch.capture(['/bin/test'], tmp_path / 'capture', {}, 'PipeUtilization', 'exact')


def test_timeout_keeps_partial_diagnostics(tmp_path):
    batch = load('scripts', 'a3_memory_batch')
    with pytest.raises(subprocess.TimeoutExpired):
        batch.execute([sys.executable, '-c',
                       'import time; print("compile started",flush=True); time.sleep(30)'],
                      tmp_path / 'timeout', None, timeout=.2)
    assert 'compile started' in (tmp_path / 'timeout/stdout.log').read_text()
    assert json.loads((tmp_path / 'timeout/failure.json').read_text())['type'] == 'TimeoutExpired'


def test_zero_exit_without_correctness_rejected(tmp_path, monkeypatch):
    batch = load('scripts', 'a3_memory_batch')
    monkeypatch.setattr(batch, 'execute', lambda *a, **kw: subprocess.CompletedProcess(a, 0, '', ''))
    with pytest.raises(RuntimeError, match='correctness evidence'):
        batch.run_case(['/bin/true'], tmp_path / 'case', {}, 'exact')
