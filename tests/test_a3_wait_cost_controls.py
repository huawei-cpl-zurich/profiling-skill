import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import a3_wait_cost_batch as controls


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []

    def build(folder, *, source):
        folder.mkdir()
        exe = folder / 'native'
        exe.write_bytes(b'compiled self-contained source')
        return exe

    def execute(argv, folder, env, *, timeout):
        calls.append(('execute', argv))
        folder.mkdir(parents=True)
        record = dict(correctness=True, mismatches=0, warmup=3 if argv[-1] == '1' else 0,
                      launch_count=7 if argv[-1] == '1' else 1,
                      samples_us=[1, 2, 3, 4, 5, 6, 7] if argv[-1] == '1' else [])
        (folder / 'stdout.log').write_text(json.dumps(record))
        return subprocess.CompletedProcess(argv, 0, json.dumps(record), '')

    def capture(argv, folder, env, metric, selector):
        calls.append(('capture', argv, metric))
        assert argv[-1] == '0'  # The actual application never profiles timing/warmup loops.
        folder.mkdir(parents=True)
        (folder / 'retained.log').write_text(metric)
        return dict(metric=metric, selector=selector,
                    basic_rows=[{'Kernel Name': selector, 'Duration(us)': '2.5'}],
                    **({'availability': 'unsupported'} if metric == 'Memory' else {}))

    monkeypatch.setattr(controls.a3_native_build, 'build', build)
    monkeypatch.setattr(controls.base, 'execute', execute)
    monkeypatch.setattr(controls.base, 'capture', capture)
    monkeypatch.setattr(controls, 'memory_capture',
                        lambda argv, folder, env, selector: capture(argv, folder, env, 'Memory', selector))
    return calls


def test_equal_work_oracle_and_sweep():
    assert len(controls.cases()) == 32
    for start in range(0, 32, 4):
        transfer, compute, serial, overlap = controls.cases()[start:start + 4]
        a, b = controls.expected(transfer), controls.expected(compute)
        assert controls.expected(serial) == controls.expected(overlap)
        assert controls.expected(serial) == [x + y for x, y in zip(a, b)]
        assert len(a) == transfer['n'] * 2
        assert len(set(a[:transfer['n']])) == 17
        assert set(b[compute['n']:]) == {compute['rounds'] * compute['repetitions']}


def test_capture_counts_interleaving_resume_and_provenance(tmp_path, fake_runtime):
    selected = controls.cases(True)
    output = tmp_path / 'run'
    result = controls.run(selected, output, {}, physical=7)
    assert len(result['capture_order']) == 44
    for cell in result['cells'].values():
        assert cell['event_timing']['median_us'] == 4
        assert cell['profile_timing']['samples_us'] == [2.5] * 5
        assert cell['event_evidence']['physical_device'] == 7
        assert len(cell['captures']) == 11
        assert cell['captures'][-1]['availability'] == 'unsupported'
    assert result['capture_order'][0]['case'].endswith('/transfer')
    assert result['capture_order'][4]['case'].endswith('/compute')
    comparison = result['comparisons']['1024/1/1']
    assert comparison['paired_speedup'] == 1
    assert comparison['isolated_stage_sum_us'] == 8
    old_calls = list(fake_runtime)
    resumed = controls.run(selected, output, {}, physical=9, resume=True)
    assert fake_runtime == old_calls
    assert resumed['cells'][selected[0]['id']]['event_evidence']['physical_device'] == 7
    assert resumed['capture_order'][0]['physical_device'] == 7


def test_failure_stops_before_profiling_and_keeps_logs(tmp_path, fake_runtime, monkeypatch):
    execute = controls.base.execute

    def failed(*args, **kwargs):
        result = execute(*args, **kwargs)
        return subprocess.CompletedProcess(result.args, 1, result.stdout, 'runtime failure')

    monkeypatch.setattr(controls.base, 'execute', failed)
    with pytest.raises(RuntimeError, match='workload process failed'):
        controls.run(controls.cases(True), tmp_path / 'run', {}, physical=0)
    assert not any(c[0] == 'capture' for c in fake_runtime)
    assert list((tmp_path / 'run').rglob('stdout.log'))


def test_bad_event_samples_are_not_zero_filled(tmp_path, fake_runtime, monkeypatch):
    execute = controls.base.execute

    def invalid(*args, **kwargs):
        result = execute(*args, **kwargs)
        record = json.loads(result.stdout)
        if record['samples_us']:
            record['samples_us'][0] = 0
        return subprocess.CompletedProcess(result.args, 0, json.dumps(record), '')

    monkeypatch.setattr(controls.base, 'execute', invalid)
    with pytest.raises(ValueError, match='finite and positive'):
        controls.run(controls.cases(True), tmp_path / 'run', {}, physical=0)


def test_resume_rejects_source_config_binary_and_evidence_changes(tmp_path, fake_runtime):
    output = tmp_path / 'run'
    controls.run(controls.cases(True), output, {}, physical=0)
    with pytest.raises(RuntimeError, match='configuration/source mismatch'):
        controls.run(controls.cases(), output, {}, physical=0, resume=True)
    exe = output / 'build/native'
    exe.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='executable changed'):
        controls.run(controls.cases(True), output, {}, physical=0, resume=True)
    exe.write_bytes(b'compiled self-contained source')
    next(output.rglob('retained.log')).write_text('changed')
    with pytest.raises(RuntimeError, match='evidence changed'):
        controls.run(controls.cases(True), output, {}, physical=0, resume=True)


def test_cli_rejects_invalid_group_before_device_probe(tmp_path):
    result = subprocess.run([sys.executable, str(Path(controls.__file__)), '--output', str(tmp_path),
                             '--canary', '--group', '8192/32/16'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'unknown or excluded group' in result.stderr


@pytest.mark.parametrize('noisy_mode', ['serialized', 'overlap', 'transfer', None])
def test_pair_confidence_preserves_medians_and_noise(noisy_mode):
    cells = {}
    originals = {}
    for case in controls.cases(True):
        samples = [5.94, 27.24, 7.36, 5.76, 5.88, 5.2, 29.46] \
            if case['mode'] == noisy_mode else [6.0] * 7
        cells[case['id']] = dict(case=case, event_timing=controls.base.summarize(samples))
        originals[case['id']] = list(samples)
    comparison = controls.conclusions(cells)['1024/1/1']
    noisy_pair = noisy_mode in ('serialized', 'overlap')
    assert comparison['low_confidence'] is noisy_pair
    assert (comparison['status'] == 'inconclusive_noisy_event_timing') is noisy_pair
    assert comparison['event_low_confidence'] == {m: m == noisy_mode for m in controls.MODES}
    assert comparison['paired_speedup'] == comparison['event_medians_us']['serialized'] / \
        comparison['event_medians_us']['overlap']
    assert {key: cell['event_timing']['samples_us'] for key, cell in cells.items()} == originals
