import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from blind_native_wait_tasks import TASKS, export


def load_runner(folder):
    sys.path.insert(0, str(folder))
    definition = importlib.util.spec_from_file_location('work_oracle', folder/'runner.py')
    runner = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    return runner


@pytest.mark.parametrize('task', TASKS)
def test_exported_oracle_runs_standalone_and_changes_with_seed(tmp_path, task):
    folder = tmp_path/task
    spec = export(task, folder)
    records = []
    for seed in (7, 8):
        ran = subprocess.run([sys.executable, str(folder/'runner.py'), '--cpu-only',
                              '--checks', '4', '--seed', str(seed)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
        assert ran.returncode == 0, ran.stderr
        records.append([json.loads(line) for line in ran.stdout.splitlines()])
    assert len(records[0]) == 4
    assert {r['variant'] for r in records[0]} == set(spec['variants'])
    assert all(a['expected'] != b['expected'] for a, b in zip(*records))
    for record in records[0]:
        v = spec['variants'][record['variant']]
        assert len(record['expected']) == v['n']*(4 if task == 'intercore' else 2)
        assert record['oracle_only'] is True  # CPU formula is not native correctness evidence.
    with pytest.raises(FileExistsError):
        export(task, folder)


def test_bad_variant_fails_before_compilation(tmp_path):
    folder = tmp_path/'task'
    export('phase', folder)
    result = subprocess.run([sys.executable, str(folder/'runner.py'), '--variant', 'missing'],
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'unknown variant' in result.stderr


def test_native_runner_compiles_candidate_and_checks_all_variants(tmp_path, monkeypatch):
    folder = tmp_path/'task'
    export('phase', folder)
    sys.path.insert(0, str(folder))
    definition = importlib.util.spec_from_file_location('wait_runner', folder/'runner.py')
    runner = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(runner)
    calls = []
    def build(output, *, source, extra_flags):
        calls.append(('build', source, extra_flags))
        output.mkdir(parents=True)
        (output/'a3_sync_control').touch()
    monkeypatch.setattr(runner, 'build', build)
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kw: calls.append(('run', command, kw)))
    monkeypatch.setattr(sys, 'argv', ['runner', '--checks', '4', '--event-timing'])
    runner.main()
    assert calls[0][0] == 'build'
    assert '-DIMPLEMENTATION="candidate.asc"' in calls[0][2]
    assert '-DTASK_KIND=3' in calls[0][2]
    commands = [c[1] for c in calls[1:]]
    assert [int(c[2]) for c in commands] == [1024, 8192, 1024, 8192]
    assert all(c[-1] == '1' and c[-3] == '1' for c in commands)
    monkeypatch.setattr(sys, 'argv', ['runner', '--variant', 'long'])
    runner.main()
    assert calls[-1][1][-1] == '0'  # Normal profiler application has no event loop.
    assert len([c for c in calls if c[0] == 'build']) == 1


@pytest.mark.parametrize('task', TASKS)
def test_old_constant_work_collapse_fails_seeded_oracle(tmp_path, task):
    folder = tmp_path/task
    spec = export(task, folder)
    runner = load_runner(folder)
    for variant in spec['variants'].values():
        kind, n = spec['native_wait_kind'], variant['n']
        actual = np.array(runner.oracle(kind, variant, 17), dtype=np.float32)
        collapsed = actual.copy()
        if kind == 1:
            values = runner.seeded_inputs(kind, variant, 17)
            for block in range(2):
                peer = np.add(values[1-block], np.float32(32 if variant['imbalanced'] and block == 1 else 0))
                collapsed[block*2*n+n:(block+1)*2*n] = np.add(peer, np.float32(variant['rounds']))
        else:
            collapsed[n:] = np.float32(variant['rounds']*variant['repetitions'])
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(collapsed, actual)


def test_phase_every_payload_and_each_cores_work_seed_affect_outputs(tmp_path, monkeypatch):
    folder = tmp_path/'phase'
    spec = export('phase', folder)
    runner = load_runner(folder)
    variant = spec['variants']['long']
    original = runner.seeded_inputs(3, variant, 71)
    expected = np.array(runner.oracle(3, variant, 71), dtype=np.float32)
    # Keep the last payload/seed unchanged; dropping an earlier copy is not equivalent.
    changed = original.copy()
    changed[0] = np.float32(.125)
    monkeypatch.setattr(runner, 'seeded_inputs', lambda *args: changed)
    ignored = np.array(runner.oracle(3, variant, 71), dtype=np.float32)
    n = variant['n']
    np.testing.assert_array_equal(expected[:n], ignored[:n])
    assert np.any(expected[n:] != ignored[n:])
    inter = tmp_path/'intercore'
    spec = export('intercore', inter)
    runner = load_runner(inter)
    variant = spec['variants']['default']
    original = runner.seeded_inputs(1, variant, 19)
    assert np.any(original[2] != original[3])
    expected = np.array(runner.oracle(1, variant, 19), dtype=np.float32)
    changed = original.copy()
    changed[3] = changed[2]  # Reusing the other core's recurrence is observably wrong.
    monkeypatch.setattr(runner, 'seeded_inputs', lambda *args: changed)
    reused = np.array(runner.oracle(1, variant, 19), dtype=np.float32)
    assert np.any(expected != reused)
