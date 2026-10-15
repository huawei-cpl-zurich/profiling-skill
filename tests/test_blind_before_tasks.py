import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from blind_before_tasks import TASKS, export


def load_runner(folder):
    spec = importlib.util.spec_from_file_location('task_runner', folder / 'runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('task', TASKS)
def test_export_runs_outside_repository(task, tmp_path):
    folder = tmp_path / task
    spec = export(task, folder)
    result = subprocess.run([sys.executable, str(folder / 'runner.py'), '--cpu-only', '--checks', '3'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records and all(record['correctness'] is True for record in records)
    expected_count = spec['n'] ** 2 if task == 'task03' else spec['n'] * (2 if task == 'task06' else 1)
    assert all(record['elements'] == expected_count for record in records)
    extension = 'asc' if task == 'task06' else 'py'
    assert (folder / ('baseline.' + extension)).read_bytes() == (folder / ('candidate.' + extension)).read_bytes()
    with pytest.raises(FileExistsError):
        export(task, folder)


@pytest.mark.parametrize('task', TASKS)
def test_seeded_oracle_is_input_dependent(task, tmp_path):
    folder = tmp_path / task
    spec = export(task, folder)
    runner = load_runner(folder)
    _, expected = runner.inputs_and_reference(spec, 87)
    _, same = runner.inputs_and_reference(spec, 87)
    _, other = runner.inputs_and_reference(spec, 88)
    assert torch.equal(expected, same)
    assert not torch.equal(expected, other)


def test_runtime_contract_inputs(tmp_path):
    for task in ('task01', 'task02', 'task04'):
        spec = export(task, tmp_path / task)
        runner = load_runner(tmp_path / task)
        inputs0, expected0 = runner.inputs_and_reference(spec, 9, 0)
        inputs1, expected1 = runner.inputs_and_reference(spec, 9, 1)
        assert not torch.equal(expected0, expected1)
        if task == 'task01':
            assert inputs0['work'].item() == 64 and inputs1['work'].item() == 0
        if task == 'task02':
            assert torch.equal(expected1, inputs1['x'][inputs1['index'].long()])
            duplicate, expected2 = runner.inputs_and_reference(spec, 9, 2)
            assert duplicate['index'].unique().numel() <= 64
            assert torch.equal(expected2, duplicate['x'][duplicate['index'].long()])
        if task == 'task04':
            assert torch.equal(expected1, torch.where(inputs1['selected'], inputs1['x'] * 1.5 + 2, 0))


def test_invalid_launch_count_fails_before_device_import(tmp_path):
    export('task01', tmp_path / 'bundle')
    result = subprocess.run([sys.executable, str(tmp_path / 'bundle/runner.py'), '--launches', '0'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'must be positive' in result.stderr


def test_omitting_small_loop_contribution_is_rejected(tmp_path):
    spec = export('task01', tmp_path / 'bundle')
    runner = load_runner(tmp_path / 'bundle')
    inputs, expected = runner.inputs_and_reference(spec, 87)
    runner.validate_output(spec, expected.clone(), expected)
    with pytest.raises(AssertionError):
        runner.validate_output(spec, inputs['x'] * 1.5 + 2, expected)
