import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from natural_before_tasks import TASKS, export, qualification_variants


def bundle(tmp_path, task):
    folder = tmp_path / task
    spec = export(task, folder)
    loader = importlib.util.spec_from_file_location('natural_runner', folder / 'runner.py')
    runner = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runner)
    return spec, runner, folder


@pytest.mark.parametrize('task', TASKS)
def test_standalone_bundle(task, tmp_path):
    spec, runner, folder = bundle(tmp_path, task)
    result = subprocess.run([sys.executable, str(folder / 'runner.py'), '--cpu-only', '--checks', '3'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 3 and all(record['correctness'] for record in records)
    _, expected = runner.inputs_and_reference(spec, 3)
    assert all(record['elements'] == expected.numel() for record in records)
    assert torch.isfinite(expected).all()
    with pytest.raises(FileExistsError):
        export(task, folder)
    for variant in qualification_variants(task):
        compile(variant, '<qualification>', 'exec')


@pytest.mark.parametrize('task', TASKS)
def test_each_runtime_input_contributes_and_bad_output_fails(task, tmp_path):
    spec, runner, _ = bundle(tmp_path, task)
    # Small instances exercise the same numerical interfaces without large CPU matmuls.
    spec.update(rows=4, width=16, table=64, m=8, n=8, k=16)
    inputs, expected = runner.inputs_and_reference(spec, 87, 1)
    _, same = runner.inputs_and_reference(spec, 87, 1)
    _, other = runner.inputs_and_reference(spec, 88, 1)
    assert torch.equal(expected, same) and not torch.equal(expected, other)
    runner.validate_output(spec, expected.clone(), expected)
    for key, value in inputs.items():
        changed = {name: tensor.clone() for name, tensor in inputs.items()}
        if key == 'index':
            changed[key] = (value + 1) % spec['table']
        elif value.dtype == torch.bool:
            changed[key] = ~value
        else:
            changed[key] = value + .5
        mutant = runner.reference(spec, changed)
        assert not torch.allclose(expected, mutant, rtol=.003, atol=.003), key
        with pytest.raises(AssertionError):
            runner.validate_output(spec, mutant, expected)


@pytest.mark.parametrize('task', ['task03', 'task04'])
def test_masked_empty_singleton_and_normal_rows(task, tmp_path):
    spec, runner, _ = bundle(tmp_path, task)
    spec.update(rows=8, width=16)
    for check in (0, 1):
        inputs, expected = runner.inputs_and_reference(spec, 9, check)
        assert torch.equal(expected[0], torch.zeros(16))
        assert expected[1, -1] == 1 and expected[1].sum() == 1
        assert torch.equal(expected[~inputs['selected']], torch.zeros_like(expected[~inputs['selected']]))
        torch.testing.assert_close(expected[2:].sum(-1), inputs['selected'][2:].any(-1).float())
        with pytest.raises(AssertionError):
            runner.validate_output(spec, torch.softmax(inputs['x'], -1), expected)


def test_invalid_launch_count_rejected_without_device(tmp_path):
    _, _, folder = bundle(tmp_path, 'task01')
    result = subprocess.run([sys.executable, str(folder / 'runner.py'), '--launches', '0'],
                            capture_output=True, text=True)
    assert result.returncode != 0
