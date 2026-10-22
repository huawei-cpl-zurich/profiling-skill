"""Observable contract tests; normalized fixtures are not vendor schema claims."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / 'skills' /
          'ascend-model-benchmarking/scripts/profile_accounting.py')


@pytest.fixture
def accounting():
    spec = importlib.util.spec_from_file_location('profile_accounting', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(identity, correlation='launch', name='same_kernel', kind='device',
         start=0, duration=2):
    return dict(task_id=identity, correlation_id=correlation, name=name,
                kind=kind, start_us=start, duration_us=duration)


def iteration(identity='i0', correlations=('launch',), expected=('a',)):
    return dict(iteration_id=identity, correlation_ids=list(correlations),
                expected_task_ids=list(expected))


def account(module, iterations=None, tasks=None, **kwargs):
    return module.account_profile(
        iterations if iterations is not None else [iteration()],
        tasks if tasks is not None else [task('a')],
        capture_complete=kwargs.pop('capture_complete', True), **kwargs)


def test_components_multiple_launches_and_overlap_are_not_elapsed(accounting):
    result = account(accounting, [iteration(correlations=('l1', 'l2'),
                                           expected=('a', 'b', 'c', 'd'))], [
        task('a', 'l1', 'compute_mix_aic', start=0, duration=8),
        task('b', 'l1', 'compute_mix_aiv', start=2, duration=8),
        task('c', 'l2', 'cast', start=10, duration=3),
        task('d', 'l2', 'cpu', kind='aicpu', duration=5)])
    row = result['iterations'][0]
    assert row['device_busy_us'] == 19
    assert row['aicpu_busy_us'] == 5
    assert row['device_active_union_us'] == 13
    assert row['device_overlap_us'] == 6
    assert 'latency_us' not in row
    assert len(result['tasks']) == 4


@pytest.mark.parametrize('origin', [10**16, float(10**16)])
def test_small_duration_at_large_absolute_origin_is_not_lost(accounting, origin):
    result = account(accounting, tasks=[task('a', start=origin, duration=1)])
    row = result['iterations'][0]
    assert row['device_busy_us'] == 1
    assert row['device_active_union_us'] == 1
    assert row['device_overlap_us'] == 0
    assert json.loads(json.dumps(result))['tasks'][0]['start_us'] == origin


def test_large_integer_origin_preserves_adjacent_and_overlapping_intervals(accounting):
    origin = 10**16
    result = account(accounting, [iteration(expected=('a', 'b', 'c'))], [
        task('a', start=origin, duration=2),
        task('b', start=origin + 1, duration=2),
        task('c', start=origin + 3, duration=1)])
    row = result['iterations'][0]
    assert row['device_busy_us'] == 5
    assert row['device_active_union_us'] == 4
    assert row['device_overlap_us'] == 1
    assert result['tasks'][1]['start_us'] == origin + 1
    assert isinstance(json.loads(json.dumps(result))['tasks'][1]['start_us'], int)


def test_same_names_do_not_determine_iteration_or_launch_count(accounting):
    result = account(accounting, [iteration('first', ('x',), ('a',)),
                                 iteration('second', ('y',), ('b', 'c'))], [
        task('c', 'y', duration=5), task('a', 'x', duration=2),
        task('b', 'y', duration=3)])
    assert [r['device_busy_us'] for r in result['iterations']] == [2, 8]
    assert [r['task_count'] for r in result['iterations']] == [1, 2]


def test_filter_flags_only_and_does_not_mutate_source(accounting):
    tasks = [task(str(i), duration=d) for i, d in enumerate([1, 1, 1, 100])]
    original = copy.deepcopy(tasks)
    result = account(accounting, [iteration(expected=tuple(str(i) for i in range(4)))], tasks)
    assert result['iterations'][0]['device_busy_us'] == 103
    assert result['tasks'][-1]['diagnostic_flags'] == ['duration_above_median_threshold']
    assert result['tasks'][-1]['diagnostic_threshold_us'] == 50
    assert len(result['tasks']) == 4
    assert tasks == original


def test_recurring_legitimate_slow_casts_remain_counted(accounting):
    tasks = [task('a', name='cast', duration=500),
             task('b', name='cast', duration=500),
             task('c', name='dot', duration=1)]
    result = account(accounting, [iteration(expected=('a', 'b', 'c'))], tasks)
    assert result['iterations'][0]['device_busy_us'] == 1001
    assert all(not row['diagnostic_flags'] for row in result['tasks'])


def test_filter_absolute_floor(accounting):
    tasks = [task(str(i), duration=d) for i, d in enumerate([0, 0, 0, 11])]
    result = account(accounting, [iteration(expected=('0', '1', '2', '3'))], tasks)
    assert result['tasks'][-1]['diagnostic_threshold_us'] == 10
    assert result['iterations'][0]['device_busy_us'] == 11


@pytest.mark.parametrize('complete', [False, None, 'true', 1])
def test_incomplete_capture_is_not_measurement(accounting, complete):
    with pytest.raises(accounting.AccountingError, match='complete'):
        account(accounting, capture_complete=complete)


@pytest.mark.parametrize('change,match', [
    (lambda rows: rows.append(task('a')), 'duplicate task'),
    (lambda rows: rows[0].update(correlation_id='unknown'), 'unknown correlation'),
    (lambda rows: rows[0].pop('correlation_id'), 'correlation_id'),
    (lambda rows: rows[0].update(duration_us=-1), 'duration_us'),
    (lambda rows: rows[0].update(duration_us=float('nan')), 'duration_us'),
    (lambda rows: rows[0].update(duration_us=True), 'duration_us'),
    (lambda rows: rows[0].update(kind='host'), 'kind'),
])
def test_bad_tasks_fail_clearly(accounting, change, match):
    rows = [task('a')]
    change(rows)
    with pytest.raises(accounting.AccountingError, match=match):
        account(accounting, tasks=rows)


def test_missing_expected_task_and_wrong_iteration_rejected(accounting):
    with pytest.raises(accounting.AccountingError, match='inventory'):
        account(accounting, [iteration(expected=('a', 'missing'))])
    with pytest.raises(accounting.AccountingError, match='inventory'):
        account(accounting, [iteration('one', ('x',), ('a',)),
                             iteration('two', ('y',), ('b',))],
                [task('a', 'y'), task('b', 'x')])


def test_ambiguous_correlation_is_rejected_even_with_different_names(accounting):
    with pytest.raises(accounting.AccountingError, match='ambiguous correlation'):
        account(accounting, [iteration('one'), iteration('two', expected=('b',))],
                [task('a'), task('b', name='different')])


@pytest.mark.parametrize('iterations', [[],
    [iteration(), iteration()], [iteration(correlations=())]])
def test_empty_or_duplicate_iteration_contract_rejected(accounting, iterations):
    with pytest.raises(accounting.AccountingError):
        account(accounting, iterations=iterations)


def test_inventory_is_optional_not_claimed_as_completeness_proof(accounting):
    marker = iteration()
    del marker['expected_task_ids']
    result = account(accounting, [marker])
    assert result['iterations'][0]['device_busy_us'] == 2
    assert result['iterations'][0]['inventory_checked'] is False


def test_explicit_zero_task_iteration_is_preserved(accounting):
    result = account(accounting, [iteration(expected=())], [])
    assert result['iterations'][0]['device_busy_us'] == 0
    assert result['iterations'][0]['task_count'] == 0


def test_cli_emits_json_and_fails_nonzero_for_incomplete(tmp_path):
    path = tmp_path / 'capture.json'
    payload = dict(iterations=[iteration()], tasks=[task('a')], capture_complete=True)
    path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['iterations'][0]['device_busy_us'] == 2
    payload['capture_complete'] = False
    path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'complete' in json.loads(result.stderr)['error']
