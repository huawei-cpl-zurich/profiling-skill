"""Functional history contracts and a multi-round controller integration."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyzer_history as history
from actor_analyzer_controller import Campaign, digest


def summary(source='a', duration=10, activity=20, selector='matmul'):
    return dict(source_sha256=source, state='completed', correctness=True,
        cases=['shape1'], timing={'mean_us': duration, 'samples': 3},
        captures=[dict(selector=selector, pipe_rows=[{'mac_ratio': activity,
            'Device_id': 4, 'Start Time(us)': 1234}], basic_rows=[])])


def receipt(device=0):
    return dict(target='gz-a3', runtime='native', physical_device=device,
                request=dict(verb='pipes', pipe_captures=3, implementation='candidate'))


def report(ctx):
    return dict(contract_version=2, candidate_commit=ctx['candidate_commit'],
        source_sha256=ctx['source_sha256'], profiling=dict(status='unavailable', reason='capture failed'),
        evidence=[], observations=[], bottlenecks=[],
        optimizations=[dict(status='none', description='No optimization established')],
        utilization=[dict(status='unavailable', description='No usable capture', cases=['unknown'], kernels=['unknown'])],
        ceilings=[dict(resource='memory', status='not_established', basis='Missing measured bandwidth', assumptions=[])],
        unknowns=['No profile'], comparison=dict(status=ctx['status'],
            rounds=[ctx['previous_distinct']] if ctx['previous_distinct'] else [],
            interpretation='Missing metrics prevent a performance conclusion'))


def test_pair_nonruntime_metrics_and_same_node_devices():
    a = history.measurements(summary(), receipt(1), 'workload')
    b = history.measurements(summary(duration=8, activity=40), receipt(5), 'workload')
    result = history.pair([b], [a])
    pairs = {item['metric']: item for item in result['pairs']}
    assert pairs['timing/mean_us']['percent'] == -20
    assert pairs['kernel/matmul/shape1/pipe_rows/all/mac_ratio']['delta'] == 20
    assert not any('Device' in key or 'Start Time' in key or 'samples' in key for key in pairs)
    b['context']['runtime'] = 'different'
    assert not history.pair([b], [a])['pairs']
    b['context']['runtime'] = None
    a['context']['runtime'] = None
    assert not history.pair([b], [a])['pairs']


def test_decomposition_and_unattributed_cases_are_not_averaged():
    a = history.measurements(summary(), receipt(), 'workload')
    b = history.measurements(summary(selector='fused'), receipt(), 'workload')
    result = history.pair([b], [a])
    assert [pair['metric'] for pair in result['pairs']] == ['timing/mean_us']
    assert result['gaps']
    value = summary()
    value['cases'] = ['shape1', 'shape2']
    assert not any(key.startswith('kernel/') for key in history.measurements(value, receipt())['metrics'])


def test_selection_and_diff_attribution(tmp_path):
    first = tmp_path / 'round-1'
    first.mkdir()
    (first / 'candidate.py').write_text('old\n')
    (first / 'record.json').write_text(json.dumps(dict(round=1, source_sha256='old', acquisitions=[])))
    ctx = history.context(tmp_path, b'new\n', 'a'*40, 2)
    value = report(ctx)
    history.validate_round(value, ctx)
    for mutation in (lambda v: v['comparison'].update(rounds=[]),
                     lambda v: v['comparison'].update(rounds=[99]),
                     lambda v: v.update(source_sha256='fake'),
                     lambda v: v['comparison'].update(pairs=[]),
                     lambda v: v.update(optimizations=[dict(status='observed', description='fake', diff_lines=['+invented'])])):
        invalid = copy.deepcopy(value)
        mutation(invalid)
        with pytest.raises(ValueError): history.validate_round(invalid, ctx)
    value['optimizations'] = [dict(status='observed', description='Changed expression', diff_lines=['+new'])]
    history.validate_round(value, ctx)
    assert history.finalize(value, ctx, tmp_path, [])['comparison']['results'][0]['gaps']


@pytest.mark.parametrize('cite_controller', [True, False])
def test_controller_retains_sources_and_delivers_current_comparisons(tmp_path, cite_controller):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('value = 0\n')
    (bundle / 'task.json').write_text(json.dumps(dict(public_files=['candidate.py'])))
    observed = []
    def runner(campaign, phase, role, budget, commit):
        ctx = json.loads((campaign.history / 'context.json').read_text())
        index = int(phase.name.split('-')[1])
        if role == 'actor':
            if index > 1:
                prior = json.loads((campaign.history / f'round-{index-1}/analysis.json').read_text())
                observed.append(prior['comparison'])
                assert ctx['latest_analysis']['analysis_valid']
                assert (campaign.history / 'round-1/candidate.py').read_text() == 'value = 1\n'
            (campaign.workspace / 'candidate.py').write_text(f'value = {min(index,2)}\n')
            campaign.qualify(phase, dict(state='completed', correctness=True,
                source_sha256=digest(campaign.workspace / 'candidate.py')))
        else:
            value = report(ctx)
            data = summary(ctx['source_sha256'], 12-index, 20*index)
            # Actual production layout: public summary has no receipt; tool-state is private.
            folder = phase / 'candidate-snapshot/evidence/call/results'
            folder.mkdir(parents=True)
            (folder / 'summary.json').write_text(json.dumps(data))
            private = phase / 'tool-state/call'
            private.mkdir(parents=True)
            acquisition = receipt(index)
            acquisition['request']['source_sha256'] = ctx['source_sha256']
            (private / 'receipt.json').write_text(json.dumps(acquisition))
            path = 'evidence/call/results/summary.json'
            value.update(profiling=dict(status='acquired'),
                evidence=[dict(path=path, kind='profile', kernels=['matmul'], cases=['shape1'])],
                bottlenecks=[dict(finding='Memory movement likely limits compute', confidence='low', evidence=[path])],
                utilization=[dict(status='measured', description='MAC activity', cases=['shape1'], kernels=['matmul'], evidence=[path])])
            # A realistic analyzer may keep two successful captures of the same
            # candidate; the next round must still receive one pair per metric.
            second = phase / 'candidate-snapshot/evidence/call2/results'
            second.mkdir(parents=True)
            data['timing']['mean_us'] += 2
            (second / 'summary.json').write_text(json.dumps(data))
            second_private = phase / 'tool-state/call2'
            second_private.mkdir(parents=True)
            acquisition['call_id'] = 'call2'
            (second_private / 'receipt.json').write_text(json.dumps(acquisition))
            required = dict(source_sha256=ctx['source_sha256'], state='completed',
                evidence='evidence/call2/results/summary.json', command_dispatched=True,
                profile_attempt_arguments=['pipes'])
            (phase / 'required-profile.json').write_text(json.dumps(required))
            if cite_controller:
                value['evidence'].append(dict(path=required['evidence'],
                    kind='profile', kernels=['matmul'], cases=['shape1']))
            (phase / 'analysis.json').write_text(json.dumps(value))
        return dict(exit_code=0)
    campaign = Campaign(tmp_path / 'run', bundle, {}, ['unused'], 'task', runner=runner, rounds=3 if cite_controller else 1)
    state = campaign.run()
    retained = json.loads((campaign.history / 'round-1/record.json').read_text())
    assert retained['required_profile']['evidence'] == 'evidence/call2/results/summary.json'
    assert (campaign.workspace / 'experiments/round-1-analyzer/required-profile.json').is_file()
    if not cite_controller:
        assert not state['phases']['round-1-analyzer']['analysis_valid']
        assert 'controller-acquired profile' in state['phases']['round-1-analyzer']['analysis_error']
        return
    assert all(state['phases'][f'round-{index}-analyzer']['analysis_valid'] for index in (1,2,3))
    assert observed[0]['status'] == 'first'
    assert observed[1]['results'][0]['pairs']
    assert all(pair['current_acquisitions'] == pair['previous_acquisitions'] == 2
               for pair in observed[1]['results'][0]['pairs'])
    assert len({pair['metric'] for pair in observed[1]['results'][0]['pairs']}) == len(observed[1]['results'][0]['pairs'])
    assert observed[1]['results'][0]['pairs'][0]['current'] != observed[1]['results'][0]['pairs'][0]['previous']
    third = json.loads((campaign.history / 'round-3/analysis.json').read_text())
    assert third['comparison']['status'] == 'unchanged'
    assert third['comparison']['repeated_round'] == 2
    assert campaign.run() == state
    # Legacy configuration resumes with its original contract.
    raw = json.loads((campaign.root / 'state.json').read_text())
    raw['configuration'].pop('analysis_contract')
    (campaign.root / 'state.json').write_text(json.dumps(raw))
    legacy = Campaign(campaign.root, bundle, {}, ['unused'], 'task', rounds=3)
    assert legacy.analysis_contract == 1
    assert legacy.run()['state'] == 'completed'


def test_measured_samples_keep_busy_time_boundaries_and_capture_settings():
    value = summary()
    value.update(busy_time=dict(metric='sum_of_task_durations_not_elapsed_latency',
        samples_us=[14.72, 14.7, 13.76], active_union_samples_us=[10, 11, 12]),
        latency=dict(metric='warm synchronized full-forward wall latency', median_us=392.279))
    current = history.measurements(value, receipt(), 'workload')
    assert current['metrics']['busy_time/samples_us/median'] == 14.7
    assert current['metrics']['busy_time/active_union_samples_us/median'] == 11
    old = copy.deepcopy(current)
    old['context']['capture']['pipe_captures'] = 99
    assert not history.pair([current], [old])['pairs']
    old = copy.deepcopy(current)
    old['context']['timing_semantics']['busy_time'] = 'elapsed latency'
    assert not history.pair([current], [old])['pairs']


def test_utilization_coverage_and_empty_bottleneck_rejected(tmp_path):
    ctx = history.context(tmp_path, b'kernel', 'a'*40, 1)
    value = report(ctx)
    path = 'evidence/call/results/summary.json'
    value.update(profiling=dict(status='acquired'),
        evidence=[dict(path=path, kind='profile', cases=['a', 'b'], kernels=['k'])],
        utilization=[dict(status='measured', description='MAC', cases=['a'], kernels=['k'], evidence=[path])])
    with pytest.raises(ValueError, match='bottleneck'): history.validate_round(value, ctx)
    value['bottlenecks'] = [dict(finding='likely memory bound', confidence='low', evidence=[path])]
    with pytest.raises(ValueError, match='cover'): history.validate_round(value, ctx)
    value['utilization'].append(dict(status='unavailable', description='unsupported case', cases=['b'], kernels=['k']))
    history.validate_round(value, ctx)


def test_native_timing_without_pmu_collector_compares_retained_samples():
    value = summary()
    value.update(samples=3, warmup=20, captures=[],
        busy_time=dict(metric='sum_of_task_durations_not_elapsed_latency', samples_us=[14,15,16]))
    identity = dict(target='gz-a3', runtime='native', request=dict(operation='timing'))
    current = history.measurements(value, identity, 'workload')
    old = copy.deepcopy(current)
    old['metrics']['busy_time/samples_us/median'] = 30
    pairs = history.pair([current], [old])['pairs']
    assert next(p for p in pairs if p['metric'] == 'busy_time/samples_us/median')['percent'] == -50


def test_optional_older_selection_requires_reason_and_distinct_identity(tmp_path):
    for index in range(1, 4):
        folder = tmp_path / f'round-{index}'
        folder.mkdir()
        (folder / 'candidate.py').write_text(str(index))
        (folder / 'record.json').write_text(json.dumps(dict(round=index, source_sha256=str(index))))
    ctx = history.context(tmp_path, b'new', 'a'*40, 4)
    value = report(ctx)
    value['comparison']['rounds'] = [3, 1]
    with pytest.raises(ValueError, match='reason'): history.validate_round(value, ctx)
    value['comparison']['selection_reason'] = 'Compare against original unfused implementation'
    history.validate_round(value, ctx)
    value['comparison']['rounds'] = [3, 2, 1]
    with pytest.raises(ValueError, match='at most two'): history.validate_round(value, ctx)


@pytest.mark.parametrize('attempted', [False, True])
def test_profile_attempt_failure_retains_source_and_explicit_latest_status(tmp_path, attempted):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('kernel = 1\n')
    (bundle / 'task.json').write_text(json.dumps(dict(public_files=['candidate.py'])))
    def runner(campaign, phase, role, budget, commit):
        if role == 'actor':
            campaign.qualify(phase, dict(state='completed', correctness=True,
                source_sha256=digest(campaign.workspace / 'candidate.py')))
        else:
            ctx = json.loads((campaign.history / 'context.json').read_text())
            (phase / 'analysis.json').write_text(json.dumps(report(ctx)))
            if attempted:
                (phase / 'required-profile.json').write_text(json.dumps(dict(
                    source_sha256=ctx['source_sha256'], state='failed', error='transport unavailable',
                    command_dispatched=True, profile_attempt_arguments=['pipes'])))
        return dict(exit_code=0)
    campaign = Campaign(tmp_path / 'run', bundle, {}, ['unused'], 'task', runner=runner, rounds=1)
    state = campaign.run()
    assert state['phases']['round-1-analyzer']['analysis_valid'] == attempted
    assert (campaign.history / 'round-1/candidate.py').read_text() == 'kernel = 1\n'
    latest = json.loads((campaign.workspace / 'analysis.json').read_text())
    assert latest['profiling']['status'] == 'unavailable' if attempted else latest['status'] == 'invalid'
    assert (campaign.workspace / 'experiments/round-1-analyzer/raw-analysis.json').is_file()


def test_standalone_csv_profiles_preserve_seeded_workload_attribution():
    value = dict(source_sha256='a'*64, correctness=[dict(check=i, correctness=True) for i in range(4)],
        launch_contract=dict(profile_stream_verified=True),
        timing=dict(samples_us=[8, 10, 12], median_us=10), captures=[dict(selector='gather',
            basic_rows=[{'Duration(us)': '10', 'Device_id': '3'}],
            pipe_rows=[{'aic_mac_ratio': '12.5', 'aic_mte2_ratio': '80',
                        'Device_id': '3', 'Start Time(us)': '123', 'invalid': 'NaN', 'unsupported': 'N/A'}])])
    now = history.measurements(value, receipt(), 'standalone-bundle')
    assert now['context']['cases'] == ['0', '1', '2', '3']
    key = 'kernel/gather/seeded-workload-aggregate/pipe_rows/all/aic_mte2_ratio'
    assert now['metrics'][key] == 80
    assert not any('Device_id' in key or 'invalid' in key for key in now['metrics'])
    old = copy.deepcopy(now)
    old['metrics'][key] = 40
    result = history.pair([now], [old])
    assert next(item for item in result['pairs'] if item['metric'] == key)['percent'] == 100
    assert any(item['metric'].startswith('timing/') for item in result['pairs'])
    value['launch_contract']['profile_stream_verified'] = False
    unattributed = history.measurements(value, receipt(), 'standalone-bundle')
    assert not any(key.startswith('kernel/') for key in unattributed['metrics'])


def test_multiple_successful_acquisitions_are_paired_once_with_provenance():
    def acquisition(duration, activity, call):
        identity = receipt()
        identity.update(call_id=call, handle='remote:gz-a3:job:' + call)
        return history.measurements(summary(duration=duration, activity=activity), identity, 'workload')
    old = [acquisition(10, 20, 'old1'), acquisition(30, 40, 'old2')]
    new = [acquisition(8, 60, 'new1'), acquisition(12, 80, 'new2')]
    result = history.pair(new, old)
    pairs = {item['metric']: item for item in result['pairs']}
    assert len(pairs) == len(result['pairs'])
    timing = pairs['timing/mean_us']
    assert (timing['previous'], timing['current'], timing['percent']) == (20, 10, -50)
    assert timing['previous_acquisitions'] == timing['current_acquisitions'] == 2
    assert {item['call_id'] for item in timing['previous_provenance']} == {'old1', 'old2'}
    assert pairs['kernel/matmul/shape1/pipe_rows/all/mac_ratio']['delta'] == 40
    assert history.pair(new, list(reversed(old)))['pairs'] == result['pairs']


def test_optional_older_selection_cannot_repeat_previous_source(tmp_path):
    for index, source in enumerate((b'A', b'B', b'A'), 1):
        folder = tmp_path / f'round-{index}'
        folder.mkdir()
        (folder / 'candidate.py').write_bytes(source)
        (folder / 'record.json').write_text(json.dumps(dict(round=index,
            source_sha256=hashlib.sha256(source).hexdigest())))
    ctx = history.context(tmp_path, b'C', 'a'*40, 4)
    value = report(ctx)
    value['comparison'].update(rounds=[3, 1], selection_reason='Compare original A')
    with pytest.raises(ValueError, match='distinct source'): history.validate_round(value, ctx)
    value['comparison']['rounds'] = [3, 2]
    history.validate_round(value, ctx)


@pytest.mark.parametrize('profiling', ['unavailable', 'acquired'])
def test_first_round_controller_owns_empty_selection_preserving_raw_diagnostic(tmp_path, profiling):
    ctx = history.context(tmp_path, b'kernel', 'a'*40, 1)
    value = report(ctx)
    value['comparison'].update(rounds=[1], selection_reason='No historical kernel exists')
    if profiling == 'acquired':
        path = 'evidence/current/results/summary.json'
        value.update(profiling=dict(status='acquired'),
            evidence=[dict(path=path, kind='profile', cases=['case'], kernels=['kernel'])],
            bottlenecks=[dict(finding='Possible memory limit', evidence=[path], confidence='low')],
            utilization=[dict(status='measured', description='MAC', cases=['case'], kernels=['kernel'], evidence=[path])])
    normalized = history.finalize(value, ctx, tmp_path, [])
    assert normalized['comparison']['rounds'] == []
    assert normalized['comparison']['results'] == []
    assert normalized['controller_diagnostics']['first_round_selection']['reported_rounds'] == [1]
    assert value['comparison']['rounds'] == [1]
    value['comparison']['status'] = 'new'
    with pytest.raises(ValueError, match='status'): history.finalize(value, ctx, tmp_path, [])


def test_historical_unknown_selection_remains_invalid_after_first_round_normalization(tmp_path):
    folder = tmp_path / 'round-1'
    folder.mkdir()
    (folder / 'candidate.py').write_text('old')
    (folder / 'record.json').write_text(json.dumps(dict(round=1, source_sha256='old')))
    ctx = history.context(tmp_path, b'new', 'a'*40, 2)
    value = report(ctx)
    value['comparison']['rounds'] = [1, 2]
    with pytest.raises(ValueError, match='unknown'): history.finalize(value, ctx, tmp_path, [])
