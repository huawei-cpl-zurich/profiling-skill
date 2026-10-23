"""Functional CPU/NPU correlation tests using the acquired torch_npu schema."""
import copy
import importlib.util
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/ascend-model-benchmarking/scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


capture = load('profile_capture')
accounting = load('profile_accounting')


def event(name, pid, tid, ts, duration, **extra):
    return dict(name=name, pid=pid, tid=tid, ts=ts, dur=duration, ph='X', **extra)


@pytest.fixture
def trace():
    # Epoch strings, field names and IDs follow the actual retained A3 capture.
    result = [event('model_iteration_0', 3189398, 3189398,
                    '1792459730174020.890', 471.67, cat='cpu_op')]
    for i, (host, device, duration) in enumerate([
            ('1792459730174238.730', '1792459730174335.602', .999875),
            ('1792459730174344.340', '1792459730174365.401', 18.979625)]):
        result += [event('add_kernel', 3189398, 3189398, host, 7.57, cat='cpu_op'),
                   event('add_kernel', 3265944001, 47, device, duration,
                         args={'Task Type': 'KERNEL_AIVEC', 'Physic Stream Id': 47,
                               'Task Id': i, 'Batch Id': 0, 'Subtask Id': 0}),
                   dict(ph='s', cat='async_npu', name='torch_to_npu', id=i,
                        pid=3189398, tid=3189398, ts=host),
                   dict(ph='f', cat='async_npu', name='torch_to_npu', id=i,
                        pid=3265944001, tid=47, ts=device)]
    return result


def normalize(trace):
    return capture.normalize_trace(trace, ['model_iteration_0'], export_completed=True)


def test_identical_names_unequal_durations_are_all_retained(trace):
    result = normalize(trace)
    summary = accounting.account_profile(result['iterations'], result['tasks'],
                                        capture_complete=result['capture_complete'])
    assert summary['iterations'][0]['device_busy_us'] == pytest.approx(19.9795)
    assert summary['iterations'][0]['task_count'] == 2
    assert len(result['acquisition_evidence']['links']) == 2
    assert result['tasks'][0]['start_us'] == pytest.approx(314.712)


@pytest.mark.parametrize('mutation', ['marker', 'finish', 'all-flow', 'host', 'duplicate',
                                    'unclosed', 'unknown-kind', 'negative-duration'])
def test_incomplete_or_ambiguous_capture_fails(trace, mutation):
    if mutation == 'marker':
        trace.pop(0)
    elif mutation == 'finish':
        trace.pop()
    elif mutation == 'all-flow':
        trace[:] = [e for e in trace if not (e.get('cat') == 'async_npu' and e['id'] == 0)]
    elif mutation == 'host':
        trace.pop(1)
    elif mutation == 'duplicate':
        trace.append(copy.deepcopy(trace[2]))
    elif mutation == 'unclosed':
        trace[0]['ph'] = 'B'
    elif mutation == 'unknown-kind':
        trace[2]['args']['Task Type'] = 'undocumented'
    else:
        trace[2]['dur'] = -1
    with pytest.raises(capture.CaptureError):
        normalize(trace)


def test_export_incomplete_rejected(trace):
    with pytest.raises(capture.CaptureError, match='completed'):
        capture.normalize_trace(trace, ['model_iteration_0'], export_completed=False)


def test_aic_and_aicpu_are_separate(trace):
    trace[2]['args']['Task Type'] = 'KERNEL_AICORE'
    trace[6]['args']['Task Type'] = 'KERNEL_AICPU'
    result = normalize(trace)
    summary = accounting.account_profile(result['iterations'], result['tasks'],
                                        capture_complete=True)
    assert summary['iterations'][0]['device_busy_us'] == .999875
    assert summary['iterations'][0]['aicpu_busy_us'] == 18.979625


@pytest.mark.parametrize('kind', ['MODEL_EXECUTE', 'PROFILER_TRACE_EX',
                                'PROFILING_ENABLE', 'PROFILING_DISABLE'])
def test_vendor_administrative_tasks_are_excluded_with_raw_evidence(trace, kind):
    trace.append(event(kind, 3267914182, 47, '1792459730174333', 3.06,
                       args={'Task Type': kind}))
    result = normalize(trace)
    assert len(result['tasks']) == 2
    assert result['acquisition_evidence']['exclusions'][0]['task']['dur'] == 3.06


@pytest.mark.parametrize('kind', ['MIX_AIC', 'AI_CORE', 'AI_VECTOR_CORE'])
def test_observed_mixed_kernel_is_one_task_not_invented_components(trace, kind):
    for event in trace:
        if 'Task Type' in event.get('args', {}):
            event['args']['Task Type'] = kind
            event['name'] = 'qualified_matmul_add'
    result = normalize(trace)
    assert len(result['tasks']) == 2
    assert {t['kind'] for t in result['tasks']} == {'device'}
    assert {t['name'] for t in result['tasks']} == {'qualified_matmul_add'}


def test_actual_level1_mixed_trace_three_iterations():
    # Exact endpoint/range/duration records from native 5971ace9b099 capture;
    # omit unrelated host allocation records, not any device kernel or flow.
    rows = [
        ('1792460482362661.470',737.5,'1792460482363192.720',14.64,
         '1792460482363284.188',52.618875),
        ('1792460482363463.850',389.59,'1792460482363724.980',11.14,
         '1792460482363755.218',56.018875),
        ('1792460482363903.230',366.95,'1792460482364145.940',10.8,
         '1792460482364170.470',58.318875)]
    events = []
    for i,(start,duration,host,host_duration,device,device_duration) in enumerate(rows):
        events += [event(f'model_iteration_{i}',3194843,3194843,start,duration,cat='cpu_op'),
            event('qualified_matmul_add',3194843,3194843,host,host_duration,cat='cpu_op'),
            event('qualified_matmul_add',3271519682,47,device,device_duration,
                  args={'Task Type':'MIX_AIC','Physic Stream Id':47,'Task Id':34+i,
                        'Batch Id':0,'Subtask Id':0}),
            dict(ph='s',cat='async_npu',name='torch_to_npu',id=i,
                 pid=3194843,tid=3194843,ts=host),
            dict(ph='f',cat='async_npu',name='torch_to_npu',id=i,
                 pid=3271519682,tid=47,ts=device)]
    normalized = capture.normalize_trace(events,[f'model_iteration_{i}' for i in range(3)],
                                         export_completed=True)
    result = accounting.account_profile(normalized['iterations'], normalized['tasks'],
                                        capture_complete=True)
    assert [r['task_count'] for r in result['iterations']] == [1,1,1]
    assert [r['device_busy_us'] for r in result['iterations']] == [52.618875,56.018875,58.318875]


def test_unrelated_fully_correlated_work_retains_explicit_exclusion(trace):
    extra = copy.deepcopy(trace[1:5])
    for e in extra:
        e['tid'] = 99
        if e.get('cat') == 'async_npu':
            e['id'] = 10
        if 'Task Type' in e.get('args', {}):
            e['args']['Task Id'] = 10
    trace.extend(extra)
    result = normalize(trace)
    assert len(result['tasks']) == 2
    excluded = result['acquisition_evidence']['exclusions']
    assert len(excluded) == 1
    assert excluded[0]['task']['name'] == 'add_kernel'


def test_overlap_markers_rejected(trace):
    marker = copy.deepcopy(trace[0])
    marker['name'] = 'model_iteration_1'
    trace.append(marker)
    with pytest.raises(capture.CaptureError, match='overlapping'):
        capture.normalize_trace(trace, ['model_iteration_0', 'model_iteration_1'],
                                export_completed=True)


def fake_acquisition(trace, tmp_path, *, recompile=False, export=True):
    state = dict(calls=0, steps=0, compilations=1, active=False)

    def prepared():
        state['calls'] += 1
        if state['active'] and recompile:
            state['compilations'] += 1

    prepared.compiler = SimpleNamespace(monitor_compilation=nullcontext,
        snapshot=lambda: (state['compilations'], 0))

    class Profile:
        def __enter__(self):
            state['active'] = True
            return self

        def __exit__(self, *exception):
            state['active'] = False
            if export:
                vendor = tmp_path / 'capture/vendor'
                vendor.mkdir()
                (vendor / 'trace_view.json').write_text(json.dumps(trace))

        def step(self):
            state['steps'] += 1

    torch = SimpleNamespace(__version__='test', inference_mode=nullcontext,
        npu=SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0),
        autograd=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext())))
    def profile(**kwargs):
        state['profile_kwargs'] = kwargs
        return Profile()

    npu = SimpleNamespace(__version__='test', profiler=SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU=0, NPU=1), profile=profile,
        ProfilerLevel=SimpleNamespace(Level1='level-one'),
        _ExperimentalConfig=lambda **kwargs: kwargs,
        schedule=lambda **kwargs: None,
        tensorboard_trace_handler=lambda path: None))
    return capture.Acquisition(torch_module=torch, npu_module=npu), prepared, state


def test_acquisition_warms_outside_capture_and_retains_raw_export(trace, tmp_path):
    adapter, prepared, state = fake_acquisition(trace, tmp_path)
    result = adapter.profile(prepared, samples=1, warmup=20, output=tmp_path / 'capture')
    assert state['calls'] == 21 and state['steps'] == 1
    assert state['profile_kwargs']['experimental_config'] == {
        'profiler_level': 'level-one', 'data_simplification': False}
    assert result['capture_complete']
    raw = json.load(gzip.open(tmp_path / 'capture/trace_view.json.gz', 'rt'))
    assert raw == trace
    manifest = json.loads((tmp_path / 'capture/acquisition.json').read_text())
    assert manifest['state'] == 'complete'
    assert manifest['kernel_names'] == ['add_kernel']
    assert adapter.comparability({}, {})['comparable'] is False


@pytest.mark.parametrize('recompile,export', [(True, True), (False, False)])
def test_acquisition_failures_retain_manifest_not_valid_timing(trace, tmp_path,
                                                             recompile, export):
    adapter, prepared, state = fake_acquisition(trace, tmp_path,
                                               recompile=recompile, export=export)
    with pytest.raises(capture.CaptureError):
        adapter.profile(prepared, samples=1, warmup=20, output=tmp_path / 'capture')
    manifest = json.loads((tmp_path / 'capture/acquisition.json').read_text())
    assert manifest['state'] == 'failed'
    assert manifest['error']


def test_failed_normalization_preserves_compressed_raw_trace(trace, tmp_path):
    trace.pop()
    adapter, prepared, _ = fake_acquisition(trace, tmp_path)
    with pytest.raises(capture.CaptureError, match='incomplete'):
        adapter.profile(prepared, samples=1, warmup=20, output=tmp_path / 'capture')
    assert json.load(gzip.open(tmp_path / 'capture/trace_view.json.gz', 'rt')) == trace
    manifest = json.loads((tmp_path / 'capture/acquisition.json').read_text())
    assert manifest['state'] == 'failed' and manifest['trace_sha256']


def placeholder():
    # Exact structural identity from retained BSA baseline case0/repetition2.
    # Only coordinates are shifted into the small acquired trace above.
    return event('PLACE_HOLDER_SQE', 3265944001, 47,
        '1792459730174340.000', 0.0,
        args={'Model Id': 4294967295, 'Task Type': 'PLACE_HOLDER_SQE',
              'Physic Stream Id': 47, 'Task Id': 0, 'Batch Id': 1,
              'Subtask Id': 4294967295, 'connection_id': 18446744073709551615})


def test_observed_zero_duration_placeholder_retains_audit_without_compute(trace):
    original = placeholder()
    trace.append(original)
    result = normalize(trace)
    summary = accounting.account_profile(result['iterations'], result['tasks'], capture_complete=True)
    assert summary['iterations'][0]['device_busy_us'] == pytest.approx(19.9795)
    assert summary['iterations'][0]['task_count'] == 2
    assert result['acquisition_evidence']['observed_tasks'] == 2
    assert result['acquisition_evidence']['paired_flows'] == 2
    assert result['acquisition_evidence']['exclusions'] == [dict(
        reason='verified administrative queue placeholder with sentinel linkage', task=original)]


@pytest.mark.parametrize('field,value', [
    ('dur', 1), ('dur', -1), ('dur', 0.01), ('dur', 0.021), ('dur', 0.03), ('dur', float('inf')), ('dur', float('nan')), ('name', 'unknown_placeholder'), ('cat', 'cpu_op'),
    ('pid', -1), ('tid', 48), ('Model Id', 0), ('Task Id', 1),
    ('Batch Id', 0), ('Batch Id', 4294967295), ('Subtask Id', 0),
    ('connection_id', 0), ('Physic Stream Id', 48), ('Task Id', False)])
def test_placeholder_near_matches_remain_unsupported(trace, field, value):
    record = placeholder()
    (record if field in ('dur', 'name', 'cat', 'pid', 'tid') else record['args'])[field] = value
    trace.append(record)
    with pytest.raises(capture.CaptureError):
        normalize(trace)


def test_placeholder_missing_sentinel_identity_remains_unsupported(trace):
    record = placeholder()
    del record['args']['connection_id']
    trace.append(record)
    with pytest.raises(capture.CaptureError, match='unvalidated'):
        normalize(trace)


def test_placeholder_cannot_satisfy_a_flow_or_empty_iteration(trace):
    record = placeholder()
    trace.append(record)
    trace[4].update(pid=record['pid'], tid=record['tid'], ts=record['ts'])
    with pytest.raises(capture.CaptureError, match='exactly one CPU op and device task'):
        normalize(trace)
    with pytest.raises(capture.CaptureError, match='empty active iteration'):
        normalize([trace[0], record])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unclosed'])
def test_placeholder_never_relaxes_explicit_markers(trace, mutation):
    trace.append(placeholder())
    if mutation == 'missing':trace.pop(0)
    elif mutation == 'duplicate':trace.append(copy.deepcopy(trace[0]))
    else:trace[0]['ph'] = 'B'
    with pytest.raises(capture.CaptureError, match='explicit iteration marker'):
        normalize(trace)


@pytest.fixture
def retained_mixed_trace():
    # Compact actual GDN repetition1 ReduceSum record and exact vendor links.
    return [event('model_iteration_0', 2702806, 2702806,
                  '1792701342689803.470', 1099392.9, cat='cpu_op'),
            event('aclnnReduceSum', 2702806, 2702806,
                  '1792701342690412.980', 2.17, cat='cpu_op',
                  args={'Sequence number': -1, 'Fwd thread id': 0}),
            event('aclnnReduceSum_ReduceSumOpAiCore_ReduceSum', 2767673793, 2,
                  '1792701342690522.310', 19.9,
                  args={'Model Id': 4294967295, 'Task Type': 'MIX_AIV',
                        'Physic Stream Id': 2, 'Task Id': 14337, 'Batch Id': 0,
                        'Subtask Id': 0, 'connection_id': 1852220}),
            dict(ph='s', bp='e', name='torch_to_npu', id=1792701342690522310,
                 pid=2702806, tid=2702806, ts='1792701342690412.980', cat='async_npu'),
            dict(ph='f', bp='e', name='torch_to_npu', id=1792701342690522310,
                 pid=2767673793, tid=2, ts='1792701342690522.310', cat='async_npu')]


def test_retained_mixed_aiv_is_correlated_device_compute(retained_mixed_trace):
    result = normalize(retained_mixed_trace)
    summary = accounting.account_profile(result['iterations'], result['tasks'], capture_complete=True)
    assert result['tasks'][0]['kind'] == 'device'
    assert result['tasks'][0]['duration_us'] == 19.9
    assert result['tasks'][0]['start_us'] == pytest.approx(718.84)
    assert summary['iterations'][0]['device_busy_us'] == 19.9
    assert summary['iterations'][0]['aicpu_busy_us'] == 0
    assert result['acquisition_evidence']['paired_flows'] == 1
    assert result['acquisition_evidence']['exclusions'] == []


@pytest.mark.parametrize('mutation', ['missing-flow', 'duplicate-flow', 'missing-marker',
                                    'duplicate-marker', 'unclosed-marker', 'unknown-kind'])
def test_mixed_aiv_still_requires_complete_flow_and_markers(retained_mixed_trace, mutation):
    trace = retained_mixed_trace
    if mutation == 'missing-flow': trace.pop()
    elif mutation == 'duplicate-flow': trace.append(copy.deepcopy(trace[-1]))
    elif mutation == 'missing-marker': trace.pop(0)
    elif mutation == 'duplicate-marker': trace.append(copy.deepcopy(trace[0]))
    elif mutation == 'unclosed-marker': trace[0]['ph'] = 'B'
    else: trace[2]['args']['Task Type'] = 'UNVALIDATED_MIX'
    with pytest.raises(capture.CaptureError):
        normalize(trace)

@pytest.mark.parametrize('pid,ts', [(2967673280, '1792719785863995.715'),
                                   (2970402240, '1792719910816995.340')])
def test_retained_short_queue_placeholder_is_audited_not_compute(trace, pid, ts):
    # Exact unlinked placeholders from both fresh BSA smoke repetitions.
    record = event('PLACE_HOLDER_SQE', pid, 2, ts, 0.02,
        args={'Model Id': 4294967295, 'Task Type': 'PLACE_HOLDER_SQE',
              'Physic Stream Id': 2, 'Task Id': 0, 'Batch Id': 1,
              'Subtask Id': 4294967295, 'connection_id': 18446744073709551615})
    trace.append(record)
    result = normalize(trace)
    summary = accounting.account_profile(result['iterations'], result['tasks'], capture_complete=True)
    assert summary['iterations'][0]['device_busy_us'] == pytest.approx(19.9795)
    assert result['acquisition_evidence']['exclusions'] == [dict(
        reason='verified administrative queue placeholder with sentinel linkage', task=record)]
    trace[4].update(pid=pid, tid=2, ts=ts)
    with pytest.raises(capture.CaptureError, match='exactly one CPU op and device task'):
        normalize(trace)


@pytest.fixture
def retained_event_record():
    # Exact timing-event command and CPU/NPU link from the BSA ten-forward baseline.
    return [event('record_event', 2941004, 2941004, '1792723909142941.690', 49.7,
                  cat='cpu_op', args={'Sequence number': -1, 'Fwd thread id': 0}),
            event('EVENT_RECORD', 3011588544, 2, '1792723909143162.440', 0.02,
                  args={'Model Id': 4294967295, 'Task Type': 'EVENT_RECORD',
                        'Physic Stream Id': 2, 'Task Id': 1045, 'Batch Id': 0,
                        'Subtask Id': 4294967295, 'connection_id': 19945}),
            dict(ph='s', bp='e', name='torch_to_npu', id=1792723909143162440,
                 pid=2941004, tid=2941004, ts='1792723909142941.690', cat='async_npu'),
            dict(ph='f', bp='e', name='torch_to_npu', id=1792723909143162440,
                 pid=3011588544, tid=2, ts='1792723909143162.440', cat='async_npu')]


def test_linked_timing_event_is_audited_not_compute(trace, retained_event_record):
    result = normalize(trace + retained_event_record)
    summary = accounting.account_profile(result['iterations'], result['tasks'], capture_complete=True)
    assert summary['iterations'][0]['device_busy_us'] == pytest.approx(19.9795)
    assert len(result['tasks']) == 2
    evidence = result['acquisition_evidence']
    assert evidence['observed_tasks'] == 2 and evidence['paired_flows'] == 3
    audit = evidence['exclusions'][0]
    assert audit['task'] == retained_event_record[1]
    assert audit['host'] == retained_event_record[0]
    assert audit['link']['flow_id'] == '1792723909143162440'
    assert audit['reason'] == 'explicitly linked timing-event command, not compute'
    with pytest.raises(capture.CaptureError, match='empty active iteration'):
        normalize([trace[0]] + retained_event_record)


@pytest.mark.parametrize('mutation', ['missing-start', 'missing-finish', 'no-flow', 'duplicate-flow',
    'wrong-host', 'duration', 'model', 'batch', 'subtask', 'connection', 'task', 'stream', 'name'])
def test_event_record_requires_observed_schema_and_exact_flow(trace, retained_event_record, mutation):
    records = retained_event_record
    if mutation == 'missing-start': records.pop(2)
    elif mutation == 'missing-finish': records.pop()
    elif mutation == 'no-flow': records = records[:2]
    elif mutation == 'duplicate-flow': records.append(copy.deepcopy(records[-1]))
    elif mutation == 'wrong-host': records[0]['name'] = 'ordinary_kernel_launch'
    elif mutation == 'duration': records[1]['dur'] = 0.03
    elif mutation == 'name': records[1]['name'] = 'unknown'
    else:
        field, value = {'model': ('Model Id', 0), 'batch': ('Batch Id', -1),
            'subtask': ('Subtask Id', 0), 'connection': ('connection_id', 18446744073709551615),
            'task': ('Task Id', False), 'stream': ('Physic Stream Id', 3)}[mutation]
        records[1]['args'][field] = value
    with pytest.raises(capture.CaptureError):
        normalize(trace + records)


@pytest.mark.parametrize('duration,batch', [(0, 0), (0.02, 1), (0, 1), (0.02, 2), (0, 2), (0.02, 3)])
def test_retained_event_duration_and_batch_variants(trace, retained_event_record, duration, batch):
    # Observed across BSA/GDN captures; batch is a runtime identity, not a sentinel.
    retained_event_record[1]['dur'] = duration
    retained_event_record[1]['args']['Batch Id'] = batch
    result = normalize(trace + retained_event_record)
    assert len(result['tasks']) == 2
    audit = result['acquisition_evidence']['exclusions'][0]
    assert audit['task'] == retained_event_record[1]
    assert audit['host']['name'] == 'record_event'


@pytest.mark.parametrize('field,value', [('Batch Id', False), ('Batch Id', 4294967295),
                                       ('Task Id', -1), ('connection_id', -1)])
def test_event_identity_fields_require_real_nonnegative_integers(trace, retained_event_record, field, value):
    retained_event_record[1]['args'][field] = value
    with pytest.raises(capture.CaptureError):
        normalize(trace + retained_event_record)


@pytest.mark.parametrize('batch', [2, 3])
def test_retained_gdn_placeholder_batch_identity_variants(trace, batch):
    record = event('PLACE_HOLDER_SQE', 3011657153, 2, '1792723918825740.220', 0.0,
        args={'Model Id': 4294967295, 'Task Type': 'PLACE_HOLDER_SQE',
              'Physic Stream Id': 2, 'Task Id': 0, 'Batch Id': batch,
              'Subtask Id': 4294967295, 'connection_id': 18446744073709551615})
    result = normalize(trace + [record])
    assert len(result['tasks']) == 2
    assert result['acquisition_evidence']['exclusions'][0]['task'] == record
    trace[4].update(pid=record['pid'], tid=record['tid'], ts=record['ts'])
    with pytest.raises(capture.CaptureError, match='exactly one CPU op and device task'):
        normalize(trace + [record])
