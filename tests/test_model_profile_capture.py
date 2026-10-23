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
        reason='zero-duration queue placeholder with sentinel linkage', task=original)]


@pytest.mark.parametrize('field,value', [
    ('dur', 1), ('dur', -1), ('name', 'unknown_placeholder'), ('cat', 'cpu_op'),
    ('pid', -1), ('tid', 48), ('Model Id', 0), ('Task Id', 1),
    ('Batch Id', 0), ('Batch Id', 2), ('Subtask Id', 0),
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


def test_unproven_mixed_aiv_remains_unsupported(trace):
    trace[2]['args']['Task Type'] = 'MIX_AIV'
    with pytest.raises(capture.CaptureError, match='unvalidated device task type: MIX_AIV'):
        normalize(trace)
