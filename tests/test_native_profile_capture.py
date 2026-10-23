"""Native launch proof from filtered, retained captures; no device launches.

Fixture times are shifted by each iteration's origin, identifiers are
anonymized, and the after fixture retains the actual 43ns clock inversion.
Only unrelated events were removed; every selected task retains its links.
"""
import copy
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/ascend-model-benchmarking/scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture, accounting = load('profile_capture'), load('profile_accounting')


def fixture(arm='before'):
    return json.loads((ROOT / 'tests/fixtures/native_flow' / (arm + '.json')).read_text())['traceEvents']


def normalize(events):
    return capture.normalize_trace(events, ['model_iteration_0'], export_completed=True)


@pytest.mark.parametrize('arm,duration', [('before', 68.51875), ('after', 35.5995)])
def test_retained_native_flow_owns_one_kernel_and_preserves_clock_evidence(arm, duration):
    result = normalize(fixture(arm))
    summary = accounting.account_profile(result['iterations'], result['tasks'], capture_complete=True)
    assert summary['iterations'][0]['task_count'] == 1
    assert summary['iterations'][0]['device_busy_us'] == duration
    evidence = result['acquisition_evidence']
    assert evidence['native_flow_count'] == 1
    assert result['tasks'][0]['correlation_id'].startswith('HostToDevice:')
    link = evidence['links'][0]
    assert link['native_launch']['args']['connection_id'] in (14, 39)
    if arm == 'after':
        assert link['host_device_start_delta_us'] == '-0.043'
        assert link['host_device_clock_order'] == 'device-before-host'
    else:
        assert evidence['paired_flows'] == 2  # One Torch timing event, one native compute.
        assert len(evidence['auxiliary_native_flows']) == 1
        assert len(evidence['exclusions']) == 1
        assert evidence['exclusions'][0]['task']['name'] == 'EVENT_RECORD'


@pytest.mark.parametrize('mutation', ['start', 'finish', 'node', 'caller', 'thread',
    'connection', 'task-connection', 'name', 'flow-name', 'duplicate-start',
    'duplicate-task', 'duplicate-caller', 'peer-caller', 'second-process'])
def test_native_ownership_never_uses_names_counts_or_device_time_windows(mutation):
    events = fixture('after')
    node = next(e for e in events if e.get('name') == 'Node@launch')
    task = next(e for e in events if 'Task Type' in e.get('args', {}))
    caller = next(e for e in events if e.get('name') == 'optimization_native::launch')
    if mutation in ('start', 'finish'):
        events[:] = [e for e in events if e.get('ph') != ('s' if mutation == 'start' else 'f')]
    elif mutation == 'node': events.remove(node)
    elif mutation == 'caller': events.remove(caller)
    elif mutation == 'thread': node['args']['Thread Id'] += 1
    elif mutation == 'connection': node['args']['connection_id'] += 1
    elif mutation == 'task-connection': task['args']['connection_id'] += 1
    elif mutation == 'name': node['args']['item_id'] = 'another_kernel'
    elif mutation == 'flow-name': next(e for e in events if e.get('ph') == 's')['name'] = 'unknown'
    elif mutation == 'duplicate-start': events.append(copy.deepcopy(next(e for e in events if e.get('ph') == 's')))
    elif mutation == 'duplicate-task': events.append(copy.deepcopy(task))
    else:
        other = copy.deepcopy(caller)
        if mutation == 'peer-caller':
            # Both contain the native launch but neither contains the other.
            start, duration = Decimal(node['ts']), Decimal(str(node['dur']))
            caller['ts'], caller['dur'] = str(start - 2), float(duration + 3)
            other['ts'], other['dur'] = str(start - 1), float(duration + 3)
        elif mutation == 'second-process': other['pid'] += 1
        events.append(other)
    with pytest.raises(capture.CaptureError):
        normalize(events)


def test_nested_dispatcher_scope_is_supported_and_missing_marker_still_fails():
    events = fixture('after')
    caller = next(e for e in events if e.get('name') == 'optimization_native::launch')
    outer = copy.deepcopy(caller)
    outer['name'] = 'outer_dispatcher'
    outer['ts'], outer['dur'] = '0.01', 100
    events.append(outer)
    result = normalize(events)
    assert result['acquisition_evidence']['links'][0]['host']['name'] == caller['name']
    events[:] = [e for e in events if e.get('name') != 'model_iteration_0']
    with pytest.raises(capture.CaptureError, match='iteration marker'):
        normalize(events)


def test_native_task_outside_host_iteration_is_audited_without_reassignment():
    events = fixture('before')
    # Move only the host iteration away. Device timestamps are never used to
    # pull a fully linked launch into an unrelated host iteration.
    events[0]['ts'], events[0]['dur'] = '10000', 100
    with pytest.raises(capture.CaptureError, match='empty active iteration'):
        normalize(events)
