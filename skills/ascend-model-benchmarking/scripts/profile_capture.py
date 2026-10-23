"""Acquire warmed Model CPU/NPU traces and account explicit vendor flow links.

Validated schema: torch_npu 2.10 trace_view.json. Unknown schemas fail closed.
No remote transport, inferred kernel counts, name grouping or device-time windows.
"""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform

TASK_KINDS = dict(KERNEL_AIVEC='device', KERNEL_AICORE='device', KERNEL_AICPU='aicpu',
                 MIX_AIC='device', MIX_AIV='device', AI_CORE='device', AI_VECTOR_CORE='device')


class CaptureError(ValueError):
    """The profiler export cannot establish unambiguous iteration accounting."""


def _decimal(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise CaptureError('invalid trace timestamp/duration') from error
    if not result.is_finite() or result < 0:
        raise CaptureError('negative/nonfinite trace timestamp/duration')
    return result


def _coordinate(event):
    return event['pid'], event['tid'], _decimal(event['ts'])


def _queue_placeholder(event):
    """Recognize only observed unlinked queue placeholder schemas.

    This is not a kernel-type exemption: another duration, real connection,
    different identity or an explicit flow still fails normal validation.
    """
    args = event['args']
    sentinel_identity = {'Model Id': 2**32 - 1, 'Task Id': 0,
                         'Subtask Id': 2**32 - 1, 'connection_id': 2**64 - 1}
    return (event.get('name') == 'PLACE_HOLDER_SQE' and event.get('cat') is None
            and _decimal(event.get('dur')) in (Decimal('0'), Decimal('0.02'))
            and all(type(args.get(key)) is int and args[key] == value
                    for key, value in sentinel_identity.items())
            and type(args.get('Batch Id')) is int and 0 < args['Batch Id'] < 2**32 - 1
            and type(event.get('pid')) is int and event['pid'] >= 0
            and type(event.get('tid')) is int and event['tid'] >= 0
            and type(args.get('Physic Stream Id')) is int
            and args['Physic Stream Id'] == event['tid'])


def _event_record(event):
    """Observed timing-event command; still requires its complete host flow."""
    args = event['args']
    return (event.get('name') == 'EVENT_RECORD' and event.get('cat') is None
            and _decimal(event.get('dur')) in (Decimal('0'), Decimal('0.02'))
            and all(type(args.get(key)) is int and args[key] == value for key, value in
                    {'Model Id': 2**32 - 1, 'Subtask Id': 2**32 - 1}.items())
            and all(type(args.get(key)) is int and 0 <= args[key] < maximum for key, maximum in
                    [('Task Id', 2**32 - 1), ('Batch Id', 2**32 - 1), ('connection_id', 2**64 - 1)])
            and type(event.get('pid')) is int and event['pid'] >= 0
            and type(event.get('tid')) is int and event['tid'] >= 0
            and type(args.get('Physic Stream Id')) is int
            and args['Physic Stream Id'] == event['tid'])


def normalize_trace(events, iteration_names, *, export_completed):
    """Join exact CPU op -> async_npu flow -> device task, then host scopes.

    All observed device tasks require explicit complete flows. Fully linked tasks
    outside active host scopes are retained as exclusions, never silently erased.
    Completion covers observed records, not undetectable vendor record loss.
    """
    if export_completed is not True:
        raise CaptureError('producer must establish completed profiler export')
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise CaptureError('traceEvents must be a list of events')
    if not iteration_names or len(set(iteration_names)) != len(iteration_names):
        raise CaptureError('expected iteration names must be nonempty and unique')
    try:
        return _normalize(events, iteration_names)
    except (KeyError, TypeError) as error:
        raise CaptureError(f'unsupported/incomplete vendor trace schema: {error}') from error


def _native_host(node, task, host):
    """Bridge the observed CANN launch-thread metadata to its CPU caller.

    Device ownership still comes from an exact HostToDevice flow. Range
    containment only identifies the CPU caller on the explicitly named thread.
    """
    args = node['args']
    connection = args.get('connection_id')
    if (node.get('name') != 'Node@launch' or node.get('ph') != 'X'
            or args.get('Mode') != 'launch' or args.get('level') != 'node'
            or type(args.get('Thread Id')) is not int or args['Thread Id'] != node['tid']
            or type(connection) is not int or not 0 <= connection < 2**64 - 1
            or type(task['args'].get('connection_id')) is not int
            or task['args']['connection_id'] != connection
            or args.get('item_id') != task['name']):
        raise CaptureError('unvalidated native launch or mismatched connection identity')
    start, end = _decimal(node['ts']), _decimal(node['ts']) + _decimal(node['dur'])
    if end <= start:
        raise CaptureError('native launch must have a closed positive duration')
    callers = [op for values in host.values() for op in values
               if op['tid'] == args['Thread Id']
               and _decimal(op['ts']) <= start
               and end <= _decimal(op['ts']) + _decimal(op['dur'])]
    if not callers or len({(op['pid'], op['tid']) for op in callers}) != 1:
        raise CaptureError('native launch lacks a unique CPU caller thread')
    # Nested record_function / dispatcher scopes are ordinary. The innermost
    # caller must be unique; never choose among overlapping peer CPU ranges.
    inner = [op for op in callers if not any(other is not op
             and _decimal(op['ts']) <= _decimal(other['ts'])
             and _decimal(other['ts']) + _decimal(other['dur'])
                 <= _decimal(op['ts']) + _decimal(op['dur']) for other in callers)]
    if len(inner) != 1:
        raise CaptureError('ambiguous native CPU caller scope')
    return inner[0]


def _normalize(events, names):
    markers = [e for e in events if e.get('name') in names and e.get('cat') == 'cpu_op']
    if (len(markers) != len(names) or {e['name'] for e in markers} != set(names)
            or any(e.get('ph') != 'X' or _decimal(e.get('dur', 0)) <= 0 for e in markers)):
        raise CaptureError('missing, duplicate or unclosed explicit iteration marker')
    markers.sort(key=lambda e: _decimal(e['ts']))
    if [e['name'] for e in markers] != names:
        raise CaptureError('iteration markers are not ordered')
    for first, second in zip(markers, markers[1:]):
        if _decimal(first['ts']) + _decimal(first['dur']) > _decimal(second['ts']):
            raise CaptureError('overlapping iteration markers')
    host, task_at, endpoints = defaultdict(list), defaultdict(list), defaultdict(dict)
    native_nodes, native_endpoints = defaultdict(list), defaultdict(dict)
    tasks, administrative = [], []
    placeholder_coordinates = set()
    for position, event in enumerate(events):
        if event.get('ph') == 'X' and event.get('cat') == 'cpu_op' and event not in markers:
            host[_coordinate(event)].append(event)
        if event.get('ph') == 'X' and event.get('name') == 'Node@launch':
            native_nodes[_coordinate(event)].append(event)
        if event.get('ph') == 'X' and 'Task Type' in event.get('args', {}):
            kind = event['args']['Task Type']
            if kind in ('PROFILER_TRACE_EX', 'MODEL_EXECUTE',
                        'PROFILING_ENABLE', 'PROFILING_DISABLE'):
                administrative.append(dict(reason='vendor administrative task, not a kernel',
                                           task=dict(event)))
                continue
            if kind == 'PLACE_HOLDER_SQE' and _queue_placeholder(event):
                placeholder_coordinates.add(_coordinate(event))
                administrative.append(dict(reason='verified administrative queue placeholder with sentinel linkage',
                                           task=dict(event)))
                continue
            if kind not in TASK_KINDS and not (kind == 'EVENT_RECORD' and _event_record(event)):
                raise CaptureError(f'unvalidated device task type: {kind}')
            tasks.append((position, event))
            task_at[_coordinate(event)].append((position, event))
        if event.get('cat') == 'async_npu':
            if event.get('name') != 'torch_to_npu' or event.get('ph') not in ('s', 'f'):
                raise CaptureError('unsupported async_npu flow schema')
            key, phase = str(event['id']), event['ph']
            if phase in endpoints[key]:
                raise CaptureError('duplicate async_npu flow endpoint')
            endpoints[key][phase] = event
        if event.get('cat') == 'HostToDevice' and event.get('ph') in ('s', 'f'):
            key, phase = str(event['id']), event['ph']
            native_endpoints[key].setdefault(phase, []).append(event)
    index = {name: dict(iteration_id=name, correlation_ids=[]) for name in names}
    used, identities, normalized, links, excluded = set(), set(), [], [], []
    origin = min(_decimal(m['ts']) for m in markers)
    flows = [('async_npu', key, phases) for key, phases in endpoints.items()]
    # A Torch link already proves ownership of timing events and ordinary ops.
    # Native CANN links to those same tasks are auxiliary, not extra tasks.
    flows += [('HostToDevice', key, phases) for key, phases in native_endpoints.items()]
    torch_linked, native_link_count, auxiliary = set(), 0, []
    for schema, key, phases in flows:
        if schema == 'HostToDevice':
            if any(_coordinate(endpoint) in placeholder_coordinates
                   for values in phases.values() for endpoint in values):
                raise CaptureError('native flow must not identify a queue placeholder')
            finishes = phases.get('f', [])
            positions = {position for finish in finishes
                         for position, _ in task_at[_coordinate(finish)]}
            if positions and positions <= torch_linked:
                auxiliary.append(dict(flow_id=key, reason='auxiliary CANN flow; task already has complete Torch proof'))
                continue
            if not positions:
                # Unused vendor flows are not ownership evidence. Any uncovered
                # device task still fails the complete-coverage check below.
                continue
            if any(len(values) != 1 for values in phases.values()):
                raise CaptureError('duplicate native flow endpoint')
            phases = {phase: values[0] for phase, values in phases.items()}
        if set(phases) != {'s', 'f'}:
            raise CaptureError('incomplete CPU/NPU flow')
        start, finish = phases['s'], phases['f']
        if any(_coordinate(endpoint) in placeholder_coordinates for endpoint in (start, finish)):
            raise CaptureError('flow endpoint must identify exactly one CPU op and device task; '
                               'queue placeholder coordinate is ambiguous')
        matches = (host if schema == 'async_npu' else native_nodes)[_coordinate(start)]
        targets = task_at[_coordinate(finish)]
        if len(matches) != 1 or len(targets) != 1:
            raise CaptureError('flow endpoint must identify exactly one CPU op and device task')
        op, (position, task) = matches[0], targets[0]
        node = None
        if schema == 'HostToDevice':
            if any(endpoint.get('name') != 'HostToDevice' + key for endpoint in (start, finish)):
                raise CaptureError('unsupported native flow identity')
            node = op
            op = _native_host(node, task, host)
            native_link_count += 1
        if position in used:
            raise CaptureError('device task assigned more than once')
        used.add(position)
        if schema == 'async_npu':
            torch_linked.add(position)
        if schema == 'async_npu' and _decimal(finish['ts']) < _decimal(start['ts']):
            raise CaptureError('reversed CPU/NPU flow')
        owners = [m for m in markers if (m['pid'], m['tid']) == (op['pid'], op['tid'])
                  and _decimal(m['ts']) <= _decimal(op['ts'])
                  and _decimal(op['ts']) + _decimal(op['dur'])
                  <= _decimal(m['ts']) + _decimal(m['dur'])]
        if len(owners) > 1:
            raise CaptureError('ambiguous host iteration scope')
        kind = task['args']['Task Type']
        identity = ':'.join(str(value) for value in [task['pid']] + [task['args'][k]
            for k in ('Physic Stream Id', 'Task Id', 'Batch Id', 'Subtask Id')])
        if identity in identities:
            raise CaptureError('duplicate device task identity')
        identities.add(identity)
        duration = float(_decimal(task['dur']))
        if not math.isfinite(duration):
            raise CaptureError('unrepresentable task duration')
        link = dict(flow_id=key, task_id=identity, host_start=str(start['ts']),
                    device_start=str(task['ts']), task_type=kind)
        if node is not None:
            # CANN host/device timestamps cross clock domains. The retained
            # direct-launch trace contains a 43ns inversion despite an exact
            # flow/connection match. Preserve it; do not invent a clock offset
            # or use timestamp proximity as the ownership criterion.
            link.update(flow_schema=schema, native_launch=dict(node), host=dict(op),
                        host_device_start_delta_us=str(_decimal(task['ts']) - _decimal(node['ts'])),
                        host_device_clock_order='device-before-host' if _decimal(task['ts']) < _decimal(node['ts'])
                                               else 'host-before-device')
        if kind == 'EVENT_RECORD':
            if op.get('name') != 'record_event':
                raise CaptureError('EVENT_RECORD requires the observed record_event CPU op')
            administrative.append(dict(reason='explicitly linked timing-event command, not compute',
                                       task=dict(task), host=dict(op), link=link))
            continue
        if not owners:
            excluded.append(dict(reason='complete CPU/NPU flow outside active host scopes',
                                 task=dict(task), link=link))
            continue
        name = owners[0]['name']
        correlation = schema + ':' + key
        index[name]['correlation_ids'].append(correlation)
        normalized.append(dict(task_id=identity, correlation_id=correlation,
            name=task['name'], kind=TASK_KINDS[kind],
            start_us=float(_decimal(task['ts']) - origin), duration_us=duration))
        links.append(dict(iteration=name, **link))
    if len(used) != len(tasks):
        raise CaptureError('device tasks lack explicit CPU/NPU correlation coverage')
    if any(not item['correlation_ids'] for item in index.values()):
        raise CaptureError('empty active iteration; no explicitly linked device work')
    return dict(iterations=list(index.values()), tasks=normalized, capture_complete=True,
        acquisition_evidence=dict(export_completed=True,
            schema='torch_npu_trace_cann_native_v1' if native_link_count else 'torch_npu_trace_async_npu_v1',
            association='explicit CPU/NPU flow + same-thread host scope nesting',
            observed_tasks=sum(task['args']['Task Type'] != 'EVENT_RECORD' for _, task in tasks),
            paired_flows=len(endpoints) + native_link_count, links=links,
            native_flow_count=native_link_count, auxiliary_native_flows=auxiliary,
            exclusions=excluded + administrative, silent_vendor_record_loss_detectable=False))


class Acquisition:
    """Default generic acquisition callback; callers own physical device policy."""
    def __init__(self, *, torch_module=None, npu_module=None):
        self.torch = torch_module or importlib.import_module('torch')
        self.npu = npu_module or importlib.import_module('torch_npu')

    def synchronize(self):
        self.torch.npu.synchronize()

    def provenance(self):
        return dict(python=platform.python_version(), torch=self.torch.__version__,
                    torch_npu=self.npu.__version__, logical_device=self.torch.npu.current_device(),
                    visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                    comparability_qualification='physical clock/load telemetry not provided')

    def comparability(self, before, after):
        return dict(comparable=False, reason='adapter has no verified clock/load telemetry',
                    runtime_identity_unchanged=before == after)

    def profile(self, prepared, *, samples=50, warmup=20, output):
        if samples < 1 or warmup < 20:
            raise ValueError('require positive samples and at least 20 warmups')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(state='incomplete', samples=samples, warmup=warmup,
                        provenance=self.provenance(), profiler_level='Level1',
                        data_simplification=False)
        try:
            with self.torch.inference_mode(), prepared.compiler.monitor_compilation():
                for _ in range(warmup):
                    prepared()
                self.synchronize()
                before = prepared.compiler.snapshot()
                names = [f'model_iteration_{i}' for i in range(samples)]
                activities = [self.npu.profiler.ProfilerActivity.CPU,
                              self.npu.profiler.ProfilerActivity.NPU]
                with self.npu.profiler.profile(activities=activities, record_shapes=False,
                        experimental_config=self.npu.profiler._ExperimentalConfig(
                            profiler_level=self.npu.profiler.ProfilerLevel.Level1,
                            data_simplification=False),
                        with_stack=False, schedule=self.npu.profiler.schedule(
                            wait=0, warmup=0, active=samples, repeat=1),
                        on_trace_ready=self.npu.profiler.tensorboard_trace_handler(
                            str(output / 'vendor'))) as profiler:
                    for name in names:
                        with self.torch.autograd.profiler.record_function(name):
                            prepared()
                            self.synchronize()
                        if prepared.compiler.snapshot() != before:
                            raise CaptureError('compilation detected during active profiler replay')
                        profiler.step()
            paths = list((output / 'vendor').rglob('trace_view.json'))
            if len(paths) != 1:
                raise CaptureError('completed export must produce exactly one trace_view.json')
            raw = paths[0].read_bytes()
            with gzip.open(output / 'trace_view.json.gz', 'wb') as stream:
                stream.write(raw)
            manifest.update(trace_sha256=hashlib.sha256(raw).hexdigest(), vendor_trace=str(paths[0]))
            trace = json.loads(raw)
            events = trace.get('traceEvents') if isinstance(trace, dict) else trace
            capture = normalize_trace(events, names, export_completed=True)
            manifest.update(state='complete', kernel_names=sorted({t['name']
                                for t in capture['tasks']}))
            capture['acquisition_evidence']['manifest'] = manifest
            return capture
        except Exception as error:
            manifest.update(state='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            (output / 'acquisition.json').write_text(json.dumps(manifest, indent=2) + '\n')


def create_acquisition():
    return Acquisition()
