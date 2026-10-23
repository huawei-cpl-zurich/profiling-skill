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
    sentinel_identity = {'Model Id': 2**32 - 1, 'Task Id': 0, 'Batch Id': 1,
                         'Subtask Id': 2**32 - 1, 'connection_id': 2**64 - 1}
    return (event.get('name') == 'PLACE_HOLDER_SQE' and event.get('cat') is None
            and _decimal(event.get('dur')) in (Decimal('0'), Decimal('0.02'))
            and all(type(args.get(key)) is int and args[key] == value
                    for key, value in sentinel_identity.items())
            and type(event.get('pid')) is int and event['pid'] >= 0
            and type(event.get('tid')) is int and event['tid'] >= 0
            and type(args.get('Physic Stream Id')) is int
            and args['Physic Stream Id'] == event['tid'])


def _event_record(event):
    """Observed timing-event command; still requires its complete host flow."""
    args = event['args']
    return (event.get('name') == 'EVENT_RECORD' and event.get('cat') is None
            and _decimal(event.get('dur')) == Decimal('0.02')
            and all(type(args.get(key)) is int and args[key] == value for key, value in
                    {'Model Id': 2**32 - 1, 'Subtask Id': 2**32 - 1, 'Batch Id': 0}.items())
            and all(type(args.get(key)) is int and 0 <= args[key] < maximum for key, maximum in
                    [('Task Id', 2**32 - 1), ('connection_id', 2**64 - 1)])
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
    tasks, administrative = [], []
    for position, event in enumerate(events):
        if event.get('ph') == 'X' and event.get('cat') == 'cpu_op' and event not in markers:
            host[_coordinate(event)].append(event)
        if event.get('ph') == 'X' and 'Task Type' in event.get('args', {}):
            kind = event['args']['Task Type']
            if kind in ('PROFILER_TRACE_EX', 'MODEL_EXECUTE',
                        'PROFILING_ENABLE', 'PROFILING_DISABLE'):
                administrative.append(dict(reason='vendor administrative task, not a kernel',
                                           task=dict(event)))
                continue
            if kind == 'PLACE_HOLDER_SQE' and _queue_placeholder(event):
                _coordinate(event)  # Validate the original timestamp and coordinates.
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
    index = {name: dict(iteration_id=name, correlation_ids=[]) for name in names}
    used, identities, normalized, links, excluded = set(), set(), [], [], []
    origin = min(_decimal(m['ts']) for m in markers)
    for key, phases in endpoints.items():
        if set(phases) != {'s', 'f'}:
            raise CaptureError('incomplete async_npu flow')
        start, finish = phases['s'], phases['f']
        matches = host[_coordinate(start)]
        targets = task_at[_coordinate(finish)]
        if len(matches) != 1 or len(targets) != 1:
            raise CaptureError('flow endpoint must identify exactly one CPU op and device task')
        op, (position, task) = matches[0], targets[0]
        if position in used:
            raise CaptureError('device task assigned more than once')
        used.add(position)
        if _decimal(finish['ts']) < _decimal(start['ts']):
            raise CaptureError('reversed CPU/NPU flow')
        owners = [m for m in markers if (m['pid'], m['tid']) == (start['pid'], start['tid'])
                  and _decimal(m['ts']) <= _decimal(start['ts'])
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
        correlation = 'async_npu:' + key
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
        acquisition_evidence=dict(export_completed=True, schema='torch_npu_trace_async_npu_v1',
            association='explicit CPU/NPU flow + same-thread host scope nesting',
            observed_tasks=sum(task['args']['Task Type'] != 'EVENT_RECORD' for _, task in tasks),
            paired_flows=len(endpoints), links=links,
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
