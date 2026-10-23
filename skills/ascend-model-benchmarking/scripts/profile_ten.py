"""Operational ten-forward capture; caller owns correctness, compilation and lease."""
import hashlib
import importlib
import json
import math
import re
import statistics
from pathlib import Path


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def _error(error):
    return dict(error_type=type(error).__name__, error=str(error))


def resolve_triton_names(graphtexts):
    """Resolve recorded HOP indices to live JIT identities without executing.

    Graph text alone is not identity proof. These expected names still require
    exact matches in the ten-forward device trace before identity is complete.
    """
    indices = {int(value) for text in graphtexts
               for value in re.findall(r'\bkernel_idx\s*=\s*(\d+)\b', text)}
    if not indices:
        return None
    try:
        table = importlib.import_module('torch._higher_order_ops.triton_kernel_wrap').kernel_side_table
        names = []
        for index in sorted(indices):
            kernel = table.get_kernel(index)
            function = getattr(kernel, 'fn', None) or getattr(kernel, '_fn', None)
            name = getattr(function, '__name__', None) or getattr(kernel, '__name__', None)
            if not isinstance(name, str) or not name:
                return None
            names.append(name)
        return sorted(set(names))
    except (ImportError, AttributeError, KeyError, IndexError, TypeError):
        return None


def kernel_identity(accounted, expected):
    """Accounting assigns task iteration IDs from normalized correlation links."""
    tasks = accounted['tasks']
    observed = sorted({task['name'] for task in tasks if task['kind'] == 'device'})
    missing = sorted({name for iteration in accounted['iterations']
        for name in set(expected or []) - {task['name'] for task in tasks
            if task['kind'] == 'device' and task['iteration_id'] == iteration['iteration_id']}})
    return dict(status='observed-only' if expected is None else 'failed' if missing else 'complete',
                names=observed, missing=missing, source='same ten-forward capture')


def capture_ten(run, output, *, expected_kernel=None, torch_module=None, npu_module=None):
    """Exactly ten forwards, no warmup/discovery; event results precede export.

    Event timings are profiler-instrumented, not unprofiled latency measurements.
    Preparation must already validate correctness and strict fullgraph compilation.
    Existing vendor files are retained even when export/normalization fails.
    """
    torch = torch_module or importlib.import_module('torch')
    npu = npu_module or importlib.import_module('torch_npu')
    capture_module = importlib.import_module('profile_capture')
    accounting = importlib.import_module('profile_accounting')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if expected_kernel is not None and (not expected_kernel or not isinstance(expected_kernel, (list, tuple))
            or not all(isinstance(name, str) and name for name in expected_kernel)):
        raise ValueError('expected_kernel must be a list of exact kernel names or None')
    result = dict(protocol='single_capture_ten_forwards', forward_calls=0,
                  event_latency=dict(status='incomplete', count=0, samples_ms=[],
                                     instrumentation='torch_npu profiler active'),
                  profile=dict(status='incomplete', export_status='incomplete', accounting_status='incomplete',
                               export_trigger='step-after-ten-event-samples',
                               analysis_max_process_number=1, analysis_calls=0),
                  kernel_identity=dict(status='unavailable', names=[],
                                       expected=list(expected_kernel or []), missing=[]))
    names = [f'model_iteration_{i}' for i in range(10)]
    # All ten forwards belong to one active profiler step. Complete that step
    # only after event samples are durable; no model invocation advances it.
    try:
        with torch.inference_mode():
            with npu.profiler.profile(
                    activities=[npu.profiler.ProfilerActivity.CPU,
                                npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False, with_stack=False,
                    schedule=npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                    experimental_config=npu.profiler._ExperimentalConfig(
                        profiler_level=npu.profiler.ProfilerLevel.Level1,
                        data_simplification=False),
                    on_trace_ready=npu.profiler.tensorboard_trace_handler(
                        str(output / 'vendor'), analyse_flag=False)) as profiler:
                pairs = []
                for name in names:
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    with torch.autograd.profiler.record_function(name):
                        start.record()
                        result['forward_calls'] += 1
                        run()
                        end.record()
                    pairs.append((start, end))
                torch.npu.synchronize()
                for start, end in pairs:
                    elapsed = float(start.elapsed_time(end))
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError('invalid NPU event elapsed time')
                    result['event_latency']['samples_ms'].append(elapsed)
                    result['event_latency']['count'] += 1
                samples = result['event_latency']['samples_ms']
                result['event_latency'].update(status='complete', mean_ms=statistics.mean(samples),
                    sample_sd_ms=statistics.stdev(samples))
                # This is deliberately inside the context, before export starts.
                _save(output / 'latency.json', result['event_latency'])
                _save(output / 'forward-count.json', dict(forward_calls=result['forward_calls']))
                print(json.dumps(dict(event='profiled-event-latencies-ready', output=str(output),
                                      event_latency=result['event_latency'])), flush=True)
                profiler.step()
        # Capture/finalization is complete. Analyse retained raw data exactly once,
        # outside profiling, with an explicit bounded analysis worker count.
        analyse = importlib.import_module('torch_npu.profiler.profiler').analyse
        result['profile']['analysis_calls'] += 1
        analyse(str(output / 'vendor'), max_process_number=1)
        paths = list((output / 'vendor').rglob('trace_view.json'))
        if len(paths) != 1:
            raise capture_module.CaptureError('completed export must produce exactly one trace_view.json')
        trace_path = paths[0]
        digest = hashlib.sha256()
        with trace_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        result['profile'].update(vendor_trace=str(trace_path), trace_sha256=digest.hexdigest(),
                                 export_status='complete')
        with trace_path.open() as stream:
            raw = json.load(stream)
        events = raw.get('traceEvents') if isinstance(raw, dict) else raw
        capture = capture_module.normalize_trace(events, names, export_completed=True)
        summary = accounting.account_profile(capture['iterations'], capture['tasks'], capture_complete=True)
        _save(output / 'normalized.json', capture)
        _save(output / 'accounting.json', summary)
        result['profile']['accounting_status'] = 'complete'
        result['kernel_identity'].update(kernel_identity(summary, expected_kernel))
        result['profile'].update(status='complete', capture_path=str(output / 'normalized.json'),
                                 accounting_path=str(output / 'accounting.json'))
    except Exception as error:
        result['profile'].update(status='failed', **_error(error))
    finally:
        _save(output / 'result.json', result)
    return result
