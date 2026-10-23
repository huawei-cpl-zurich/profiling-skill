#!/usr/bin/env python3
"""Symmetric, single-correctness/ten-profile optimization experiments.

Adapters own compilation and device movement. Neither preparation nor capture may
silently execute a warmup. CPU oracle caches are private, trusted torch payloads.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys

HELPERS = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
sys.path.insert(0, str(HELPERS))
from profile_ten import capture_ten, kernel_identity
from profile_accounting import account_profile
from profile_capture import normalize_trace

PROTOCOL = 'optimization-pair-v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sources(paths):
    result = {str(Path(p).resolve()): sha(p) for p in paths}
    if not result:
        raise ValueError('source files must be declared')
    return result


def clone(value, torch):
    """Copy full storage, preserving strides/offsets instead of compacting views."""
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError('oracle fixtures must contain CPU strided tensors')
        count = value.untyped_storage().nbytes() // value.element_size()
        storage = value.detach().as_strided((count,), (1,), storage_offset=0).clone()
        return storage.as_strided(value.shape, value.stride(), value.storage_offset())
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v, torch) for v in value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: clone(v, torch) for k, v in value.items()}
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError('fixtures support nested tensors and primitive metadata only')


def fingerprint(value, torch):
    if isinstance(value, torch.Tensor):
        count = value.untyped_storage().nbytes() // value.element_size()
        raw = value.detach().as_strided((count,), (1,), storage_offset=0)
        return [str(value.dtype), list(value.shape), list(value.stride()), value.storage_offset(),
                hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, (tuple, list)):
        return [fingerprint(v, torch) for v in value]
    if isinstance(value, dict):
        return {k: fingerprint(v, torch) for k, v in value.items()}
    return value


def fixture(adapter, spec, cache, torch):
    provider = sources(getattr(adapter, 'case_sources', [inspect.getsourcefile(adapter.make_case)]))
    key = hashlib.sha256(json.dumps(dict(spec=spec, sources=provider), sort_keys=True).encode()).hexdigest()
    folder = cache / key
    if folder.exists():
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['protocol'] != PROTOCOL or manifest['sources'] != provider or manifest['case'] != spec:
            raise ValueError('oracle cache identity mismatch')
        if sha(folder / 'fixture.pt') != manifest['sha256']:
            raise ValueError('oracle cache payload changed')
        value = torch.load(folder / 'fixture.pt', map_location='cpu', weights_only=True)
        generated = 0
    else:
        value = adapter.make_case(spec)
        clone(value, torch)  # Reject arbitrary pickle objects and non-CPU data.
        if not isinstance(value, dict) or 'expected' not in value:
            raise ValueError('fixture must contain expected output')
        folder.mkdir(parents=True)
        torch.save(value, folder / 'fixture.pt')
        manifest = dict(protocol=PROTOCOL, case=spec, sources=provider, sha256=sha(folder / 'fixture.pt'))
        save(folder / 'manifest.json', manifest)
        generated = 1
    return value, dict(**manifest, cache=str(folder), oracle_calls=generated)


def verify_sources(expected):
    if sources(expected) != expected:
        raise ValueError('prepared source or binary changed during experiment')


def check_capture(result, prepared, count):
    """Independently recount normalized flows; adapters cannot assert identity."""
    latency = result['event_latency']
    samples = latency.get('samples_ms', [])
    if count != 10 or latency.get('count') != 10 or len(samples) != 10 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in samples):
        raise ValueError('exactly ten successful calls and positive samples required')
    latency.update(status='complete', mean_ms=statistics.mean(samples), sample_sd_ms=statistics.stdev(samples))
    profile = result['profile']
    # Preserve export failures and valid samples; missing accounting is distinct.
    result['kernel_identity'] = dict(status='unavailable', names=[], missing=[])
    profile['accounting_status'] = 'incomplete'
    if profile.get('export_status') != 'complete':
        profile['accounting_status'] = 'incomplete'
        return False
    trace = Path(profile['vendor_trace'])
    if sha(trace) != profile['trace_sha256']:
        raise ValueError('exported trace digest changed')
    raw = json.loads(trace.read_text())
    events = raw.get('traceEvents') if isinstance(raw, dict) else raw
    normalized = normalize_trace(events, [f'model_iteration_{i}' for i in range(10)], export_completed=True)
    if normalized != json.loads(Path(profile['capture_path']).read_text()):
        raise ValueError('normalized capture disagrees with retained vendor trace')
    if len(normalized['iterations']) != 10:
        raise ValueError('ten explicitly owned iterations required')
    accounted = account_profile(normalized['iterations'], normalized['tasks'], capture_complete=True)
    result['kernel_identity'] = kernel_identity(accounted, prepared.expected_kernels)
    profile['accounting_status'] = 'complete'
    result['operation_accounting'] = accounted
    result['artifacts'] = sources([profile[key] for key in
        ('vendor_trace', 'capture_path', 'accounting_path') if profile.get(key)])
    return result['kernel_identity']['status'] == 'complete'


def run_pair(adapter, case_spec, output, cache, provenance, order=('before', 'after'), *, torch=None):
    """Run both correctness checks first; failures retain a machine-readable report."""
    if set(order) != {'before', 'after'} or len(order) != 2:
        raise ValueError('order must contain before and after exactly once')
    torch = torch or importlib.import_module('torch')
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, status='failed', case=case_spec, provenance=provenance,
                  order=list(order), timing_kind='profiled-device-events', arms={},
                  warmup_calls=0, discovery_calls=0, unprofiled_timing_calls=0)
    try:
        shared, report['fixture'] = fixture(adapter, case_spec, cache, torch)
        prepared = {}
        for arm in order:
            folder = output / arm
            folder.mkdir()
            private = clone(shared, torch)
            before = fingerprint(private, torch)
            row = report['arms'][arm] = dict(correctness=dict(status='incomplete', calls=0))
            item = prepared[arm] = adapter.prepare(arm, case_spec, private, folder)
            if not item.expected_kernels or not all(isinstance(n, str) and n for n in item.expected_kernels):
                raise ValueError('each arm requires exact expected kernel names')
            row.update(sources=sources(item.sources), metadata=item.metadata)
            row['correctness']['calls'] += 1
            actual = item.run()
            diagnostics = adapter.validate(arm, actual, private, case_spec)
            if diagnostics is False or isinstance(diagnostics, dict) and diagnostics.get('passed') is False:
                raise ValueError('correctness rejected: ' + arm)
            if fingerprint(private, torch) != before:
                raise ValueError('correctness mutated fixture: ' + arm)
            verify_sources(row['sources'])
            row['correctness'].update(status='complete', diagnostics=diagnostics)
        for arm in order:
            item, row = prepared[arm], report['arms'][arm]
            verify_sources(row['sources'])
            state = getattr(item, 'compilation_state', lambda: None)
            compiled = json.dumps(state(), sort_keys=True, allow_nan=False)
            row['compilation_state_before_capture'] = json.loads(compiled)
            count = 0
            def counted():
                nonlocal count
                if count >= 10:
                    raise ValueError('capture attempted extra forward')
                count += 1
                return item.run()
            try:
                capture = getattr(adapter, 'capture', None)
                result = (capture(counted, item, output / arm / 'capture') if capture else
                          capture_ten(counted, output / arm / 'capture',
                                      expected_kernel=item.expected_kernels, metric_group='PipeUtilization'))
                row.update(capture=result, profiled_calls=count)
                final_state = json.dumps(state(), sort_keys=True, allow_nan=False)
                row['compilation_state_after_capture'] = json.loads(final_state)
                if compiled != final_state:
                    raise ValueError('compilation changed during capture')
                verify_sources(row['sources'])
                row['status'] = 'complete' if check_capture(result, item, count) else 'incomplete'
            except Exception as error:
                row.update(status='failed', profiled_calls=count,
                           error=dict(type=type(error).__name__, message=str(error)))
        statuses = [row['status'] for row in report['arms'].values()]
        report['status'] = 'failed' if 'failed' in statuses else ('complete' if set(statuses) == {'complete'} else 'incomplete')
        if report['status'] == 'complete':
            report['speedup_before_over_after'] = (report['arms']['before']['capture']['event_latency']['mean_ms'] /
                                                   report['arms']['after']['capture']['event_latency']['mean_ms'])
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        save(output / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'case', 'output', 'cache', 'provenance'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    module, name = args.adapter.split(':', 1)
    adapter = getattr(importlib.import_module(module), name)()
    report = run_pair(adapter, json.loads(Path(args.case).read_text()), args.output, args.cache,
                      json.loads(Path(args.provenance).read_text()))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
