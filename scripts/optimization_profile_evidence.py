#!/usr/bin/env python3
"""Collect digest-bound paired timing and PMU evidence without device execution."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HELPERS = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
sys.path.insert(0, str(HELPERS))
from profile_accounting import account_profile
from profile_capture import normalize_trace
from profile_ten import kernel_identity


class EvidenceError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise EvidenceError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def samples(values, *, positive=False):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v)
                         or (v <= 0 if positive else v < 0) for v in values):
        raise EvidenceError('invalid timing samples')
    return dict(samples=values, count=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def numeric(value):
    if value.strip().upper() in ('', 'N/A', 'NA', 'NONE', '--'):
        return None
    try:
        number = Decimal(value.strip())
    except InvalidOperation as error:
        raise EvidenceError('invalid numeric CSV value: ' + value) from error
    if not number.is_finite():
        raise EvidenceError('nonfinite CSV value')
    converted = float(number)
    if not math.isfinite(converted):
        raise EvidenceError('unrepresentable CSV value')
    return converted


def operation_windows(accounted):
    """Span includes gaps between all operation tasks on the device clock."""
    windows = []
    for iteration in accounted['iterations']:
        owned = [task for task in accounted['tasks'] if task['iteration_id'] == iteration['iteration_id']]
        if not owned:
            raise EvidenceError('operation iteration contains no tasks')
        first = min(Decimal(str(task['start_us'])) for task in owned)
        last = max(Decimal(str(task['start_us'])) + Decimal(str(task['duration_us'])) for task in owned)
        active, end = Decimal(0), first
        for start, stop in sorted((Decimal(str(t['start_us'])), Decimal(str(t['start_us'])) +
                                   Decimal(str(t['duration_us']))) for t in owned):
            active += max(Decimal(0), stop-max(end,start))
            end = max(end,stop)
        span = float(last-first)
        windows.append(dict(iteration_id=iteration['iteration_id'], device_operation_span_us=span,
                            operation_active_union_us=float(active), device_idle_gap_us=float(last-first-active)))
    return windows


def pmu_rows(path, tasks, physical_die):
    """Join identities from the same capture; never infer ownership from time/name counts."""
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    required = {'Name', 'Task ID', 'Stream ID', 'Device_id', 'Context ID'}
    if not columns or len(set(columns)) != len(columns) or not required <= set(columns):
        raise EvidenceError('unsupported kernel_details CSV schema')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise EvidenceError('CSV row does not match its header')
    candidates = defaultdict(list)
    for task in tasks:
        if task['kind'] != 'device':
            raise EvidenceError('PMU join for non-device task is unsupported')
        identity = task['task_id'].split(':')
        if len(identity) != 5:
            raise EvidenceError('unsupported device task identity')
        _, stream, task_id, batch, context = identity
        candidates[(stream, task_id, task['name'])].append((task, batch, context))
    matched, used = [], set()
    for row in rows:
        if row['Device_id'].strip() != str(physical_die):
            raise EvidenceError('CSV physical device differs from capture provenance')
        key = (row['Stream ID'].strip(), row['Task ID'].strip(), row['Name'])
        found = candidates.get(key, [])
        # N/A is retained verbatim. It supplies no additional identity constraint;
        # stream/task/name must already have a unique proven trace identity.
        context = row['Context ID'].strip()
        if context.upper() not in ('N/A', 'NA', ''):
            found = [item for item in found if item[2] == context]
        if len(found) != 1:
            raise EvidenceError('CSV row lacks a unique trace task identity')
        task, batch, trace_context = found[0]
        if task['task_id'] in used:
            raise EvidenceError('duplicate CSV row for a trace task')
        used.add(task['task_id'])
        matched.append(dict(task_id=task['task_id'], iteration_id=task['iteration_id'],
                            trace_batch_id=batch, trace_context_id=trace_context,
                            csv_context_id=row['Context ID'], raw=row))
    if used != {task['task_id'] for task in tasks}:
        raise EvidenceError('CSV does not cover every proven device task')
    kernels = {}
    for name in sorted({task['name'] for task in tasks}):
        selected = [item for item in matched if item['raw']['Name'] == name]
        fields = {}
        for column in columns:
            if not (column.startswith(('aic_', 'aiv_')) or column in
                    ('aicore_time(us)', 'cube_utilization(%)', 'Duration(us)', 'Wait Time(us)')):
                continue
            values = [numeric(item['raw'][column]) for item in selected]
            available = [value for value in values if value is not None]
            fields[column] = dict(status='complete' if len(available) == len(values) else
                                  'partial' if available else 'unavailable', samples=values,
                                  available_count=len(available), unavailable_count=len(values)-len(available),
                                  mean=statistics.mean(available) if available else None,
                                  sample_sd=statistics.stdev(available) if len(available) > 1 else None)
        kernels[name] = dict(task_count=len(selected),
                             tasks_per_iteration=dict(Counter(item['iteration_id'] for item in selected)),
                             rows=selected, numeric_columns=fields)
    return dict(status='available' if any(c.startswith(('aic_', 'aiv_')) for c in columns)
                else 'unavailable', csv_sha256=sha(path), columns=columns, kernels=kernels,
                semantics='exported per-task values; no ratios summed, no denominator or roofline inferred')


def collect_pair(result_path, *, root=None, reanalysis=None):
    root = Path(root or Path.cwd()).resolve()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    result_path = resolve(result_path)
    report = load(result_path)
    if report.get('protocol') != 'optimization-pair-v1':
        raise EvidenceError('unsupported pair protocol')
    if any(report.get(key) != 0 for key in ('warmup_calls','discovery_calls','unprofiled_timing_calls')):
        raise EvidenceError('pair must declare zero additional execution calls')
    proof_path = resolve(reanalysis) if reanalysis else None
    proof = load(proof_path) if proof_path else None
    proof_case = None
    if proof and 'cases' in proof:
        matches = [(name, value) for name,value in proof['cases'].items()
                   if value.get('original_pair_sha256') == sha(result_path)]
        if len(matches) != 1:
            raise EvidenceError('reanalysis must uniquely identify the original pair')
        proof_case, proof = matches[0]
    if proof and proof.get('original_pair_sha256') != sha(result_path):
        raise EvidenceError('reanalysis belongs to a different original pair')
    physical = report['provenance']['physical_die']
    if type(physical) is not int or physical < 0:
        raise EvidenceError('explicit physical device provenance required')
    result = dict(protocol='optimization-profile-evidence-v1', original_pair_sha256=sha(result_path),
                  reanalysis_sha256=sha(proof_path) if proof_path else None,
                  reanalysis_case=proof_case,
                  case=report['case'], provenance=report['provenance'], fixture=report['fixture'],
                  new_device_forwards=0, new_captures=0, arms={})
    for arm in ('before', 'after'):
        row = report['arms'][arm]
        capture = row['capture']
        if row['correctness']['status'] != 'complete' or row['correctness']['calls'] != 1:
            raise EvidenceError('both arms require one successful correctness call')
        if row['profiled_calls'] != 10 or capture['forward_calls'] != 10:
            raise EvidenceError('both arms require ten actual profiled calls')
        profile = capture['profile']
        if profile['export_status'] != 'complete':
            raise EvidenceError('export is incomplete')
        trace = resolve(profile['vendor_trace'])
        if sha(trace) != profile['trace_sha256']:
            raise EvidenceError('vendor trace digest mismatch')
        raw = load(trace)
        normalized = normalize_trace(raw['traceEvents'] if isinstance(raw, dict) else raw,
                                     [f'model_iteration_{i}' for i in range(10)], export_completed=True)
        accounted = account_profile(normalized['iterations'], normalized['tasks'], capture_complete=True)
        claim = proof['arms'][arm] if proof else capture
        if proof and claim.get('status') != 'complete':
            raise EvidenceError('reanalysis is incomplete')
        if claim['kernel_identity']['status'] != 'complete':
            raise EvidenceError('kernel identity proof is incomplete')
        if proof:
            if claim['trace_sha256'] != sha(trace) or claim['capture_forward_calls'] != 10:
                raise EvidenceError('reanalysis trace or call identity mismatch')
            prefix = proof_case+'-' if proof_case else ''
            normalized_path = proof_path.parent / (prefix + arm + '-normalized.json')
            accounting_path = proof_path.parent / (prefix + arm + '-accounting.json')
            if (sha(normalized_path) != claim['normalized_sha256'] or
                    sha(accounting_path) != claim['accounting_sha256']):
                raise EvidenceError('reanalysis artifact digest mismatch')
        else:
            normalized_path, accounting_path = (resolve(profile[key]) for key in ('capture_path', 'accounting_path'))
            for artifact in (normalized_path, accounting_path):
                expected = capture.get('artifacts', {}).get(str(artifact.resolve()))
                if not expected or sha(artifact) != expected:
                    raise EvidenceError('retained capture artifact digest mismatch')
        saved = load(normalized_path)
        # Evidence metadata can grow between parser versions. Actual task and
        # iteration associations must agree exactly with the retained capture.
        if any(saved[key] != normalized[key] for key in ('iterations', 'tasks')):
            raise EvidenceError('retained normalized tasks differ from vendor trace')
        if load(accounting_path) != accounted:
            raise EvidenceError('retained accounting differs from recomputed tasks')
        identity = kernel_identity(accounted, claim['kernel_identity']['names'])
        if identity['status'] != 'complete' or identity['names'] != claim['kernel_identity']['names']:
            raise EvidenceError('recomputed kernel identity differs from retained proof')
        latency = samples(capture['event_latency']['samples_ms'], positive=True)
        if latency['count'] != 10 or capture['event_latency']['count'] != 10:
            raise EvidenceError('ten positive event samples required')
        windows = operation_windows(accounted)
        detail = dict(status='complete', correctness=row['correctness'], sources=row['sources'],
                      metadata=row['metadata'], trace_sha256=sha(trace),
                      compilation_before=row.get('compilation_state_before_capture'),
                      compilation_after=row.get('compilation_state_after_capture'),
                      normalized_sha256=sha(normalized_path), accounting_sha256=sha(accounting_path),
                      kernel_identity=identity, event_latency_ms=latency,
                      event_instrumentation=capture['event_latency'].get('instrumentation'),
                      device_work_us={key: samples([item[key] for item in accounted['iterations']])
                                      for key in ('device_busy_us', 'device_active_union_us', 'device_overlap_us')},
                      device_window_us={key:samples([item[key] for item in windows])
                                        for key in ('device_operation_span_us','operation_active_union_us','device_idle_gap_us')},
                      device_windows=windows,
                      device_work_semantics='busy is summed task duration; union/overlap are activity. Device span includes inter-task gaps but omits first enqueue and host/profiler overhead; none is unprofiled end-to-end latency',
                      per_iteration=accounted['iterations'], tasks=accounted['tasks'])
        csv_path = trace.parent / 'kernel_details.csv'
        try:
            detail['metrics'] = pmu_rows(csv_path, accounted['tasks'], physical) if csv_path.exists() else dict(
                status='unavailable', reason='kernel_details.csv absent')
        except EvidenceError as error:
            detail.update(status='incomplete', metrics=dict(status='identity-failed', reason=str(error)))
        result['arms'][arm] = detail
    result['status'] = 'complete' if all(row['status'] == 'complete' for row in result['arms'].values()) else 'incomplete'
    result['event_mean_ratio_before_over_after'] = (result['arms']['before']['event_latency_ms']['mean'] /
                                                   result['arms']['after']['event_latency_ms']['mean'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--reanalysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = collect_pair(args.result, root=args.root, reanalysis=args.reanalysis)
    with args.output.open('x') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return int(evidence['status'] != 'complete')


if __name__ == '__main__':
    raise SystemExit(main())
