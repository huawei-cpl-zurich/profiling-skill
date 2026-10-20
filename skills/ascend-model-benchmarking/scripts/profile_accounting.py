#!/usr/bin/env python3
"""Account normalized correlated device tasks without inferring vendor schemas.

``account_profile(iterations, tasks, capture_complete=True)`` accepts iteration
records with ``iteration_id`` and explicit ``correlation_ids``. Optional
``expected_task_ids`` must come from an independent workload inventory, not the
same observed rows. Tasks have ``task_id``, ``correlation_id``, ``name``, ``kind``
(``device`` or ``aicpu``), ``start_us`` and ``duration_us``. IDs are nonempty
strings, timestamps use one normalized clock, and task IDs identify actual tasks
(not kernel names). Distinct AIC/AIV tasks must have distinct IDs.

The caller must validate completed acquisition/export and CPU/NPU correlation
coverage before setting capture_complete. This module cannot detect records lost
before normalization without independent inventory. Host/warmup/unrelated tasks
must be excluded explicitly by the acquisition adapter with retained evidence;
unknown correlations here are errors, never silently ignored.

CLI: python profile_accounting.py normalized-capture.json
The JSON object contains iterations, tasks, and capture_complete. Every primary
total is unfiltered. The CANNBot per-name median threshold is diagnostic ONLY.
"""
import argparse
from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
import json
import math
from numbers import Real
from pathlib import Path
import statistics
import sys


class AccountingError(ValueError):
    """The capture cannot support unambiguous per-iteration accounting."""


def _identity(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AccountingError(f'{key} must be a nonempty string')
    return value


def _identities(value, key):
    if not isinstance(value, list):
        raise AccountingError(f'{key} must be a list of unique identifiers')
    result = [_identity({key: item}, key) for item in value]
    if len(set(result)) != len(result):
        raise AccountingError(f'{key} contains duplicate identifiers')
    return result


def _number(record, key):
    value = record.get(key)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise AccountingError(f'{key} must be finite and nonnegative')
    # Keep integer timestamps exact, including those beyond float's 2**53 range.
    # Other Real inputs normalize to the JSON-compatible float representation.
    return value if isinstance(value, int) else float(value)


def _union_duration(tasks):
    """Union on the normalized device clock, not end-to-end latency."""
    # Endpoint arithmetic must not erase a short duration at an epoch-sized
    # origin. Rational decimal representations avoid both float cancellation
    # and a fixed Decimal precision limit; public outputs remain JSON numbers.
    intervals = [(Fraction(str(t['start_us'])), Fraction(str(t['duration_us'])))
                 for t in tasks]
    total, end = Fraction(0), Fraction(0)
    for start, stop in sorted((start, start + duration) for start, duration in intervals):
        total += max(Fraction(0), stop - max(end, start))
        end = max(end, stop)
    return float(total)


def account_profile(iterations, tasks, *, capture_complete):
    """Return JSON-safe sums and flag-only diagnostics; never mutate inputs.

    Busy-time is the sum of task durations, not a Model's elapsed latency. Union
    duration is a device activity diagnostic, also not end-to-end latency. The
    public result preserves input order and every normalized task, adding its
    iteration association and diagnostic annotation.
    """
    if capture_complete is not True:
        raise AccountingError('capture must have verified complete acquisition/export')
    if not isinstance(iterations, list) or not iterations:
        raise AccountingError('iterations must be a nonempty list')
    if not isinstance(tasks, list):
        raise AccountingError('tasks must be a list')
    iteration_rows, owners = {}, {}
    expected = {}
    for marker in iterations:
        if not isinstance(marker, Mapping):
            raise AccountingError('iteration must be an object')
        identity = _identity(marker, 'iteration_id')
        if identity in iteration_rows:
            raise AccountingError(f'duplicate iteration {identity}')
        correlations = _identities(marker.get('correlation_ids'), 'correlation_ids')
        if not correlations:
            raise AccountingError(f'iteration {identity} has no explicit correlations')
        for correlation in correlations:
            if correlation in owners:
                raise AccountingError(f'ambiguous correlation {correlation}')
            owners[correlation] = identity
        if 'expected_task_ids' in marker:
            expected[identity] = set(_identities(marker['expected_task_ids'],
                                                'expected_task_ids'))
        iteration_rows[identity] = []

    seen, retained, durations = set(), [], defaultdict(list)
    for source in tasks:
        if not isinstance(source, Mapping):
            raise AccountingError('task must be an object')
        identity = _identity(source, 'task_id')
        if identity in seen:
            raise AccountingError(f'duplicate task {identity}')
        seen.add(identity)
        correlation = _identity(source, 'correlation_id')
        if correlation not in owners:
            raise AccountingError(f'unknown correlation {correlation} for task {identity}')
        name = _identity(source, 'name')
        kind = source.get('kind')
        if kind not in ('device', 'aicpu'):
            raise AccountingError(f'unsupported task kind {kind!r}')
        row = dict(source)
        row.update(start_us=_number(source, 'start_us'),
                   duration_us=_number(source, 'duration_us'),
                   iteration_id=owners[correlation])
        if not math.isfinite(row['start_us'] + row['duration_us']):
            raise AccountingError('task endpoint must be finite')
        iteration_rows[owners[correlation]].append(row)
        retained.append(row)
        durations[kind, name].append(row['duration_us'])

    for identity, wanted in expected.items():
        actual = {row['task_id'] for row in iteration_rows[identity]}
        if wanted != actual:
            raise AccountingError(f'iteration {identity} inventory mismatch: '
                                  f'missing={sorted(wanted - actual)}, '
                                  f'unexpected={sorted(actual - wanted)}')
    thresholds = {key: max(50 * statistics.median(values), 10.0)
                  for key, values in durations.items()}
    for row in retained:
        threshold = thresholds[row['kind'], row['name']]
        row['diagnostic_threshold_us'] = threshold
        row['diagnostic_flags'] = (['duration_above_median_threshold']
                                   if row['duration_us'] > threshold else [])
    result = []
    for identity, rows in iteration_rows.items():
        device = [row for row in rows if row['kind'] == 'device']
        busy = math.fsum(row['duration_us'] for row in device)
        union = _union_duration(device)
        result.append(dict(iteration_id=identity, task_count=len(rows),
                           inventory_checked=identity in expected,
                           device_busy_us=busy, device_active_union_us=union,
                           device_overlap_us=max(0.0, busy - union),
                           aicpu_busy_us=math.fsum(row['duration_us'] for row in rows
                                                  if row['kind'] == 'aicpu')))
    return dict(schema_version=1, metric='sum_of_task_durations_not_elapsed_latency',
                filter_policy='diagnostic_only_no_rows_removed',
                iterations=result, tasks=retained)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.capture.read_text())
        if not isinstance(payload, dict):
            raise AccountingError('capture must be an object')
        result = account_profile(payload.get('iterations'), payload.get('tasks'),
                                 capture_complete=payload.get('capture_complete'))
    except (AccountingError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
