#!/usr/bin/env python3
"""Ten independent full paired Model acquisitions, with between-run statistics."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(case):
    values = {}
    for arm in ('baseline', 'candidate'):
        values[arm + '_latency_ms'] = case['latency'][arm]['median_ms']
        for kind in ('all_kernel', 'device', 'aicpu'):
            values[arm + '_' + kind + '_busy_us'] = case[kind + '_busy_median_us'][arm]
    values['latency_speedup'] = case['latency']['latency_speedup']
    values['busy_time_speedup'] = case['busy_time_speedup']
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('non-finite/negative metric: ' + name)
        if value == 0 and 'aicpu' not in name:
            raise ValueError('nonpositive metric: ' + name)
    return values


def aggregate(reports, requested=10):
    """Summarize complete paired runs only; retain every failed run explicitly."""
    if type(requested) is not int or requested < 1 or len(reports) > requested:
        raise ValueError('requested repetitions must be positive and cover all supplied reports')
    result = dict(state='incomplete', requested_repetitions=requested,
        completed_repetitions=0, failed_repetitions=0, failures=[], cases=[],
        estimator='mean and sample standard deviation (ddof=1) of per-repetition medians',
        comparability=dict(comparable=False, reasons=[]))
    identity = None
    rows = {}
    comparable = True
    for index, report in enumerate(reports, 1):
        try:
            if not isinstance(report, dict):
                raise ValueError('single-run report must be an object')
            if report.get('state') != 'complete':
                raise ValueError(report.get('error', 'single-run acquisition did not complete'))
            cases = report['cases']
            if not cases or any(row.get('state') != 'complete' for row in cases):
                raise ValueError('missing or incomplete cases')
            keyed = {row['id']: row for row in cases}
            if len(keyed) != len(cases):
                raise ValueError('duplicate case identifiers')
            values = {name: metrics(row) for name, row in keyed.items()}
            current = dict(sources=report['sources'], cases={name: row['specialization']['sha256']
                           for name, row in keyed.items()})
            if identity is not None and identity != current:
                raise ValueError('source or case specialization changed between repetitions')
            identity = current
            for name, measurement in values.items():
                target = rows.setdefault(name, {})
                for key, value in measurement.items():
                    target.setdefault(key, []).append(value)
            result['completed_repetitions'] += 1
            if report.get('comparability', {}).get('comparable') is not True:
                comparable = False
                result['comparability']['reasons'].append(dict(repetition=index,
                    assessment=report.get('comparability', {'reasons': ['missing telemetry']})))
        except (KeyError, TypeError, ValueError) as error:
            result['failures'].append(dict(repetition=index, error=str(error)))
    result['failed_repetitions'] = len(result['failures'])
    result['unattempted_repetitions'] = requested - len(reports)
    for name, measurements in rows.items():
        summary = {}
        for key, values in measurements.items():
            summary[key] = dict(values=values, n=len(values), mean=statistics.mean(values),
                sample_stddev=statistics.stdev(values) if len(values) > 1 else None,
                unit='ms' if key.endswith('_ms') else 'us' if key.endswith('_us') else 'ratio')
        result['cases'].append(dict(id=name, metrics=summary))
    if result['completed_repetitions'] == requested:
        result['state'] = 'complete'
    result['comparability']['comparable'] = comparable and result['state'] == 'complete'
    # Per-run telemetry does not establish matching conditions ACROSS runs. Keep
    # this qualification even when every individual paired acquisition is sound.
    result['cross_repetition_comparability'] = dict(comparable=False,
        reasons=['verify physical device, clock/load policy and runtime across retained runs'])
    return result


def run_repeated(output, arguments, repetitions=10, *, command=None):
    """Fresh child processes avoid Dynamo cache/recompilation state accumulation.

    Run inside one externally selected/masked device lease. Children inherit that
    same environment; this controller never selects devices or changes runtimes.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    command = command or [sys.executable, str(Path(__file__).with_name('model_benchmark.py'))]
    reports, runs = [], []
    initial = aggregate([], repetitions)
    initial['runs'] = []
    save(output / 'summary.json', initial)
    for index in range(1, repetitions + 1):
        child_output = output / f'run-{index:04d}'
        invocation = [*command, *arguments, '--output', str(child_output)]
        returncode = None
        with (output / f'run-{index:04d}.log').open('w') as log:
            try:
                returncode = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT,
                                            check=False).returncode
                report = json.loads((child_output / 'report.json').read_text())
                if not isinstance(report, dict):
                    raise ValueError('single-run report must be an object')
                if returncode != 0:
                    report = dict(state='failed', error=report.get('error',
                        f'child exited {returncode} despite report state {report.get("state")}'))
            except (OSError, ValueError) as error:
                report = dict(state='failed', error=f'{type(error).__name__}: {error}')
        reports.append(report)
        runs.append(dict(repetition=index, command=invocation, returncode=returncode,
            report=str(child_output.relative_to(output) / 'report.json'),
            log=f'run-{index:04d}.log'))
        result = aggregate(reports, repetitions)
        result['runs'] = runs
        save(output / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Other arguments are forwarded unchanged to model_benchmark.py.')
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--output', required=True, help='new parent evidence directory')
    options, arguments = parser.parse_known_args(argv)
    try:
        report = run_repeated(options.output, arguments, options.repetitions)
    except (OSError, ValueError) as error:
        print(json.dumps(dict(state='failed', error=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(state=report['state'], summary=str(Path(options.output) / 'summary.json'))))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
