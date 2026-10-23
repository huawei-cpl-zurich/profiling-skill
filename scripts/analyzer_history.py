"""Controller-owned round history and conservative retained-measurement pairing."""
from __future__ import annotations

import difflib
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def records(history):
    return [json.loads(path.read_text()) for path in sorted(history.glob('round-*/record.json'),
            key=lambda path: int(path.parent.name.split('-')[1]))]


def context(history, source, commit, round_index):
    prior = records(history)
    sha = hashlib.sha256(source).hexdigest()
    distinct = [item for item in prior if item['source_sha256'] != sha]
    previous = distinct[-1]['round'] if distinct else None
    repeated = next((item['round'] for item in reversed(prior) if item['source_sha256'] == sha), None)
    status = 'unchanged' if repeated is not None else ('new' if prior else 'first')
    result = dict(contract_version=2, round=round_index, candidate_commit=commit,
                  source_sha256=sha, status=status, previous_distinct=previous,
                  repeated_round=repeated, history=[{key: item.get(key) for key in
                      ('round', 'candidate_commit', 'source_sha256', 'analysis_valid', 'analysis_error')} for item in prior])
    if previous is not None:
        old = (history / f'round-{previous}/candidate.py').read_text()
        result['source_diff'] = ''.join(difflib.unified_diff(old.splitlines(True),
            source.decode().splitlines(True), fromfile=f'round-{previous}', tofile=f'round-{round_index}'))
    return result


def validate_round(report, ctx):
    if report.get('contract_version') != 2:
        raise ValueError('fresh campaigns require analysis contract_version 2')
    if report.get('source_sha256') != ctx['source_sha256']:
        raise ValueError('analysis source_sha256 differs from frozen kernel')
    for field in ('optimizations', 'utilization'):
        items = report.get(field)
        if not isinstance(items, list) or not items:
            raise ValueError(field + ' requires entries, including explicit unavailable states')
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('description'), str) or not item['description'].strip():
                raise ValueError(field + ' requires description')
            allowed = ('observed', 'none') if field == 'optimizations' else ('measured', 'unavailable')
            if item.get('status') not in allowed:
                raise ValueError(field + ' has invalid status')
            if field == 'optimizations' and item['status'] == 'observed' and not item.get('diff_lines'):
                raise ValueError('observed optimization requires source diff_lines')
            if field == 'optimizations' and item['status'] == 'observed':
                lines = item['diff_lines']
                allowed_lines = ctx.get('source_diff', '').splitlines()
                if not isinstance(lines, list) or not all(isinstance(line, str) and
                        line.startswith(('+', '-')) and not line.startswith(('+++', '---'))
                        and line in allowed_lines for line in lines):
                    raise ValueError('optimization diff_lines must identify actual source changes')
            if field == 'utilization':
                if not all(isinstance(item.get(key), list) and item[key] and
                           all(isinstance(label, str) and label.strip() for label in item[key])
                           for key in ('cases', 'kernels')):
                    raise ValueError('utilization requires cases and kernels (or explicit unknown label)')
                if item['status'] == 'measured' and not set(item.get('evidence', [])):
                    raise ValueError('measured utilization requires evidence')
                if not set(item.get('evidence', [])) <= {e['path'] for e in report['evidence']}:
                    raise ValueError('utilization refers to unlisted evidence')
    if report['profiling']['status'] == 'acquired':
        if not report['bottlenecks']:
            raise ValueError('acquired profile requires bottleneck assessment with confidence')
        covered = {(str(case), str(kernel)) for item in report['utilization']
                   for case in item['cases'] for kernel in item['kernels']}
        expected = {(str(case), str(kernel)) for item in report['evidence'] if item['kind'] == 'profile'
                    for case in item['cases'] for kernel in item['kernels']}
        if not expected <= covered:
            raise ValueError('utilization must cover all cited profile cases/kernels')
    if not report['ceilings']:
        raise ValueError('roofline requires estimate or explicit not_established ceiling')
    comparison = report.get('comparison')
    if not isinstance(comparison, dict) or comparison.get('status') != ctx['status']:
        raise ValueError('comparison status must match controller history')
    selections = comparison.get('rounds')
    if not isinstance(selections, list) or len(selections) > 2 or not all(type(i) is int for i in selections) or len(set(selections)) != len(selections):
        raise ValueError('comparison permits at most two distinct historical rounds')
    previous = ctx['previous_distinct']
    if previous is not None and previous not in selections:
        raise ValueError('comparison must include previous distinct verified kernel')
    eligible = {item['round'] for item in ctx['history'] if item['source_sha256'] != ctx['source_sha256']}
    if not set(selections) <= eligible:
        raise ValueError('comparison selects unknown or unchanged kernel')
    sources = {item['round']: item['source_sha256'] for item in ctx['history']}
    if len({sources[index] for index in selections}) != len(selections):
        raise ValueError('comparison selections must identify distinct source kernels')
    if len(selections) > (1 if previous is not None else 0) and not comparison.get('selection_reason'):
        raise ValueError('optional older comparison requires selection_reason')
    if not isinstance(comparison.get('interpretation'), str) or not comparison['interpretation'].strip():
        raise ValueError('comparison requires interpretation and uncertainty')
    # Numbers belong to the controller; do not accept actor-authored deltas.
    if 'pairs' in comparison or 'results' in comparison:
        raise ValueError('comparison pairs are computed by controller')


def _numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def measurements(summary, receipt, workload_identity=None):
    """Extract only measured numerics; retain original units via exact metric names.

    Per-capture rows are averaged per exported selector. Never sum pipeline ratios
    or invent whole-workload utilization when the kernel decomposition changes.
    """
    metrics = {}
    def flatten(value, prefix):
        if isinstance(value, dict):
            for name, item in value.items():
                if name.lower() not in ('count', 'samples', 'warmup', 'iterations'):
                    flatten(item, prefix + '/' + name)
        elif isinstance(value, list) and value and all(_numeric(item) for item in value):
            metrics[prefix + '/median'] = median(value)
        elif _numeric(value):
            metrics[prefix] = value
    for boundary in ('timing', 'latency', 'busy_time'):
        flatten(summary.get(boundary), boundary)
    cases = summary.get('cases')
    seeded = cases is None and isinstance(summary.get('correctness'), list)
    if seeded:
        cases = [str(item['check']) for item in summary['correctness'] if 'check' in item]
    rows = {}
    for capture in summary.get('captures', []):
        selector = capture.get('selector')
        for category in ('basic_rows', 'pipe_rows'):
            for row in capture.get(category, []):
                # Hardware sub-blocks must remain separate.
                block = str(row.get('sub_block_id', 'all'))
                case = capture.get('case', row.get('case'))
                if case is None and seeded and summary.get('launch_contract', {}).get('profile_stream_verified'):
                    # Checks identify seeded validation cases, not individual profile
                    # rows. The verified standalone stream is one workload launch.
                    case = 'seeded-workload-aggregate'
                if case is None and cases and len(cases) == 1:
                    case = str(cases[0])
                if case is None:
                    continue  # Never average unrelated workload cases.
                for name, value in row.items():
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                    normalized = name.lower().replace('_', ' ')
                    if _numeric(value) and not any(token in normalized for token in
                            (' id', 'start time', 'block num', 'sub block', 'task type')):
                        key = f'kernel/{selector}/{case}/{category}/{block}/{name}'
                        rows.setdefault(key, []).append(value)
    metrics.update({key: mean(values) for key, values in rows.items()})
    request = receipt.get('request', {})
    # Physical device intentionally excluded: devices on the same node are equivalent.
    ctx = {name: receipt.get(name) for name in ('target', 'runtime')}
    ctx.update({name: summary.get(name) for name in
        ('cases', 'collector', 'granularity', 'warmup', 'samples')})
    ctx['cases'] = cases
    ctx['workload_identity'] = workload_identity
    ctx['capture'] = {name: request.get(name) for name in
        ('operation', 'verb', 'implementation', 'timing_captures', 'pipe_captures')}
    ctx['timing_semantics'] = {name: summary.get(name, {}).get('metric')
        for name in ('latency', 'busy_time') if isinstance(summary.get(name), dict)}
    return dict(context=ctx, metrics=metrics, source_sha256=summary['source_sha256'],
                provenance={key: receipt.get(key, summary.get(key)) for key in ('call_id', 'handle')},
                kernels=sorted({str(c.get('selector')) for c in summary.get('captures', [])}))


def _aggregate(acquisitions):
    grouped = {}
    for item in acquisitions:
        key = json.dumps(item['context'], sort_keys=True)
        grouped.setdefault(key, []).append(item)
    results = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: json.dumps(item.get('provenance', {}), sort_keys=True))
        values = {}
        for item in items:
            for name, value in item['metrics'].items():
                values.setdefault(name, []).append(value)
        results.append(dict(context=items[0]['context'],
            metrics={name: median(samples) for name, samples in values.items()},
            metric_counts={name: len(samples) for name, samples in values.items()},
            kernels=sorted({kernel for item in items for kernel in item['kernels']}),
            acquisition_count=len(items), provenance=[item.get('provenance', {}) for item in items]))
    return results


def pair(current, previous):
    pairs, gaps = [], []
    previous = _aggregate(previous)
    for now in _aggregate(current):
        compatible = [old for old in previous if old['context'] == now['context']
                      and all(now['context'].get(key) is not None for key in ('target', 'runtime', 'cases', 'workload_identity'))
                      and bool(now['context']['cases'])
                      and any(now['context']['capture'].get(key) for key in ('operation', 'verb'))
                      and (now['context']['capture'].get('pipe_captures') is not None
                           or (all(now['context'].get(key) is not None for key in ('samples', 'warmup'))
                               and (all(now['context'].get(key) is not None for key in ('collector', 'granularity'))
                                    or any(now['context']['timing_semantics'].values()))))]
        if len(compatible) != 1:
            gaps.append('No unique compatible acquisition: runtime/node/cases/capture semantics missing or different')
            continue
        old = compatible[0]
        for metric in sorted(set(now['metrics']) | set(old['metrics'])):
            if metric not in now['metrics'] or metric not in old['metrics']:
                gaps.append('Unavailable paired metric: ' + metric)
                continue
            if '/pipe_rows/' in metric and now['context']['capture'].get('pipe_captures') is None and not all(
                    now['context'].get(key) is not None for key in ('collector', 'granularity')):
                gaps.append('Missing profiler metric semantics: ' + metric)
                continue
            if metric.startswith('kernel/') and now['kernels'] != old['kernels']:
                gaps.append('Changed kernel decomposition; use workload timing: ' + metric)
                continue
            before, after = old['metrics'][metric], now['metrics'][metric]
            pairs.append(dict(metric=metric, previous=before, current=after,
                              delta=after-before, percent=(after-before)/before*100 if before else None,
                              aggregation='median_of_compatible_acquisitions',
                              current_acquisitions=now['metric_counts'][metric],
                              previous_acquisitions=old['metric_counts'][metric],
                              current_provenance=now['provenance'], previous_provenance=old['provenance']))
    if not pairs:
        gaps.append('No comparable numeric measurements retained')
    return dict(pairs=pairs, gaps=sorted(set(gaps)))


def finalize(report, ctx, history, acquisitions):
    """Pure report enrichment, also usable for retrospective retained-evidence checks."""
    report = json.loads(json.dumps(report))
    if (ctx['status'] == 'first' and not ctx['history']
            and ctx['previous_distinct'] is None and ctx['repeated_round'] is None):
        comparison = report.get('comparison')
        if isinstance(comparison, dict) and comparison.get('rounds') != []:
            report['controller_diagnostics'] = dict(first_round_selection=dict(
                reported_rounds=comparison.get('rounds'), normalized_rounds=[],
                reason='No retained historical kernel exists; controller owns the empty selection'))
            comparison['rounds'] = []
    validate_round(report, ctx)
    report['comparison']['repeated_round'] = ctx['repeated_round']
    report['comparison']['results'] = []
    for index in report['comparison']['rounds']:
        old = json.loads((history / f'round-{index}/record.json').read_text())
        result = dict(round=index, **pair(acquisitions, old.get('acquisitions', [])),
                      previous_bottlenecks=old.get('bottlenecks', []), estimated_rooflines=[])
        old_report = history / f'round-{index}/analysis.json'
        ceilings = json.loads(old_report.read_text())['ceilings'] if old_report.is_file() else []
        for ceiling in report['ceilings']:
            matches = [item for item in ceilings if item['status'] == ceiling['status'] == 'estimated'
                       and all(item.get(key) == ceiling.get(key) for key in ('resource', 'basis', 'assumptions'))
                       and item.get('measured_us') and ceiling.get('measured_us')]
            if len(matches) == 1 and result['pairs']:
                before = matches[0]['lower_bound_us'] / matches[0]['measured_us']
                after = ceiling['lower_bound_us'] / ceiling['measured_us']
                result['estimated_rooflines'].append(dict(resource=ceiling['resource'],
                    previous_ideal_fraction=before, current_ideal_fraction=after, delta=after-before))
            else:
                result['gaps'].append('Roofline not paired: missing or incompatible estimated ceiling for ' + ceiling['resource'])
        report['comparison']['results'].append(result)
    for ceiling in report['ceilings']:
        if ceiling['status'] == 'estimated' and ceiling.get('measured_us'):
            ceiling['ideal_fraction'] = ceiling['lower_bound_us'] / ceiling['measured_us']
    return report
