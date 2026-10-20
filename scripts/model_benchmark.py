#!/usr/bin/env python3
"""Private controller for paired compiled Model measurements; never an agent fixture."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import traceback

HELPERS = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
sys.path.insert(0, str(HELPERS))
from model_execution import CapabilityError, benchmark_pair, prepare_model
from profile_accounting import account_profile


def load(spec):
    module, separator, name = spec.partition(':')
    if not separator:
        raise ValueError('imports must have module:callable form')
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise ValueError(f'{spec} is not callable')
    return value


def identity(value):
    path = Path(inspect.getsourcefile(value) or '')
    if not path.is_file():
        raise ValueError('all benchmark hooks require retained Python source files')
    return {'module': value.__module__, 'name': value.__qualname__,
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def run(options, *, compiler_factory=None, clock=None):
    """Adapter hooks are trusted private code, not candidate-supplied callbacks."""
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'incomplete', 'configuration': vars(options), 'cases': [],
              'invocation': list(sys.argv), 'python_version': sys.version}
    try:
        hooks = {name: load(getattr(options, name)) for name in (
            'baseline', 'candidate', 'cases_provider', 'validator', 'acquisition')}
        report['sources'] = {name: identity(hook) for name, hook in hooks.items()}
        adapter = hooks['acquisition']()
        report['environment_before'] = adapter.provenance()
        cases = list(hooks['cases_provider']())
        if not cases or len({case['id'] for case in cases}) != len(cases):
            raise ValueError('provide at least one case with unique string id')
        for index, case in enumerate(cases):
            if not isinstance(case['id'], str) or not case['id']:
                raise ValueError('case id must be a nonempty string')
            case_dir = output / f'case-{index:04d}'
            case_dir.mkdir()
            row = {'id': case['id'], 'state': 'incomplete'}
            report['cases'].append(row)
            models = {}
            for arm in ('baseline', 'candidate'):
                factory = hooks[arm]
                models[arm] = prepare_model(
                    lambda factory=factory: factory(*case.get('init_args', ()),
                                                    **case.get('init_kwargs', {})),
                    case['args'], case.get('kwargs', {}),
                    lambda result, arm=arm: hooks['validator'](case, arm, result),
                    compiler=compiler_factory() if compiler_factory else None,
                    synchronize=adapter.synchronize)
            # Separate profiling replay proves identity before unprofiled timing.
            profiles = {}
            for arm, prepared in models.items():
                before = prepared.compiler.snapshot()
                executions = prepared.compiler.executions
                with prepared.compiler.torch.inference_mode(), prepared.compiler.monitor_compilation():
                    capture = adapter.profile(prepared, samples=50, warmup=20,
                                              output=case_dir / arm)
                save(case_dir / f'{arm}-normalized-capture.json', capture)
                if prepared.compiler.snapshot() != before:
                    raise CapabilityError('compilation detected during profiler replay')
                if prepared.compiler.executions - executions < 70:
                    raise CapabilityError('profiler replay lacks compiled execution evidence')
                accounted = account_profile(capture.get('iterations'), capture.get('tasks'),
                                             capture_complete=capture.get('capture_complete'))
                if len(accounted['iterations']) != 50:
                    raise CapabilityError('profile must identify exactly 50 active iterations')
                if any(not item['task_count'] for item in accounted['iterations']):
                    raise CapabilityError('profile contains an empty active iteration')
                profiles[arm] = accounted
                save(case_dir / f'{arm}-busy-time.json', accounted)
            # Exact task identities, not graph-text guesses or name-count grouping.
            wanted = set(options.expected_kernel)
            candidate = profiles['candidate']
            for iteration in candidate['iterations']:
                observed = {task['name'] for task in candidate['tasks']
                            if task['iteration_id'] == iteration['iteration_id']
                            and task['kind'] == 'device'}
                if not wanted.issubset(observed):
                    raise CapabilityError('submitted kernel identity absent from compiled replay')
            models['candidate'].evidence['custom_kernel_identity_verified'] = True
            arguments = {'synchronize': adapter.synchronize}
            if clock is not None:
                arguments['clock'] = clock
            latency = benchmark_pair(models['baseline'], models['candidate'], **arguments)
            save(case_dir / 'latency.json', latency)
            busy = {arm: statistics.median(item['device_busy_us']
                    for item in profile['iterations']) for arm, profile in profiles.items()}
            if busy['candidate'] <= 0:
                raise CapabilityError('candidate device busy-time must be positive')
            row.update(state='complete', latency=latency,
                       device_busy_median_us=busy,
                       busy_time_speedup=busy['baseline'] / busy['candidate'])
            row['latency_variability_flags'] = [arm for arm in ('baseline', 'candidate')
                if latency.get(arm, {}).get('between_batch_relative_spread', 0) > .05]
            save(output / 'report.json', report)
        report['environment_after'] = adapter.provenance()
        # The trusted adapter records actual physical device, clock/load and backend
        # conditions. Missing evidence qualifies comparison; stability != equal clocks.
        report['comparability'] = adapter.comparability(
            report['environment_before'], report['environment_after'])
        if report['comparability'].get('comparable') is True:
            report['geomean_latency_speedup'] = statistics.geometric_mean(
                row['latency']['latency_speedup'] for row in report['cases'])
            report['geomean_busy_time_speedup'] = statistics.geometric_mean(
                row['busy_time_speedup'] for row in report['cases'])
        report['state'] = 'complete'
        return report
    except Exception as error:
        report.update(state='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if hasattr(error, 'details'):
            report['failure_details'] = error.details
        return report
    finally:
        save(output / 'report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'cases-provider', 'validator', 'acquisition'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--expected-kernel', action='append', required=True,
                        help='exact submitted device task name; repeat for required components')
    parser.add_argument('--output', required=True, help='new evidence directory')
    options = parser.parse_args(argv)
    try:
        report = run(options)
    except OSError as error:
        print(json.dumps({'state': 'failed', 'error': str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({'state': report['state'], 'report': str(Path(options.output) / 'report.json')}))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
