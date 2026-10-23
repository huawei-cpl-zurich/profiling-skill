#!/usr/bin/env python3
"""One paired acquisition primitive; use model_benchmark_repeated.py for reports.

Private controller for compiled Models; never an agent fixture.
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
import traceback

HELPERS = Path(__file__).resolve().parents[1] / 'skills/ascend-model-benchmarking/scripts'
sys.path.insert(0, str(HELPERS))
from model_execution import CapabilityError, GraphCompiler, MeasurementError, benchmark_pair, prepare_model, prepare_eager_model
from profile_accounting import AccountingError, account_profile
from profile_capture import CaptureError


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


def describe(value):
    """JSON primitives/configuration and Tensor descriptors, never Tensor contents."""
    torch = sys.modules.get('torch')
    tensor_type = getattr(torch, 'Tensor', ())
    if tensor_type and isinstance(value, tensor_type):
        return dict(kind='tensor', shape=list(value.shape), dtype=str(value.dtype),
                    stride=list(value.stride()), device=str(value.device),
                    requires_grad=bool(value.requires_grad))
    for name in ('dtype', 'device', 'Size'):
        known_type = getattr(torch, name, ())
        if known_type and isinstance(value, known_type):
            return dict(kind='torch.' + name,
                        value=[describe(item) for item in value] if name == 'Size' else str(value))
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [describe(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: describe(item) for key, item in value.items()}
    raise ValueError('case metadata/configuration must be finite JSON primitives or tensors; '
                     'unsupported type ' + type(value).__name__)


def case_identity(case, expected):
    """Bind runtime specialization/init/provider metadata alongside provider source."""
    payload = describe(dict(args=case['args'], kwargs=case.get('kwargs', {}),
        init_args=case.get('init_args', ()), init_kwargs=case.get('init_kwargs', {}),
        expected_kernel=expected, metadata=case.get('metadata')))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(descriptor=payload, sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                provider_metadata_supplied='metadata' in case)


class InputGuard:
    """Private exact snapshots catch both versioned and raw-pointer mutations."""
    def __init__(self, values, synchronize):
        self.values, self.synchronize = values, synchronize
        self.structure = describe(values)
        self.tensors = []
        torch = sys.modules.get('torch')
        tensor_type = getattr(torch, 'Tensor', ())
        def visit(value):
            if tensor_type and isinstance(value, tensor_type):
                try:
                    version = value._version
                except RuntimeError as error:
                    raise CapabilityError('input mutation tracking unavailable for inference tensors') from error
                self.tensors.append((value, version, value.detach().clone()))
            elif type(value) in (list, tuple):
                for item in value:
                    visit(item)
            elif type(value) is dict:
                for item in value.values():
                    visit(item)
        visit(values)
        synchronize()

    def __call__(self, stage):
        self.synchronize()
        if describe(self.values) != self.structure:
            raise MeasurementError('Model mutated input structure or metadata', dict(stage=stage))
        torch = sys.modules.get('torch')
        for value, version, snapshot in self.tensors:
            try:
                if value._version != version:
                    raise AssertionError('input version changed')
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0, equal_nan=True,
                                           check_stride=False)
            except AssertionError as error:
                raise MeasurementError('Model mutated an input tensor', dict(stage=stage)) from error


def run(options, *, compiler_factory=None, clock=None):
    """Adapter hooks are trusted private code, not candidate-supplied callbacks."""
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'incomplete', 'configuration': vars(options), 'cases': [],
              'invocation': list(sys.argv), 'python_version': sys.version}
    try:
        policy = getattr(options, 'profile_policy', 'required')
        if policy not in ('required', 'optional'):
            raise ValueError('unknown profile policy')
        report['profile_policy'] = policy
        hooks = {name: load(getattr(options, name)) for name in (
            'baseline', 'candidate', 'cases_provider', 'validator', 'acquisition')}
        compiler_factory = compiler_factory or (lambda: GraphCompiler(backend_name=options.backend))
        mode = getattr(options, 'baseline_execution', 'compiled')
        if mode not in ('eager', 'compiled'):
            raise ValueError('baseline_execution must be eager or compiled')
        report['execution_modes'] = dict(baseline=mode, candidate='compiled')
        report['sources'] = {name: identity(hook) for name, hook in hooks.items()}
        adapter = hooks['acquisition']()
        report['environment_before'] = adapter.provenance()
        cases = list(hooks['cases_provider']())
        if not cases or len({case['id'] for case in cases}) != len(cases):
            raise ValueError('provide at least one case with unique string id')
        for index, case in enumerate(cases):
            if not isinstance(case['id'], str) or not case['id']:
                raise ValueError('case id must be a nonempty string')
            expected = case.get('expected_kernel', options.expected_kernel)
            if (not isinstance(expected, list) or not expected
                    or any(not isinstance(name, str) or not name for name in expected)):
                raise ValueError('provide nonempty exact expected_kernel names per case or CLI')
            case_dir = output / f'case-{index:04d}'
            case_dir.mkdir()
            specialization = case_identity(case, expected)
            private_path = case_dir / 'private-specialization.json'
            save(private_path, specialization)
            row = {'id': case['id'], 'state': 'incomplete',
                   'specialization': dict(sha256=specialization['sha256'],
                                         private_artifact=str(private_path.relative_to(output)))}
            report['cases'].append(row)
            guard = InputGuard((case['args'], case.get('kwargs', {}), case.get('init_args', ()),
                                case.get('init_kwargs', {})), adapter.synchronize)
            models = {}
            for arm in ('baseline', 'candidate'):
                factory = hooks[arm]
                eager = arm == 'baseline' and getattr(options, 'baseline_execution', 'compiled') == 'eager'
                prepare = prepare_eager_model if eager else prepare_model
                preparation = {} if eager else {'compiler': compiler_factory()}
                models[arm] = prepare(
                    lambda factory=factory: factory(*case.get('init_args', ()),
                                                    **case.get('init_kwargs', {})),
                    case['args'], case.get('kwargs', {}),
                    lambda result, arm=arm: hooks['validator'](case, arm, result),
                    **preparation,
                    synchronize=adapter.synchronize)
                guard(arm + ':prepare')
            profiles = {}
            row['profile_status'] = {}

            def acquire(arm):
                """Capture errors may qualify metrics; execution invariants never do."""
                prepared = models[arm]
                before = prepared.compiler.snapshot()
                executions = prepared.compiler.executions
                capture_error = None
                try:
                    with prepared.compiler.torch.inference_mode(), prepared.compiler.monitor_compilation():
                        capture = adapter.profile(prepared, samples=50, warmup=20,
                                                  output=case_dir / arm)
                    save(case_dir / f'{arm}-normalized-capture.json', capture)
                except Exception as error:
                    capture_error = error
                guard(arm + ':profile')
                if prepared.compiler.snapshot() != before:
                    raise CapabilityError('compilation detected during profiler replay')
                if (capture_error is None
                        and prepared.evidence.get('execution_mode', 'compiled') == 'compiled'
                        and prepared.compiler.executions - executions < 70):
                    raise CapabilityError('profiler replay lacks compiled execution evidence')
                try:
                    if capture_error is not None:
                        raise capture_error
                    accounted = account_profile(capture.get('iterations'), capture.get('tasks'),
                                                 capture_complete=capture.get('capture_complete'))
                    if len(accounted['iterations']) != 50:
                        raise AccountingError('profile must identify exactly 50 active iterations')
                    if any(not item['task_count'] for item in accounted['iterations']):
                        raise AccountingError('profile contains an empty active iteration')
                    for iteration in accounted['iterations']:
                        iteration['all_kernel_busy_us'] = (iteration['device_busy_us']
                                                           + iteration['aicpu_busy_us'])
                    profiles[arm] = accounted
                    save(case_dir / f'{arm}-busy-time.json', accounted)
                    row['profile_status'][arm] = dict(state='complete')
                except (CaptureError, AccountingError) as error:
                    row['profile_status'][arm] = dict(state='unavailable',
                        error_type=type(error).__name__, error=str(error))
                    save(case_dir / f'{arm}-profile-status.json', row['profile_status'][arm])
                    if policy == 'required' or arm == 'candidate':
                        raise

            # Optional baseline profiling cannot obstruct submitted-kernel identity
            # or already-validated unprofiled latency. Required retains its order.
            if policy == 'required':
                acquire('baseline')
            acquire('candidate')
            candidate_profile = profiles['candidate']
            wanted = set(expected)
            for iteration in candidate_profile['iterations']:
                observed = {task['name'] for task in candidate_profile['tasks']
                            if task['iteration_id'] == iteration['iteration_id']
                            and task['kind'] == 'device'}
                if not wanted.issubset(observed):
                    raise CapabilityError('submitted kernel identity absent from compiled replay')
            models['candidate'].evidence['custom_kernel_identity_verified'] = True
            row['candidate_identity'] = dict(state='verified', expected_kernel=expected)
            if policy == 'optional':
                # Match candidate replay depth without making baseline trace
                # support a prerequisite for valid paired latency.
                prepared = models['baseline']
                before = prepared.compiler.snapshot()
                executions = prepared.compiler.executions
                with prepared.compiler.torch.inference_mode(), prepared.compiler.monitor_compilation():
                    for _ in range(70):
                        prepared()
                    adapter.synchronize()
                guard('baseline:pre-latency-replay')
                if prepared.compiler.snapshot() != before:
                    raise CapabilityError('compilation detected during baseline pre-latency replay')
                if (prepared.evidence.get('execution_mode', 'compiled') == 'compiled'
                        and prepared.compiler.executions - executions < 70):
                    raise CapabilityError('baseline pre-latency replay lacks compiled execution evidence')
                row['pre_latency_replay_calls'] = dict(baseline=70, candidate=70)
            arguments = {'synchronize': adapter.synchronize, 'validate_inputs': guard}
            if clock is not None:
                arguments['clock'] = clock
            latency = benchmark_pair(models['baseline'], models['candidate'], **arguments)
            save(case_dir / 'latency.json', latency)
            row['latency'] = latency
            if policy == 'optional':
                acquire('baseline')
            row['metric_status'] = {name: 'complete' for name in (
                'baseline_latency_ms', 'candidate_latency_ms', 'latency_speedup')}
            for kind, field in (('device', 'device_busy_us'), ('aicpu', 'aicpu_busy_us'),
                                ('all_kernel', 'all_kernel_busy_us')):
                row[kind + '_busy_median_us'] = {}
                for arm in ('baseline', 'candidate'):
                    status = 'complete' if arm in profiles else 'unavailable'
                    row['metric_status'][arm + '_' + kind + '_busy_us'] = status
                    if arm in profiles:
                        row[kind + '_busy_median_us'][arm] = statistics.median(
                            item[field] for item in profiles[arm]['iterations'])
            busy = row['device_busy_median_us']
            total = row['all_kernel_busy_median_us']
            if busy['candidate'] <= 0:
                raise CapabilityError('candidate device busy-time must be positive')
            row['metric_status']['busy_time_speedup'] = (
                'complete' if 'baseline' in profiles and policy == 'required' else 'unavailable')
            row['profile_replay_depth_paired'] = policy == 'required'
            if policy == 'optional':
                row['busy_time_comparison_unavailable_reason'] = (
                    'baseline and candidate profiles have different replay depths around paired latency')
            row.update(state='complete',
                busy_time_metric='per_iteration_sum_of_device_and_aicpu_task_durations')
            if 'baseline' in profiles and policy == 'required':
                row['device_busy_time_speedup'] = busy['baseline'] / busy['candidate']
                row['busy_time_speedup'] = total['baseline'] / total['candidate']
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
            if all('busy_time_speedup' in row for row in report['cases']):
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
    parser.add_argument('--backend', choices=('torchair-ge', 'npu-inductor'), default='torchair-ge',
                        help='explicit backend for compiled arms; never falls back automatically')
    parser.add_argument('--baseline-execution', choices=('eager', 'compiled'), default='compiled',
                        help='explicit reference mode; candidate always compiled; no fallback')
    parser.add_argument('--expected-kernel', action='append', default=[],
                        help='exact submitted task name; private per-case expected_kernel overrides')
    parser.add_argument('--profile-policy', choices=('required', 'optional'), default='required',
                        help='optional retains latency when baseline profiling is unavailable; candidate identity stays mandatory')
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
