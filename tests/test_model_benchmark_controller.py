from argparse import Namespace
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'benchmark_controller', Path(__file__).parents[1] / 'scripts/model_benchmark.py')
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def baseline():
    return lambda value: value + 1


def candidate():
    return lambda value: value + 1


def cases():
    return [{'id': 'generic-add', 'args': (2,)}]


def validate(case, arm, result):
    if result != case['args'][0] + 1:
        raise AssertionError('wrong output')


class Prepared:
    def __init__(self, factory, args, kwargs, validate, **unused):
        self.callable = factory()
        self.args, self.kwargs = args, kwargs
        self.compiler = SimpleNamespace(
            compilations=1, triton_compilations=0, executions=1,
            torch=SimpleNamespace(inference_mode=nullcontext),
            monitor_compilation=lambda **kwargs: nullcontext())
        self.compiler.snapshot = lambda: (self.compiler.compilations,
                                          self.compiler.triton_compilations)
        self.evidence = {'custom_kernel_identity_verified': False}
        validate(self.callable(*args, **kwargs))

    def __call__(self):
        self.compiler.executions += 1
        return self.callable(*self.args, **self.kwargs)


class Adapter:
    def __init__(self, *, name='submitted', complete=True, comparable=True,
                 recompile=False):
        self.name, self.complete, self.comparable = name, complete, comparable
        self.recompile = recompile

    def synchronize(self):
        pass

    def provenance(self):
        return {'device': 'fake-functional-test', 'clock_mhz': 1000}

    def comparability(self, before, after):
        return {'comparable': self.comparable, 'reason': 'functional fixture only'}

    def profile(self, prepared, *, samples, warmup, output):
        for _ in range(warmup + samples):
            prepared()
        if self.recompile:
            prepared.compiler.compilations += 1
        return {'capture_complete': self.complete,
                'iterations': [{'iteration_id': str(i), 'correlation_ids': [str(i)]}
                               for i in range(samples)],
                'tasks': [{'task_id': str(i), 'correlation_id': str(i),
                           'name': self.name, 'kind': 'device', 'start_us': i * 1000,
                           'duration_us': 100 if i < samples - 1 else 10000}
                          for i in range(samples)]}


def acquisition():
    return Adapter()


class ControllerTests(unittest.TestCase):
    def run_fixture(self, root, adapter=None, validator=validate, baseline_execution="compiled", policy=None):
        hooks = dict(baseline=baseline, candidate=candidate, cases_provider=cases,
                     validator=validator, acquisition=acquisition)
        adapter = adapter or Adapter()
        hooks['acquisition'] = lambda: adapter
        options = Namespace(**{name: name for name in hooks},
                            output=str(root / 'results'), expected_kernel=['submitted'],
                            baseline_execution=baseline_execution)
        if policy is not None:
            options.profile_policy = policy
        ticks = iter(range(0, 10**8, 1000))
        with patch.object(controller, 'load', side_effect=hooks.__getitem__), \
                patch.object(controller, 'prepare_model', side_effect=Prepared):
            return controller.run(options, compiler_factory=lambda: None, clock=lambda: next(ticks))

    def test_complete_report_retains_flags_and_both_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # A deterministic increasing timer gives positive complete samples.
            ticks = iter(range(0, 10**7, 1000))
            with patch.object(controller, 'benchmark_pair', wraps=controller.benchmark_pair) as call:
                original = call._mock_wraps
                call.side_effect = lambda *args, **kwargs: original(
                    *args, **dict(kwargs, clock=lambda: next(ticks)))
                report = self.run_fixture(root)
            self.assertEqual(report['state'], 'complete')
            self.assertEqual(report['geomean_busy_time_speedup'], 1)
            row = report['cases'][0]
            private = json.loads((root / 'results' / row['specialization']['private_artifact']).read_text())
            self.assertEqual(private['descriptor']['args'], [2])
            self.assertEqual(row['specialization']['sha256'], private['sha256'])
            self.assertNotIn('descriptor', row['specialization'])
            self.assertEqual([len(batch) for batch in row['latency']['candidate']['batches_ms']], [50] * 3)
            profile = json.loads((root / 'results/case-0000/candidate-busy-time.json').read_text())
            self.assertEqual(len(profile['tasks']), 50)
            self.assertTrue(profile['tasks'][-1]['diagnostic_flags'])
            self.assertEqual(profile['iterations'][-1]['device_busy_us'], 10000)
            self.assertTrue(row['latency']['execution_evidence']['candidate']['custom_kernel_identity_verified'])

    def test_explicit_eager_profile_and_timing_retain_compiled_candidate_checks(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(controller, 'benchmark_pair', return_value={'latency_speedup': 1}):
            report = self.run_fixture(Path(temporary), baseline_execution='eager')
            self.assertEqual(report['state'], 'complete', report.get('error'))
            self.assertEqual(report['execution_modes'], dict(baseline='eager', candidate='compiled'))
            self.assertEqual(len(json.loads((Path(temporary) /
                'results/case-0000/baseline-busy-time.json').read_text())['iterations']), 50)
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_fixture(Path(temporary), Adapter(name='wrong'), baseline_execution='eager')
            self.assertEqual(report['state'], 'failed')
            self.assertIn('identity absent', report['error'])

    def test_optional_baseline_failure_retains_real_paired_latency(self):
        class BrokenBaseline(Adapter):
            def profile(self, prepared, **kwargs):
                if kwargs['output'].name == 'baseline':
                    self.assert_latency_saved = (kwargs['output'].parent / 'latency.json').exists()
                    raise controller.CaptureError('unsupported baseline trace')
                return super().profile(prepared, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = BrokenBaseline()
            report = self.run_fixture(root, adapter, policy='optional')
            self.assertEqual(report['state'], 'complete')
            self.assertTrue(adapter.assert_latency_saved)
            row = report['cases'][0]
            self.assertEqual(row['candidate_identity']['state'], 'verified')
            self.assertGreater(row['latency']['candidate']['median_ms'], 0)
            self.assertEqual(row['metric_status']['baseline_device_busy_us'], 'unavailable')
            self.assertEqual(row['metric_status']['candidate_device_busy_us'], 'complete')
            self.assertEqual(row['profile_status']['baseline']['error'], 'unsupported baseline trace')
            self.assertNotIn('baseline', row['device_busy_median_us'])
            self.assertNotIn('busy_time_speedup', row)
            self.assertNotIn('geomean_busy_time_speedup', report)
            self.assertIn('geomean_latency_speedup', report)

    def test_optional_matches_stateful_replay_depth_before_paired_timing(self):
        counts = dict(baseline=0, candidate=0)
        def factory(arm):
            def model(value):
                counts[arm] += 1
                return value + 1
            return model
        original = controller.benchmark_pair
        def paired(baseline, candidate, **kwargs):
            self.assertEqual(counts, dict(baseline=71, candidate=71))
            return original(baseline, candidate, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(globals(), baseline=lambda: factory('baseline'),
                           candidate=lambda: factory('candidate')), \
                patch.object(controller, 'benchmark_pair', side_effect=paired):
            report = self.run_fixture(Path(temporary), policy='optional')
        self.assertEqual(report['state'], 'complete', report.get('error'))
        self.assertEqual(report['cases'][0]['pre_latency_replay_calls'], dict(baseline=70, candidate=70))

    def test_optional_success_retains_arm_metrics_without_unpaired_busy_speedups(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_fixture(Path(temporary), policy='optional')
        self.assertEqual(report['state'], 'complete', report.get('error'))
        row = report['cases'][0]
        self.assertEqual(row['profile_status'], dict(baseline=dict(state='complete'), candidate=dict(state='complete')))
        self.assertEqual(set(row['all_kernel_busy_median_us']), {'baseline', 'candidate'})
        self.assertFalse(row['profile_replay_depth_paired'])
        self.assertEqual(row['metric_status']['busy_time_speedup'], 'unavailable')
        self.assertNotIn('busy_time_speedup', row)
        self.assertNotIn('device_busy_time_speedup', row)
        self.assertNotIn('geomean_busy_time_speedup', report)
        self.assertIn('geomean_latency_speedup', report)

    def test_optional_eager_reference_failure_keeps_compiled_candidate_identity(self):
        class BrokenBaseline(Adapter):
            def profile(self, prepared, **kwargs):
                if kwargs['output'].name == 'baseline':
                    raise controller.CaptureError('eager baseline accounting unavailable')
                return super().profile(prepared, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_fixture(Path(temporary), BrokenBaseline(),
                                      policy='optional', baseline_execution='eager')
            self.assertEqual(report['state'], 'complete', report.get('error'))
            self.assertEqual(report['execution_modes'], dict(baseline='eager', candidate='compiled'))
            row = report['cases'][0]
            self.assertEqual(row['candidate_identity']['state'], 'verified')
            self.assertEqual(row['profile_status']['baseline']['state'], 'unavailable')
            self.assertEqual(row['metric_status']['candidate_device_busy_us'], 'complete')
            self.assertNotIn('busy_time_speedup', row)

    def test_optional_baseline_unexpected_runtime_failure_is_fatal(self):
        class Broken(Adapter):
            def profile(self, prepared, **kwargs):
                if kwargs['output'].name == 'baseline':
                    raise RuntimeError('device execution failed')
                return super().profile(prepared, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.run_fixture(root, Broken(), policy='optional')
            self.assertEqual(report['state'], 'failed')
            self.assertTrue((root / 'results/case-0000/latency.json').exists())
            self.assertNotIn('geomean_latency_speedup', report)

    def test_optional_capture_failure_cannot_hide_input_mutation(self):
        import torch
        class Broken(Adapter):
            def profile(self, prepared, **kwargs):
                if kwargs['output'].name == 'baseline':
                    prepared.args[0].add_(1)
                    raise controller.CaptureError('unsupported trace')
                return super().profile(prepared, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(globals(), cases=lambda: [dict(id='scalar', args=(torch.tensor(2.),))]):
            report = self.run_fixture(Path(temporary), Broken(), policy='optional')
            self.assertEqual(report['state'], 'failed')
            self.assertIn('mutated an input', report['error'])

    def test_optional_baseline_recompile_then_capture_failure_is_fatal(self):
        class Broken(Adapter):
            def profile(self, prepared, **kwargs):
                if kwargs['output'].name == 'baseline':
                    prepared.compiler.compilations += 1
                    raise controller.CaptureError('unsupported trace after compilation')
                return super().profile(prepared, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_fixture(Path(temporary), Broken(), policy='optional')
            self.assertEqual(report['state'], 'failed')
            self.assertIn('compilation detected', report['error'])

    def test_optional_still_rejects_identity_and_candidate_capture_failure(self):
        class BrokenCandidate(Adapter):
            def profile(self, prepared, **kwargs):
                raise RuntimeError('candidate trace unavailable')
        for adapter in (Adapter(name='other'), BrokenCandidate()):
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(controller, 'benchmark_pair') as timing:
                report = self.run_fixture(Path(temporary), adapter, policy='optional')
                self.assertEqual(report['state'], 'failed')
                self.assertNotIn('geomean_latency_speedup', report)
                timing.assert_not_called()

    def test_optional_does_not_swallow_wrong_output_or_profile_recompilation(self):
        def reject(*args):
            raise AssertionError('independent mismatch')
        for validator, adapter in ((reject, Adapter()), (validate, Adapter(recompile=True))):
            with tempfile.TemporaryDirectory() as temporary:
                report = self.run_fixture(Path(temporary), adapter, validator, policy='optional')
                self.assertEqual(report['state'], 'failed')

    def test_required_remains_strict_when_baseline_capture_throws(self):
        class Broken(Adapter):
            def profile(self, prepared, **kwargs):
                raise RuntimeError('no baseline profile')
        for policy in (None, 'required'):
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(controller, 'benchmark_pair') as timing:
                report = self.run_fixture(Path(temporary), Broken(), policy=policy)
                self.assertEqual(report['state'], 'failed')
                timing.assert_not_called()

    def test_missing_kernel_identity_fails_before_latency(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(controller, 'benchmark_pair') as timing:
            report = self.run_fixture(Path(temporary), Adapter(name='other'))
            self.assertEqual(report['state'], 'failed')
            self.assertIn('identity absent', report['error'])
            timing.assert_not_called()

    def test_incomplete_capture_and_recompilation_fail_with_evidence(self):
        for adapter, message in [(Adapter(complete=False), 'complete acquisition'),
                                 (Adapter(recompile=True), 'compilation detected')]:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report = self.run_fixture(root, adapter)
                self.assertEqual(report['state'], 'failed')
                self.assertIn(message, report['error'])
                self.assertTrue((root / 'results/case-0000/baseline-normalized-capture.json').exists())

    def test_wrong_output_stops_before_acquisition(self):
        def reject(*args):
            raise AssertionError('independent reference mismatch')
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Adapter()
            with patch.object(adapter, 'profile', wraps=adapter.profile) as profile:
                report = self.run_fixture(Path(temporary), adapter, reject)
            self.assertEqual(report['state'], 'failed')
            profile.assert_not_called()

    def test_uncertain_clocks_withhold_aggregate_ranking(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(controller, 'benchmark_pair', return_value={'latency_speedup': 1}):
            report = self.run_fixture(Path(temporary), Adapter(comparable=False))
            self.assertEqual(report['state'], 'complete')
            self.assertNotIn('geomean_latency_speedup', report)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'results').mkdir()
            with self.assertRaises(FileExistsError):
                self.run_fixture(root)

    def test_aicpu_is_summed_per_iteration_before_median(self):
        class Crossed(Adapter):
            def profile(self, prepared, **kwargs):
                capture = super().profile(prepared, **kwargs)
                for index, task in enumerate(list(capture['tasks'])):
                    device, cpu = (1, 100) if index < 17 else ((2, 2) if index < 33 else (100, 1))
                    task['duration_us'] = device
                    capture['tasks'].append(dict(task, task_id='cpu-' + str(index),
                                                 kind='aicpu', duration_us=cpu))
                return capture
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(controller, 'benchmark_pair', return_value={'latency_speedup': 1}):
            report = self.run_fixture(Path(temporary), Crossed())
            self.assertEqual(report['state'], 'complete')
            row = report['cases'][0]
            self.assertEqual(row['device_busy_median_us']['candidate'], 2)
            self.assertEqual(row['aicpu_busy_median_us']['candidate'], 2)
            self.assertEqual(row['all_kernel_busy_median_us']['candidate'], 101)
            self.assertEqual(row['busy_time_speedup'], 1)
            self.assertEqual(row['device_busy_time_speedup'], 1)

    def test_private_case_can_select_its_actual_entrypoint(self):
        varied = [dict(cases()[0], expected_kernel=['different-entrypoint'])]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(controller, 'benchmark_pair', return_value={'latency_speedup': 1}), \
                patch.dict(globals(), cases=lambda: varied):
            report = self.run_fixture(Path(temporary), Adapter(name='different-entrypoint'))
            self.assertEqual(report['state'], 'complete')

    def test_backend_selection_and_explicit_acquisition_reach_runner(self):
        args = ['--baseline', 'r:Model', '--candidate', 'c:Model',
                '--cases-provider', 'p:cases', '--validator', 'p:validate',
                '--acquisition', 'trusted_capture:create', '--output', 'new']
        for extra, expected in [([], 'torchair-ge'), (['--backend', 'npu-inductor'], 'npu-inductor')]:
            with patch.object(controller, 'run', return_value={'state': 'complete'}) as run:
                self.assertEqual(controller.main(args + extra), 0)
                self.assertEqual(run.call_args.args[0].backend, expected)
                self.assertEqual(run.call_args.args[0].profile_policy, 'required')
                self.assertEqual(run.call_args.args[0].acquisition, 'trusted_capture:create')

    def test_missing_acquisition_is_rejected_before_runner(self):
        with patch.object(controller, 'run') as run, self.assertRaises(SystemExit) as error:
            controller.main(['--baseline', 'r:Model', '--candidate', 'c:Model',
                '--cases-provider', 'p:cases', '--validator', 'p:validate', '--output', 'new'])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_specialization_binds_tensor_descriptors_and_private_runtime_config(self):
        class Tensor:
            shape = (8, 16)
            dtype = 'torch.float16'
            device = 'npu:0'
            requires_grad = False

            def stride(self):
                return (16, 1)

            def tolist(self):
                raise AssertionError('tensor values must never be read')

        case = dict(id='opaque', args=(Tensor(),), init_kwargs={'mode': 'first'},
                    metadata={'seed': 123, 'generator_environment': {'layout': 'dense'}})
        with patch.dict(sys.modules, torch=SimpleNamespace(Tensor=Tensor)):
            first = controller.case_identity(case, ['submitted'])
            tensor = first['descriptor']['args'][0]
            self.assertEqual(tensor, dict(kind='tensor', shape=[8, 16], dtype='torch.float16',
                stride=[16, 1], device='npu:0', requires_grad=False))
            for updated in (dict(case, init_kwargs={'mode': 'second'}),
                            dict(case, metadata={'seed': 124})):
                self.assertNotEqual(first['sha256'], controller.case_identity(updated, ['submitted'])['sha256'])
            case['args'][0].shape = (4, 32)
            self.assertNotEqual(first['sha256'], controller.case_identity(case, ['submitted'])['sha256'])
            self.assertEqual(first['sha256'], controller.case_identity(
                dict(id='other-label', args=(Tensor(),), init_kwargs={'mode': 'first'},
                     metadata={'generator_environment': {'layout': 'dense'}, 'seed': 123}),
                ['submitted'])['sha256'])

    def test_metadata_rejects_arbitrary_serialization_and_nonfinite_values(self):
        class PrivateObject:
            def __repr__(self):
                raise AssertionError('must not serialize arbitrary objects')
        for value in (PrivateObject(), float('nan'), {1: 'non-string-key'}):
            with self.assertRaisesRegex(ValueError, 'JSON primitives'):
                controller.case_identity(dict(args=(2,), metadata=value), ['submitted'])

    def test_common_torch_constructor_configuration_has_typed_identity(self):
        import torch
        case = dict(args=(torch.ones(2),), init_kwargs=dict(
            dtype=torch.float16, device=torch.device('cpu'), shape=torch.Size([2, 4])))
        value = controller.case_identity(case, ['submitted'])
        config = value['descriptor']['init_kwargs']
        self.assertEqual(config['dtype'], dict(kind='torch.dtype', value='torch.float16'))
        self.assertEqual(config['device'], dict(kind='torch.device', value='cpu'))
        self.assertEqual(config['shape'], dict(kind='torch.Size', value=[2, 4]))
        changed = dict(case, init_kwargs=dict(case['init_kwargs'], dtype=torch.float32))
        self.assertNotEqual(value['sha256'], controller.case_identity(changed, ['submitted'])['sha256'])

    def test_nested_raw_pointer_mutation_is_detected_without_version_change(self):
        import torch
        tensor = torch.ones(3)
        guard = controller.InputGuard(({'nested': [tensor]},), lambda: None)
        version = tensor._version
        tensor.data.add_(1)
        self.assertEqual(tensor._version, version)
        with self.assertRaisesRegex(controller.MeasurementError, 'mutated an input') as error:
            guard('candidate:batch-0')
        self.assertEqual(error.exception.details['stage'], 'candidate:batch-0')

    def test_inference_input_tracking_is_explicitly_unsupported(self):
        import torch
        with torch.inference_mode():
            tensor = torch.ones(2)
        with self.assertRaisesRegex(controller.CapabilityError, 'inference tensors'):
            controller.InputGuard((tensor,), lambda: None)

    def test_upstream_gate_rejects_before_timing_with_noop_validator(self):
        import torch
        tensor = torch.ones(3)
        def wrong_candidate():
            return lambda value: value + 9
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(globals(), {'candidate': wrong_candidate,
                    'cases': lambda: [dict(id='wrong', args=(tensor,))]}), \
                patch.object(controller, 'benchmark_pair') as timing:
            report = self.run_fixture(Path(temporary), validator=lambda *args: None)
            self.assertEqual(report['state'], 'failed')
            self.assertIn('verification failed', report['cases'][0]['correctness']['error_msg'])
            timing.assert_not_called()

    def test_mutating_baseline_and_candidate_fail_before_timing(self):
        import torch
        for arm in ('baseline', 'candidate'):
            tensor = torch.ones(3)
            def mutator():
                def forward(value):
                    value.data.add_(1)
                    return value
                return forward
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(globals(), {arm: mutator, 'cases': lambda: [dict(id='nested', args=(tensor,))]}), \
                    patch.object(controller, 'benchmark_pair') as timing:
                report = self.run_fixture(Path(temporary), validator=lambda *args: None)
                self.assertEqual(report['state'], 'failed')
                self.assertEqual(report['error_type'], 'MeasurementError')
                self.assertEqual(report['failure_details']['stage'], arm + ':prepare')
                timing.assert_not_called()


if __name__ == '__main__':
    unittest.main()
