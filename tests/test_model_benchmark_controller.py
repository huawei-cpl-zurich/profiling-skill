from argparse import Namespace
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
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
            monitor_compilation=nullcontext)
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
    def run_fixture(self, root, adapter=None, validator=validate):
        hooks = dict(baseline=baseline, candidate=candidate, cases_provider=cases,
                     validator=validator, acquisition=acquisition)
        adapter = adapter or Adapter()
        hooks['acquisition'] = lambda: adapter
        options = Namespace(**{name: name for name in hooks},
                            output=str(root / 'results'), expected_kernel=['submitted'])
        with patch.object(controller, 'load', side_effect=hooks.__getitem__), \
                patch.object(controller, 'prepare_model', side_effect=Prepared):
            return controller.run(options, compiler_factory=lambda: None, clock=lambda: 1000)

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
            self.assertEqual([len(batch) for batch in row['latency']['candidate']['batches_ms']], [50] * 3)
            profile = json.loads((root / 'results/case-0000/candidate-busy-time.json').read_text())
            self.assertEqual(len(profile['tasks']), 50)
            self.assertTrue(profile['tasks'][-1]['diagnostic_flags'])
            self.assertEqual(profile['iterations'][-1]['device_busy_us'], 10000)
            self.assertTrue(row['latency']['execution_evidence']['candidate']['custom_kernel_identity_verified'])

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

    def test_backend_selection_and_default_acquisition_reach_runner(self):
        args = ['--baseline', 'r:Model', '--candidate', 'c:Model',
                '--cases-provider', 'p:cases', '--validator', 'p:validate', '--output', 'new']
        for extra, expected in [([], 'torchair-ge'), (['--backend', 'npu-inductor'], 'npu-inductor')]:
            with patch.object(controller, 'run', return_value={'state': 'complete'}) as run:
                self.assertEqual(controller.main(args + extra), 0)
                self.assertEqual(run.call_args.args[0].backend, expected)
                self.assertEqual(run.call_args.args[0].acquisition, 'profile_capture:create_acquisition')


if __name__ == '__main__':
    unittest.main()
