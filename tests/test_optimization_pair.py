import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import optimization_pair as pair
from profile_capture import normalize_trace


class Adapter:
    case_sources = [__file__]

    def __init__(self, *, bad_arm=None, calls=10, missing=False, export_error=False, mutate=False):
        self.oracle_calls = 0
        self.log = []
        self.bad_arm, self.calls, self.missing = bad_arm, calls, missing
        self.export_error, self.mutate = export_error, mutate
        self.private = []

    def make_case(self, spec):
        self.oracle_calls += 1
        tensor = torch.arange(20)[2:18:3]
        return dict(inputs=tensor, expected=tensor + 1)

    def prepare(self, arm, spec, fixture, output):
        self.private.append(fixture)
        def run():
            self.log.append(('run', arm))
            if self.mutate:
                fixture['inputs'].add_(1)
            return fixture['inputs'] + 1
        return N(run=run, expected_kernels=['kernel'], sources=[__file__], metadata={'backend': 'fixture'})

    def validate(self, arm, actual, fixture, spec):
        self.log.append(('validate', arm))
        torch.testing.assert_close(actual, fixture['expected'])
        return dict(passed=arm != self.bad_arm)

    def capture(self, run, prepared, output):
        output.mkdir()
        self.log.append(('capture', None))
        for _ in range(self.calls):
            run()
        result = dict(event_latency=dict(count=10, samples_ms=[.25] * 10),
                      profile=dict(export_status='failed' if self.export_error else 'complete'),
                      kernel_identity=dict(status='complete'))  # Deliberately untrusted.
        if self.export_error:
            return result
        events = []
        names = [f'model_iteration_{i}' for i in range(10)]
        for i, name in enumerate(names):
            base = i * 1000
            kernel = 'wrong' if self.missing and i == 9 else 'kernel'
            events.extend([
                dict(name=name, pid=1, tid=1, ts=base, dur=900, ph='X', cat='cpu_op'),
                dict(name=kernel, pid=1, tid=1, ts=base+20, dur=5, ph='X', cat='cpu_op'),
                dict(name=kernel, pid=2, tid=2, ts=base+100, dur=10, ph='X',
                     args={'Task Type': 'KERNEL_AIVEC', 'Physic Stream Id': 2, 'Task Id': i,
                           'Batch Id': 0, 'Subtask Id': 0}),
                dict(name='torch_to_npu', pid=1, tid=1, ts=base+20, ph='s', cat='async_npu', id=i),
                dict(name='torch_to_npu', pid=2, tid=2, ts=base+100, ph='f', cat='async_npu', id=i)])
        trace = output / 'trace_view.json'
        pair.save(trace, dict(traceEvents=events))
        result['profile'].update(vendor_trace=str(trace), trace_sha256=pair.sha(trace))
        path = output / 'normalized.json'
        pair.save(path, normalize_trace(events, names, export_completed=True))
        result['profile']['capture_path'] = str(path)
        return result


class PairTests(unittest.TestCase):
    def run_case(self, adapter, folder, name='out'):
        return pair.run_pair(adapter, {'id': 'one'}, folder / name, folder / 'cache', {'device': 'CPU fixture'})

    def test_symmetric_counts_oracle_reuse_and_layout_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, adapter = Path(directory), Adapter()
            result = self.run_case(adapter, folder)
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(adapter.oracle_calls, 1)
            self.assertEqual(adapter.log[:4], [('run', 'before'), ('validate', 'before'),
                                               ('run', 'after'), ('validate', 'after')])
            self.assertEqual(adapter.log.count(('run', 'before')), 11)
            self.assertEqual(adapter.log.count(('run', 'after')), 11)
            self.assertEqual(adapter.private[0]['inputs'].stride(), (3,))
            self.assertEqual(adapter.private[0]['inputs'].storage_offset(), 2)
            self.assertNotEqual(adapter.private[0]['inputs'].data_ptr(), adapter.private[1]['inputs'].data_ptr())
            for arm in ('before', 'after'):
                row = result['arms'][arm]
                self.assertEqual(row['correctness']['calls'], 1)
                self.assertEqual(row['profiled_calls'], 10)
                self.assertEqual(row['capture']['kernel_identity']['status'], 'complete')
                self.assertEqual(row['capture']['operation_accounting']['tasks'][0]['iteration_id'],
                                 'model_iteration_0')
            self.assertEqual(self.run_case(adapter, folder, 'reuse')['status'], 'complete')
            self.assertEqual(adapter.oracle_calls, 1)

    def test_no_capture_when_either_correctness_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Adapter(bad_arm='after')
            result = self.run_case(adapter, Path(directory))
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn(('capture', None), adapter.log)
            self.assertEqual(len(adapter.log), 4)

    def test_failed_correctness_diagnostics_and_counts_survive_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, adapter = Path(directory), Adapter()
            diagnostics = dict(passed=False, max_abs_error=0.125, mismatches=3,
                               reason='numeric tolerance exceeded')
            adapter.validate = lambda *args: diagnostics
            result = self.run_case(adapter, folder)
            retained = json.loads((folder / 'out/result.json').read_text())
            self.assertEqual(result, retained)
            self.assertEqual(retained['status'], 'failed')
            self.assertEqual(retained['arms']['before']['correctness'], dict(
                status='failed', calls=1, diagnostics=diagnostics, reason='validator rejected output'))
            self.assertEqual(retained['counts'], dict(correctness_calls=1, profiled_calls=0, capture_calls=0))
            self.assertEqual(adapter.log, [('run', 'before')])

    def test_capture_counts_and_identity_cannot_be_self_asserted(self):
        for kwargs, status in ((dict(calls=9), 'failed'), (dict(calls=11), 'failed'),
                               (dict(missing=True), 'incomplete')):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                result = self.run_case(Adapter(**kwargs), Path(directory))
                self.assertEqual(result['status'], status, result)
                if kwargs.get('missing'):
                    self.assertEqual(result['arms']['before']['capture']['kernel_identity']['missing'], ['kernel'])

    def test_export_failure_retains_timing_and_runs_other_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(Adapter(export_error=True), Path(directory))
            self.assertEqual(result['status'], 'incomplete')
            for row in result['arms'].values():
                self.assertEqual(row['profiled_calls'], 10)
                self.assertEqual(row['capture']['event_latency']['mean_ms'], .25)
                self.assertEqual(row['capture']['profile']['export_status'], 'failed')

    def test_cache_corruption_rejected_before_any_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.run_case(Adapter(), folder)
            next((folder / 'cache').rglob('fixture.pt')).write_bytes(b'changed')
            adapter = Adapter()
            result = self.run_case(adapter, folder, 'corrupt')
            self.assertEqual(result['status'], 'failed')
            self.assertIn('payload changed', result['error']['message'])
            self.assertEqual(adapter.log, [])

    def test_source_changes_and_capture_compilation_are_rejected(self):
        for mode in ('source', 'compile'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                folder, adapter = Path(directory), Adapter()
                source = folder / 'kernel.bin'
                source.write_bytes(b'original')
                prepare = adapter.prepare
                counter = {'compiles': 0}
                def prepare_checked(*args):
                    item = prepare(*args)
                    item.sources = [source]
                    item.compilation_state = lambda: counter
                    return item
                adapter.prepare = prepare_checked
                capture = adapter.capture
                def capture_changed(*args):
                    result = capture(*args)
                    if mode == 'source':
                        source.write_bytes(b'changed')
                    else:
                        counter['compiles'] += 1
                    return result
                adapter.capture = capture_changed
                result = self.run_case(adapter, folder)
                self.assertEqual(result['status'], 'failed')
                self.assertIn('changed', result['arms']['before']['error']['message'])

    def test_input_mutation_rejected_and_report_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            adapter = Adapter(mutate=True)
            adapter.validate = lambda *args: {'passed': True}
            result = self.run_case(adapter, folder)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('mutated fixture', result['error']['message'])
            self.assertEqual(json.loads((folder / 'out/result.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
