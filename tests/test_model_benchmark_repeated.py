import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_benchmark_repeated as repeated


def report(value=2, comparable=True):
    return dict(state='complete', sources={'candidate': {'source_sha256': 'abc'}},
        comparability=dict(comparable=comparable, reasons=[]), cases=[dict(
            id='shape', state='complete', specialization={'sha256': 'same'},
            latency=dict(baseline={'median_ms': 4}, candidate={'median_ms': value},
                         latency_speedup=4/value),
            all_kernel_busy_median_us=dict(baseline=8, candidate=value),
            device_busy_median_us=dict(baseline=7, candidate=value),
            aicpu_busy_median_us=dict(baseline=1, candidate=0),
            busy_time_speedup=8/value)])


class RepeatedBenchmarkTests(unittest.TestCase):
    def test_known_variance_and_constant_mean(self):
        result = repeated.aggregate([report(1), report(2), report(3)], 3)
        row = result['cases'][0]
        metric = row['metrics']['candidate_latency_ms']
        self.assertEqual(metric['values'], [1, 2, 3])
        self.assertEqual(metric['mean'], 2)
        self.assertEqual(metric['sample_stddev'], 1)
        self.assertEqual(row['metrics']['baseline_latency_ms']['sample_stddev'], 0)
        self.assertEqual(row['metrics']['candidate_all_kernel_busy_us']['unit'], 'us')
        self.assertEqual(result['state'], 'complete')

    def test_failure_and_partial_cases_not_silently_replaced(self):
        failed = report(9)
        failed.update(state='failed', error='numeric mismatch')
        result = repeated.aggregate([report(), failed, report()], 3)
        self.assertEqual(result['state'], 'incomplete')
        self.assertEqual(result['completed_repetitions'], 2)
        self.assertEqual(result['failed_repetitions'], 1)
        self.assertEqual(result['cases'][0]['metrics']['candidate_latency_ms']['values'], [2, 2])
        self.assertEqual(result['failures'][0]['error'], 'numeric mismatch')

    def test_one_sample_has_undefined_sd(self):
        result = repeated.aggregate([report()], 10)
        self.assertIsNone(result['cases'][0]['metrics']['candidate_latency_ms']['sample_stddev'])
        self.assertEqual(result['state'], 'incomplete')

    def test_missing_cases_or_changed_identity_rejects_run(self):
        for mutation in ('case', 'source', 'specialization', 'nan'):
            with self.subTest(mutation=mutation):
                changed = report()
                if mutation == 'case':
                    changed['cases'] = []
                elif mutation == 'source':
                    changed['sources']['candidate']['source_sha256'] = 'different'
                elif mutation == 'specialization':
                    changed['cases'][0]['specialization']['sha256'] = 'different'
                else:
                    changed['cases'][0]['latency']['candidate']['median_ms'] = float('nan')
                result = repeated.aggregate([report(), changed], 2)
                self.assertEqual(result['failed_repetitions'], 1)
                self.assertEqual(result['completed_repetitions'], 1)

    def test_missing_telemetry_still_qualifies_statistics(self):
        result = repeated.aggregate([report(), report(comparable=False)], 2)
        self.assertEqual(result['state'], 'complete')
        self.assertFalse(result['comparability']['comparable'])

    def test_malformed_report_is_counted_not_crash(self):
        result = repeated.aggregate([[], report()], 2)
        self.assertEqual(result['failed_repetitions'], 1)
        self.assertEqual(result['completed_repetitions'], 1)

    def test_real_cli_forwards_backend_and_reports_failed_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'runs'
            invocation = [sys.executable, repeated.__file__, '--output', str(output),
                '--repetitions', '1', '--backend', 'npu-inductor']
            for name in ('baseline', 'candidate', 'cases-provider', 'validator', 'acquisition'):
                invocation.extend(['--' + name, 'missing_benchmark_hook:factory'])
            child = subprocess.run(invocation, capture_output=True, text=True)
            self.assertEqual(child.returncode, 2, child.stderr)
            result = json.loads((output / 'summary.json').read_text())
            raw = json.loads((output / 'run-0001/report.json').read_text())
            self.assertEqual(raw['configuration']['backend'], 'npu-inductor')
            self.assertEqual(result['failed_repetitions'], 1)
            self.assertIn('missing_benchmark_hook', result['failures'][0]['error'])

    def test_fresh_children_default_ten_and_preserve_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            child = base / 'child.py'
            child.write_text('''import json, os, pathlib, sys
output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir()
payload = json.loads(sys.argv[1])
payload['pid'] = os.getpid()
(output / 'report.json').write_text(json.dumps(payload))
(output / 'raw.txt').write_text('untouched')
''')
            output = base / 'runs'
            result = repeated.run_repeated(output, [],
                command=[sys.executable, str(child), json.dumps(report())])
            self.assertEqual(result['completed_repetitions'], 10)
            self.assertEqual(len({json.loads((output / f'run-{i:04d}' / 'report.json').read_text())['pid']
                                  for i in range(1, 11)}), 10)
            self.assertEqual((output / 'run-0001/raw.txt').read_text(), 'untouched')
            self.assertTrue((output / 'summary.json').is_file())
            with self.assertRaises(FileExistsError):
                repeated.run_repeated(output, [], repetitions=1)

    def test_process_failure_without_report_is_retained_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            result = repeated.run_repeated(Path(directory) / 'runs', [], repetitions=2,
                command=[sys.executable, '-c', 'import sys; print("compile error"); sys.exit(2)'])
            self.assertEqual(result['failed_repetitions'], 2)
            self.assertEqual(len(result['runs']), 2)
            self.assertEqual(result['state'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
