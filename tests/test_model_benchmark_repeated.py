import json
from pathlib import Path
import sys
import subprocess
import os
import signal
import time
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertTrue(result['within_repetition_comparable'])
        self.assertFalse(result['comparability']['comparable'])

    def test_ten_latency_repetitions_with_independent_profile_counts(self):
        reports = [report() for _ in range(10)]
        for index, item in enumerate(reports):
            row = item['cases'][0]
            row['metric_status'] = {}
            if index < 7:
                for kind in ('all_kernel', 'device', 'aicpu'):
                    row[kind + '_busy_median_us'].pop('baseline')
                    row['metric_status']['baseline_' + kind + '_busy_us'] = 'unavailable'
                row.pop('busy_time_speedup')
                row['metric_status']['busy_time_speedup'] = 'unavailable'
        result = repeated.aggregate(reports, 10)
        self.assertEqual(result['state'], 'complete')
        metrics = result['cases'][0]['metrics']
        self.assertEqual(metrics['latency_speedup']['n'], 10)
        self.assertEqual(metrics['candidate_device_busy_us']['n'], 10)
        self.assertEqual(metrics['baseline_device_busy_us']['n'], 3)
        self.assertEqual(metrics['busy_time_speedup']['unavailable_repetitions'], 7)
        self.assertEqual(metrics['busy_time_speedup']['state'], 'incomplete')
        self.assertEqual(metrics['baseline_latency_ms']['state'], 'complete')

    def test_all_profiles_unavailable_retains_latency_but_not_fabricated_statistics(self):
        reports = [report() for _ in range(10)]
        for item in reports:
            row = item['cases'][0]
            row['metric_status'] = {'busy_time_speedup': 'unavailable'}
            row.pop('busy_time_speedup')
        result = repeated.aggregate(reports)
        metric = result['cases'][0]['metrics']['busy_time_speedup']
        self.assertEqual(metric['n'], 0)
        self.assertIsNone(metric['mean'])
        self.assertIsNone(metric['sample_stddev'])
        self.assertEqual(result['completed_repetitions'], 10)

    def test_unavailable_latency_is_not_a_complete_repetition(self):
        item = report()
        item['cases'][0]['metric_status'] = {'candidate_latency_ms': 'unavailable'}
        self.assertEqual(repeated.aggregate([item], 1)['failed_repetitions'], 1)

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

    def test_malformed_cases_count_failed_and_do_not_abort_series(self):
        for cases in ('not-a-list', {'id': 'shape'}, [None], [1], [], ['case']):
            with self.subTest(cases=cases):
                malformed = report()
                malformed['cases'] = cases
                result = repeated.aggregate([malformed, report()], 2)
                self.assertEqual(result['failed_repetitions'], 1)
                self.assertEqual(result['completed_repetitions'], 1)
                self.assertEqual(result['attempted_repetitions'], 2)
                self.assertEqual(result['running_repetitions'], 0)
                self.assertEqual(result['unattempted_repetitions'], 0)

    def test_malformed_child_preserves_report_and_continues_next_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            child = base / 'child.py'
            child.write_text('''import json, pathlib, sys
output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir()
payload = json.loads(sys.argv[1])
if output.name == 'run-0001': payload['cases'] = [None]
(output / 'report.json').write_text(json.dumps(payload))
''')
            output = base / 'runs'
            result = repeated.run_repeated(output, [], repetitions=3,
                command=[sys.executable, str(child), json.dumps(report())])
            self.assertEqual(result['completed_repetitions'], 2)
            self.assertEqual(result['failed_repetitions'], 1)
            self.assertEqual(result['attempted_repetitions'], 3)
            self.assertEqual(result['running_repetitions'], 0)
            self.assertEqual(result['unattempted_repetitions'], 0)
            self.assertEqual([row['state'] for row in result['runs']], ['failed', 'complete', 'complete'])
            self.assertEqual(json.loads((output / 'run-0001/report.json').read_text())['cases'], [None])

    def test_failed_atomic_replace_preserves_previous_valid_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            repeated.save(path, {'running': 1})
            with patch.object(repeated.os, 'replace', side_effect=OSError('interrupted replace')):
                with self.assertRaises(OSError):
                    repeated.save(path, {'completed': 1})
            self.assertEqual(json.loads(path.read_text()), {'running': 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    @unittest.skipUnless(hasattr(os, 'killpg'), 'requires POSIX process groups')
    def test_interruption_retains_running_entry_before_child_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            child = base / 'child.py'
            child.write_text('''import os, pathlib, sys, time
output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
output.mkdir()
print('retained running log', flush=True)
(output / 'pid').write_text(str(os.getpid()))
time.sleep(30)
''')
            output = base / 'runs'
            driver = ('import sys; sys.path.insert(0, sys.argv[1]); '
                      'import model_benchmark_repeated as r; '
                      'r.run_repeated(sys.argv[2], [], command=[sys.executable, sys.argv[3]])')
            process = subprocess.Popen([sys.executable, '-c', driver,
                str(Path(repeated.__file__).parent), str(output), str(child)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    pidfile = output / 'run-0001/pid'
                    if pidfile.exists():
                        try:
                            child_pid = int(pidfile.read_text())
                            break
                        except ValueError:
                            pass
                    if process.poll() is not None:
                        self.fail(process.stderr.read().decode())
                    time.sleep(.01)
                self.assertIsNotNone(child_pid, 'bounded child did not start')
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
                result = json.loads((output / 'summary.json').read_text())
                self.assertEqual(result['attempted_repetitions'], 1)
                self.assertEqual(result['running_repetitions'], 1)
                self.assertEqual(result['completed_repetitions'], 0)
                self.assertEqual(result['failed_repetitions'], 0)
                self.assertEqual(result['unattempted_repetitions'], 9)
                entry = result['runs'][0]
                self.assertEqual(entry['state'], 'running')
                self.assertEqual(entry['command'][-1], str(output / 'run-0001'))
                self.assertEqual(entry['report'], 'run-0001/report.json')
                self.assertIn('retained running log', (output / entry['log']).read_text())
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                process.stderr.close()
                stat = Path(f'/proc/{child_pid}/stat')
                deadline = time.monotonic() + 2
                while child_pid is not None and stat.exists() and time.monotonic() < deadline:
                    if stat.read_text().split()[2] == 'Z':
                        break
                    time.sleep(.01)
                if child_pid is not None and stat.exists():
                    # A reparented zombie is already dead; it cannot retain NPU work.
                    self.assertEqual(stat.read_text().split()[2], 'Z')

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
