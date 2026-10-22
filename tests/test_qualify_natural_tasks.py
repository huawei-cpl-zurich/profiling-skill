"""Exercise private qualification acquisition, failure retention and compact output."""
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/qualify_natural_tasks.py'
spec = importlib.util.spec_from_file_location('qualification', SCRIPT)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)

EXPORTER = '''
import json
def qualification_variants(task):
    return ('alternative-one', 'alternative-two')
def export(task, folder):
    folder.mkdir()
    for name in ('baseline.py','candidate.py'):
        (folder/name).write_text('baseline')
    (folder/'runner.py').write_text('trusted numerical oracle')
    (folder/'task.json').write_text(json.dumps({'id':task}))
'''


class Acquisition:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def execute(self, argv, folder, env, **kwargs):
        self.calls.append(argv)
        folder.mkdir(parents=True)
        (folder/'stdout.log').write_text('retained diagnostic')
        if '--version' in argv:
            return subprocess.CompletedProcess(argv, 0, 'test-profiler', '')
        if '--checks=4' in argv:
            records = '\n'.join(json.dumps({'correctness': True, 'seed': i}) for i in range(4))
            return subprocess.CompletedProcess(argv, int(self.fail == 'candidate'), records, 'compiler diagnostic')
        if '--launch-count=5000' in argv:
            report = folder/'report'
            report.mkdir()
            with (report/'OpBasicInfo.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Op Name', 'Task Duration(us)'])
                writer.writeheader()
                for _ in range(2 if self.fail == 'extra-kernel' else 1):
                    writer.writerow({'Op Name': 'actual_exported_symbol', 'Task Duration(us)': '10'})
            return subprocess.CompletedProcess(argv, 0, 'ok', '')
        raise AssertionError(argv)

    def capture(self, command, folder, env, metric, selector):
        self.calls.append([metric, selector])
        folder.mkdir()
        (folder/'evidence.json').write_text(json.dumps({'metric': metric}))
        (folder/'raw.bin').write_bytes(b'bulky-vendor-data')
        item = dict(metric=metric, selector=selector,
                    basic_rows=[{'Op Name': selector, 'Task Duration(us)': '10'}])
        if metric == 'PipeUtilization':
            if self.fail == 'pipes':
                raise RuntimeError('missing pipe rows despite exit zero')
            item['pipe_rows'] = [{'Op Name': selector, 'aiv_vec_ratio': '.6'}]
        return item

    def helpers(self):
        return dict(execute=self.execute, capture=self.capture,
                    duration=lambda row: float(row['Task Duration(us)']),
                    summarize=lambda values: dict(samples_us=values, median_us=statistics.median(values)),
                    validate_capture=lambda code, text: None)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        (self.source/'benchmarks').mkdir(parents=True)
        (self.source/'scripts').mkdir()
        (self.source/'benchmarks/natural_before_tasks.py').write_text(EXPORTER)
        for name in ('a3_memory_batch.py','profile_device.py'):
            (self.source/'scripts'/name).touch()
        self.bundle = self.root/'bundle'

    def prepare(self):
        return qualification.prepare(self.source, 'task01', self.bundle)

    def run_batch(self, acquisition):
        return qualification.run(self.bundle, self.bundle/'batch-1',
            helpers=acquisition.helpers(), device=(7, {'ASCEND_DEVICE_ID': '0'}))

    def test_preparation_preserves_oracles_and_exports_three_scoped_sources(self):
        manifest = self.prepare()
        self.assertEqual(len(manifest['variants']), 3)
        self.assertEqual([(self.bundle/f'variant-{i}/candidate.py').read_text() for i in range(3)],
                         ['baseline', 'alternative-one', 'alternative-two'])
        self.assertTrue(all((self.bundle/f'variant-{i}/runner.py').read_text() == 'trusted numerical oracle'
                            for i in range(3)))
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_success_keeps_all_samples_and_uses_discovered_selector(self):
        self.prepare()
        acquisition = Acquisition()
        result = self.run_batch(acquisition)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['physical_device'], 7)
        for item in result['variants'].values():
            self.assertEqual(len(item['correctness']), 4)
            self.assertEqual(item['timing']['samples_us'], [10.] * 5)
            self.assertEqual(len(item['pipe_captures']), 3)
            self.assertEqual(item['selector'], 'actual_exported_symbol')
        self.assertEqual(sum(call == ['PipeUtilization','actual_exported_symbol'] for call in acquisition.calls), 9)
        self.assertEqual(sum('--launch-count=5000' in call for call in acquisition.calls), 3)
        self.assertFalse(any(any(arg.startswith('--kernel-name') for arg in call)
                             for call in acquisition.calls if '--launch-count=5000' in call))
        with tarfile.open(self.bundle/'batch-1.tar.gz') as archive:
            self.assertIn('batch-1/summary.json', archive.getnames())
            self.assertTrue(any(name.endswith('stdout.log') for name in archive.getnames()))
            self.assertFalse(any(name.endswith('.bin') for name in archive.getnames()))
        with self.assertRaises(FileExistsError):
            self.run_batch(acquisition)

    def test_candidate_error_stops_before_profiles_and_retains_diagnostics(self):
        self.prepare()
        acquisition = Acquisition('candidate')
        result = self.run_batch(acquisition)
        first = result['variants']['variant-0']
        self.assertEqual(first['failure_stage'], 'candidate')
        self.assertIn('compiler diagnostic', first['error'])
        self.assertEqual(len(result['variants']), 1)
        self.assertFalse(any('--launch-count=5000' in call for call in acquisition.calls))
        self.assertTrue((self.bundle/'batch-1.tar.gz').is_file())

    def test_extra_compute_or_missing_pipes_never_accepts_qualification(self):
        for failure in ('extra-kernel', 'pipes'):
            with self.subTest(failure=failure):
                self.bundle = self.root/failure
                self.prepare()
                result = self.run_batch(Acquisition(failure))
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(len(result['variants']), 1)
                self.assertEqual(result['variants']['variant-0']['failure_stage'], 'profiling')

    def test_changed_source_rejected_before_output_or_device_work(self):
        self.prepare()
        (self.bundle/'variant-1/candidate.py').write_text('unexpected changed source')
        acquisition = Acquisition()
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.run_batch(acquisition)
        self.assertFalse((self.bundle/'batch-1').exists())
        self.assertEqual(acquisition.calls, [])

    def test_missing_dependency_does_not_reserve_output(self):
        (self.source/'scripts/profile_device.py').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_changed_trusted_inputs_are_rejected_before_device_reservation(self):
        for name in ('variant-0/runner.py', 'variant-1/task.json',
                     'a3_memory_batch.py', 'profile_device.py', 'qualify_natural_tasks.py'):
            with self.subTest(name=name):
                self.bundle = self.root/name.replace('/', '-')
                self.prepare()
                (self.bundle/name).write_text('changed trusted input')
                acquisition = Acquisition()
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.run_batch(acquisition)
                self.assertFalse((self.bundle/'batch-1').exists())
                self.assertEqual(acquisition.calls, [])

    def test_popen_compatible_device_probe_and_lease_release(self):
        self.prepare()
        # The real helper has a Popen-compatible interface, not subprocess.run.
        from io import StringIO
        lease = StringIO()
        def bounded(argv, **kwargs):
            timeout = kwargs.pop('timeout')
            with subprocess.Popen([sys.executable, '-c', 'print("probe-ok")'], **kwargs) as process:
                stdout, stderr = process.communicate(timeout=timeout)
                return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        def wait(probe):
            self.assertEqual(probe({}), (True, ''))
            return 3, {}, lease
        with patch.object(qualification.runpy, 'run_path', return_value=dict(
                bounded_run=bounded, wait_for_device=wait)):
            result = qualification.run(self.bundle, self.bundle/'batch-1',
                                       helpers=Acquisition().helpers())
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['physical_device'], 3)
        self.assertTrue(lease.closed)


if __name__ == '__main__':
    unittest.main()
