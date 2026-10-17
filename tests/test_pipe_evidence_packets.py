import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/export_pipe_evidence_packets.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = []
        self.csv = b'block_id,sub_block_id,aiv_vec_ratio,aiv_vec_time(us)\n0,vector0,0.25,NA\n1,vector1,0.5,2\n'
        for task in ('task01', 'task02', 'task03', 'task04', 'task05', 'task06'):
            for variant in range(3):
                folder = self.root / task / f'variant-{variant}'
                folder.mkdir(parents=True)
                (folder / 'candidate.py').write_text(f'# source {variant}\n')
                (folder / 'task.json').write_text(json.dumps(dict(
                    id=task, family='gather', selector='task_kernel', rows=2,
                    public_files=['candidate.py', 'runner.py'])))
            for batch in (1, 2):
                case = dict(task=task, batch=batch, state='completed', task_spec=dict(family='gather', rows=2),
                            archive=f'evidence-{task}-batch{batch}.tar.gz', variants={})
                files = {}
                for index in range(3):
                    name = f'variant-{index}'
                    source = (self.root / task / name / 'candidate.py').read_bytes()
                    record = dict(source_sha256=digest(source), state='completed', selector='task_kernel',
                                  timing=dict(samples_us=[1., 2., 3., 4., 5.], median_us=3.),
                                  correctness_checks_passed=4, correctness_seeds=[1, 2, 3, 4],
                                  compute_discovery_count=1, pipe_capture_row_counts=[2, 2, 2],
                                  compiler_artifacts={})
                    result = dict(record, correctness=[dict(correctness=True, seed=x) for x in range(1, 5)],
                                  discovered_compute_rows=[{'Op Name': 'task_kernel'}])
                    prefix = f'batch-{batch}/{name}'
                    files[f'{prefix}/result.json'] = json.dumps(result).encode()
                    files[f'{prefix}/discovery/report/OpBasicInfo.csv'] = b'Op Name,Task Duration(us)\ntask_kernel,1\n'
                    for capture in range(5):
                        files[f'{prefix}/basic-{capture}/report/OpBasicInfo.csv'] = (
                            f'Op Name,Task Duration(us)\ntask_kernel,{capture + 1}\n'.encode())
                    for capture in range(3):
                        files[f'{prefix}/pipe-{capture}/report/PipeUtilization.csv'] = self.csv
                    for kind in ('ttir', 'ttadapter'):
                        member = f'{name}/compiler-evidence/candidate/{kind}.txt'
                        files[member] = b'PRIVATE COMPILER EVIDENCE'
                        record['compiler_artifacts'][kind] = dict(member=member, sha256=digest(files[member]))
                    case['variants'][name] = record
                archive = self.root / case['archive']
                with tarfile.open(archive, 'w:gz') as stream:
                    for name, data in files.items():
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        stream.addfile(info, io.BytesIO(data))
                case['archive_sha256'] = digest(archive.read_bytes())
                self.cases.append(case)
        self.qualification = self.root / 'qualification.json'
        self.write_qualification()

    def write_qualification(self):
        self.qualification.write_text(json.dumps(dict(cases=self.cases)))

    def export(self):
        return runpy.run_path(str(SCRIPT))['export'](
            self.qualification, self.root, self.root, self.root / 'output')

    def test_exports_six_complete_packets_without_private_evidence(self):
        manifest = self.export()
        self.assertEqual(manifest['tasks'], [f'task{x:02}' for x in range(1, 7)])
        output = self.root / 'output'
        for task in manifest['tasks']:
            public = output / 'public' / task
            metadata = json.loads((public / 'task.json').read_text())
            self.assertEqual(set(metadata['public_files']), {p.name for p in public.iterdir()})
            self.assertEqual((public / 'candidate-a.py').read_text(), '# source 1\n')
            self.assertEqual((public / 'candidate-b.py').read_text(), '# source 2\n')
            self.assertEqual(json.loads((public / 'timing.json').read_text())['samples_us'], [1, 2, 3, 4, 5])
            for capture in range(3):
                self.assertEqual((output / 'pipes' / task / f'pipe-{capture}.csv').read_bytes(), self.csv)
        for path in (output / 'public').rglob('*'):
            if path.is_file():
                self.assertNotIn(b'PRIVATE COMPILER EVIDENCE', path.read_bytes())
                self.assertNotIn(b'compiler_artifacts', path.read_bytes())
        gold = json.loads((output / 'private/gold.json').read_text())
        self.assertEqual(len(gold['cases']), 12)
        self.assertEqual(len(manifest['files']), 61)

    def test_bad_archive_hash_rejected_before_output(self):
        self.cases[0]['archive_sha256'] = 'bad'
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'archive'):
            self.export()
        self.assertFalse((self.root / 'output').exists())

    def test_changed_candidate_rejected_before_output(self):
        (self.root / 'task06/variant-2/candidate.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source'):
            self.export()
        self.assertFalse((self.root / 'output').exists())

    def test_inconsistent_timing_rejected(self):
        self.cases[-1]['variants']['variant-2']['timing']['samples_us'][0] = 99
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'timing'):
            self.export()

    def test_missing_capture_rejected(self):
        self.cases[0]['variants']['variant-0']['pipe_capture_row_counts'] = [2, 2, 3]
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'pipe'):
            self.export()

    def test_wrong_selector_rejected(self):
        self.cases[0]['variants']['variant-0']['selector'] = 'wrong'
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'selector'):
            self.export()

    def test_wrong_compiler_hash_rejected(self):
        self.cases[0]['variants']['variant-0']['compiler_artifacts']['ttir']['sha256'] = 'bad'
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'compiler'):
            self.export()

    def test_wrong_task_shape_rejected_before_output(self):
        metadata = self.root / 'task06/variant-2/task.json'
        spec = json.loads(metadata.read_text())
        spec['rows'] = 999
        metadata.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'specification'):
            self.export()
        self.assertFalse((self.root / 'output').exists())

    def test_missing_seeded_check_rejected(self):
        self.cases[0]['variants']['variant-0']['correctness_checks_passed'] = 3
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'seeded'):
            self.export()

    def test_duplicate_batch_and_existing_output_fail(self):
        self.cases[-1] = self.cases[0]
        self.write_qualification()
        with self.assertRaisesRegex(ValueError, 'batches'):
            self.export()

    def test_cli(self):
        command = [sys.executable, str(SCRIPT), '--qualification', str(self.qualification),
                   '--archives', str(self.root), '--bundles', str(self.root),
                   '--output', str(self.root / 'output')]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)['tasks'], [f'task{x:02}' for x in range(1, 7)])
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)


if __name__ == '__main__':
    unittest.main()
