"""Exercise the cohort controller without model credentials or a device."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_blind_optimization_pilot.py'
SPEC = importlib.util.spec_from_file_location('pilot', SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)

EXPORTER = '''
import json
def export(task, output):
    output.mkdir()
    (output / 'candidate.py').write_text('starting kernel')
    (output / 'private.py').write_text('trusted oracle')
    (output / 'task.json').write_text(json.dumps({'public_files': ['candidate.py', 'task.json']}))
'''
LAUNCHER = '''
PROMPT = 'shared prompt v2 with 600 second budget'
if __name__ == '__main__':
    import argparse, json, os, time
    from pathlib import Path
    parser = argparse.ArgumentParser()
    for name in ('bundle', 'prompt', 'output', 'model-json', 'tool-command-json', 'skill'):
        parser.add_argument('--' + name)
    parser.add_argument('--forbidden', action='append')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir()
    events = output.parent / 'events.jsonl'
    def event(kind):
        with events.open('a') as stream:
            stream.write(json.dumps([kind, output.name, time.time()]) + '\\n')
    event('start')
    time.sleep(.08)
    bundle = Path(args.bundle)
    public = json.loads((bundle / 'task.json').read_text())['public_files']
    workspace = output / 'workspace'
    workspace.mkdir()
    for name in public:
        (workspace / name).write_bytes((bundle / name).read_bytes())
    receipt = dict(prompt=Path(args.prompt).read_text(),
                   model=json.loads(Path(args.model_json).read_text()),
                   tool=json.loads(Path(args.tool_command_json).read_text()),
                   guidance=bool(args.skill), forbidden=args.forbidden)
    (output / 'receipt.json').write_text(json.dumps(receipt))
    print('retained agent diagnostics', flush=True)
    event('end')
    raise SystemExit(7 if output.name == 'task02-control' else 0)
'''


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for relative in [*('scripts/' + name for name in pilot.HELPERS), *pilot.GUIDANCE,
                         'benchmarks/blind_before_tasks.py',
                         'benchmarks/blind_task_support/runner.py',
                         'benchmarks/blind_task_support/native_host.asc']:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        (self.source / 'scripts/blind_optimization_launcher.py').write_text(LAUNCHER)
        (self.source / 'benchmarks/blind_before_tasks.py').write_text(EXPORTER)
        # These source-only answers must not enter the guidance snapshot.
        (self.source / 'references/answers.json').write_text('private')
        self.output = self.root / 'cohort'
        self.model = dict(name='explicit-model', reasoning_effort='high')
        self.forbidden = [self.source, self.root / 'orchestration']

    def prepare(self):
        return pilot.prepare(self.source, self.output, self.model, self.forbidden)

    def test_paired_execution_retains_failures_without_retries(self):
        self.prepare()
        self.assertEqual(pilot.run_cohort(self.output), 1)
        results = [json.loads(path.read_text()) for path in self.output.glob('*-result.json')]
        self.assertEqual(len(results), 12)
        self.assertEqual(sum(result['controller_exit'] != 0 for result in results), 1)
        self.assertEqual(next(r for r in results if r['cell'] == 'task02-control')['controller_exit'], 7)
        for index in range(1, 7):
            guidance, control = (next(r for r in results if r['cell'] == f'task{index:02}-{arm}')
                                 for arm in ('guidance', 'control'))
            for key in ('prompt', 'model', 'tool', 'forbidden'):
                self.assertEqual(guidance['receipt'][key], control['receipt'][key])
            self.assertTrue(guidance['receipt']['guidance'])
            self.assertFalse(control['receipt']['guidance'])
            self.assertEqual(guidance['command'][:-2],
                             [arg.replace('-control', '-guidance') for arg in control['command']])
            self.assertIn('runtime/blind_optimization_tools.py', control['receipt']['tool'][1])
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        active = peak = 0
        for kind, name, timestamp in events:
            active += 1 if kind == 'start' else -1
            peak = max(peak, active)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 3)
        self.assertEqual(len([e for e in events if e[0] == 'start']), 12)
        for wave, expected in enumerate(pilot.WAVES, 1):
            cells = json.loads((self.output / f'wave-{wave}.json').read_text())
            self.assertEqual([r['cell'] for r in cells],
                             [f'task{i:02}-' + ('guidance' if g else 'control') for i, g in expected])
            if wave > 1:
                previous = json.loads((self.output / f'wave-{wave-1}.json').read_text())
                self.assertGreaterEqual(min(r['started'] for r in cells), max(r['ended'] for r in previous))
        self.assertFalse(any(self.output.glob('task*-*/workspace/private.py')))
        self.assertIn('retained agent diagnostics',
                      (self.output / 'task02-control-controller.log').read_text())
        with self.assertRaises(FileExistsError):
            pilot.run_cohort(self.output)
        self.assertEqual(len((self.output / 'events.jsonl').read_text().splitlines()), 24)

    def test_prepare_cli_freezes_only_allowlist_and_rejects_reuse(self):
        model = self.root / 'model.json'
        model.write_text(json.dumps(self.model))
        command = [sys.executable, str(SCRIPT), '--source-root', str(self.source),
                   '--output', str(self.output), '--model-json', str(model),
                   '--forbidden-workspace', str(self.source), '--prepare-only']
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertFalse((self.output / 'events.jsonl').exists())
        self.assertEqual({str(p.relative_to(self.output / 'skill')) for p in
                          (self.output / 'skill').rglob('*') if p.is_file()}, set(pilot.GUIDANCE))
        self.assertFalse(any(self.output.rglob('cpl-remote')))
        self.assertEqual(json.loads((self.output / 'model.json').read_text()), self.model)
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        (self.output / 'runtime/profile_device.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed frozen input'):
            pilot.run_cohort(self.output)
        self.assertFalse((self.output / 'started.json').exists())

    def test_invalid_model_and_forbidden_output_fail_before_creation(self):
        self.model.pop('reasoning_effort')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.model['reasoning_effort'] = 'high'
        self.output = self.source / 'cohort'
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
