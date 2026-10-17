import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('diagnostic', Path(__file__).resolve().parents[1] / 'scripts/run_pipe_evidence_diagnostic.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def dump(path, value):
    path.write_text(json.dumps(value))


def fixtures(tmp):
    packets = tmp / 'packets'
    packets.mkdir()
    tasks = [f'task{i:02}' for i in range(1, 7)]
    for task in tasks:
        public = packets / 'public' / task
        public.mkdir(parents=True)
        files = ['task.json', 'TASK.md', 'baseline.py', 'candidate-a.py', 'candidate-b.py', 'timing.json']
        dump(public / 'task.json', {'public_files': files})
        for name in files[1:]:
            (public / name).write_text('{}' if name.endswith('.json') else 'baseline input')
        pipes = packets / 'pipes' / task
        pipes.mkdir(parents=True)
        for name in ['pipe-manifest.json', 'pipe-0.csv', 'pipe-1.csv', 'pipe-2.csv']:
            (pipes / name).write_text('raw metric')
    private = packets / 'private'
    private.mkdir()
    (private / 'gold.json').write_text('SECRET')
    dump(packets / 'manifest.json', {'schema_version': 1, 'tasks': tasks, 'files': {
        str(p.relative_to(packets)): module.digest(p) for p in packets.rglob('*') if p.is_file()}})
    skill = tmp / 'skill'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('frozen guidance')
    return packets, skill


ANSWER = {'diagnosis': 'uncertain', 'decision': 'a', 'predictions': [
    {'candidate': key, 'latency': 'indeterminate', 'execution_change': 'unknown', 'evidence': ['timing.json']}
    for key in ['a', 'b']], 'uncertainty': 'limited evidence', 'next_measurement': 'measure both'}


class FakeLauncher(module.DiagnosticLauncher):
    calls = []
    mode = None

    def launch(self):
        self.prepare()
        stage = 'before-answer.json' in json.loads((self.bundle / 'task.json').read_text())['public_files']
        has_pipes = (self.workspace / 'pipe-0.csv').exists()
        self.calls.append((stage, has_pipes, bool(self.skill_source)))
        assert not any('SECRET' in f.read_text() for f in self.workspace.glob('*') if f.is_file())
        if stage:
            assert json.loads((self.workspace / 'before-answer.json').read_text()) == ANSWER
        (self.workspace / 'answer.json').write_text(json.dumps(ANSWER))
        if self.mode == 'source':
            (self.workspace / 'candidate-a.py').write_text('changed')
        git = ['git', '-C', str(self.workspace)]
        if self.mode != 'uncommitted':
            subprocess.run([*git, 'add', 'answer.json'], check=True, capture_output=True)
            subprocess.run([*git, 'commit', '-m', 'diagnosis'], check=True, capture_output=True)
        self.socket_temporary.cleanup()
        return {'exit_code': 124 if self.mode == 'timeout' else 0, 'timed_out': self.mode == 'timeout'}


def run(tmp, **kw):
    packets, skill = fixtures(tmp)
    auth = tmp / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    FakeLauncher.calls = []
    return module.run(packets, skill, {'name': 'explicit', 'reasoning_effort': 'low'}, tmp / 'output',
                      launcher=FakeLauncher, launcher_kwargs={'auth_home': auth}, **kw)


def test_complete_36_stage_case_and_frozen_identity(tmp_path):
    result = run(tmp_path)
    assert result['completed'] == 18
    assert len(FakeLauncher.calls) == 36
    assert FakeLauncher.calls.count((False, False, False)) == 18
    assert FakeLauncher.calls.count((True, False, False)) == 6
    assert FakeLauncher.calls.count((True, True, False)) == 6
    assert FakeLauncher.calls.count((True, True, True)) == 6
    manifest = json.loads((tmp_path / 'output' / 'manifest.json').read_text())
    assert manifest['budget_seconds'] == 240
    assert manifest['model']['name'] == 'explicit'
    assert len(manifest['files']) > 50
    with pytest.raises(ValueError, match='already exists'):
        module.run(tmp_path / 'packets', tmp_path / 'skill', manifest['model'], tmp_path / 'output')


@pytest.mark.parametrize('mode', ['uncommitted', 'source', 'timeout'])
def test_failures_retained_and_stage_two_not_dispatched(tmp_path, mode):
    FakeLauncher.mode = mode
    try:
        result = run(tmp_path)
    finally:
        FakeLauncher.mode = None
    assert result['completed'] == 0
    assert len(FakeLauncher.calls) == 18
    assert (tmp_path / 'output' / 'task01-timing' / 'stage1' / 'workspace' / 'answer.json').exists()


def test_bad_packet_rejected_before_output(tmp_path):
    packets, skill = fixtures(tmp_path)
    (packets / 'pipes' / 'task06' / 'pipe-2.csv').unlink()
    with pytest.raises(ValueError):
        module.run(packets, skill, {'name': 'x', 'reasoning_effort': 'low'}, tmp_path / 'output', launcher=FakeLauncher)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('failure', ['missing-map', 'wrong-schema', 'tampered-input'])
def test_exporter_provenance_required_before_output(tmp_path, failure):
    packets, skill = fixtures(tmp_path)
    path = packets / 'manifest.json'
    manifest = json.loads(path.read_text())
    if failure == 'missing-map':
        del manifest['files']
    elif failure == 'wrong-schema':
        manifest['schema_version'] = 2
    else:
        (packets / 'public' / 'task03' / 'baseline.py').write_text('changed after export')
    dump(path, manifest)
    with pytest.raises(ValueError):
        module.run(packets, skill, {'name': 'x', 'reasoning_effort': 'low'}, tmp_path / 'output', launcher=FakeLauncher)
    assert not (tmp_path / 'output').exists()


def test_offline_operations_fail_clearly(tmp_path):
    script = str(Path(module.__file__))
    for verb in ['timing', 'pipes', 'ir', 'run']:
        proc = subprocess.run([sys.executable, script, 'offline', '--workspace', str(tmp_path), verb], capture_output=True, text=True)
        assert proc.returncode == 2
        assert 'unavailable' in proc.stderr
    assert subprocess.run([sys.executable, script, 'offline', '--workspace', str(tmp_path), 'help'], capture_output=True).returncode == 0


@pytest.mark.parametrize('edit', ['missing', 'prediction', 'blank'])
def test_answer_schema_rejects_incomplete_diagnoses(edit):
    answer = json.loads(json.dumps(ANSWER))
    if edit == 'missing':
        del answer['uncertainty']
    elif edit == 'blank':
        answer['diagnosis'] = ''
    else:
        answer['predictions'][1]['candidate'] = 'a'
    with pytest.raises(ValueError):
        module.validate_answer(answer)


def test_resume_never_duplicates_existing_stages(tmp_path):
    run(tmp_path)
    FakeLauncher.calls = []
    manifest = json.loads((tmp_path / 'output' / 'manifest.json').read_text())
    result = module.run(tmp_path / 'packets', tmp_path / 'skill', manifest['model'], tmp_path / 'output',
                        launcher=FakeLauncher, resume=True)
    assert result['completed'] == 18
    assert not FakeLauncher.calls
    (tmp_path / 'output' / 'task01-timing' / 'stage2-result.json').unlink()
    result = module.run(tmp_path / 'packets', tmp_path / 'skill', manifest['model'], tmp_path / 'output',
                        launcher=FakeLauncher, resume=True)
    assert result['completed'] == 17
    assert not FakeLauncher.calls


@pytest.mark.parametrize('arm', module.ARMS)
def test_real_bwrap_readonly_and_visibility(tmp_path, arm):
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('local sandbox runtime unavailable')
    packets, skill = fixtures(tmp_path)
    bundle = tmp_path / 'bundle'
    shutil.copytree(packets / 'public' / 'task01', bundle)
    declaration = json.loads((bundle / 'task.json').read_text())
    if arm != 'timing':
        for name in module.PIPES:
            shutil.copy2(packets / 'pipes' / 'task01' / name, bundle / name)
            declaration['public_files'].append(name)
    dump(bundle / 'before-answer.json', ANSWER)
    declaration['public_files'].append('before-answer.json')
    dump(bundle / 'task.json', declaration)
    auth = tmp_path / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    prompt = tmp_path / 'prompt.md'
    prompt.write_text(module.AFTER)
    instance = module.DiagnosticLauncher(tmp_path / 'sandbox', bundle, prompt,
        {'name': 'explicit', 'reasoning_effort': 'low'}, [sys.executable, str(Path(module.__file__)), 'offline'],
        skill=skill if arm == 'pipes-skill' else None, auth_home=auth, forbidden=(packets,))
    instance.prepare()
    try:
        instance.preflight()
        code = '''from pathlib import Path
import sys
assert Path('/workspace/pipe-0.csv').exists() == (sys.argv[1] != 'timing')
assert Path('/workspace/.agents/skills/ascend-profiling/SKILL.md').exists() == (sys.argv[1] == 'pipes-skill')
assert not Path(sys.argv[2]).exists()
for name in ['baseline.py', 'candidate-a.py', 'candidate-b.py', 'before-answer.json']:
    try:
        Path('/workspace', name).write_text('tamper')
    except OSError:
        pass
    else:
        raise AssertionError(name + ' writable')
Path('/workspace/answer.json').write_text('answer writable')
'''
        result = subprocess.run([*instance.base_command(), 'python3', '-c', code, arm, str(packets)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    finally:
        instance.socket_temporary.cleanup()
