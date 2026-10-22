import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('ablation', Path(__file__).resolve().parents[1] / 'scripts/run_compiler_evidence_ablation.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

ANSWER = {'diagnosis': 'uncertain', 'decision': 'a', 'predictions': [
    {'candidate': key, 'latency': 'indeterminate', 'execution_change': 'unknown', 'evidence': ['timing.json']}
    for key in ['a', 'b']], 'uncertainty': 'limited evidence', 'next_measurement': 'measure both'}


def dump(path, value):
    path.write_text(json.dumps(value))


def fixtures(tmp):
    packets = tmp / 'packets'
    packets.mkdir()
    tasks = [f'task{i:02}' for i in range(1, 4)]
    evidence = {}
    for task in tasks:
        public = packets / 'public' / task
        public.mkdir(parents=True)
        for name in module.PUBLIC:
            (public / name).write_text('public input')
        dump(public / 'task.json', {'public_files': sorted(module.PUBLIC)})
        dump(public / 'environment.json', {'soc': 'test-soc', 'compiler': 'test-compiler', 'profiler': 'test-profiler'})
        evidence[task] = {'pipes': ['capture.csv'], 'compiler': ['ir/kernel.mlir', 'capture-manifest.json']}
        for kind, names in evidence[task].items():
            for name in names:
                path = packets / kind / task / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(kind + ' evidence')
        (packets / 'compiler' / task / 'not-whitelisted.txt').write_text('SECRET')
    private = packets / 'private'
    private.mkdir()
    (private / 'gold.json').write_text('SECRET')
    dump(packets / 'manifest.json', {'schema_version': 1, 'tasks': tasks, 'evidence': evidence,
        'files': {p.relative_to(packets).as_posix(): module.digest(p) for p in packets.rglob('*') if p.is_file()}})
    skill = tmp / 'skill'
    (skill / 'references').mkdir(parents=True)
    (skill / 'SKILL.md').write_text('frozen guidance')
    (skill / 'references' / 'reasoning.md').write_text('frozen reference')
    auth = tmp / 'auth'
    auth.mkdir()
    (auth / 'auth.json').write_text('{}')
    return packets, skill, auth


class FakeLauncher(module.DiagnosticLauncher):
    calls = []
    mode = None

    def launch(self):
        self.prepare()
        files = {p.relative_to(self.workspace).as_posix() for p in self.workspace.rglob('*') if p.is_file() and '.git' not in p.parts}
        self.calls.append((any(n.startswith('pipes/') for n in files), any(n.startswith('compiler/') for n in files)))
        assert self.skill_source
        assert not any('SECRET' in (self.workspace / n).read_text() for n in files)
        (self.workspace / 'answer.json').write_text(json.dumps(ANSWER))
        if self.mode == 'source':
            (self.workspace / 'candidate-a.py').write_text('changed')
        git = ['git', '-C', str(self.workspace)]
        if self.mode != 'uncommitted':
            subprocess.run([*git, 'add', 'answer.json'], check=True, capture_output=True)
            subprocess.run([*git, 'commit', '-m', 'diagnosis'], check=True, capture_output=True)
        self.socket_temporary.cleanup()
        return {'exit_code': 124 if self.mode == 'timeout' else 0, 'timed_out': self.mode == 'timeout'}


def run(tmp, **kwargs):
    packets, skill, auth = fixtures(tmp)
    FakeLauncher.calls = []
    return module.run(packets, skill, {'name': 'explicit', 'reasoning_effort': 'low'}, tmp / 'output',
                      launcher=FakeLauncher, launcher_kwargs={'auth_home': auth}, **kwargs)


def test_twelve_fresh_cells_with_common_skill_and_no_private_files(tmp_path):
    assert run(tmp_path)['completed'] == 12
    assert len(FakeLauncher.calls) == 12
    for condition in ((False, False), (True, False), (False, True), (True, True)):
        assert FakeLauncher.calls.count(condition) == 3
    manifest = json.loads((tmp_path / 'output' / 'manifest.json').read_text())
    assert manifest['budget_seconds'] == 300
    assert len(manifest['skill_files']) == 2
    assert not list((tmp_path / 'output' / 'frozen').rglob('gold.json'))
    assert not list((tmp_path / 'output' / 'frozen').rglob('not-whitelisted.txt'))


@pytest.mark.parametrize('mode', ['source', 'uncommitted', 'timeout'])
def test_invalid_results_retained_and_not_automatically_retried(tmp_path, mode):
    FakeLauncher.mode = mode
    try:
        assert run(tmp_path)['completed'] == 0
    finally:
        FakeLauncher.mode = None
    FakeLauncher.calls = []
    result = module.run(tmp_path / 'packets', tmp_path / 'skill', {'name': 'explicit', 'reasoning_effort': 'low'},
                        tmp_path / 'output', launcher=FakeLauncher, resume=True)
    assert result['completed'] == 0
    assert not FakeLauncher.calls
    assert (tmp_path / 'output' / 'task01-timing' / 'agent' / 'workspace' / 'answer.json').exists()


@pytest.mark.parametrize('failure', ['hash', 'environment', 'tasks', 'traversal', 'symlink'])
def test_bad_packets_fail_before_dispatch(tmp_path, failure):
    packets, skill, _ = fixtures(tmp_path)
    manifest = json.loads((packets / 'manifest.json').read_text())
    if failure == 'hash':
        (packets / 'public/task01/baseline.py').write_text('changed')
    elif failure == 'environment':
        dump(packets / 'public/task01/environment.json', {'soc': ''})
    elif failure == 'tasks':
        manifest['tasks'].pop()
    elif failure == 'traversal':
        manifest['evidence']['task01']['compiler'] = ['../task02/capture-manifest.json']
    else:
        path = packets / 'pipes/task01/capture.csv'
        path.unlink()
        path.symlink_to(packets / 'private/gold.json')
    dump(packets / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        module.run(packets, skill, {'name': 'x', 'reasoning_effort': 'low'}, tmp_path / 'output', launcher=FakeLauncher)
    assert not (tmp_path / 'output').exists()


def test_resume_revalidates_commit_and_partial_cells_without_duplication(tmp_path):
    run(tmp_path)
    cell = tmp_path / 'output/task01-timing'
    subprocess.run(['git', '-C', str(cell / 'agent/workspace'), 'commit', '--allow-empty', '-m', 'new head'], check=True, capture_output=True)
    (tmp_path / 'output/task01-pipes/result.json').unlink()
    FakeLauncher.calls = []
    result = module.run(tmp_path / 'packets', tmp_path / 'skill', {'name': 'explicit', 'reasoning_effort': 'low'},
                        tmp_path / 'output', launcher=FakeLauncher, resume=True)
    assert result['completed'] == 10
    assert not FakeLauncher.calls


@pytest.mark.parametrize('change', ['skill-extra', 'packet', 'prompt'])
def test_resume_rejects_frozen_input_changes(tmp_path, change):
    run(tmp_path)
    frozen = tmp_path / 'output/frozen'
    if change == 'skill-extra':
        (frozen / 'skill/EXTRA.md').write_text('new guidance')
    elif change == 'packet':
        (frozen / 'packets/compiler/task01/ir/kernel.mlir').write_text('changed')
    else:
        (frozen / 'prompt.md').write_text('new prompt')
    FakeLauncher.calls = []
    if change == 'prompt':
        # Completed retained cells are not dispatched; the frozen prompt still
        # must be rejected rather than silently certifying a changed campaign.
        expected = 'frozen prompt'
    else:
        expected = 'frozen'
    with pytest.raises(ValueError, match=expected):
        module.run(tmp_path / 'packets', tmp_path / 'skill', {'name': 'explicit', 'reasoning_effort': 'low'},
                   tmp_path / 'output', launcher=FakeLauncher, resume=True)
    assert not FakeLauncher.calls


def test_input_overlap_and_nonpositive_budget_fail_before_creation(tmp_path):
    packets, skill, _ = fixtures(tmp_path)
    for output, budget in ((packets / 'output', 300), (skill / 'output', 300), (tmp_path / 'output', 0)):
        with pytest.raises(ValueError):
            module.run(packets, skill, {'name': 'x', 'reasoning_effort': 'low'}, output, budget=budget)
        assert not output.exists()


@pytest.mark.parametrize('arm', module.ARMS)
def test_real_sandbox_evidence_visibility_and_readonly_inputs(tmp_path, arm):
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('local sandbox runtime unavailable')
    packets, skill, auth = fixtures(tmp_path)
    bundle = tmp_path / 'bundle'
    shutil.copytree(packets / 'public/task01', bundle)
    files = sorted(module.PUBLIC)
    for kind in ('pipes', 'compiler'):
        if arm in (kind, 'both'):
            names = ['capture.csv'] if kind == 'pipes' else ['ir/kernel.mlir', 'capture-manifest.json']
            for name in names:
                target = bundle / kind / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packets / kind / 'task01' / name, target)
                files.append(kind + '/' + name)
    dump(bundle / 'task.json', {'public_files': files})
    prompt = tmp_path / 'prompt.md'
    prompt.write_text(module.PROMPT)
    instance = module.DiagnosticLauncher(tmp_path / 'sandbox', bundle, prompt,
        {'name': 'explicit', 'reasoning_effort': 'low'}, [sys.executable, '-c', 'print("offline")'],
        skill=skill, auth_home=auth, forbidden=(packets,), timeout=300)
    instance.prepare()
    try:
        instance.preflight()
        code = '''from pathlib import Path
import json, sys
p = Path('/workspace')
assert (p/'pipes/capture.csv').exists() == (sys.argv[1] in ('pipes','both'))
assert (p/'compiler/ir/kernel.mlir').exists() == (sys.argv[1] in ('compiler','both'))
assert (p/'.agents/skills/ascend-profiling/SKILL.md').is_file()
assert not Path(sys.argv[2]).exists()
for name in json.loads((p/'task.json').read_text())['public_files']:
    try: (p/name).write_text('tamper')
    except OSError: pass
    else: raise AssertionError(name+' writable')
(p/'answer.json').write_text('writable')
'''
        result = subprocess.run([*instance.base_command(), 'python3', '-c', code, arm, str(packets)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    finally:
        instance.socket_temporary.cleanup()
