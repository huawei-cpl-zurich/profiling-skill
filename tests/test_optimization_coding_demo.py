import json
from pathlib import Path
import shutil
import socketserver
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import optimization_coding_demo as m
from blind_optimization_launcher import ToolHandler, execute


def manifest(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'candidate.py').write_text('def run(inputs): return inputs["x"]\n')
    (bundle / 'secret.py').write_text('hidden_answer = 17')
    (bundle / 'task.json').write_text(json.dumps({'public_files': ['candidate.py']}))
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Implement the public operation and use experiment-tool for measurements.')
    skills = {}
    for name in ['cannbot', 'library']:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'SKILL.md').write_text('---\nname: ' + name + '\ndescription: guidance\n---\n' + name)
        skills[name] = str(folder)
    return dict(model=dict(name='test-model', reasoning_effort='low'), seed=5,
                budget_seconds=60, max_tool_calls=6, common_skills={'cannbot': skills['cannbot']},
                library_skill=dict(name='library', path=skills['library']), tasks=[dict(
                    id='task-1', bundle=str(bundle), prompt=str(prompt), tool_command=['host-tool'],
                    evaluator_command=['host-evaluator'], expected_case_ids=['a', 'b'],
                    minimum_geomean_speedup=1.5, maximum_case_regression=1.25)])


def report(candidate_hash='source', *, reverse=False):
    rows = [dict(id=i, correct=True, identity_verified=True,
                 baseline_samples_us=[20.] * 10, candidate_samples_us=[value] * 10)
            for i, value in [('a', 5.), ('b', 10.)]]
    return dict(schema='optimization-coding-evaluation-v1', candidate_sha256=candidate_hash,
                measurement_kind=m.MEASUREMENT, cases=list(reversed(rows)) if reverse else rows)


def test_actual_speedup_gate_uses_all_cases_and_stable_identities(tmp_path):
    task = manifest(tmp_path)['tasks'][0]
    result = m.adjudicate(report(reverse=True), task, 'source')
    assert result['solved'] and result['case_speedups'] == {'a': 4, 'b': 2}
    slow = report()
    slow['cases'][0]['candidate_samples_us'] = [30.] * 10
    assert not m.adjudicate(slow, task, 'source')['solved']
    failed = report()
    failed['cases'][0]['correct'] = False
    assert m.adjudicate(failed, task, 'source')['reason'] == 'correctness or identity failure'


@pytest.mark.parametrize('change', ['hash', 'boundary', 'missing', 'duplicate', 'nan', 'bool', 'count'])
def test_invalid_terminal_evidence_never_becomes_a_win(tmp_path, change):
    task = manifest(tmp_path)['tasks'][0]
    data = report()
    if change == 'hash': data['candidate_sha256'] = 'other'
    if change == 'boundary': data['measurement_kind'] = 'unprofiled_latency'
    if change == 'missing': data['cases'].pop()
    if change == 'duplicate': data['cases'][1]['id'] = 'a'
    if change in ('nan', 'bool'): data['cases'][0]['candidate_samples_us'][0] = float('nan') if change == 'nan' else True
    if change == 'count': data['cases'][0]['candidate_samples_us'].pop()
    with pytest.raises(ValueError): m.adjudicate(data, task, 'source')


def test_real_candidate_sessions_are_paired_frozen_and_all_failures_retained(tmp_path):
    config = manifest(tmp_path)
    calls = []
    class Actor:
        def __init__(self, output, bundle, prompt, model, command, **kwargs):
            self.output, self.bundle, self.kwargs = output, bundle, kwargs
            calls.append((model, command, kwargs))
        def launch(self):
            workspace = self.output / 'workspace'
            workspace.mkdir(parents=True)
            assert not (self.bundle / 'secret.py').exists()
            (workspace / 'candidate.py').write_text('def run(inputs): return inputs["x"] * 2\n')
            return dict(returncode=0, timed_out=False)
    def evaluator(command, request, session):
        assert request['action'] == 'evaluate' and request['phase'] == 'terminal'
        assert Path(request['candidate_path']).read_text().endswith('* 2\n')
        return report(request['candidate_sha256'])
    output = tmp_path / 'run'
    results = m.run_suite(config, output, launcher=Actor, evaluator=evaluator)
    assert len(results) == 2 and all(row['result']['solved'] for row in results)
    assert {row['with_library'] for row in results} == {True, False}
    assert sorted(len(c[2]['guidance']) for c in calls) == [1, 2]
    assert all(c[2]['timeout'] == 60 and c[2]['max_tool_calls'] == 6 for c in calls)
    assert calls[0][0:2] == calls[1][0:2]
    (Path(config['common_skills']['cannbot']) / 'SKILL.md').write_text('changed after freeze')
    assert 'changed' not in (output / 'frozen/guidance/cannbot/SKILL.md').read_text()
    with pytest.raises(FileExistsError): m.run_suite(config, output, launcher=Actor, evaluator=evaluator)
    def broken(*args): raise RuntimeError('retained backend failure')
    failed = m.run_suite(config, tmp_path / 'failed', launcher=Actor, evaluator=broken)
    assert len(failed) == 2 and all(row['status'] == 'failed' for row in failed)
    def unavailable(command, request, session):
        return {**report(request['candidate_sha256']), 'outcome': 'infrastructure_failure'}
    invalid = m.run_suite(config, tmp_path / 'invalid', launcher=Actor, evaluator=unavailable)
    assert all(row['status'] == 'invalid-comparison' and row['result']['solved'] is None
               for row in invalid)


def test_terminal_evaluator_executes_exact_json_stdin_and_retains_diagnostics(tmp_path):
    script = tmp_path / 'evaluator.py'
    script.write_text('import json,sys\nrequest=json.load(sys.stdin)\n'
                      'print("backend diagnostic",file=sys.stderr)\nprint(json.dumps(request))\n')
    request = dict(action='evaluate', phase='terminal', candidate_sha256='exact-source')
    actual = m.evaluate([sys.executable, str(script)], request, tmp_path)
    assert actual == request
    assert 'backend diagnostic' in (tmp_path / 'evaluation.stderr').read_text()


def test_bad_public_bundle_is_rejected_before_model_output_reservation(tmp_path):
    config = manifest(tmp_path)
    bundle = Path(config['tasks'][0]['bundle'])
    (bundle / 'task.json').write_text(json.dumps({'public_files': ['../prompt.md']}))
    with pytest.raises(ValueError): m.freeze(config, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_real_bubblewrap_named_guidance_and_tool_budget(tmp_path):
    if not shutil.which('bwrap') or not shutil.which('codex'):
        pytest.skip('Bubblewrap/Codex runtime unavailable')
    config = manifest(tmp_path)
    auth = tmp_path / 'auth'; auth.mkdir(); (auth / 'auth.json').write_text('{}')
    helper = tmp_path / 'tool.py'
    helper.write_text('print("actual backend executed")')
    actor = m.CodingLauncher(tmp_path / 'actor', Path(config['tasks'][0]['bundle']),
        Path(config['tasks'][0]['prompt']), config['model'], [sys.executable, str(helper)],
        guidance={**config['common_skills'], 'library': config['library_skill']['path']},
        max_tool_calls=1, auth_home=auth)
    actor.prepare(); actor.preflight()
    server = socketserver.UnixStreamServer(str(actor.socket_dir / 'tool.sock'), ToolHandler)
    server.owner = actor
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        code = '''from pathlib import Path
import subprocess
assert not Path('/workspace/secret.py').exists()
for name in ('cannbot','library'):
 p=Path('/workspace/.agents/skills')/name/'SKILL.md'
 assert p.is_file()
 try:p.write_text('tampered')
 except OSError:pass
 else:raise AssertionError('guidance mutable')
Path('/workspace/candidate.py').write_text('new kernel source')
a=subprocess.run(['experiment-tool','evaluate'],capture_output=True,text=True)
b=subprocess.run(['experiment-tool','evaluate'],capture_output=True,text=True)
assert a.returncode==0 and 'actual backend executed' in a.stdout
assert b.returncode==75 and 'budget exhausted' in b.stderr
'''
        result = execute([*actor.base_command(), 'python3', '-c', code], actor.root, 30)
        assert result['returncode'] == 0, result
        assert actor.tool_calls == 1
        entries = [json.loads(x) for x in actor.journal.read_text().splitlines()]
        assert [e['dispatched'] for e in entries] == [True, False]
    finally:
        server.shutdown(); server.server_close(); thread.join(); actor.socket_temporary.cleanup()


def test_primary_gate_retains_all_case_regression_guard(tmp_path):
    task = manifest(tmp_path)['tasks'][0]
    task.update(performance_case_ids=['a'], minimum_geomean_speedup=3)
    data = report()
    result = m.adjudicate(data, task, 'source')
    assert result['solved'] and result['primary_geomean_speedup'] == 4
    assert result['all_case_geomean_speedup'] == pytest.approx(8 ** .5)
    data['cases'][1]['candidate_samples_us'] = [30.] * 10
    assert not m.adjudicate(data, task, 'source')['solved']


@pytest.mark.parametrize('patch', [dict(performance_case_ids=[]),
    dict(performance_case_ids=['a', 'a']), dict(performance_case_ids=['missing']),
    dict(minimum_geomean_speedup=float('nan')), dict(maximum_case_regression=float('inf'))])
def test_bad_gate_rejected_before_actor_dispatch(tmp_path, patch):
    config = manifest(tmp_path); config['tasks'][0].update(patch)
    with pytest.raises(ValueError): m.freeze(config, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_infrastructure_failure_is_not_candidate_failure(tmp_path):
    task = manifest(tmp_path)['tasks'][0]
    data = report(); data['outcome'] = 'infrastructure_failure'
    for row in data['cases']: row.update(correct=False, identity_verified=False)
    result = m.adjudicate(data, task, 'source')
    assert result['solved'] is None and not result['comparison_valid']
    data['outcome'] = 'candidate_failure'
    result = m.adjudicate(data, task, 'source')
    assert result['solved'] is False and result['comparison_valid']
    assert result['outcome'] == 'candidate_failure'
    data['cases'].pop()
    with pytest.raises(ValueError): m.adjudicate(data, task, 'source')
