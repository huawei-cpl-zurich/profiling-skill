from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time

import pytest

SPEC = importlib.util.spec_from_file_location("blind_launcher", Path(__file__).resolve().parents[1] / "scripts/blind_optimization_launcher.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def make_launcher(tmp_path, *, skill=False):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidate.py").write_text("value = 1\n")
    (bundle / "baseline.py").write_text("value = 1\n")
    (bundle / "secret-oracle.py").write_text("secret = 42\n")
    (bundle / "task.json").write_text(json.dumps({"public_files": ["candidate.py", "baseline.py", "task.json"]}))
    prompt = tmp_path / "prompt.md"
    prompt.write_text(module.PROMPT)
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    snapshot = None
    if skill:
        snapshot = tmp_path / "skill"
        snapshot.mkdir()
        (snapshot / "SKILL.md").write_text("---\nname: ascend-profiling\ndescription: profiling\n---\n")
    helper = tmp_path / "helper.py"
    helper.write_text("import sys\nprint(' '.join(sys.argv[1:]))\nprint('full diagnostic', file=sys.stderr)\nraise SystemExit(3 if 'fail' in sys.argv else 0)\n")
    return module.BlindLauncher(tmp_path / "evaluation", bundle, prompt,
                                {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
                                [sys.executable, str(helper)], skill=snapshot, auth_home=auth,
                                forbidden=(bundle,))


def test_preparation_exports_only_declared_files_and_retains_git(tmp_path):
    instance = make_launcher(tmp_path)
    instance.prepare()
    assert not (instance.workspace / "secret-oracle.py").exists()
    log = subprocess.check_output(["git", "-C", str(instance.workspace), "log", "--format=%s"], text=True)
    assert log.strip() == "Starting implementation"
    with pytest.raises(module.LaunchError, match="already exists"):
        instance.prepare()


def test_public_file_escape_rejected(tmp_path):
    instance = make_launcher(tmp_path)
    (instance.bundle / "task.json").write_text(json.dumps({"public_files": ["../prompt.md"]}))
    with pytest.raises(module.LaunchError, match="invalid public"):
        instance.prepare()
    assert not instance.root.exists()


@pytest.mark.parametrize("failure", ["manifest", "prompt", "skill"])
def test_invalid_input_does_not_reserve_output_and_valid_retry_succeeds(tmp_path, failure):
    instance = make_launcher(tmp_path, skill=failure == "skill")
    path = {"manifest": instance.bundle / "task.json", "prompt": instance.prompt_source,
            "skill": instance.skill_source / "SKILL.md" if instance.skill_source else None}[failure]
    original = path.read_text()
    if failure == "manifest":
        path.write_text("{not json")
    else:
        path.unlink()
    with pytest.raises(module.LaunchError):
        instance.prepare()
    assert not instance.root.exists()
    path.write_text(original)
    instance.prepare()
    assert (instance.workspace / "candidate.py").is_file()


def test_timeout_keeps_partial_diagnostics_and_stops_child_group(tmp_path):
    result = module.execute([sys.executable, "-u", "-c", "import time; print('compile details'); time.sleep(20)"], tmp_path, .1)
    assert result["timed_out"]
    assert "compile details" in result["stdout"]
    assert result["returncode"] != 0


@pytest.mark.parametrize("skill", [False, True])
def test_real_boundary_visibility_immutable_baseline_and_tool_diagnostics(tmp_path, skill):
    if not shutil.which("bwrap") or not shutil.which("codex"):
        pytest.skip("Bubblewrap and Codex runtime required")
    instance = make_launcher(tmp_path, skill=skill)
    instance.prepare()
    instance.preflight()
    server = socketserver.UnixStreamServer(str(instance.socket_dir / "tool.sock"), module.ToolHandler)
    server.owner = instance
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        code = """import os, subprocess
from pathlib import Path
assert not Path('/workspace/secret-oracle.py').exists()
assert Path('/workspace/.agents/skills/ascend-profiling/SKILL.md').exists() == bool(int(os.environ['EXPECTED_SKILL']))
try:
    Path('/workspace/baseline.py').write_text('tampered')
except OSError:
    pass
else:
    raise AssertionError('baseline writable')
Path('/workspace/candidate.py').write_text('value = 2')
result = subprocess.run(['experiment-tool', 'fail'], capture_output=True, text=True)
assert result.returncode == 3, result
assert 'full diagnostic' in result.stderr
assert '--workspace' in result.stdout
"""
        result = module.execute([*instance.base_command(), "--setenv", "EXPECTED_SKILL", str(int(skill)),
                                 "python3", "-c", code], instance.root, 30)
        assert result["returncode"] == 0, result
        assert (instance.workspace / "candidate.py").read_text() == "value = 2"
        assert json.loads(instance.journal.read_text())["returncode"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_existing_output_is_never_relaunched(tmp_path):
    instance = make_launcher(tmp_path)
    instance.root.mkdir()
    with pytest.raises(module.LaunchError, match="already exists"):
        instance.launch()


def test_output_inside_workspace_forbidden(tmp_path):
    instance = make_launcher(tmp_path)
    with pytest.raises(module.LaunchError, match="outside"):
        module.BlindLauncher(instance.bundle / "run", instance.bundle, instance.prompt_source,
                             instance.model, instance.tool_command, auth_home=instance.auth_home,
                             forbidden=(instance.bundle,))


def fake_model(instance, behavior):
    runtime = instance.root.parent / "fake-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    (runtime / "bin" / "node").touch()
    instance.codex = runtime / "bin" / "codex"
    instance.codex.write_text("#!/usr/bin/python3\n" + behavior)
    instance.codex.chmod(0o755)


@pytest.mark.parametrize("behavior, expected", [
    ("raise SystemExit(7)\n", 7),
    ("print('no commit')\n", 3),
    ("import time\nprint('partial', flush=True)\ntime.sleep(20)\n", 124),
])
def test_real_launch_receipt_rejects_failed_uncommitted_and_timed_out_agents(tmp_path, behavior, expected):
    instance = make_launcher(tmp_path)
    fake_model(instance, behavior)
    instance.timeout = .15 if expected == 124 else 5
    result = instance.launch()
    assert result["exit_code"] == expected
    assert json.loads((instance.root / "receipt.json").read_text())["exit_code"] == expected
    if expected == 124:
        assert "partial" in (instance.root / "agent.jsonl").read_text()


def test_long_evidence_path_real_tool_call_and_budget_visible_to_agent(tmp_path):
    instance = make_launcher(tmp_path)
    instance.root = tmp_path / ("long-" + "a" * 90) / ("nested-" + "b" * 90)
    fake_model(instance, """import json, subprocess
from datetime import datetime
from pathlib import Path
budget = json.loads(Path('/workspace/execution.json').read_text())
assert budget['wall_budget_seconds'] == 600
assert budget['finish_reserve_seconds'] == 120
assert (datetime.fromisoformat(budget['utc_deadline']) - datetime.fromisoformat(budget['utc_start'])).total_seconds() == 600
subprocess.run(['experiment-tool', 'help'], check=True)
Path('notes.md').write_text('Noted start ' + budget['utc_start'])
subprocess.run(['git', 'add', 'notes.md'], check=True)
subprocess.run(['git', 'commit', '-m', 'Record validated work early'], check=True)
""")
    result = instance.launch()
    assert result["exit_code"] == 0
    assert result["candidate_commits"] == 1
    assert len(str(instance.socket_dir / "tool.sock").encode()) < 108
    assert json.loads(instance.journal.read_text())["returncode"] == 0


def test_preflight_does_not_require_agent_commit(tmp_path):
    instance = make_launcher(tmp_path)
    result = instance.launch(preflight_only=True)
    assert result["exit_code"] == 0
    assert result["candidate_commits"] == 0


def test_wrapper_managed_flags_are_recoverable_without_dispatch_and_help_is_public(tmp_path):
    instance = make_launcher(tmp_path)
    helper = Path(instance.tool_command[1])
    helper.write_text("""import argparse
p = argparse.ArgumentParser()
p.add_argument('--workspace', required=True)
p.add_argument('operation')
a = p.parse_args()
assert a.operation == 'help'
print('experiment-tool check|timing|pipes|ir|sanitize')
""")
    fake_model(instance, """import subprocess
for args in (['--workspace', '/workspace', 'timing'], ['--bundle=other', 'timing'],
             ['--target', 'other', 'check'], ['--runtime=other', 'pipes']):
    p = subprocess.run(['experiment-tool', *args], capture_output=True, text=True)
    assert p.returncode == 2, p
    assert 'Managed configuration' in p.stderr and 'experiment-tool help' in p.stderr
for args in (['--help'], ['-h'], ['timing', '--help'], ['help']):
    p = subprocess.run(['experiment-tool', *args], capture_output=True, text=True)
    assert p.returncode == 0, p
    assert '--workspace' not in p.stdout and '--bundle' not in p.stdout
subprocess.run(['git', 'commit', '--allow-empty', '-m', 'Record recovered interface test'], check=True)
""")
    result = instance.launch()
    assert result["exit_code"] == 0
    rows = [json.loads(line) for line in instance.journal.read_text().splitlines()]
    assert len(rows) == 8
    assert all(row["dispatched"] is False and "command" not in row for row in rows[:4])
    assert all(row["command"][-1] == "help" for row in rows[4:])


def test_cli_propagates_receipt_exit_code(tmp_path):
    instance = make_launcher(tmp_path)
    fake_model(instance, "raise SystemExit(9)\n")
    model = tmp_path / "model.json"
    model.write_text(json.dumps(instance.model))
    tool = tmp_path / "tool.json"
    tool.write_text(json.dumps(instance.tool_command))
    environment = dict(os.environ, CODEX_HOME=str(instance.auth_home),
                       PATH=str(instance.codex.parent) + ":" + os.environ["PATH"])
    run = subprocess.run([sys.executable, str(Path(module.__file__)),
                          "--bundle", str(instance.bundle), "--prompt", str(instance.prompt_source),
                          "--output", str(instance.root), "--model-json", str(model),
                          "--tool-command-json", str(tool)], env=environment, capture_output=True, text=True)
    assert run.returncode == 9, run.stderr
    assert json.loads(run.stdout)["exit_code"] == 9


def test_live_output_survives_launcher_interruption_before_completion(tmp_path):
    stdout, stderr = tmp_path / "agent.jsonl", tmp_path / "agent.stderr"
    script = f"""import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('launcher', {str(Path(module.__file__))!r})
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.execute({[sys.executable, '-u', '-c', "import os, sys, time; print(os.getpid()); print('live error', file=sys.stderr); time.sleep(30)"]!r}, Path({str(tmp_path)!r}), 60,
                 (Path({str(stdout)!r}), Path({str(stderr)!r})))
"""
    process = subprocess.Popen([sys.executable, "-c", script])
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if stdout.exists() and stdout.read_text().strip() and stderr.exists() and "live error" in stderr.read_text():
                child_pid = int(stdout.read_text().strip())
                break
            time.sleep(.02)
        assert child_pid is not None
        assert process.poll() is None  # Evidence exists before communicate returns.
        process.terminate()
        process.wait(timeout=5)
        assert stdout.read_text().strip() == str(child_pid)
        assert "live error" in stderr.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid:
            os.killpg(child_pid, signal.SIGTERM)
