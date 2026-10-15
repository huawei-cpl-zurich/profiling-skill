from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import socketserver
import subprocess
import sys
import threading

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
