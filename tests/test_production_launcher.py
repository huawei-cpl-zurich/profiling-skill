from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "production_launcher", ROOT / "scripts" / "production_launcher.py"
)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def launcher_fixture(tmp_path: Path, **kwargs):
    runtime = tmp_path / "node"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(runtime / "bin" / "codex", "#!/bin/sh\nexit 0\n")
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "real-auth"
    auth.mkdir()
    (auth / "auth.json").write_text("secret-token")
    bwrap = executable(tmp_path / "bwrap", "#!/bin/sh\nexit 0\n")
    return launcher.ProductionLauncher(
        ["controller"], codex=str(codex), bwrap=str(bwrap), auth_home=auth, **kwargs
    )


def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "cell"
    (root / "workspace" / ".agents" / "skills" / "ascend-profiling").mkdir(parents=True)
    (root / "PROMPT.md").write_text("identical prompt\n")
    return root


def test_bwrap_argv_mounts_only_workspace_minimal_state_and_readonly_auth(tmp_path: Path):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    argv = instance._base_command(root, root / "attempt")
    joined = "\0".join(map(str, argv))
    assert "--unshare-all" in argv and "--share-net" in argv
    assert f"{root.resolve()}/workspace\0/workspace" in joined
    assert f"{instance.auth_home}/auth.json\0/codex-home/auth.json" in joined
    assert "--ro-bind\0" + str(instance.auth_home / "auth.json") in joined
    assert str(instance.auth_home / "skills") not in joined
    # The request-gate file is mounted alone; its source repository is not mounted.
    assert f"{ROOT}\0" not in joined
    assert argv[argv.index("--setenv") + 1:argv.index("--setenv") + 3] == ["HOME", "/home/agent"]


def test_preflight_checks_auth_local_skills_and_forbidden_host_paths(tmp_path: Path, monkeypatch):
    forbidden = tmp_path / "orchestration"
    forbidden.mkdir()
    instance = launcher_fixture(tmp_path, forbidden_paths=[forbidden, Path.home() / ".codex" / "skills"])
    root = sandbox(tmp_path)
    base = instance._base_command(root, root / "attempt")
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    state = root / "controller-state"
    state.mkdir()
    instance._preflight(base, state)
    shell = observed["argv"][-1]
    assert "test -r /codex-home/auth.json" in shell
    assert "test -d /workspace/.agents/skills" in shell
    assert str(forbidden.resolve()) in shell
    assert "test ! -e /codex-home/skills" in shell


def test_preflight_failure_is_clear(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    base = instance._base_command(root, root / "attempt")
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "outside path visible"),
    )
    state = root / "controller-state"
    state.mkdir()
    with pytest.raises(launcher.LaunchError, match="outer isolation preflight failed"):
        instance._preflight(base, state)


def test_budget_controller_is_opaque_and_enforces_limit(tmp_path: Path):
    backend = executable(
        tmp_path / "backend",
        "#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n",
    )
    socket_path = tmp_path / "state" / "controller.sock"
    workspace = tmp_path / "cell" / "workspace"
    workspace.mkdir(parents=True)
    with launcher.BudgetController(socket_path, [str(backend)], 2, workspace,
                                   {"cell_id": "gdn-project-only", "device": 0}) as controller:
        def request(arguments):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
                return json.loads(client.makefile("rb").readline())

        assert json.loads(request(["one"])["stdout"]) == ["one"]
        assert json.loads(request(["two"])["stdout"]) == ["two"]
        exhausted = request(["three"])
        assert exhausted["exit_code"] == 75
        assert "budget exhausted" in exhausted["stderr"]
        assert controller.used == 2
    assert not socket_path.exists()


def test_controller_client_forwards_exit_and_streams(tmp_path: Path, capsys):
    backend = executable(tmp_path / "backend", "#!/bin/sh\necho output\necho diagnostic >&2\nexit 7\n")
    socket_path = tmp_path / "controller.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with launcher.BudgetController(socket_path, [str(backend)], 1, workspace,
                                   {"cell_id": "bsa-cannbot", "device": 1}):
        assert launcher.controller_client(socket_path, ["argument"]) == 7
    captured = capsys.readouterr()
    assert captured.out == "output\n"
    assert captured.err == "diagnostic\n"


def test_three_round_persistent_session_uses_fixed_model_and_prompt(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    calls = []

    class FakeBudget:
        used = 4
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-2:] and "sh" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        output = json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(launcher, "BudgetController", FakeBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, {"rounds": 3, "request_budget": 12})
    codex_calls = calls[1:]
    assert len(codex_calls) == 3
    assert codex_calls[0][1]["input"] == "identical prompt\n"
    assert all("gpt-5.6-sol" in call[0] for call in codex_calls)
    assert all('model_reasoning_effort="low"' in call[0] for call in codex_calls)
    assert all("thread-123" in call[0] for call in codex_calls[1:])
    assert all("--ignore-rules" not in call[0] for call in codex_calls)
    assert result["rounds_completed"] == 3 and result["controller_requests"] == 4


def test_controller_binds_cells_and_maps_candidate_workspace_paths(tmp_path: Path):
    backend = executable(
        tmp_path / "backend",
        "#!/usr/bin/python3\nimport json,os,sys\nprint(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd()}))\n",
    )
    for cell_id, device in (("gdn-project-only", 0), ("bsa-cannbot", 1)):
        workspace = tmp_path / cell_id / "workspace"
        workspace.mkdir(parents=True)
        socket_path = tmp_path / cell_id / "controller.sock"
        command = [str(backend), "--cell", "{cell_id}", "--device", "{device}"]
        with launcher.BudgetController(socket_path, command, 2, workspace,
                                       {"cell_id": cell_id, "device": device}):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                request = {"arguments": ["profile", "--candidate=/workspace/kernel.py"]}
                client.sendall(json.dumps(request).encode() + b"\n")
                response = json.loads(client.makefile("rb").readline())
        observed = json.loads(response["stdout"])
        assert observed["argv"] == ["--cell", cell_id, "--device", str(device),
                                     "profile", f"--candidate={workspace}/kernel.py"]
        assert observed["cwd"] == str(workspace)


def test_controller_rejects_nonworkspace_absolute_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    controller = launcher.BudgetController(tmp_path / "sock", ["true"], 1, workspace,
                                           {"cell_id": "gdn", "device": 0})
    with pytest.raises(ValueError, match="outside /workspace"):
        controller.map_arguments(["--candidate=/tmp/leak.py"])


def test_launch_rejects_noncanonical_limits_before_codex(tmp_path: Path):
    instance = launcher_fixture(tmp_path)
    with pytest.raises(launcher.LaunchError, match="three rounds"):
        instance.launch(sandbox(tmp_path), {"rounds": 2, "request_budget": 12})


def test_dry_run_builds_plan_without_starting_bwrap_or_reading_auth(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path, dry_run=True)
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *a, **k: pytest.fail("dry-run must not start a process"),
    )
    result = instance.launch(sandbox(tmp_path), {"rounds": 3, "request_budget": 12})
    assert result["dry_run"] is True
    assert result["rounds"] == 3 and result["request_budget"] == 12
    assert "secret-token" not in json.dumps(result)


def test_retry_uses_fresh_attempt_state(tmp_path: Path):
    instance = launcher_fixture(tmp_path, dry_run=True)
    root = sandbox(tmp_path)
    first = instance.launch(root, {"rounds": 3, "request_budget": 12})
    second = instance.launch(root, {"rounds": 3, "request_budget": 12})
    assert first["attempt_id"] != second["attempt_id"]
    assert (root / ".launcher-attempts" / first["attempt_id"] / "codex-state").is_dir()
    assert (root / ".launcher-attempts" / second["attempt_id"] / "codex-state").is_dir()


def test_nonzero_codex_turn_is_retained_as_infrastructure_error(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)

    class FakeBudget:
        used = 1
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    calls = 0
    def fake_run(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:  # isolation preflight
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 17, "partial output", "transport lost")

    monkeypatch.setattr(launcher, "BudgetController", FakeBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, {"cell_id": "gdn-project-only", "device": 0,
                                    "rounds": 3, "request_budget": 12})
    assert result["status"] == "infrastructure_error"
    assert result["rounds_completed"] == 0
    assert result["turns"] == [{"round": 1, "exit_code": 17,
                                "stdout": "partial output", "stderr": "transport lost"}]


def test_missing_auth_or_bwrap_fails_without_reading_credentials(tmp_path: Path):
    runtime = tmp_path / "node"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(runtime / "bin" / "codex", "#!/bin/sh\nexit 0\n")
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"
    auth.mkdir()
    with pytest.raises(launcher.LaunchError, match="Bubblewrap"):
        launcher.ProductionLauncher(["ctl"], codex=str(codex), bwrap=str(tmp_path / "missing"), auth_home=auth)
    bwrap = executable(tmp_path / "bwrap", "#!/bin/sh\nexit 0\n")
    with pytest.raises(launcher.LaunchError, match="authenticated"):
        launcher.ProductionLauncher(["ctl"], codex=str(codex), bwrap=str(bwrap), auth_home=auth)


def test_real_bwrap_with_functional_fake_codex_runs_persistent_rounds(tmp_path: Path):
    bwrap = __import__("shutil").which("bwrap")
    if not bwrap:
        pytest.skip("Bubblewrap is unavailable")
    runtime = tmp_path / "runtime"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(
        runtime / "bin" / "codex",
        """#!/bin/sh
test "$CODEX_HOME" = /codex-home || exit 21
test -d /workspace/.agents/skills/ascend-profiling || exit 22
test ! -e /codex-home/skills || exit 23
printf '%s\n' '{"type":"thread.started","thread_id":"fake-thread"}'
""",
    )
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}\n")
    root = sandbox(tmp_path)
    instance = launcher.ProductionLauncher(
        ["/bin/true"], codex=str(codex), bwrap=bwrap, auth_home=auth,
        forbidden_paths=[ROOT, Path.home() / ".codex" / "skills"],
    )
    try:
        result = instance.launch(root, {"cell_id": "fake-project-only", "device": 0,
                                        "rounds": 3, "request_budget": 12})
    except launcher.LaunchError as error:
        if "outer isolation preflight failed" in str(error) and "Operation not permitted" in str(error):
            pytest.skip("user namespaces are disabled")
        raise
    assert result["exit_code"] == 0
    assert result["session_id"] == "fake-thread"
    assert result["rounds_completed"] == 3
