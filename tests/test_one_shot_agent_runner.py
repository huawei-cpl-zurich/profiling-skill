from __future__ import annotations

import hashlib
import importlib.util
import base64
import json
import os
import runpy
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("one_shot_agent_runner", ROOT / "scripts/one_shot_agent_runner.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(module)


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode() + b"\0")
            digest.update(f"{stat.S_IMODE(item.stat().st_mode) & 0o111:o}".encode() + b"\0")
            digest.update(item.read_bytes() + b"\0")
    return digest.hexdigest()


class Client:
    def __init__(self, result=None):
        self.requests = []
        self.result = result or {"status": "ok", "diagnostics": "passed", "passed": True}

    def run(self, request):
        self.requests.append(request)
        return self.result


def executable(path: Path, body="#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body); path.chmod(0o755)
    return path


def fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / "prompt.md"; prompt.write_text("write kernel\n")
    baseline = tmp_path / "source/baseline.py"; baseline.parent.mkdir()
    baseline.write_bytes((ROOT / "benchmarks/matmul/baseline.py").read_bytes())
    cases = tmp_path / "source/cases.jsonl"
    cases.write_text('{"name":"tail","kind":"correctness","m":3,"n":5,"k":7}\n')
    remote_runner = tmp_path / "source/runner.py"; remote_runner.write_text("# runner\n")
    skills = {}
    hashes = {}
    for name in ("ascend-profiling", "triton-guarded-kernel"):
        root = tmp_path / "skills" / name; root.mkdir(parents=True)
        (root / "SKILL.md").write_text(f"# {name}\n")
        skills[name] = str(root); hashes[name] = tree_digest(root)
    workspace = tmp_path / "workspace"; workspace.mkdir()
    model = {"name": "test-model", "reasoning_effort": "low"}
    request = {
        "protocol_version": 2, "operation": "one_shot", "cell_id": "wave-1-project-guarded",
        "wave": 1, "attempt": 1, "treatment": "project-guarded", "workspace": str(workspace),
        "prompt": {"encoding": "base64", "data": base64.b64encode(prompt.read_bytes()).decode(),
                   "sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()},
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "model": model, "model_sha256": hashlib.sha256(json.dumps(model, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "skills": list(skills),
        "skill_sha256": hashes, "controller_contract": {"billed_limit": 1, "command": module.CHECK},
    }
    runtime = tmp_path / "node"; executable(runtime / "bin/node"); codex = executable(runtime / "bin/codex")
    (runtime / "lib/node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"; auth.mkdir(); (auth / "auth.json").write_text("secret")
    bwrap = executable(tmp_path / "bwrap")
    assets = {"baseline": str(baseline), "case_spec": str(cases), "runner": str(remote_runner)}
    placements = {"project-guarded": [{"profile": "bz-a3-1", "device": 2},
                                       {"profile": "bz-a3-2", "device": 3}]}
    client = Client()
    runner = module.OneShotRunner(skill_sources=skills, assets=assets, placements=placements,
                                  client=client, codex=str(codex), bwrap=str(bwrap), auth_home=auth)
    return runner, request, client


def call_socket(socket_path: str, arguments: list[str]) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path); sock.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
        return json.loads(sock.makefile("rb").readline())


def test_real_runner_materializes_minimum_and_enforces_controller(monkeypatch, tmp_path: Path):
    runner, request, client = fixture(tmp_path)
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        socket_dir = argv[argv.index("/experiment-state") - 1]
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        assert json.loads(call_socket(str(Path(socket_dir) / "controller.sock"), ["help"])["stdout"])["billed"] is False
        assert json.loads(call_socket(str(Path(socket_dir) / "controller.sock"), ["budget"])["stdout"])["remaining"] == 1
        checked = call_socket(str(Path(socket_dir) / "controller.sock"), module.CHECK)
        assert checked["exit_code"] == 0
        assert call_socket(str(Path(socket_dir) / "controller.sock"), module.CHECK)["exit_code"] == 75
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", lambda argv, prompt, timeout:
                        fake_run(argv, input=prompt, timeout=timeout))
    result = runner.run(request)

    assert result["status"] == "ok", result
    assert result["controller_usage"]["billed"] == 1
    assert result["controller_usage"]["calls"] == [{"arguments": module.CHECK}]
    assert result["controller_usage"]["over_budget"] == 1
    assert client.requests[0]["cases"] == [1]
    assert client.requests[0]["device"] == 2 and client.requests[0]["profile"] == "bz-a3-1"
    assert client.requests[0]["logical_device"] == 0
    assert Path(client.requests[0]["candidate"]).parent.name == "development-check"
    workspace = Path(request["workspace"])
    assert {p.name for p in workspace.iterdir()} == {
        "baseline.py", "baseline.json", "cases.jsonl", "AGENTS.md", ".agents",
        "candidate.py", "candidate.manifest.json"
    }
    assert (workspace / "baseline.json").read_bytes() == (workspace / "cases.jsonl").read_bytes()
    assert len(runpy.run_path(str(workspace / "baseline.py"))["get_input_groups"]()) == 1
    argv = observed["argv"]
    joined = "\0".join(map(str, argv))
    for name in request["skills"]:
        assert f"/workspace/.agents/skills/{name}" in argv
        assert f"--ro-bind\0{runner.skill_sources[name]}\0/workspace/.agents/skills/{name}" in joined
    assert "/codex-home/skills" not in joined and "/codex-home/plugins" not in joined
    assert "reference_repos" not in joined and "controller.sock" not in json.dumps(client.requests)


def test_compile_diagnostic_is_bounded_and_returned_to_agent(tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    client = Client({"status": "compile_error", "diagnostics": "traceback:" + "x" * 70000})
    workspace = Path(request["workspace"])
    (workspace / "candidate.py").write_text("bad\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    socket_path = tmp_path / "ctl/controller.sock"
    with module.Controller(socket_path, client, request, {"profile": "bz-a3-2", "device": 9},
                           runner.assets, tmp_path / "snapshots") as ctl:
        response = call_socket(str(socket_path), module.CHECK)
    document = json.loads(response["stdout"])
    assert response["exit_code"] == 2 and document["status"] == "compile_error"
    assert document["diagnostics"].startswith("traceback:")
    assert document["diagnostics"].endswith("[diagnostic truncated]")
    assert len(document["diagnostics"]) < 66000 and ctl.used == 1


def test_standalone_sandbox_controller_client_needs_no_backend_module(tmp_path: Path):
    script = tmp_path / "runner.py"
    script.write_bytes((ROOT / "scripts/one_shot_agent_runner.py").read_bytes())
    socket_path = tmp_path / "ctl/controller.sock"
    runner, request, client = fixture(tmp_path / "inputs")
    with module.Controller(socket_path, client, request,
                           {"profile": "bz-a3-1", "device": 0}, runner.assets,
                           tmp_path / "snapshots"):
        run = subprocess.run([sys.executable, str(script), "controller-client", str(socket_path), "help"],
                             text=True, capture_output=True, check=False)
    assert run.returncode == 0
    assert json.loads(run.stdout)["operation"] == "help"


def test_frozen_hash_and_fresh_workspace_drift_fail_before_codex(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request["skill_sha256"]["ascend-profiling"] = "0" * 64
    monkeypatch.setattr(module, "_run_group", lambda *a, **k: pytest.fail("Codex started"))
    result = runner.run(request)
    assert result["status"] == "setup_error" and "frozen skill changed" in result["diagnostics"]

    runner, request, _ = fixture(tmp_path / "second")
    Path(request["workspace"], "leak").write_text("x")
    result = runner.run(request)
    assert result["status"] == "setup_error" and "fresh empty workspace" in result["diagnostics"]


def test_timeout_is_observed_and_preserves_controller_usage(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def timeout(command, prompt, timeout):
        raise subprocess.TimeoutExpired(command, timeout, stderr="agent exceeded turn")

    monkeypatch.setattr(module, "_run_group", timeout)
    result = runner.run(request, timeout=1)
    assert result["status"] == "timeout"
    assert "agent exceeded turn" in result["diagnostics"]
    assert result["controller_usage"] == {"billed": 0, "calls": []}


def test_model_service_failure_is_infrastructure(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    monkeypatch.setattr(module, "_run_group", lambda argv, prompt, timeout:
                        subprocess.CompletedProcess(argv, 1, "", "API error: service unavailable"))
    result = runner.run(request)
    assert result["status"] == "model_service_error"


def test_billed_controller_infrastructure_overrides_zero_codex_exit(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    runner.client = Client({"status": "infrastructure_error", "failure_type": "device_error",
                            "diagnostics": "device unavailable", "handle": None})

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 2
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "device_error"
    assert result["diagnostics"] == "device unavailable"
    assert result["handle"] is None and result["controller_result"]["failure_type"] == "device_error"
    assert result["controller_usage"]["billed"] == 1


def test_billed_controller_infrastructure_overrides_agent_timeout(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    runner.client = Client({"status": "infrastructure_error", "failure_type": "transport_error",
                            "diagnostics": "observer unavailable", "handle": "bz-a3-1:kept"})

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        call_socket(str(socket_dir / "controller.sock"), module.CHECK)
        raise subprocess.TimeoutExpired(argv, timeout, stderr="agent exceeded turn")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "transport_error"
    assert result["diagnostics"] == "observer unavailable"
    assert result["handle"] == "bz-a3-1:kept"
    assert result["controller_result"]["handle"] == "bz-a3-1:kept"
    assert result["agent_timeout"] == "agent exceeded turn"


def test_process_group_timeout_kills_descendants(tmp_path: Path):
    child = tmp_path / "child.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        module._run_group(["bash", "-c", f"sleep 30 & echo $! >{child}; wait"], "", 1)
    pid = int(child.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True,
                               capture_output=True, check=False).stdout.strip()
        assert not state or state.startswith("Z")


def test_in_band_prompt_hash_drift_fails_before_codex(tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request["prompt"]["data"] = base64.b64encode(b"different").decode()
    result = runner.run(request)
    assert result["status"] == "setup_error" and "prompt hash mismatch" in result["diagnostics"]
