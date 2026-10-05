from __future__ import annotations

import hashlib
import importlib.util
import base64
import json
import os
import runpy
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnostic_campaign as diagnostic
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


def docker_fixture(tmp_path: Path, monkeypatch, image_id="sha256:frozen",
                   inspected_image_id=None):
    runner, request, client = fixture(tmp_path)
    docker = executable(tmp_path / "docker")

    def inspect(argv, **kwargs):
        assert argv[1:3] == ["image", "inspect"]
        return subprocess.CompletedProcess(argv, 0, (inspected_image_id or image_id) + "\n", "")

    monkeypatch.setattr(module.subprocess, "run", inspect)
    configured = module.OneShotRunner(
        skill_sources={name: str(path) for name, path in runner.skill_sources.items()},
        assets=runner.assets, placements=runner.placements, client=client,
        codex=str(runner.codex), auth_home=runner.auth_home,
        sandbox_backend="docker", docker=str(docker), docker_image="python:3.10",
        docker_image_id=image_id,
    )
    return configured, request, client


def call_socket(socket_path: str, arguments: list[str]) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path); sock.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
        return json.loads(sock.makefile("rb").readline())


def test_real_runner_materializes_minimum_and_enforces_controller(monkeypatch, tmp_path: Path):
    runner, request, client = fixture(tmp_path)
    starter = tmp_path / "starter.py"; starter.write_text("starter\n")
    manifest = tmp_path / "candidate.manifest.json"; manifest.write_text('{"kernel_name":"starter"}\n')
    runner.assets.update(starter=str(starter), candidate_manifest=str(manifest))
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        shim = Path(argv[argv.index("/experiment/controller-client.py") - 1])
        observed["shim"] = shim.read_text()
        socket_dir = argv[argv.index("/experiment-state") - 1]
        workspace = Path(request["workspace"])
        assert (workspace / "candidate.py").read_text() == "starter\n"
        assert json.loads((workspace / "candidate.manifest.json").read_text()) == {
            "kernel_name": "starter"
        }
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        assert json.loads(call_socket(str(Path(socket_dir) / "controller.sock"), ["help"])["stdout"])["billed"] is False
        assert json.loads(call_socket(str(Path(socket_dir) / "controller.sock"), ["budget"])["stdout"])["remaining"] == 1
        checked = call_socket(str(Path(socket_dir) / "controller.sock"), module.CHECK)
        assert checked["exit_code"] == 0
        snapshot = Path(client.requests[0]["candidate"])
        observed["snapshot"] = snapshot
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
        assert str(snapshot.parent.parent) not in map(str, argv)
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
    assert "--clearenv" in argv
    assert "/experiment/runner.py" not in joined
    assert "class Controller" not in observed["shim"] and "socket.AF_UNIX" in observed["shim"]
    assert "reference_repos" not in joined and "controller.sock" not in json.dumps(client.requests)
    assert str(observed["snapshot"].parent.parent) not in joined
    assert not observed["snapshot"].exists()


@pytest.mark.parametrize("benchmark", ["bsa", "gdn"])
def test_two_shot_runner_resumes_session_and_requires_modified_candidate(
        monkeypatch, tmp_path: Path, benchmark: str):
    runner, request, client = fixture(tmp_path)
    metadata = tmp_path / "source/baseline.json"
    metadata.write_text('{"case": 47}\n')
    runner.assets["supplementary"] = {"baseline.json": str(metadata)}
    request["operation"] = "two_shot"
    request["controller_contract"] = {
        "billed_limit": 2,
        "commands": [module.check_command(1), module.check_command(2)],
    }
    request["cases"] = [0, 1, 2, 3, 4, 5, 6]
    request["benchmark"] = benchmark
    calls = []

    def fake_group(argv, prompt, timeout):
        calls.append((argv, prompt))
        workspace = Path(request["workspace"])
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        round_number = len(calls)
        (workspace / "candidate.py").write_text(f"candidate {round_number}\n")
        (workspace / "candidate.manifest.json").write_text(
            json.dumps({"round": round_number}) + "\n")
        checked = call_socket(
            str(socket_dir / "controller.sock"), module.check_command(round_number))
        assert checked["exit_code"] == 0
        output = ('{"type":"thread.started","thread_id":"thread-1"}\n'
                  if round_number == 1 else '{"type":"turn.completed"}\n')
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)

    assert result["status"] == "ok", result
    assert result["rounds_completed"] == 2
    assert result["controller_usage"]["calls"] == [
        {"arguments": module.check_command(1)},
        {"arguments": module.check_command(2)},
    ]
    assert result["candidate_sha256"]["1"] != result["candidate_sha256"]["2"]
    resume_index = calls[1][0].index("resume")
    assert calls[1][0][resume_index - 1:resume_index + 2] == ["exec", "resume", "--json"]
    assert "thread-1" in calls[1][0]
    assert "modify" in calls[1][1].lower()
    assert [sent["cases"] for sent in client.requests] == [request["cases"], request["cases"]]
    assert [sent["benchmark"] for sent in client.requests] == [benchmark, benchmark]
    assert [sent["supplementary_assets"] for sent in client.requests] == [
        {"baseline.json": str(metadata)}, {"baseline.json": str(metadata)}]
    assert (Path(request["workspace"]) / "baseline.json").read_bytes() == metadata.read_bytes()
    assert [item["status"] for item in result["controller_results"]] == ["ok", "ok"]


@pytest.mark.parametrize("destination_kind", ["traversal", "absolute", "reserved"])
def test_runner_rejects_unsafe_supplementary_destination_before_copy(
        monkeypatch, tmp_path: Path, destination_kind: str):
    runner, request, _client = fixture(tmp_path)
    workspace = Path(request["workspace"])
    destinations = {
        "traversal": "../escaped.py",
        "absolute": str(tmp_path / "absolute-escape.py"),
        "reserved": "candidate.py",
    }
    destination = destinations[destination_kind]
    runner.assets["supplementary"] = {
        destination: runner.assets["case_spec"]}
    monkeypatch.setattr(module, "_run_group", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError("agent must not launch")))

    result = runner.run(request)

    assert result["status"] == "setup_error"
    assert "supplementary asset" in result["diagnostics"]
    assert list(workspace.iterdir()) == []
    assert not (workspace.parent / "escaped.py").exists()
    assert not (tmp_path / "absolute-escape.py").exists()


def test_two_shot_runner_counts_unchanged_second_submission(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request["operation"] = "two_shot"
    request["cases"] = [1]
    request["benchmark"] = "matmul"
    request["controller_contract"] = {
        "billed_limit": 2,
        "commands": [module.check_command(1), module.check_command(2)],
    }

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("same candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        round_number = 1 if "resume" not in argv else 2
        response = call_socket(
            str(socket_dir / "controller.sock"), module.check_command(round_number))
        if round_number == 1:
            assert response["exit_code"] == 0
            return subprocess.CompletedProcess(
                argv, 0, '{"type":"thread.started","thread_id":"thread-1"}\n', "")
        assert response["exit_code"] == 2
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)

    assert result["status"] == "submission_error"
    assert "must modify candidate.py" in result["diagnostics"]
    assert result["controller_usage"]["billed"] == 2


def test_two_shot_stops_after_round_one_infrastructure(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request.update(operation="two_shot", benchmark="matmul", cases=[0, 1])
    request["controller_contract"] = {
        "billed_limit": 2,
        "commands": [module.check_command(1), module.check_command(2)],
    }
    runner.client = Client({"status": "infrastructure_error", "failure_type": "device_error",
                            "diagnostics": "device unavailable", "handle": None})
    calls = []

    def fake_group(argv, prompt, timeout):
        calls.append(argv)
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("round one\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"),
                           module.check_command(1))["exit_code"] == 2
        return subprocess.CompletedProcess(
            argv, 0, '{"type":"thread.started","thread_id":"thread-1"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)

    assert len(calls) == 1
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "device_error"
    assert result["controller_results"] == [{
        "status": "infrastructure_error", "failure_type": "device_error",
        "diagnostics": "device unavailable", "handle": None,
    }]


@pytest.mark.parametrize("name", ["candidate.py", "candidate.manifest.json"])
def test_post_check_symlink_is_counted_and_canonical_output_restored(
        monkeypatch, tmp_path: Path, name: str):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text('{"checked":true}\n')
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 0
        outside = tmp_path / "outside"; outside.write_text("outside\n")
        (workspace / name).unlink(); (workspace / name).symlink_to(outside)
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    workspace = Path(request["workspace"])
    assert result["status"] == "submission_error"
    assert not (workspace / name).is_symlink()
    assert (workspace / "candidate.py").read_text() == "checked candidate\n"
    assert json.loads((workspace / "candidate.manifest.json").read_text()) == {"checked": True}


def test_post_check_regular_mutation_restores_checked_bytes(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text('{"checked":true}\n')
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 0
        (workspace / "candidate.py").write_text("unchecked mutation\n")
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "ok"
    assert (Path(request["workspace"]) / "candidate.py").read_text() == "checked candidate\n"


@pytest.mark.parametrize("remote_status", ["compile_error", "runtime_error", "correctness_error"])
def test_failed_check_still_restores_checked_bytes(monkeypatch, tmp_path: Path, remote_status: str):
    runner, request, _ = fixture(tmp_path)
    runner.client = Client({"status": remote_status, "diagnostics": "failed"})

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 2
        (workspace / "candidate.py").write_text("unchecked mutation\n")
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    runner.run(request)
    assert (Path(request["workspace"]) / "candidate.py").read_text() == "checked candidate\n"


@pytest.mark.parametrize("name", ["candidate.py", "candidate.manifest.json"])
def test_post_check_missing_output_is_counted_and_restored(monkeypatch, tmp_path: Path, name: str):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text('{"checked":true}\n')
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 0
        (workspace / name).unlink()
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    workspace = Path(request["workspace"])
    assert result["status"] == "submission_error"
    assert diagnostic.classify(result, None, workspace) == ("submission_error", "counted")
    assert (workspace / "candidate.py").read_text() == "checked candidate\n"
    assert json.loads((workspace / "candidate.manifest.json").read_text()) == {"checked": True}


def test_post_check_mutation_is_restored_before_agent_timeout(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text('{"checked":true}\n')
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 0
        (workspace / "candidate.py").write_text("unchecked mutation\n")
        raise subprocess.TimeoutExpired(argv, timeout, stderr="agent exceeded turn")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "timeout"
    assert (Path(request["workspace"]) / "candidate.py").read_text() == "checked candidate\n"


def test_post_check_mutation_is_restored_before_nonzero_exit(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("checked candidate\n")
        (workspace / "candidate.manifest.json").write_text('{"checked":true}\n')
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 0
        (workspace / "candidate.py").write_text("unchecked mutation\n")
        return subprocess.CompletedProcess(argv, 1, "", "protocol failed")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "protocol_error"
    assert (Path(request["workspace"]) / "candidate.py").read_text() == "checked candidate\n"


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


@pytest.mark.parametrize(("operation", "round_two_visible"), [
    ("one_shot", False), ("two_shot", True),
])
def test_controller_help_matches_operation(tmp_path: Path, operation: str,
                                           round_two_visible: bool):
    runner, request, client = fixture(tmp_path)
    request["operation"] = operation
    socket_path = tmp_path / "ctl/controller.sock"
    with module.Controller(socket_path, client, request,
                           {"profile": "bz-a3-1", "device": 0}, runner.assets,
                           tmp_path / "snapshots"):
        response = call_socket(str(socket_path), ["help"])
    usage = json.loads(response["stdout"])["usage"]
    assert ("--round 2" in usage) is round_two_visible


def test_late_check_is_billed_without_starting_remote_work(tmp_path: Path):
    runner, request, client = fixture(tmp_path)
    workspace = Path(request["workspace"])
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    socket_path = tmp_path / "ctl/controller.sock"
    with module.Controller(socket_path, client, request, {"profile": "bz-a3-1", "device": 0},
                           runner.assets, tmp_path / "snapshots",
                           deadline=time.monotonic() + 19) as controller:
        response = call_socket(str(socket_path), module.CHECK)
    result = json.loads(response["stdout"])
    assert response["exit_code"] == 2 and result["status"] == "submission_error"
    assert controller.used == 1 and not client.requests


def test_check_dispatches_at_client_minimum_timeout(tmp_path: Path):
    runner, request, client = fixture(tmp_path)
    workspace = Path(request["workspace"])
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    socket_path = tmp_path / "ctl/controller.sock"
    with module.Controller(socket_path, client, request, {"profile": "bz-a3-1", "device": 0},
                           runner.assets, tmp_path / "snapshots",
                           deadline=time.monotonic() + 47) as controller:
        response = call_socket(str(socket_path), module.CHECK)
    assert response["exit_code"] == 0 and controller.used == 1
    assert client.requests[0]["timeout"] == 26


@pytest.mark.parametrize("name", ["candidate.py", "candidate.manifest.json"])
@pytest.mark.parametrize("absolute", [False, True])
def test_controller_rejects_submission_symlink_outside_workspace(
        tmp_path: Path, name: str, absolute: bool):
    runner, request, client = fixture(tmp_path)
    workspace = Path(request["workspace"])
    outside = tmp_path / "outside"; outside.write_text("host data\n")
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    (workspace / name).unlink()
    (workspace / name).symlink_to(outside if absolute else "../outside")
    socket_path = tmp_path / "ctl/controller.sock"
    with module.Controller(socket_path, client, request, {"profile": "bz-a3-1", "device": 0},
                           runner.assets, tmp_path / "snapshots") as controller:
        response = call_socket(str(socket_path), module.CHECK)
    result = json.loads(response["stdout"])
    assert response["exit_code"] == 2 and result["status"] == "submission_error"
    assert controller.used == 1 and controller.last_result == result
    assert not client.requests
    assert not (tmp_path / "snapshots/development-check" / name).exists()


def test_symlink_submission_is_counted_when_codex_exits_zero(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        outside = tmp_path / "outside.py"; outside.write_text("host data\n")
        (workspace / "candidate.py").symlink_to(outside)
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 2
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "submission_error"
    assert result["controller_usage"]["billed"] == 1
    assert result["controller_result"]["status"] == "submission_error"


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


def test_billed_snapshot_exception_overrides_zero_codex_exit(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fail_copy(*_):
        raise OSError("snapshot storage unavailable")

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        (workspace / "candidate.py").write_text("candidate\n")
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 74
        return subprocess.CompletedProcess(argv, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(module, "_copy_regular", fail_copy)
    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "controller_error"
    assert "snapshot storage unavailable" in result["diagnostics"]
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


def test_billed_submission_overrides_agent_timeout(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def fake_group(argv, prompt, timeout):
        workspace = Path(request["workspace"])
        outside = tmp_path / "outside.py"; outside.write_text("host data\n")
        (workspace / "candidate.py").symlink_to(outside)
        (workspace / "candidate.manifest.json").write_text("{}\n")
        socket_dir = Path(argv[argv.index("/experiment-state") - 1])
        assert call_socket(str(socket_dir / "controller.sock"), module.CHECK)["exit_code"] == 2
        raise subprocess.TimeoutExpired(argv, timeout, stderr="agent exceeded turn")

    monkeypatch.setattr(module, "_run_group", fake_group)
    result = runner.run(request)
    assert result["status"] == "submission_error"
    assert result["controller_usage"]["billed"] == 1
    assert result["controller_result"]["status"] == "submission_error"
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


def test_process_output_is_bounded_while_captured():
    run = module._run_group(
        [sys.executable, "-c", "import sys; print('x' * 1000000); print('y' * 1000000, file=sys.stderr)"],
        "", 10)
    assert len(run.stdout) < 66000 and "diagnostic truncated" in run.stdout
    assert len(run.stderr) < 66000 and "diagnostic truncated" in run.stderr


def test_in_band_prompt_hash_drift_fails_before_codex(tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request["prompt"]["data"] = base64.b64encode(b"different").decode()
    result = runner.run(request)
    assert result["status"] == "setup_error" and "prompt hash mismatch" in result["diagnostics"]


def _docker_inputs(runner, request, tmp_path: Path):
    socket_dir = tmp_path / "socket"; socket_dir.mkdir()
    shim = tmp_path / "controller-client.py"; shim.write_text("# opaque client\n")
    prompt = tmp_path / "PROMPT.md"; prompt.write_text("write kernel\n")
    workspace = Path(request["workspace"])
    skill_root = workspace / ".agents/skills"; skill_root.mkdir(parents=True)
    for name in request["skills"]: (skill_root / name).mkdir()
    return workspace, prompt, socket_dir, shim


def test_docker_argv_is_an_explicit_minimal_allowlist(monkeypatch, tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    workspace, prompt, socket_dir, shim = _docker_inputs(runner, request, tmp_path)
    state = tmp_path / "state"; state.mkdir()
    argv = runner._docker_command(workspace, prompt, socket_dir, shim, state,
                                  request, "frozen-cell")
    joined = "\0".join(map(str, argv))
    assert argv[:3] == [str(runner.docker), "run", "--rm"]
    for item in ("--interactive", "--read-only", "--cap-drop", "ALL", "--pids-limit",
                 "512", "--memory", "8g", "no-new-privileges"):
        assert item in argv
    assert f"type=bind,src={workspace.resolve()},dst=/workspace" in argv
    assert f"type=bind,src={prompt.resolve()},dst=/experiment/PROMPT.md,readonly" in argv
    assert f"type=bind,src={runner.auth_home / 'auth.json'},dst=/codex-home/auth.json,readonly" in argv
    assert f"type=bind,src={shim.resolve()},dst=/experiment/controller-client.py,readonly" in argv
    for name in request["skills"]:
        assert (f"type=bind,src={runner.skill_sources[name]},"
                f"dst=/workspace/.agents/skills/{name},readonly") in argv
    assert runner.docker_image_id == argv[-1]
    for forbidden in ("/var/run/docker.sock", f"src={ROOT},", str(Path.home() / ".agents"),
                      str(Path.home() / ".ssh"), "reference_repos", "reference-library",
                      "/experiment/runner.py"):
        assert forbidden not in joined
    read_write = [value for value in argv if value.startswith("type=bind")
                  and not value.endswith(",readonly")]
    assert read_write == [f"type=bind,src={workspace.resolve()},dst=/workspace",
                          f"type=bind,src={state.resolve()},dst=/codex-home",
                          f"type=bind,src={socket_dir.resolve()},dst=/experiment-state"]


def test_two_docker_turns_share_writable_codex_state(monkeypatch, tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    request["operation"] = "two_shot"
    workspace, prompt, socket_dir, shim = _docker_inputs(runner, request, tmp_path)
    state = tmp_path / "cell-codex-state"
    state.mkdir()

    commands = [runner._docker_command(
        workspace, prompt, socket_dir, shim, state, request, "frozen-cell")
        for _ in range(2)]

    state_mount = f"type=bind,src={state.resolve()},dst=/codex-home"
    auth_mount = (f"type=bind,src={runner.auth_home / 'auth.json'},"
                  "dst=/codex-home/auth.json,readonly")
    assert all(state_mount in command and auth_mount in command for command in commands)
    assert commands[0].count(state_mount) == commands[1].count(state_mount) == 1
    assert all("/codex-home:rw" not in item for command in commands for item in command)


def test_docker_rejects_drifted_image_and_symlink_mount(monkeypatch, tmp_path: Path):
    with pytest.raises(module.RunnerError, match="frozen image ID"):
        docker_fixture(tmp_path / "drift", monkeypatch, image_id="sha256:expected",
                       inspected_image_id="sha256:actual")
    runner, request, _ = docker_fixture(tmp_path / "link", monkeypatch)
    workspace, prompt, socket_dir, shim = _docker_inputs(runner, request, tmp_path / "link")
    linked = prompt.with_name("linked-prompt.md"); linked.symlink_to(prompt)
    with pytest.raises(module.RunnerError, match="must not be a symlink"):
        state = tmp_path / "link/state"; state.mkdir()
        runner._docker_command(workspace, linked, socket_dir, shim, state, request, "cell")


@pytest.mark.parametrize("cancelled", [False, True])
def test_docker_timeout_or_cancel_force_removes_container(monkeypatch, tmp_path: Path,
                                                           cancelled: bool):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    removed = []
    failure = KeyboardInterrupt() if cancelled else subprocess.TimeoutExpired(["docker"], 1)
    monkeypatch.setattr(module, "_run_group",
                        lambda argv, prompt, timeout: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(module.subprocess, "run",
        lambda argv, **kwargs: (removed.append(argv) or subprocess.CompletedProcess(argv, 0, "", "")))
    if cancelled:
        with pytest.raises(KeyboardInterrupt): runner.run(request, timeout=1)
    else:
        result = runner.run(request, timeout=1)
        assert result["status"] == "timeout" and result["sandbox"]["backend"] == "docker"
    assert len(removed) == 1 and removed[0][1:3] == ["rm", "-f"]
    assert removed[0][3].startswith("triton-one-shot-")


def test_docker_sigterm_routes_through_cleanup_and_restores_handlers(monkeypatch,
                                                                    tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    handlers = {}
    removed = []

    def install(watched, handler):
        previous = handlers.get(watched, signal.SIG_DFL)
        handlers[watched] = handler
        return previous

    def interrupt(_argv, _prompt, _timeout):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(module.signal, "signal", install)
    monkeypatch.setattr(module, "_run_group", interrupt)
    monkeypatch.setattr(module.subprocess, "run",
        lambda argv, **kwargs: (removed.append(argv) or subprocess.CompletedProcess(argv, 0, "", "")))
    with pytest.raises(KeyboardInterrupt):
        runner.run(request)
    assert handlers[signal.SIGTERM] == signal.SIG_DFL
    assert handlers[signal.SIGHUP] == signal.SIG_DFL
    assert len(removed) == 1 and removed[0][1:3] == ["rm", "-f"]


def test_real_docker_probe_has_controller_but_no_host_privilege(tmp_path: Path):
    docker = shutil.which("docker")
    if not docker: pytest.skip("Docker is unavailable")
    inspected = subprocess.run([docker, "image", "inspect", "python:3.10", "--format", "{{.Id}}"],
                               text=True, capture_output=True, check=False)
    if inspected.returncode: pytest.skip("the pinned local Python image is unavailable")
    runner, request, client = fixture(tmp_path)
    executable(runner._runtime() / "bin/codex", "#!/bin/sh\n"
        "test \"$CODEX_HOME\" = /codex-home || exit 21\n"
        "test ! -e /var/run/docker.sock || exit 22\n"
        "test ! -e /root/.ssh || exit 23\n"
        "sh -c \"$EXPERIMENT_CONTROLLER help\" >/tmp/help.json || exit 24\n"
        "grep -q '\"operation\": \"help\"' /tmp/help.json || exit 25\n"
        "! touch /workspace/.agents/skills/ascend-profiling/WRITE-LEAK 2>/dev/null || exit 26\n"
        "touch /workspace/workspace-is-writable || exit 27\n"
        "sh -c \"$EXPERIMENT_CONTROLLER profile\" >/tmp/profile.out 2>/tmp/profile.err; test $? -eq 4 || exit 28\n"
        "printf '%s\\n' '{\"type\":\"turn.completed\"}'\n")
    isolated = module.OneShotRunner(
        skill_sources={name: str(path) for name, path in runner.skill_sources.items()},
        assets=runner.assets, placements=runner.placements, client=client,
        codex=str(runner.codex), auth_home=runner.auth_home,
        sandbox_backend="docker", docker=docker, docker_image="python:3.10",
        docker_image_id=inspected.stdout.strip())
    result = isolated.run(request, timeout=30)
    assert result["status"] == "ok", result
    assert result["controller_usage"]["billed"] == 0
    assert result["controller_usage"]["invalid"] == 1
    assert result["sandbox"]["image_id"] == inspected.stdout.strip()
    assert Path(request["workspace"], "workspace-is-writable").is_file()
    assert not Path(runner.skill_sources["ascend-profiling"], "WRITE-LEAK").exists()
