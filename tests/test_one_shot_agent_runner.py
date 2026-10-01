from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import stat
import subprocess
import sys
import shutil
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
    baseline = tmp_path / "source/baseline.py"; baseline.parent.mkdir(); baseline.write_text("reference = True\n")
    cases = tmp_path / "source/cases.jsonl"; cases.write_text('{"case":1}\n')
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
        "protocol_version": 1, "operation": "one_shot", "cell_id": "wave-1-project-guarded",
        "wave": 1, "attempt": 1, "treatment": "project-guarded", "workspace": str(workspace),
        "prompt": str(prompt), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
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
        return subprocess.CompletedProcess(
            argv, 0, (inspected_image_id or image_id) + "\n", "",
        )

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

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = runner.run(request)

    assert result["status"] == "ok", result
    assert result["controller_usage"]["billed"] == 1
    assert result["controller_usage"]["calls"] == [{"arguments": module.CHECK}]
    assert result["controller_usage"]["over_budget"] == 1
    assert result["controller_evidence"] == [{
        "status": "ok", "diagnostics": "passed", "passed": True,
    }]
    assert client.requests[0]["cases"] == [1]
    assert client.requests[0]["device"] == 2 and client.requests[0]["profile"] == "bz-a3-1"
    assert client.requests[0]["logical_device"] == 0
    workspace = Path(request["workspace"])
    assert {p.name for p in workspace.iterdir()} == {
        "baseline.py", "cases.jsonl", "AGENTS.md", ".agents", "candidate.py", "candidate.manifest.json"
    }
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
    with module.Controller(socket_path, client, request, {"profile": "bz-a3-2", "device": 9}, runner.assets) as ctl:
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
                           {"profile": "bz-a3-1", "device": 0}, runner.assets):
        run = subprocess.run([sys.executable, str(script), "controller-client", str(socket_path), "help"],
                             text=True, capture_output=True, check=False)
    assert run.returncode == 0
    assert json.loads(run.stdout)["operation"] == "help"


def test_frozen_hash_and_fresh_workspace_drift_fail_before_codex(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    request["skill_sha256"]["ascend-profiling"] = "0" * 64
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Codex started"))
    result = runner.run(request)
    assert result["status"] == "setup_error" and "frozen skill changed" in result["diagnostics"]

    runner, request, _ = fixture(tmp_path / "second")
    Path(request["workspace"], "leak").write_text("x")
    result = runner.run(request)
    assert result["status"] == "setup_error" and "fresh empty workspace" in result["diagnostics"]


def test_timeout_is_observed_and_preserves_controller_usage(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr="agent exceeded turn")

    monkeypatch.setattr(module.subprocess, "run", timeout)
    result = runner.run(request, timeout=1)
    assert result["status"] == "timeout"
    assert "agent exceeded turn" in result["diagnostics"]
    assert result["controller_usage"] == {"billed": 0, "calls": []}


def test_model_service_failure_is_infrastructure(monkeypatch, tmp_path: Path):
    runner, request, _ = fixture(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 1, "", "API error: service unavailable"))
    result = runner.run(request)
    assert result["status"] == "model_service_error"


def test_docker_argv_is_an_explicit_minimal_allowlist(monkeypatch, tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    socket_dir = tmp_path / "socket"; socket_dir.mkdir()
    workspace = Path(request["workspace"])
    skill_root = workspace / ".agents/skills"; skill_root.mkdir(parents=True)
    for name in request["skills"]:
        (skill_root / name).mkdir()
    argv = runner._docker_command(
        workspace, Path(request["prompt"]), socket_dir, request, "frozen-cell",
    )
    joined = "\0".join(map(str, argv))
    assert argv[:3] == [str(runner.docker), "run", "--rm"]
    assert "--interactive" in argv
    for item in ("--read-only", "--cap-drop", "ALL", "--pids-limit", "512",
                 "--memory", "8g", "no-new-privileges"):
        assert item in argv
    assert f"type=bind,src={workspace.resolve()},dst=/workspace" in argv
    assert f"type=bind,src={Path(request['prompt']).resolve()},dst=/experiment/PROMPT.md,readonly" in argv
    assert f"type=bind,src={runner.auth_home / 'auth.json'},dst=/codex-home/auth.json,readonly" in argv
    assert any(value.endswith("dst=/usr/local/bin/controller,readonly") for value in argv)
    for name in request["skills"]:
        assert (f"type=bind,src={runner.skill_sources[name]},"
                f"dst=/workspace/.agents/skills/{name},readonly") in argv
    assert runner.docker_image == argv[-1]
    assert "/var/run/docker.sock" not in joined
    assert f"src={ROOT}," not in joined
    for forbidden in (str(Path.home() / ".agents"), str(Path.home() / ".ssh"),
                      "reference_repos", "reference-library"):
        assert forbidden not in joined
    read_write_binds = [value for value in argv if value.startswith("type=bind")
                        and not value.endswith(",readonly")]
    assert read_write_binds == [
        f"type=bind,src={workspace.resolve()},dst=/workspace",
        f"type=bind,src={socket_dir.resolve()},dst=/experiment-state",
    ]


def test_docker_rejects_drifted_image_and_symlink_mount(monkeypatch, tmp_path: Path):
    with pytest.raises(module.RunnerError, match="frozen image ID"):
        docker_fixture(tmp_path / "drift", monkeypatch, image_id="sha256:expected",
                       inspected_image_id="sha256:actual")

    runner, request, _ = docker_fixture(tmp_path / "link", monkeypatch)
    real = Path(request["prompt"])
    linked = real.with_name("linked-prompt.md"); linked.symlink_to(real)
    request["prompt"] = str(linked)
    socket_dir = tmp_path / "link/socket"; socket_dir.mkdir()
    workspace = Path(request["workspace"])
    skill_root = workspace / ".agents/skills"; skill_root.mkdir(parents=True)
    for name in request["skills"]: (skill_root / name).mkdir()
    with pytest.raises(module.RunnerError, match="must not be a symlink"):
        runner._docker_command(workspace, linked, socket_dir, request, "cell")


def test_docker_timeout_force_removes_named_container(monkeypatch, tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr="turn expired")

    monkeypatch.setattr(module.subprocess, "run", run)
    result = runner.run(request, timeout=1)
    assert result["status"] == "timeout"
    launched = calls[0]
    name = launched[launched.index("--name") + 1]
    assert calls[1] == [str(runner.docker), "rm", "-f", name]
    assert result["sandbox"] == {
        "backend": "docker", "image": "python:3.10", "image_id": "sha256:frozen",
    }


def test_docker_cancel_force_removes_named_container(monkeypatch, tmp_path: Path):
    runner, request, _ = docker_fixture(tmp_path, monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(KeyboardInterrupt):
        runner.run(request)
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == [str(runner.docker), "rm", "-f", name]


def test_real_docker_probe_has_controller_but_no_host_privilege(tmp_path: Path):
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker is unavailable")
    inspected = subprocess.run(
        [docker, "image", "inspect", "python:3.10", "--format", "{{.Id}}"],
        text=True, capture_output=True, check=False,
    )
    if inspected.returncode:
        pytest.skip("the pinned local Python image is unavailable")
    runner, request, client = fixture(tmp_path)
    executable(
        runner._runtime() / "bin/codex",
        "#!/bin/sh\n"
        "test \"$CODEX_HOME\" = /codex-home || exit 21\n"
        "test ! -e /var/run/docker.sock || exit 22\n"
        "test ! -e /root/.ssh || exit 23\n"
        "sh -c \"$EXPERIMENT_CONTROLLER help\" >/tmp/help.json || exit 24\n"
        "grep -q '\"operation\": \"help\"' /tmp/help.json || exit 25\n"
        "printf '%s\\n' '{\"type\":\"turn.completed\"}'\n",
    )
    isolated = module.OneShotRunner(
        skill_sources={name: str(path) for name, path in runner.skill_sources.items()},
        assets=runner.assets, placements=runner.placements, client=client,
        codex=str(runner.codex), auth_home=runner.auth_home,
        sandbox_backend="docker", docker=docker, docker_image="python:3.10",
        docker_image_id=inspected.stdout.strip(),
    )
    result = isolated.run(request, timeout=30)
    assert result["status"] == "ok", result
    assert result["controller_usage"]["billed"] == 0
    assert result["sandbox"]["image_id"] == inspected.stdout.strip()
