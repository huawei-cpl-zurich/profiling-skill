from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audited_contract as contract  # noqa: E402
import audited_runtime as runtime  # noqa: E402


@pytest.fixture(autouse=True)
def pending_contract_redactor(monkeypatch):
    """Keep this stacked PR runnable until the public PR1 API is rebased in."""
    if not hasattr(contract, "credential_values"):
        current = contract.redact_text
        monkeypatch.setattr(contract, "credential_values", lambda command: (), raising=False)
        monkeypatch.setattr(
            contract, "redact_text",
            lambda text, extra_secrets=(): current(text),
        )


def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo, auth, state, node = (
        tmp_path / "repo", tmp_path / "auth", tmp_path / "state", tmp_path / "node",
    )
    repo.mkdir()
    auth.mkdir()
    (auth / "auth.json").write_text('{"token":"private"}\n')
    (auth / "config.toml").write_text("must_not_copy=true\n")
    (auth / "skills").mkdir()
    executable(node / "bin" / "node", "#!/bin/sh\necho v20.20.0\n")
    executable(node / "bin" / "codex", "#!/bin/sh\necho 'codex-cli 0.160.0'\n")
    (node / "lib" / "node_modules").mkdir(parents=True)
    return repo, auth, state, node


def stream(thread: str = "thread-1") -> str:
    return json.dumps({"type": "thread.started", "thread_id": thread}) + "\n"


def test_direct_codex_invocation_is_explicit_and_persistent(tmp_path: Path, monkeypatch):
    repo, auth, state, node = fixture_paths(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        if command[-1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli test\n", "")
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stream(), "")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    invoke = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, runtime_mode="direct", agent_id="agent-a",
    )
    invoke(1, None, None)
    invoke(2, "thread-1", "repair this experiment")

    assert calls[0][0] == [
        "codex", "exec", "--json", "--ignore-user-config", "-m", "gpt-5.6-sol",
        "-c", 'model_reasoning_effort="low"', "--sandbox", "workspace-write",
        "-C", str(repo), "-",
    ]
    assert calls[1][0] == [
        "codex", "exec", "resume", "--json", "--ignore-user-config", "-m",
        "gpt-5.6-sol", "-c", 'model_reasoning_effort="low"', "thread-1", "-",
    ]
    assert all("--dangerously-bypass-approvals-and-sandbox" not in call[0] for call in calls)
    assert all(call[1]["cwd"] == repo for call in calls)
    assert all(call[1]["env"]["CODEX_HOME"] == str(state) for call in calls)
    assert "experiment 1" in calls[0][1]["input"]
    assert calls[1][1]["input"] == "repair this experiment\n"


def test_private_state_copies_only_auth_scrubs_and_reopens(tmp_path: Path, monkeypatch):
    repo, auth, state, node = fixture_paths(tmp_path)
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "codex-cli test\n", ""),
    )
    first = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, runtime_mode="direct", agent_id="agent-a",
    )
    assert (state / "auth.json").read_text() == (auth / "auth.json").read_text()
    assert not (state / "config.toml").exists() and not (state / "skills").exists()
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    (state / "sessions.jsonl").write_text("retained\n")
    (state / "config.toml").write_text("generated=true\n")
    first.scrub_auth()
    assert not (state / "auth.json").exists()

    reopened = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, runtime_mode="direct", agent_id="agent-a",
    )
    assert (state / "sessions.jsonl").read_text() == "retained\n"
    assert (state / "config.toml").read_text() == "generated=true\n"
    assert (state / "auth.json").is_file()
    assert reopened.agent_id == "agent-a"

    other = tmp_path / "other-repo"
    other.mkdir()
    with pytest.raises(contract.AuditError, match="different repository or owner"):
        runtime.CodexInvoker(
            other, auth_home=auth, state_dir=state, runtime_mode="direct", agent_id="agent-b",
        )

    agent_a = runtime.CodexInvoker(repo, auth_home=auth, runtime_mode="direct", agent_id="a")
    agent_b = runtime.CodexInvoker(repo, auth_home=auth, runtime_mode="direct", agent_id="b")
    assert agent_a.state_dir != agent_b.state_dir
    assert not agent_a.state_dir.is_relative_to(repo)
    agent_a.scrub_auth()
    agent_b.scrub_auth()


def test_state_and_auth_failures_are_clear(tmp_path: Path, monkeypatch):
    repo, auth, state, node = fixture_paths(tmp_path)
    (auth / "auth.json").unlink()
    with pytest.raises(contract.AuditError, match="auth.json is unavailable"):
        runtime.CodexInvoker(repo, auth_home=auth, runtime_mode="direct")

    (auth / "auth.json").write_text("secret")
    with pytest.raises(contract.AuditError, match="outside the experiment repository"):
        runtime.CodexInvoker(
            repo, auth_home=auth, state_dir=repo / ".codex", runtime_mode="direct",
        )

    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    with pytest.raises(contract.AuditError, match="Codex executable is unavailable"):
        runtime.CodexInvoker(
            repo, codex="missing-codex", auth_home=auth, state_dir=state,
            runtime_mode="direct",
        )


def test_constructor_failure_scrubs_copied_auth(tmp_path: Path):
    repo, auth, state, node = fixture_paths(tmp_path)

    with pytest.raises(contract.AuditError, match="runtime mode"):
        runtime.CodexInvoker(
            repo, auth_home=auth, state_dir=state, runtime_mode="unsupported",
        )

    assert not (state / "auth.json").exists()
    assert (state / ".audited-state.json").is_file()


def test_docker_is_default_and_mounts_only_declared_writable_state(tmp_path: Path, monkeypatch):
    repo, auth, state, node = fixture_paths(tmp_path)
    docker = executable(tmp_path / "docker", "#!/bin/sh\nexit 0\n")
    calls = []

    def fake_run(command, **kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "sha256:resolved-image\n", "")
        if command[-1:] == ["--version"]:
            result = "v20.20.0\n" if command[0].endswith("node") else "codex-cli 0.160.0\n"
            return subprocess.CompletedProcess(command, 0, result, "")
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stream(), "")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    invoke = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, node_runtime=node, docker=str(docker),
        agent_id="agent-a",
    )
    invoke(1, None, None)
    invoke(2, "thread-1", None)

    first, resumed = calls[0][0], calls[1][0]
    joined = "\0".join(first)
    assert first[:5] == [str(docker.resolve()), "run", "--rm", "--interactive", "--init"]
    assert first[5:7] == ["--user", f"{os.getuid()}:{os.getgid()}"]
    assert "--read-only" in first
    assert "--cap-drop\0ALL" in joined
    assert "--security-opt\0no-new-privileges" in joined
    assert f"src={repo},dst=/workspace" in joined
    assert f"src={state},dst=/codex-home" in joined
    assert f"src={node},dst=/runtime/node,readonly" in joined
    assert str(auth) not in joined
    assert "--dangerously-bypass-approvals-and-sandbox" in first
    assert "--dangerously-bypass-approvals-and-sandbox" in resumed
    assert "sha256:resolved-image" in first
    assert calls[0][1]["input"].endswith("\n") and calls[0][1]["env"] is None


def test_codex_metadata_and_runtime_errors(tmp_path: Path, monkeypatch):
    repo, auth, state, node = fixture_paths(tmp_path)
    docker = executable(tmp_path / "docker", "#!/bin/sh\nexit 0\n")

    def fake_run(command, **kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "sha256:image-id\n", "")
        if command[-1:] == ["--version"]:
            output = "v20.20.0\n" if command[0].endswith("node") else "codex-cli 0.160.0\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        raise subprocess.TimeoutExpired(command, 3)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    invoke = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, node_runtime=node, docker=str(docker),
        model="test-model", reasoning_effort="high",
    )
    metadata = invoke.reproducibility_metadata()
    expected = {
        "adapter": "CodexInvoker:docker", "model": "test-model",
        "reasoning_effort": "high", "codex_version": "codex-cli 0.160.0",
        "node_version": "v20.20.0", "docker_image": "python:3.10",
        "docker_image_id": "sha256:image-id",
    }
    assert {key: metadata[key] for key in expected} == expected
    assert len(metadata["identity_sha256"]) == 64
    with pytest.raises(contract.AuditError, match="timed out"):
        invoke(1, None, None)


@pytest.mark.parametrize("failure_kind", ["timeout", "nonzero"])
def test_codex_failure_preserves_safe_bounded_session_event(tmp_path: Path, monkeypatch,
                                                            failure_kind: str):
    repo, auth, state, node = fixture_paths(tmp_path)
    secret = "DO_NOT_RETAIN_THIS_SECRET"
    output = (
        stream("thread-survives")
        + json.dumps({"type": "item.completed", "item": {
            "type": "command_execution", "aggregated_output": secret,
        }}) + "\n"
        + (secret * 10_000)
    )

    def fake_run(command, **kwargs):
        if command[-1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "codex-cli test\n", "")
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(command, 3, output=output.encode(), stderr=secret)
        return subprocess.CompletedProcess(command, 7, output, secret)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    invoke = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, runtime_mode="direct",
    )

    with pytest.raises(runtime.CodexTurnError) as raised:
        invoke(1, None, None)

    failure = raised.value
    assert isinstance(failure, contract.AuditError)
    assert contract.extract_thread_id(failure.structured_stdout, None) == "thread-survives"
    assert len(failure.structured_stdout.splitlines()) == 1
    assert failure.timed_out is (failure_kind == "timeout")
    assert failure.exit_code == (None if failure_kind == "timeout" else 7)
    assert failure.diagnostics["stdout_truncated"] is True
    assert failure.diagnostics["stdout_bytes"] == len(output.encode())
    assert secret not in str(failure)
    assert secret not in failure.structured_stdout
    assert secret not in json.dumps(failure.diagnostics)


def test_controller_executes_compact_json_and_records_sanitized_identity(tmp_path: Path):
    controller = executable(
        tmp_path / "controller",
        "#!/bin/sh\n"
        "if [ \"$1\" = --version ]; then echo controller-v1; exit; fi\n"
        "printf '{\"status\":\"ok\",\"handle\":\"job:1\"}\\n'\n",
    )
    adapter = runtime.CommandController(
        [str(controller), "--token", "SECRET", "--mode=profile"], tmp_path, timeout=3,
    )
    metadata = adapter.reproducibility_metadata()
    assert metadata["argv"] == [
        str(controller), "--token", "<redacted>", "--mode=profile",
    ]
    assert metadata["executable_sha256"] == hashlib.sha256(controller.read_bytes()).hexdigest()
    assert metadata["executable_version"] == "controller-v1"
    assert "SECRET" not in json.dumps(metadata)
    result = adapter(2, "candidate", "manifest")
    assert result == {"status": "ok", "handle": "job:1", "controller_exit_code": 0}


def test_controller_provenance_pins_interpreter_script_and_config(tmp_path: Path):
    script = tmp_path / "controller.py"
    config = tmp_path / "controller.json"
    script.write_text("print('v1')\n")
    config.write_text('{"mode":"profile"}\n')
    command = [sys.executable, str(script), "--config", str(config)]

    first = runtime.CommandController(command, tmp_path).reproducibility_metadata()
    script.write_text("print('v2')\n")
    second = runtime.CommandController(command, tmp_path).reproducibility_metadata()

    assert [item["argument_index"] for item in first["file_arguments"]] == [1, 3]
    assert first["file_arguments"][0]["sha256"] == hashlib.sha256(b"print('v1')\n").hexdigest()
    assert first["file_arguments"][1]["sha256"] == hashlib.sha256(
        config.read_bytes()
    ).hexdigest()
    assert first["identity_sha256"] != second["identity_sha256"]
    assert first["argv"] == command


def test_controller_provenance_identity_binds_sanctioned_state_directory(tmp_path: Path):
    script = tmp_path / "controller.py"
    script.write_text("print('ok')\n")
    state = tmp_path / "mutable-controller-state"
    state.mkdir()
    command = [sys.executable, str(script), "--state-dir", str(state)]

    metadata = runtime.CommandController(command, tmp_path).reproducibility_metadata()

    assert [item["argument_index"] for item in metadata["file_arguments"]] == [1]
    assert metadata["mutable_directories"] == [{
        "argument_index": 3, "option": "--state-dir", "path": str(state),
        "device": state.stat().st_dev, "inode": state.stat().st_ino,
        "uid": state.stat().st_uid, "mode": state.stat().st_mode & 0o7777,
    }]


def test_controller_provenance_rejects_missing_sanctioned_state_directory(tmp_path: Path):
    script = tmp_path / "controller.py"
    script.write_text("print('ok')\n")
    adapter = runtime.CommandController(
        [sys.executable, str(script), "--state-dir", str(tmp_path / "missing")], tmp_path,
    )

    with pytest.raises(contract.AuditError, match="mutable controller directory"):
        adapter.reproducibility_metadata()


def test_controller_provenance_rejects_unpinned_interpreter_module(tmp_path: Path):
    adapter = runtime.CommandController([sys.executable, "-m", "controller"], tmp_path)

    with pytest.raises(contract.AuditError, match="unpinned interpreter"):
        adapter.reproducibility_metadata()


@pytest.mark.parametrize(
    ("effect", "message"),
    [
        (subprocess.TimeoutExpired(["controller"], 1), "timed out"),
        (subprocess.CompletedProcess(["controller"], 0, "not-json", ""), "invalid JSON"),
        (subprocess.CompletedProcess(["controller"], 7, '{"status":"ok"}', ""), "exited 7"),
        (subprocess.CompletedProcess(["controller"], 0, "[]", ""), "JSON object"),
    ],
)
def test_controller_classifies_transport_and_json_failures(tmp_path: Path, monkeypatch,
                                                            effect, message):
    adapter = runtime.CommandController(["controller"], tmp_path, timeout=1)

    def fake_run(*args, **kwargs):
        if isinstance(effect, BaseException):
            raise effect
        return effect

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    result = adapter(1, "candidate", "manifest")
    assert result["status"] == "infrastructure_error"
    assert message in result["reason"]


def test_controller_timeout_preserves_compact_partial_receipt(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path, timeout=1)
    partial = json.dumps({"status": "running", "handle": "job:retained"})
    failure = subprocess.TimeoutExpired(
        ["controller"], 1, output=partial, stderr="waiting\n" + "x" * 1000,
    )
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    result = adapter(2, "candidate", "manifest")

    assert result["status"] == "infrastructure_error"
    assert result["terminal"] is False
    assert result["handle"] == "job:retained"
    assert result["experiment"] == 2
    assert result["candidate_sha256"] == "candidate"
    assert result["manifest_sha256"] == "manifest"
    assert result["partial_stdout"]["excerpt"] == partial
    assert result["partial_stderr"]["truncated"] is True
    assert len(result["partial_stderr"]["excerpt"]) == contract.MAX_EXCERPT


def test_controller_failure_excerpts_use_contract_redaction(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path, timeout=1)
    secrets = ("openai-secret", "database-secret", "cli-secret")
    stdout = (
        '{"status":"running","handle":"job:retained"}\n'
        "OPENAI_API_KEY=openai-secret\n"
    )
    stderr = '{"database_password":"database-secret"}\n--token=cli-secret\n'
    calls = []

    def redact(text, extra_secrets=()):
        calls.append(text)
        for secret in (*secrets, *extra_secrets):
            text = text.replace(secret, "<redacted>")
        return text

    monkeypatch.setattr(contract, "redact_text", redact)
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], 1, output=stdout, stderr=stderr)
        ),
    )

    result = adapter(1, "candidate", "manifest")
    retained = json.dumps(result)

    assert result["handle"] == "job:retained"
    assert all(secret not in retained for secret in secrets)
    assert retained.count("<redacted>") == 3
    assert result["partial_stdout"]["sha256"] == hashlib.sha256(stdout.encode()).hexdigest()
    assert result["partial_stdout"]["bytes"] == len(stdout.encode())
    assert result["partial_stderr"]["sha256"] == hashlib.sha256(stderr.encode()).hexdigest()
    assert calls == [stdout, stderr]


def test_controller_nonzero_diagnostics_are_redacted(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path)
    secret = "controller-secret"
    monkeypatch.setattr(
        contract, "redact_text",
        lambda text, extra_secrets=(): text.replace(secret, "<redacted>"),
    )
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 9, f"not-json OPENAI_API_KEY={secret}", f"--token={secret}",
        ),
    )

    result = adapter(1, "candidate", "manifest")

    assert secret not in json.dumps(result)
    assert result["partial_stdout"]["excerpt"].endswith("<redacted>")
    assert result["partial_stderr"]["excerpt"].endswith("<redacted>")


def test_controller_redacts_bare_echo_of_configured_argv_secret(tmp_path: Path,
                                                                monkeypatch):
    extracted = []

    def credential_values(command):
        extracted.append(command)
        return ("cli-secret",)

    def redact(text, extra_secrets=()):
        for secret in extra_secrets:
            text = text.replace(secret, "<redacted>")
        return text

    monkeypatch.setattr(contract, "credential_values", credential_values)
    monkeypatch.setattr(contract, "redact_text", redact)
    adapter = runtime.CommandController(
        ["controller", "--token", "cli-secret", "--label", "public"], tmp_path,
    )
    stderr = "authentication failed for cli-secret"
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], 1, output="", stderr=stderr)
        ),
    )

    result = adapter(1, "candidate", "manifest")

    assert extracted == ["controller --token cli-secret --label public"]
    assert adapter.command == (
        "controller", "--token", "cli-secret", "--label", "public",
    )
    assert result["partial_stderr"]["excerpt"] == "authentication failed for <redacted>"
    assert result["partial_stderr"]["sha256"] == hashlib.sha256(stderr.encode()).hexdigest()
    assert result["partial_stderr"]["bytes"] == len(stderr.encode())
    assert "cli-secret" not in json.dumps(result)


def test_controller_observe_uses_exact_handle_without_submission(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller", "--mode", "profile"], tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, '{"status":"ok","handle":"job:retained"}', "",
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    result = adapter.observe(2, "candidate", "manifest", "job:retained")

    assert result["handle"] == "job:retained"
    assert calls == [[
        "controller", "--mode", "profile", "--observe-handle", "job:retained",
        "--experiment", "2", "--candidate-sha256", "candidate",
        "--manifest-sha256", "manifest",
    ]]
    assert "--submit" not in calls[0]


def test_controller_observe_rejects_replaced_handle(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path)
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, '{"status":"ok","handle":"job:other"}', "",
        ),
    )

    result = adapter.observe(1, "candidate", "manifest", "job:retained")

    assert result["status"] == "infrastructure_error"
    assert result["handle"] == "job:retained"
    assert "different handle" in result["reason"]


def test_controller_remeasure_uses_exact_pending_handle(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller", "--mode", "profile"], tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, '{"status":"ok","handle":"job:pending"}', "",
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    result = adapter.remeasure(3, "candidate", "manifest", "job:pending")

    assert result["handle"] == "job:pending"
    assert calls == [[
        "controller", "--mode", "profile", "--remeasure-handle", "job:pending",
        "--experiment", "3", "--candidate-sha256", "candidate",
        "--manifest-sha256", "manifest",
    ]]


def test_controller_remeasure_rejects_missing_or_replaced_handle(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path)
    with pytest.raises(runtime.AuditError, match="remeasure requires a durable handle"):
        adapter.remeasure(1, "candidate", "manifest", "")
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, '{"status":"ok","handle":"job:other"}', "",
        ),
    )

    result = adapter.remeasure(1, "candidate", "manifest", "job:pending")

    assert result["status"] == "infrastructure_error"
    assert result["handle"] == "job:pending"
    assert "different handle" in result["reason"]


def test_controller_bounds_receipt(tmp_path: Path, monkeypatch):
    adapter = runtime.CommandController(["controller"], tmp_path)
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "x" * (runtime.MAX_RECEIPT_BYTES + 1), "",
        ),
    )
    assert "too large" in adapter(1, "candidate", "manifest")["reason"]


def test_controller_preserves_classified_candidate_failure(tmp_path: Path, monkeypatch):
    secret = "cli-secret"
    adapter = runtime.CommandController(["controller", "--token", secret], tmp_path)
    stdout = '{"status":"candidate_error","reason":"compile failed"}'
    stderr = f"SyntaxError near token; authentication failed for {secret}"
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, stdout, stderr,
        ),
    )
    result = adapter(1, "candidate", "manifest")

    assert result["status"] == "candidate_error"
    assert result["reason"] == "compile failed"
    assert result["controller_exit_code"] == 7
    assert result["partial_stdout"]["sha256"] == hashlib.sha256(stdout.encode()).hexdigest()
    assert result["partial_stdout"]["bytes"] == len(stdout.encode())
    assert "SyntaxError near token" in result["partial_stderr"]["excerpt"]
    assert secret not in json.dumps(result)
    assert "<redacted>" in result["partial_stderr"]["excerpt"]


@pytest.mark.parametrize("status", ["compile_error", "runtime_error"])
def test_controller_normalizes_native_candidate_failures_with_diagnostics(
        tmp_path: Path, monkeypatch, status: str):
    adapter = runtime.CommandController(["controller"], tmp_path)
    stdout = json.dumps({"status": status, "reason": "kernel failed"})
    monkeypatch.setattr(
        runtime.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stdout, "compiler/runtime diagnostic",
        ),
    )

    result = adapter(1, "candidate", "manifest")

    assert result["status"] == "candidate_error"
    assert result["failure_type"] == status
    assert result["partial_stderr"]["excerpt"] == "compiler/runtime diagnostic"
    assert result["partial_stdout"]["sha256"] == hashlib.sha256(stdout.encode()).hexdigest()


def test_generated_docker_argv_passes_real_parser_and_stdin(tmp_path: Path):
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker is unavailable")
    image = subprocess.run([docker, "image", "inspect", "python:3.10"], capture_output=True)
    if image.returncode:
        pytest.skip("python:3.10 image is unavailable")
    repo, auth, state, node = fixture_paths(tmp_path)
    executable(
        node / "bin" / "codex",
        "#!/bin/sh\nIFS= read -r line\nprintf 'PROMPT:%s\\n' \"$line\"\n",
    )
    invoke = runtime.CodexInvoker(
        repo, auth_home=auth, state_dir=state, node_runtime=node, docker=docker,
    )

    run = subprocess.run(
        invoke.docker_command(["exec", "-"]), input="HELLO_DOCKER_STDIN\n",
        text=True, capture_output=True,
    )

    invoke.scrub_auth()
    assert run.returncode == 0, run.stderr
    assert run.stdout == "PROMPT:HELLO_DOCKER_STDIN\n"
