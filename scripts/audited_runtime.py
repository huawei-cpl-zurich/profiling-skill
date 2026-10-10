#!/usr/bin/env python3
"""Isolated Codex and target-neutral controller runtime adapters."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import audited_contract as evidence_contract
from audited_contract import AuditError, MAX_EXCERPT, sanitize_argv, sha256_bytes

MAX_RECEIPT_BYTES = 65_536
MAX_CODEX_FAILURE_BYTES = 65_536
MAX_TRANSIENT_RETRIES = 5
CAPACITY_ERROR_MESSAGE = "Selected model is at capacity. Please try a different model."
CODEX_RETRY_SCHEMA = "profiling-skill/codex-transient-retry/v1"
_AGENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_INTERPRETER = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?|bash|sh|node|env")
_PATH_SUFFIXES = {".py", ".sh", ".json", ".toml", ".yaml", ".yml"}


def _version(command: Sequence[str], label: str) -> str:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise AuditError(f"{label} executable is unavailable: {command[0]}") from failure
    version = (result.stdout.strip() or result.stderr.strip())[:MAX_EXCERPT]
    if result.returncode or not version:
        raise AuditError(f"{label} version is unavailable")
    return version


def _file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.is_file() else "unavailable"


def _stream_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode()


def _compact_stream(value: str | bytes | None,
                    extra_secrets: Sequence[str] = ()) -> dict:
    data = _stream_bytes(value)
    text = data.decode(errors="replace")
    safe_text = evidence_contract.redact_text(text, extra_secrets)
    return {
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "excerpt": safe_text[:MAX_EXCERPT],
        "truncated": len(safe_text) > MAX_EXCERPT,
    }


def _redact_document(document: dict, extra_secrets: Sequence[str] = ()) -> dict:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    redacted = json.loads(evidence_contract.redact_text(encoded, extra_secrets))
    if not isinstance(redacted, dict):  # pragma: no cover - JSON shape cannot change safely
        raise AuditError("contract redaction changed the controller receipt shape")
    return redacted


def _process_excerpts(stdout: str | bytes | None,
                      stderr: str | bytes | None,
                      extra_secrets: Sequence[str] = ()) -> dict:
    return {
        "partial_stdout": _compact_stream(stdout, extra_secrets),
        "partial_stderr": _compact_stream(stderr, extra_secrets),
    }


def _partial_receipt(value: str | bytes | None) -> dict | None:
    text = _stream_bytes(value).decode(errors="replace")
    candidates = [text, *reversed(text.splitlines())]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _thread_event_stream(value: str | bytes | None) -> str:
    """Retain only bounded, non-content session identity events."""
    data = _stream_bytes(value)[:MAX_CODEX_FAILURE_BYTES]
    retained = []
    for line in data.decode(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        thread_id = event.get("thread_id") if isinstance(event, dict) else None
        if (isinstance(event, dict) and event.get("type") == "thread.started"
                and isinstance(thread_id, str) and 0 < len(thread_id) <= MAX_EXCERPT):
            retained.append(json.dumps(
                {"type": "thread.started", "thread_id": thread_id},
                sort_keys=True, separators=(",", ":"),
            ))
    return "\n".join(retained) + ("\n" if retained else "")


def _terminal_error_message(event: dict) -> str | None:
    error = event.get("error")
    container = error if isinstance(error, dict) else event
    message = container.get("message")
    return message if isinstance(message, str) else None


def _retryable_terminal_failure(value: str | bytes | None,
                                expected_thread: str | None
                                ) -> tuple[str, str] | None:
    """Classify an explicit terminal service failure with proven zero progress."""
    data = _stream_bytes(value)
    if not data or len(data) > MAX_CODEX_FAILURE_BYTES:
        return None
    threads = set()
    messages = set()
    terminal = False
    for line in data.decode(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        event_type = event.get("type")
        if event_type == "thread.started":
            thread = event.get("thread_id")
            if not isinstance(thread, str) or not thread or len(thread) > MAX_EXCERPT:
                return None
            threads.add(thread)
        elif event_type == "turn.started":
            continue
        elif event_type in {"turn.failed", "error"}:
            terminal = True
            message = _terminal_error_message(event)
            if message is not None:
                messages.add(message)
        else:
            # Item, tool, command, file-change, assistant, completed-turn, and
            # unknown events all prove or may conceal progress.
            return None
    if len(threads) > 1 or messages != {CAPACITY_ERROR_MESSAGE} or not terminal:
        return None
    thread = next(iter(threads), expected_thread)
    if thread is None or (expected_thread is not None and thread != expected_thread):
        return None
    return "server_overloaded", thread


def _retry_evidence(stdout: str | bytes | None, stderr: str | bytes | None,
                    attempt: int, action: str) -> dict:
    stdout_bytes, stderr_bytes = _stream_bytes(stdout), _stream_bytes(stderr)
    return {
        "schema": CODEX_RETRY_SCHEMA, "terminal_error": "server_overloaded",
        "action": action, "attempt": attempt,
        "stdout_sha256": sha256_bytes(stdout_bytes), "stdout_bytes": len(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes), "stderr_bytes": len(stderr_bytes),
    }


class CodexTurnError(AuditError):
    """Failed Codex turn with safe session evidence for same-thread recovery.

    `structured_stdout` (also available as `stdout`) contains only bounded
    canonical `thread.started` events. Raw model/command output and stderr are
    represented solely by hashes and sizes in `diagnostics`.
    """

    def __init__(self, message: str, *, stdout: str | bytes | None,
                 stderr: str | bytes | None, timed_out: bool,
                 exit_code: int | None, thread_id: str | None = None,
                 terminal_error: str | None = None,
                 transient_retries: int = 0,
                 retry_evidence: Sequence[dict] = ()):
        super().__init__(message)
        stdout_bytes, stderr_bytes = _stream_bytes(stdout), _stream_bytes(stderr)
        self.structured_stdout = _thread_event_stream(stdout)
        if thread_id and not self.structured_stdout:
            self.structured_stdout = json.dumps(
                {"thread_id": thread_id, "type": "thread.started"},
                sort_keys=True, separators=(",", ":"),
            ) + "\n"
        self.stdout = self.structured_stdout
        self.timed_out = timed_out
        self.exit_code = exit_code
        self.retry_evidence = tuple(retry_evidence)
        self.diagnostics = {
            "stdout_sha256": sha256_bytes(stdout_bytes),
            "stdout_bytes": len(stdout_bytes),
            "stdout_truncated": len(stdout_bytes) > MAX_CODEX_FAILURE_BYTES,
            "stderr_sha256": sha256_bytes(stderr_bytes),
            "stderr_bytes": len(stderr_bytes),
            "stderr_truncated": len(stderr_bytes) > MAX_CODEX_FAILURE_BYTES,
            "terminal_error": terminal_error,
            "transient_retries": transient_retries,
        }


class CodexInvoker:
    """Run one persistent Codex thread in an isolated per-agent environment.

    Docker is the supported default. Direct execution must be selected explicitly
    and still uses Codex's workspace-write sandbox.
    """

    def __init__(
        self,
        repo: Path,
        *,
        codex: str = "codex",
        timeout: int = 900,
        state_dir: Path | None = None,
        auth_home: Path | None = None,
        runtime_mode: str = "docker",
        docker: str = "docker",
        image: str = "python:3.10",
        node_runtime: Path | None = None,
        model: str = "gpt-5.6-sol",
        reasoning_effort: str = "low",
        agent_id: str = "agent",
        transient_retry_limit: int = 2,
        transient_retry_backoff_seconds: float = 1.0,
    ):
        self.repo = repo.resolve()
        self.timeout = timeout
        self.runtime_mode = runtime_mode
        self.image = image
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex = codex
        if (type(transient_retry_limit) is not int or transient_retry_limit < 0
                or transient_retry_limit > MAX_TRANSIENT_RETRIES):
            raise AuditError(
                f"transient_retry_limit must be between 0 and {MAX_TRANSIENT_RETRIES}"
            )
        if (not isinstance(transient_retry_backoff_seconds, (int, float))
                or isinstance(transient_retry_backoff_seconds, bool)
                or not math.isfinite(transient_retry_backoff_seconds)
                or transient_retry_backoff_seconds < 0):
            raise AuditError("transient_retry_backoff_seconds must be finite and nonnegative")
        self.transient_retry_limit = transient_retry_limit
        self.transient_retry_backoff_seconds = float(transient_retry_backoff_seconds)
        self.last_retry_evidence: tuple[dict, ...] = ()
        if not _AGENT_ID.fullmatch(agent_id):
            raise AuditError("agent_id must contain only letters, numbers, dot, dash, or underscore")
        self.agent_id = agent_id
        default_state = self.repo.parent / ".codex-state" / self.repo.name / agent_id
        self.state_dir = (state_dir or default_state).resolve()
        if self.state_dir == self.repo or self.state_dir.is_relative_to(self.repo):
            raise AuditError("Codex state directory must be outside the experiment repository")

        source_home = (
            auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        ).resolve()
        source_auth = source_home / "auth.json"
        if not source_auth.is_file():
            raise AuditError(f"Codex auth.json is unavailable in {source_home}")
        self._copied_auth = False
        try:
            self._open_state(source_auth)
            self._configure_runtime(codex, docker, node_runtime)
        except BaseException:
            if self._copied_auth:
                self.auth_path.unlink(missing_ok=True)
            raise

    def _configure_runtime(self, codex: str, docker: str,
                           node_runtime: Path | None) -> None:
        if self.runtime_mode == "docker":
            docker_binary = shutil.which(docker)
            if not docker_binary:
                raise AuditError("Docker runtime is unavailable; install Docker or use --runtime direct")
            self.docker = str(Path(docker_binary).resolve())
            self.node_runtime = (node_runtime or self._find_node_runtime(codex)).resolve()
            node = self.node_runtime / "bin" / "node"
            codex_binary = self.node_runtime / "bin" / "codex"
            if not node.is_file():
                raise AuditError(f"Node executable is unavailable in {self.node_runtime}")
            if not codex_binary.is_file():
                raise AuditError(f"Codex executable is unavailable in {self.node_runtime}")
            try:
                inspected = subprocess.run(
                    [self.docker, "image", "inspect", self.image, "--format", "{{.Id}}"],
                    text=True, capture_output=True, check=False, timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as failure:
                raise AuditError(f"Docker image {self.image} could not be inspected") from failure
            self.docker_image_id = inspected.stdout.strip()
            if inspected.returncode or not self.docker_image_id.startswith("sha256:"):
                raise AuditError(
                    f"Docker image {self.image} has no resolved immutable image ID"
                )
            self.node_version = _version([str(node), "--version"], "Node")
            self.codex_version = _version([str(codex_binary), "--version"], "Codex")
            self.executable_hash = _file_hash(codex_binary.resolve())
        elif self.runtime_mode == "direct":
            resolved = shutil.which(codex)
            executable = Path(resolved or codex)
            self.docker_image_id = "not-applicable"
            self.node_runtime = None
            self.node_version = "direct-runtime"
            self.codex_version = _version([codex, "--version"], "Codex")
            self.executable_hash = _file_hash(executable.resolve())
        else:
            raise AuditError("runtime mode must be docker or direct")

    def _open_state(self, source_auth: Path) -> None:
        existed = self.state_dir.exists()
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        sentinel = self.state_dir / ".audited-state.json"
        identity = {
            "schema": "profiling-skill/codex-state/v1",
            "repo": str(self.repo),
            "state_dir": str(self.state_dir),
            "agent_id": self.agent_id,
            "uid": os.getuid(),
            "gid": os.getgid(),
        }
        if existed and any(self.state_dir.iterdir()):
            if not sentinel.is_file():
                raise AuditError("existing Codex state is unowned: missing isolation sentinel")
            try:
                retained = json.loads(sentinel.read_text())
            except (OSError, json.JSONDecodeError) as failure:
                raise AuditError("existing Codex state has an invalid isolation sentinel") from failure
            if retained != identity:
                raise AuditError("existing Codex state belongs to a different repository or owner")
        else:
            sentinel.write_text(json.dumps(identity, sort_keys=True) + "\n")
            sentinel.chmod(0o600)
        if self.state_dir.stat().st_uid != os.getuid():
            raise AuditError("Codex state directory is not owned by the current user")
        self.auth_path = self.state_dir / "auth.json"
        shutil.copyfile(source_auth, self.auth_path)
        self._copied_auth = True
        self.auth_path.chmod(0o600)
        self.private_home = self.state_dir / "home"
        self.runtime_dir = self.state_dir / "runtime"
        self.temp_dir = self.state_dir / "tmp"
        for directory in (self.private_home, self.runtime_dir, self.temp_dir):
            directory.mkdir(mode=0o700, exist_ok=True)
            directory.chmod(0o700)

    @staticmethod
    def _find_node_runtime(codex: str) -> Path:
        executable = Path(shutil.which(codex) or codex).resolve()
        for parent in executable.parents:
            if (parent / "bin" / "node").is_file() and (parent / "lib" / "node_modules").is_dir():
                return parent
        raise AuditError(f"could not locate the Node runtime containing {codex}")

    def reproducibility_metadata(self) -> dict:
        identity = {
            "adapter": f"CodexInvoker:{self.runtime_mode}",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "codex_version": self.codex_version,
            "node_version": self.node_version,
            "codex_executable_sha256": self.executable_hash,
            "docker_image": self.image if self.runtime_mode == "docker" else "not-applicable",
            "docker_image_id": self.docker_image_id,
        }
        identity["identity_sha256"] = sha256_bytes(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )
        return identity

    def scrub_auth(self) -> None:
        """Remove copied credentials without deleting resumable Codex state."""
        self.auth_path.unlink(missing_ok=True)

    def __call__(self, number: int, session_id: str | None, repair: str | None) -> str:
        active_session = session_id

        def invocation(active: str | None) -> tuple[list[str], dict | None]:
            if active:
                arguments = [
                    "exec", "resume", "--json", "--ignore-user-config", "-m", self.model,
                    "-c", f'model_reasoning_effort="{self.reasoning_effort}"', active, "-",
                ]
                if self.runtime_mode == "docker":
                    arguments[4:4] = ["--dangerously-bypass-approvals-and-sandbox"]
            else:
                arguments = [
                    "exec", "--json", "--ignore-user-config", "-m", self.model,
                    "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                ]
                if self.runtime_mode == "docker":
                    arguments += [
                        "--dangerously-bypass-approvals-and-sandbox", "-C", "/workspace", "-",
                    ]
                else:
                    arguments += ["--sandbox", "workspace-write", "-C", str(self.repo), "-"]
            command = ([self.codex, *arguments] if self.runtime_mode == "direct"
                       else self.docker_command(arguments))
            environment = None
            if self.runtime_mode == "direct":
                environment = {
                    **os.environ,
                    "CODEX_HOME": str(self.state_dir),
                    "HOME": str(self.private_home),
                    "XDG_RUNTIME_DIR": str(self.runtime_dir),
                    "TMPDIR": str(self.temp_dir),
                }
            return command, environment

        instruction = repair or (
            f"Read PROMPT.md and TASK.md. Prepare only experiment {number}: make one material "
            "candidate change, run local checks, summarize readiness, and stop for host profiling."
        )
        retries = 0
        retained_retries = []
        self.last_retry_evidence = ()
        while True:
            command, environment = invocation(active_session)
            try:
                result = subprocess.run(
                    command, cwd=self.repo, input=instruction + "\n", text=True,
                    capture_output=True, check=False, timeout=self.timeout, env=environment,
                )
            except subprocess.TimeoutExpired as failure:
                raise CodexTurnError(
                    f"Codex turn timed out after {self.timeout} seconds",
                    stdout=failure.stdout, stderr=failure.stderr, timed_out=True,
                    exit_code=None, thread_id=active_session,
                    transient_retries=retries, retry_evidence=retained_retries,
                ) from failure
            except OSError as failure:
                raise AuditError(f"Codex runtime could not start: {failure}") from failure
            if not result.returncode:
                return result.stdout
            transient = _retryable_terminal_failure(result.stdout, active_session)
            if transient is not None:
                terminal_error, active_session = transient
                if retries < self.transient_retry_limit:
                    retained_retries.append(_retry_evidence(
                        result.stdout, result.stderr, retries + 1, "retry",
                    ))
                    self.last_retry_evidence = tuple(retained_retries)
                    delay = self.transient_retry_backoff_seconds * (2 ** retries)
                    retries += 1
                    time.sleep(delay)
                    continue
                retained_retries.append(_retry_evidence(
                    result.stdout, result.stderr, retries + 1, "exhausted",
                ))
                self.last_retry_evidence = tuple(retained_retries)
            else:
                terminal_error = None
            label = "Docker Codex" if self.runtime_mode == "docker" else "Codex"
            raise CodexTurnError(
                f"{label} turn failed with exit {result.returncode}",
                stdout=result.stdout, stderr=result.stderr, timed_out=False,
                exit_code=result.returncode, thread_id=active_session,
                terminal_error=terminal_error, transient_retries=retries,
                retry_evidence=retained_retries,
            )

    def docker_command(self, codex_arguments: Sequence[str]) -> list[str]:
        """Build a Docker argv suitable for direct execution (no shell involved)."""
        return [
            self.docker, "run", "--rm", "--interactive", "--init",
            "--user", f"{os.getuid()}:{os.getgid()}", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
            "--tmpfs", f"/home/agent:rw,nosuid,nodev,mode=700,uid={os.getuid()},gid={os.getgid()}",
            "--mount", f"type=bind,src={self.repo},dst=/workspace",
            "--mount", f"type=bind,src={self.state_dir},dst=/codex-home",
            "--mount", f"type=bind,src={self.node_runtime},dst=/runtime/node,readonly",
            "--workdir", "/workspace",
            "--env", "HOME=/home/agent", "--env", "CODEX_HOME=/codex-home",
            "--env", "XDG_RUNTIME_DIR=/codex-home/runtime", "--env", "TMPDIR=/tmp",
            "--env", "PATH=/runtime/node/bin:/usr/bin:/bin",
            self.docker_image_id, "/runtime/node/bin/codex", *codex_arguments,
        ]


class CommandController:
    """Execute a target-neutral controller and return one compact JSON receipt."""

    def __init__(self, command: Sequence[str], repo: Path, *, timeout: int = 900):
        if not command:
            raise AuditError("controller command is required")
        if timeout <= 0:
            raise AuditError("controller timeout must be positive")
        self.command = tuple(command)
        self._credential_values = evidence_contract.credential_values(
            shlex.join(self.command)
        )
        self.repo = repo.resolve()
        self.timeout = timeout

    def _executable(self) -> Path:
        resolved = shutil.which(self.command[0])
        executable = Path(resolved) if resolved else Path(self.command[0])
        if not executable.is_absolute():
            executable = self.repo / executable
        return executable.resolve()

    def _mutable_directory_arguments(self) -> list[dict]:
        retained = []
        for index, argument in enumerate(self.command[1:], 1):
            if argument == "--state-dir":
                value_index = index + 1
                value = self.command[value_index] if value_index < len(self.command) else ""
            elif argument.startswith("--state-dir="):
                value_index = index
                value = argument.partition("=")[2]
            else:
                continue
            path = Path(value)
            if not path.is_absolute() or not path.is_dir():
                raise AuditError("mutable controller directory must be an existing absolute directory")
            resolved = path.resolve()
            metadata = resolved.stat()
            retained.append({
                "argument_index": value_index, "option": "--state-dir",
                "path": str(resolved), "device": metadata.st_dev,
                "inode": metadata.st_ino, "uid": metadata.st_uid,
                "mode": metadata.st_mode & 0o7777,
            })
        return retained

    def _file_arguments(self, executable: Path,
                        mutable_indices: set[int] | None = None) -> list[dict]:
        sanitized = sanitize_argv(self.command)
        mutable_indices = mutable_indices or set()
        pinned = []
        missing = []
        for index, argument in enumerate(self.command[1:], 1):
            if index in mutable_indices or "<redacted>" in sanitized[index]:
                continue
            option, separator, value = argument.partition("=")
            candidate = value if separator and option.startswith("-") else argument
            if candidate.startswith("-") or not candidate:
                continue
            path = Path(candidate)
            resolved = path if path.is_absolute() else self.repo / path
            resolved = resolved.resolve()
            if resolved.is_file():
                pinned.append({
                    "argument_index": index,
                    "path": str(resolved),
                    "sha256": _file_hash(resolved),
                })
            elif "/" in candidate or path.suffix.lower() in _PATH_SUFFIXES:
                missing.append(candidate)
        if missing:
            raise AuditError(
                f"controller file argument is unavailable: {missing[0]}"
            )
        interpreter = _INTERPRETER.fullmatch(executable.name.lower()) is not None
        inline = any(argument in {"-c", "-e"} for argument in self.command[1:])
        if interpreter and not pinned and not inline:
            raise AuditError(
                "unpinned interpreter controller requires an existing script argument"
            )
        return pinned

    def reproducibility_metadata(self) -> dict:
        executable = self._executable()
        mutable_directories = self._mutable_directory_arguments()
        file_arguments = self._file_arguments(
            executable, {item["argument_index"] for item in mutable_directories})
        version = "unavailable"
        if executable.is_file():
            try:
                checked = subprocess.run(
                    [str(executable), "--version"], text=True, capture_output=True,
                    check=False, timeout=5,
                )
                if checked.returncode == 0:
                    version = (checked.stdout.strip() or checked.stderr.strip())[:MAX_EXCERPT]
            except (OSError, subprocess.TimeoutExpired):
                pass
        identity = {
            "adapter": "CommandController",
            "argv": sanitize_argv(self.command),
            "executable_sha256": _file_hash(executable),
            "executable_version": version,
            "file_arguments": file_arguments,
            "mutable_directories": mutable_directories,
            "model": "not-applicable",
            "reasoning_effort": "not-applicable",
            "codex_version": "not-applicable",
            "node_version": "not-applicable",
            "docker_image_id": "not-applicable",
        }
        identity["identity_sha256"] = sha256_bytes(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        )
        return identity

    def __call__(self, number: int, candidate_hash: str, manifest_hash: str) -> dict:
        command = [
            *self.command, "--experiment", str(number),
            "--candidate-sha256", candidate_hash, "--manifest-sha256", manifest_hash,
        ]
        return self._execute(command, number, candidate_hash, manifest_hash)

    def observe(self, number: int, candidate_hash: str, manifest_hash: str,
                handle: str) -> dict:
        """Observe an existing durable handle without submitting a replacement."""
        if not isinstance(handle, str) or not handle:
            raise AuditError("controller observe requires a durable handle")
        command = [
            *self.command, "--observe-handle", handle, "--experiment", str(number),
            "--candidate-sha256", candidate_hash, "--manifest-sha256", manifest_hash,
        ]
        return self._execute(
            command, number, candidate_hash, manifest_hash, expected_handle=handle,
            allow_observe_transaction=True,
        )

    def remeasure(self, number: int, candidate_hash: str, manifest_hash: str,
                  handle: str, pending_receipt: dict) -> dict:
        """Remeasure a pending receipt without preparing another candidate."""
        if not isinstance(handle, str) or not handle:
            raise AuditError("controller remeasure requires a durable handle")
        try:
            evidence_contract.validate_controller_receipt(
                pending_receipt, candidate_hash, manifest_hash,
            )
        except AuditError as failure:
            raise AuditError("controller remeasure requires an authenticated pending receipt") \
                from failure
        if (pending_receipt.get("status") != "measurement_pending"
                or pending_receipt.get("handle") != handle
                or pending_receipt.get("experiment") != number):
            raise AuditError("controller remeasure requires an authenticated pending receipt")
        command = [
            *self.command, "--remeasure-handle", handle, "--experiment", str(number),
            "--candidate-sha256", candidate_hash, "--manifest-sha256", manifest_hash,
        ]
        return self._execute(
            command, number, candidate_hash, manifest_hash, expected_handle=handle,
            remeasure_source=pending_receipt,
        )

    def _execute(self, command: Sequence[str], number: int, candidate_hash: str,
                 manifest_hash: str, *, expected_handle: str | None = None,
                 allow_observe_transaction: bool = False,
                 remeasure_source: dict | None = None) -> dict:
        try:
            result = subprocess.run(
                command, cwd=self.repo, text=True, capture_output=True,
                check=False, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as failure:
            partial = _partial_receipt(failure.stdout)
            partial_handle = partial.get("handle") if partial else None
            handle = expected_handle or partial_handle
            reason = "controller timed out"
            if expected_handle and partial_handle not in {None, expected_handle}:
                reason += "; partial receipt reported a different handle"
            elif handle:
                reason += "; reobserve the same durable handle"
            receipt = {
                "status": "infrastructure_error",
                "terminal": False,
                "reason": reason,
                "experiment": number,
                "candidate_sha256": candidate_hash,
                "manifest_sha256": manifest_hash,
                **_process_excerpts(
                    failure.stdout, failure.stderr, self._credential_values,
                ),
            }
            if handle:
                receipt["handle"] = handle
            return receipt
        except OSError as failure:
            return {"status": "infrastructure_error", "reason": f"controller failed: {failure}"}
        if len(result.stdout.encode()) > MAX_RECEIPT_BYTES:
            return {"status": "infrastructure_error", "reason": "controller receipt is too large"}
        try:
            receipt = json.loads(result.stdout)
        except json.JSONDecodeError as failure:
            classified = {
                "status": "infrastructure_error",
                "reason": f"controller returned invalid JSON: {failure}",
            }
            if result.returncode:
                classified.update(_process_excerpts(
                    result.stdout, result.stderr, self._credential_values,
                ))
            return classified
        if not isinstance(receipt, dict):
            classified = {
                "status": "infrastructure_error",
                "reason": "controller receipt must be a JSON object",
            }
            if result.returncode:
                classified.update(_process_excerpts(
                    result.stdout, result.stderr, self._credential_values,
                ))
            return classified
        receipt = _redact_document(receipt, self._credential_values)
        classification = receipt.get("status")
        if classification in {"compile_error", "compilation_error", "runtime_error"}:
            receipt.setdefault("failure_type", classification)
            receipt["status"] = "candidate_error"
        allowed_failure = receipt.get("status") in {"candidate_error", "infrastructure_error"}
        if result.returncode and not allowed_failure:
            return {
                "status": "infrastructure_error",
                "reason": f"controller exited {result.returncode} without a failure classification",
                **_process_excerpts(
                    result.stdout, result.stderr, self._credential_values,
                ),
            }
        observed_transaction = (expected_handle is None or (
            not allow_observe_transaction and remeasure_source is None
            and receipt.get("handle") == expected_handle
        ))
        if allow_observe_transaction and expected_handle is not None:
            try:
                if receipt.get("status") in {
                        "ok", "candidate_error", "measurement_pending"}:
                    evidence_contract.validate_controller_receipt(
                        receipt, candidate_hash, manifest_hash,
                    )
                    evidence_contract.validate_observe_transaction(
                        receipt, expected_handle,
                    )
                elif receipt.get("handle") != expected_handle:
                    evidence_contract.validate_observe_transition(
                        receipt, expected_handle, candidate_hash, manifest_hash, number,
                    )
                observed_transaction = True
            except AuditError:
                pass
        if remeasure_source is not None and expected_handle is not None:
            try:
                if receipt.get("status") in {
                        "ok", "candidate_error", "measurement_pending"}:
                    evidence_contract.validate_remeasure_transition(
                        receipt, remeasure_source, candidate_hash, manifest_hash, number,
                    )
                    observed_transaction = True
                elif (receipt.get("status") == "infrastructure_error"
                      and receipt.get("handle") == expected_handle):
                    observed_transaction = True
            except AuditError:
                pass
        if expected_handle is not None and not observed_transaction:
            return {
                "status": "infrastructure_error",
                "terminal": False,
                "handle": expected_handle,
                "reason": (
                    "controller observation lacks valid transaction proof"
                    if allow_observe_transaction else
                    "controller remeasurement lacks valid transition proof"
                    if remeasure_source is not None else
                    "controller observation returned a different handle"
                ),
                "experiment": number,
                "candidate_sha256": candidate_hash,
                "manifest_sha256": manifest_hash,
            }
        if result.returncode:
            receipt.update(_process_excerpts(
                result.stdout, result.stderr, self._credential_values,
            ))
        receipt.setdefault("controller_exit_code", result.returncode)
        return receipt
