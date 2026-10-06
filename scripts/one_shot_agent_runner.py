#!/usr/bin/env python3
"""Run one frozen diagnostic Codex turn in an explicit outer sandbox."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def check_command(round_number: int) -> list[str]:
    return ["check", "--scope", "development", "--round", str(round_number)]


CHECK = check_command(1)
def controller_help(limit: int) -> str:
    checks = "".join(
        f"  $EXPERIMENT_CONTROLLER check --scope development --round {round_number}\n"
        for round_number in range(1, limit + 1))
    return "usage:\n  $EXPERIMENT_CONTROLLER help\n  $EXPERIMENT_CONTROLLER budget\n" + checks
MAX_DIAGNOSTIC = 64 * 1024
INNER_TURN_TIMEOUT = 330
CONTROLLER_CLIENT = '''#!/usr/bin/env python3
import json, socket, sys
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect(sys.argv[1])
    client.sendall(json.dumps({"arguments": sys.argv[2:]}).encode() + b"\\n")
    response = json.loads(client.makefile("rb").readline())
print(response.get("stdout", ""), end="")
print(response.get("stderr", ""), end="", file=sys.stderr)
raise SystemExit(int(response["exit_code"]))
'''


class RunnerError(RuntimeError):
    pass


class SubmissionError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not item.is_file():
            if item.is_symlink():
                raise RunnerError(f"skill contains a symlink: {item}")
            continue
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        digest.update(f"{stat.S_IMODE(item.stat().st_mode) & 0o111:o}".encode() + b"\0")
        digest.update(item.read_bytes() + b"\0")
    return digest.hexdigest()


def _bounded(value: object) -> str:
    text = str(value or "")
    return text if len(text) <= MAX_DIAGNOSTIC else text[:MAX_DIAGNOSTIC] + "\n...[diagnostic truncated]"


def _prompt_bytes(request: dict) -> bytes:
    prompt = request.get("prompt")
    if request.get("protocol_version") != 2 or not isinstance(prompt, dict):
        raise RunnerError("protocol-v2 in-band prompt is required")
    if prompt.get("encoding") != "base64":
        raise RunnerError("prompt must use base64 encoding")
    try:
        content = base64.b64decode(prompt.get("data", ""), validate=True)
    except (TypeError, ValueError) as error:
        raise RunnerError("prompt is not valid base64") from error
    digest = hashlib.sha256(content).hexdigest()
    if prompt.get("sha256") != digest or request.get("prompt_sha256") != digest:
        raise RunnerError("prompt hash mismatch")
    return content


def _validated_supplementary_assets(assets: dict) -> dict[str, str]:
    supplementary = assets.get("supplementary", {})
    reserved = {"baseline.py", "cases.jsonl", "runner.py", "candidate.py",
                "candidate.manifest.json", "AGENTS.md"}
    if (not isinstance(supplementary, dict)
            or any(not isinstance(name, str) or not name or name in {".", ".."}
                   or name in reserved or Path(name).name != name
                   or not isinstance(source, str) or not Path(source).is_file()
                   for name, source in supplementary.items())):
        raise RunnerError("supplementary assets require safe filenames and regular sources")
    return supplementary


def _run_group(command: list[str], prompt: str, timeout: int) -> subprocess.CompletedProcess:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, \
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout_file,
                                   stderr=stderr_file, text=True, start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
            stdout_file.seek(0); stderr_file.seek(0)
            return subprocess.CompletedProcess(command, process.returncode,
                _bounded(stdout_file.read(MAX_DIAGNOSTIC + 1)),
                _bounded(stderr_file.read(MAX_DIAGNOSTIC + 1)))
        except subprocess.TimeoutExpired as error:
            _kill_group(process.pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.05)
                try: os.killpg(process.pid, 0)
                except ProcessLookupError: break
            else: _kill_group(process.pid, signal.SIGKILL)
            process.communicate(); stdout_file.seek(0); stderr_file.seek(0)
            raise subprocess.TimeoutExpired(command, timeout,
                output=_bounded(stdout_file.read(MAX_DIAGNOSTIC + 1)),
                stderr=_bounded(stderr_file.read(MAX_DIAGNOSTIC + 1))) from error
        finally:
            _kill_group(process.pid, signal.SIGKILL)


def _kill_group(pid: int, sig: signal.Signals) -> None:
    try: os.killpg(pid, sig)
    except ProcessLookupError: pass


def _copy_regular(source: Path, destination: Path, workspace: Path) -> None:
    if stat.S_ISLNK(source.lstat().st_mode):
        raise SubmissionError(f"submission is a symlink: {source.name}")
    resolved = source.resolve(strict=True)
    if resolved == workspace or not resolved.is_relative_to(workspace):
        raise SubmissionError(f"submission escaped workspace: {source.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SubmissionError(f"submission is not a regular file: {source.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    finally:
        os.close(descriptor)
    destination.chmod(0o444)


def _restore_regular(source: Path, destination: Path, source_root: Path) -> None:
    temporary = destination.with_name(destination.name + ".controller-tmp")
    temporary.unlink(missing_ok=True)
    _copy_regular(source, temporary, source_root)
    os.replace(temporary, destination)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        owner: Controller = self.server.owner  # type: ignore[attr-defined]
        try:
            arguments = json.loads(self.rfile.readline())["arguments"]
            if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
                raise ValueError("arguments must be a string array")
            response = owner.call(arguments)
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as error:
            response = {"exit_code": 74, "stdout": "", "stderr": f"controller transport failed: {error}\n"}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class Controller:
    """Opaque one-request controller backed by the approved BZ client."""

    def __init__(self, socket_path: Path, client: Any, request: dict,
                 placement: dict, assets: dict, snapshot_root: Path,
                 deadline: float | None = None):
        self.socket_path, self.client, self.request = socket_path, client, request
        self.placement, self.assets = placement, assets
        self.snapshot_root = snapshot_root
        self.deadline = deadline or float("inf")
        self.limit = 2 if request.get("operation") == "two_shot" else 1
        self.used = self.invalid = self.over_budget = 0
        self.calls: list[dict] = []
        self.results: list[dict] = []
        self.free_calls: list[dict] = []
        self.last_result: dict | None = None
        self.lock = threading.Lock()
        self.server: socketserver.UnixStreamServer | None = None

    @staticmethod
    def _wire(document: dict, code: int = 0) -> dict:
        return {"exit_code": code, "stdout": json.dumps(document, sort_keys=True) + "\n", "stderr": ""}

    def call(self, arguments: list[str]) -> dict:
        with self.lock:
            if arguments == ["help"]:
                self.free_calls.append({"arguments": arguments})
                return self._wire({"status": "ok", "operation": "help",
                                   "usage": controller_help(self.limit), "billed": False})
            if arguments == ["budget"]:
                self.free_calls.append({"arguments": arguments})
                return self._wire({"status": "ok", "operation": "budget", "limit": self.limit,
                                   "used": self.used, "remaining": self.limit - self.used,
                                   "billed": False})
            valid = [check_command(round_number) for round_number in range(1, self.limit + 1)]
            if arguments not in valid:
                self.invalid += 1
                return self._wire({"status": "config_error", "diagnostics": "unsupported controller command",
                                   "usage": controller_help(self.limit), "billed": False}, 4)
            if self.used >= self.limit:
                self.over_budget += 1
                return {"exit_code": 75, "stdout": "", "stderr": "remote request budget exhausted\n"}
            expected = check_command(self.used + 1)
            if arguments != expected:
                self.invalid += 1
                return self._wire({"status": "config_error", "diagnostics": "unsupported controller command",
                                   "usage": controller_help(self.limit), "billed": False}, 4)
            self.used += 1
            self.calls.append({"arguments": arguments})
            try:
                response = self._development_check(self.used)
            except SubmissionError as error:
                self.last_result = {"status": "submission_error",
                                    "diagnostics": _bounded(error), "billed": True}
                response = self._wire(self.last_result, 2)
            except Exception as error:
                self.last_result = {
                    "status": "infrastructure_error", "failure_type": "controller_error",
                    "diagnostics": _bounded(f"controller failed: {type(error).__name__}: {error}"),
                    "handle": getattr(error, "handle", None),
                }
                response = self._wire(self.last_result, 74)
            assert self.last_result is not None
            self.results.append(json.loads(json.dumps(self.last_result)))
            return response

    def _development_check(self, round_number: int) -> dict:
        remote_timeout = (240 if self.deadline == float("inf") else
                          min(240, int(self.deadline - time.monotonic() - 20)))
        if remote_timeout < 26:
            raise SubmissionError("development check requested too late to complete")
        workspace = Path(self.request["workspace"])
        snapshot = self.snapshot_root / (f"development-check-{round_number}"
                                         if self.limit == 2 else "development-check")
        snapshot.mkdir(parents=True, exist_ok=False)
        for name in ("candidate.py", "candidate.manifest.json"):
            source = workspace / name
            if not source.is_file():
                self.last_result = {"status": "submission_error",
                    "diagnostics": f"missing {name}", "billed": True}
                return self._wire(self.last_result, 2)
            _copy_regular(source, snapshot / name, workspace)
        if round_number == 2:
            previous = self.snapshot_root / "development-check-1" / "candidate.py"
            if not previous.is_file():
                raise SubmissionError("round 1 candidate snapshot is missing")
            if _digest(previous) == _digest(snapshot / "candidate.py"):
                raise SubmissionError("round 2 must modify candidate.py")
        campaign = "agent-" + hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
        benchmark = self.request.get("benchmark", "matmul")
        if benchmark == "streaming-matmul-add":
            benchmark = "matmul"
        result = self.client.run({
            "campaign": campaign, "wave": self.request["wave"],
            "cell": f'{self.request["treatment"]}-round-{round_number}',
            **self.placement, "logical_device": 0, "timeout": remote_timeout,
            "candidate": str(snapshot / "candidate.py"),
            "candidate_manifest": str(snapshot / "candidate.manifest.json"),
            "baseline": self.assets["baseline"], "case_spec": self.assets["case_spec"],
            "runner": self.assets["runner"],
            **({"supplementary_assets": self.assets["supplementary"]}
               if self.assets.get("supplementary") else {}),
            "benchmark": benchmark,
            "cases": list(self.request.get("cases", [1])),
        })
        result["diagnostics"] = _bounded(result.get("diagnostics"))
        self.last_result = result
        return self._wire(result, 0 if result.get("status") == "ok" else 2)

    def finalize_outputs(self) -> None:
        with self.lock:
            available = ([self.snapshot_root / f"development-check-{round_number}"
                          for round_number in range(1, self.limit + 1)]
                         if self.limit == 2 else [self.snapshot_root / "development-check"])
            snapshot = next((path for path in reversed(available) if path.is_dir()), available[0])
            if (not self.used or not self.last_result
                    or not all((snapshot / name).is_file()
                               for name in ("candidate.py", "candidate.manifest.json"))):
                return
            workspace = Path(self.request["workspace"])
            try:
                for name in ("candidate.py", "candidate.manifest.json"):
                    try:
                        (workspace / name).lstat()
                    except FileNotFoundError as error:
                        raise SubmissionError(f"missing {name}") from error
                    probe = self.snapshot_root / (name + ".final-probe")
                    _copy_regular(workspace / name, probe, workspace)
                    probe.unlink()
            except SubmissionError as error:
                self.last_result = {"status": "submission_error",
                                    "diagnostics": _bounded(error), "billed": True}
            except Exception as error:
                self.last_result = {
                    "status": "infrastructure_error", "failure_type": "controller_error",
                    "diagnostics": _bounded(f"controller failed: {type(error).__name__}: {error}"),
                    "handle": getattr(error, "handle", None),
                }
            try:
                for name in ("candidate.py", "candidate.manifest.json"):
                    _restore_regular(snapshot / name, workspace / name, self.snapshot_root)
            except Exception as error:
                self.last_result = {
                    "status": "infrastructure_error", "failure_type": "controller_error",
                    "diagnostics": _bounded(f"controller failed: {type(error).__name__}: {error}"),
                    "handle": getattr(error, "handle", None),
                }

    def __enter__(self) -> "Controller":
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.server = socketserver.UnixStreamServer(str(self.socket_path), _Handler)
        self.server.owner = self  # type: ignore[attr-defined]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_: object) -> None:
        assert self.server
        self.server.shutdown(); self.server.server_close()
        self.socket_path.unlink(missing_ok=True)


def controller_client(socket_path: Path, arguments: list[str]) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
        response = json.loads(client.makefile("rb").readline())
    print(response.get("stdout", ""), end="")
    print(response.get("stderr", ""), end="", file=sys.stderr)
    return int(response["exit_code"])


class OneShotRunner:
    def __init__(self, *, skill_sources: dict[str, str], assets: dict[str, str], placements: dict,
                 client: BzA3DiagnosticClient, codex: str = "codex", bwrap: str = "bwrap",
                 auth_home: Path | None = None, sandbox_backend: str = "bubblewrap",
                 docker: str = "docker", docker_image: str | None = None,
                 docker_image_id: str | None = None):
        raw_skill_sources = {name: Path(path) for name, path in skill_sources.items()}
        self.skill_sources = {name: Path(path).resolve() for name, path in skill_sources.items()}
        self.assets, self.placements, self.client = assets, placements, client
        self.codex = Path(shutil.which(codex) or codex).resolve()
        self.bwrap = shutil.which(bwrap) or bwrap
        self.sandbox_backend = sandbox_backend
        self.docker = shutil.which(docker) or docker
        self.docker_image = docker_image
        self.docker_image_id = docker_image_id
        raw_auth_home = auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        self.auth_home = raw_auth_home.resolve()
        if sandbox_backend not in {"bubblewrap", "docker"}:
            raise RunnerError("sandbox backend must be bubblewrap or docker")
        if not self.codex.is_file() or not (self.auth_home / "auth.json").is_file():
            raise RunnerError("Codex and authenticated state are required")
        if sandbox_backend == "bubblewrap" and not Path(self.bwrap).is_file():
            raise RunnerError("Bubblewrap is required")
        if sandbox_backend == "docker":
            if any(path.is_symlink() for path in raw_skill_sources.values()):
                raise RunnerError("skill mount source must not be a symlink")
            if raw_auth_home.is_symlink():
                raise RunnerError("Codex auth mount source must not be a symlink")
            if not Path(self.docker).is_file() or not docker_image or not docker_image_id:
                raise RunnerError("Docker executable, image, and frozen image ID are required")
            inspected = subprocess.run(
                [self.docker, "image", "inspect", docker_image, "--format", "{{.Id}}"],
                text=True, capture_output=True, check=False,
            )
            if inspected.returncode or inspected.stdout.strip() != docker_image_id:
                raise RunnerError("local Docker image does not match frozen image ID")

    def _runtime(self) -> Path:
        for parent in self.codex.parents:
            if (parent / "bin/node").is_file() and (parent / "lib/node_modules").is_dir():
                return parent
        return self.codex.parent

    @staticmethod
    def _resolver_mounts() -> list[str]:
        resolv = Path("/etc/resolv.conf")
        if not resolv.is_symlink():
            if not resolv.is_file(): raise RunnerError("resolver configuration is unavailable")
            return []
        destination = Path(os.path.normpath(str(Path("/etc") / os.readlink(resolv))))
        allowed = Path("/run/systemd/resolve")
        if not destination.is_relative_to(allowed):
            raise RunnerError(f"unsupported resolver target: {destination}")
        source = resolv.resolve(strict=True)
        return ["--dir", "/run", "--dir", "/run/systemd", "--dir", str(allowed),
                "--ro-bind", str(source), str(destination)]

    def _validate(self, request: dict) -> tuple[Path, dict]:
        required = {"protocol_version", "operation", "workspace", "prompt", "prompt_sha256",
                    "model", "model_sha256", "skills", "skill_sha256", "controller_contract",
                    "wave", "attempt", "treatment", "cell_id"}
        operation = request.get("operation")
        if (not required.issubset(request) or request["protocol_version"] != 2
                or operation not in {"one_shot", "two_shot"}):
            raise RunnerError("invalid one-shot request")
        prompt = _prompt_bytes(request)
        model_hash = hashlib.sha256(json.dumps(request["model"], sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        if model_hash != request["model_sha256"]:
            raise RunnerError("frozen model changed")
        expected_contract = ({"billed_limit": 1, "command": CHECK}
                             if operation == "one_shot" else
                             {"billed_limit": 2,
                              "commands": [check_command(1), check_command(2)]})
        if request["controller_contract"] != expected_contract:
            raise RunnerError("controller contract changed")
        if operation == "two_shot":
            cases = request.get("cases")
            if (not isinstance(cases, list) or not cases
                    or any(isinstance(case, bool) or not isinstance(case, int) or case < 0
                           for case in cases)):
                raise RunnerError("two-shot cases must be non-empty nonnegative integers")
            if request.get("benchmark") not in {"matmul", "bsa", "gdn"}:
                raise RunnerError("two-shot benchmark must be matmul, bsa, or gdn")
        _validated_supplementary_assets(self.assets)
        if set(request["skills"]) != set(request["skill_sha256"]):
            raise RunnerError("skill inventory and hashes differ")
        from campaign import TREATMENT_SKILLS
        if request["skills"] != list(TREATMENT_SKILLS.get(request["treatment"], ())):
            raise RunnerError("noncanonical treatment skill inventory")
        for name in request["skills"]:
            source = self.skill_sources.get(name)
            if source is None or not source.is_dir() or _digest(source) != request["skill_sha256"][name]:
                raise RunnerError(f"frozen skill changed: {name}")
        raw_workspace = Path(request["workspace"])
        if self.sandbox_backend == "docker" and raw_workspace.is_symlink():
            raise RunnerError("workspace mount source must not be a symlink")
        workspace = raw_workspace.resolve()
        if not workspace.is_dir() or any(workspace.iterdir()):
            raise RunnerError("a fresh empty workspace is required")
        choices = self.placements.get(request["treatment"])
        if not isinstance(choices, list) or len(choices) != 2:
            raise RunnerError("treatment placement is unavailable")
        return workspace, choices[min(int(request["attempt"]) - 1, 1)], prompt

    @staticmethod
    def _mount_source(path: Path, *, kind: str, within: Path | None = None) -> Path:
        """Resolve one allowlisted bind without following a caller-controlled link."""
        if path.is_symlink():
            raise RunnerError(f"{kind} mount source must not be a symlink")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise RunnerError(f"{kind} mount source is unavailable") from error
        if within is not None and not resolved.is_relative_to(within.resolve(strict=True)):
            raise RunnerError(f"{kind} mount source escaped its expected root")
        if any(character in str(resolved) for character in (",", "\n", "\r")):
            raise RunnerError(f"{kind} mount source contains unsupported characters")
        return resolved

    @staticmethod
    def _docker_mount(source: Path, destination: str, *, readonly: bool) -> list[str]:
        value = f"type=bind,src={source},dst={destination}"
        if readonly:
            value += ",readonly"
        return ["--mount", value]

    def _docker_command(self, workspace: Path, prompt: Path, socket_dir: Path,
                        shim: Path, state: Path, request: dict, container_name: str) -> list[str]:
        """Construct the complete allowlisted Docker boundary for one cell."""
        workspace = self._mount_source(workspace, kind="workspace")
        prompt = self._mount_source(prompt, kind="prompt")
        auth = self._mount_source(self.auth_home / "auth.json", kind="Codex auth")
        runtime = self._mount_source(self._runtime(), kind="Codex runtime")
        shim = self._mount_source(shim, kind="controller client")
        socket_dir = self._mount_source(socket_dir, kind="controller socket parent")
        state = self._mount_source(state, kind="Codex state")
        uid, gid = os.getuid(), os.getgid()
        command = [
            self.docker, "run", "--rm", "--interactive", "--name", container_name,
            "--user", f"{uid}:{gid}", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--memory", "8g", "--tmpfs", f"/tmp:rw,nosuid,nodev,mode=1777,uid={uid},gid={gid}",
            "--tmpfs", f"/home/agent:rw,nosuid,nodev,mode=700,uid={uid},gid={gid}",
            "--workdir", "/workspace", "--env", "HOME=/home/agent",
            "--env", "CODEX_HOME=/codex-home",
            "--env", "PATH=/runtime/node/bin:/usr/local/bin:/usr/bin:/bin",
            "--env", ("EXPERIMENT_CONTROLLER=/usr/local/bin/python3 "
                      "/experiment/controller-client.py /experiment-state/controller.sock"),
            *self._docker_mount(workspace, "/workspace", readonly=False),
            *self._docker_mount(state, "/codex-home", readonly=False),
            *self._docker_mount(prompt, "/experiment/PROMPT.md", readonly=True),
            *self._docker_mount(auth, "/codex-home/auth.json", readonly=True),
            *self._docker_mount(runtime, "/runtime/node", readonly=True),
            *self._docker_mount(shim, "/experiment/controller-client.py", readonly=True),
            *self._docker_mount(socket_dir, "/experiment-state", readonly=False),
        ]
        skill_root = workspace / ".agents" / "skills"
        for name in request["skills"]:
            source = self._mount_source(self.skill_sources[name], kind=f"skill {name}")
            target = self._mount_source(skill_root / name, kind=f"skill target {name}",
                                        within=workspace)
            command += self._docker_mount(source, f"/workspace/.agents/skills/{name}", readonly=True)
            assert target.is_dir()
        command.append(str(self.docker_image_id))
        return command

    def _remove_container(self, name: str) -> None:
        subprocess.run([self.docker, "rm", "-f", name], text=True,
                       capture_output=True, check=False, timeout=30)

    def _sandbox_evidence(self) -> dict:
        evidence = {"backend": self.sandbox_backend}
        if self.sandbox_backend == "docker":
            evidence.update(image=self.docker_image, image_id=self.docker_image_id)
        return evidence

    @staticmethod
    def _session_id(output: str) -> str:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                return event["thread_id"]
        raise RunnerError("Codex did not report a persistent thread id")

    def run(self, request: dict, timeout: int = 360) -> dict:
        started = time.monotonic(); milestones = [{"name": "request_validating", "elapsed_seconds": 0.0}]
        state = socket_dir = snapshot_dir = shim_dir = None
        try:
            workspace, placement, prompt = self._validate(request)
            for name, source in (("baseline.py", self.assets["baseline"]),
                                 ("cases.jsonl", self.assets["case_spec"])):
                shutil.copy2(source, workspace / name)
            supplementary = _validated_supplementary_assets(self.assets)
            for name, source in supplementary.items():
                shutil.copy2(source, workspace / name)
            if "baseline.json" not in supplementary:
                shutil.copy2(self.assets["case_spec"], workspace / "baseline.json")
            two_shot = request["operation"] == "two_shot"
            (workspace / "AGENTS.md").write_text(
                "Write candidate.py and candidate.manifest.json. Use only declared local skills. "
                f"You have {'two turns and one billed development check per turn' if two_shot else 'one turn and exactly one billed controller check'}. "
                "Do not profile or pursue broad optimization.\n"
            )
            skill_mounts: list[str] = []
            skill_root = workspace / ".agents/skills"; skill_root.mkdir(parents=True)
            for name in request["skills"]:
                target = skill_root / name; target.mkdir()
                skill_mounts += ["--ro-bind", str(self.skill_sources[name]), f"/workspace/.agents/skills/{name}"]
            state = Path(tempfile.mkdtemp(prefix="oneshot-codex-")); (state / "auth.json").touch(mode=0o600)
            socket_dir = Path(tempfile.mkdtemp(prefix="oneshot-ctl-")); socket_path = socket_dir / "controller.sock"
            snapshot_dir = Path(tempfile.mkdtemp(prefix="oneshot-snapshot-"))
            shim_dir = Path(tempfile.mkdtemp(prefix="oneshot-shim-"))
            shim = shim_dir / "controller-client.py"; shim.write_text(CONTROLLER_CLIENT); shim.chmod(0o444)
            prompt_file = shim_dir / "PROMPT.md"; prompt_file.write_bytes(prompt); prompt_file.chmod(0o444)
            runtime = self._runtime()
            container_name = f"triton-one-shot-{uuid.uuid4().hex}"
            if self.sandbox_backend == "bubblewrap":
                command = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--share-net", "--clearenv",
                           "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
                for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
                    if Path(source).exists(): command += ["--ro-bind", source, source]
                command += self._resolver_mounts()
                command += ["--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime", "--dir", "/experiment",
                            "--bind", str(workspace), "/workspace", *skill_mounts,
                            "--bind", str(state), "/codex-home", "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
                            "--ro-bind", str(runtime), "/runtime/node", "--ro-bind", str(shim), "/experiment/controller-client.py",
                            "--bind", str(socket_dir), "/experiment-state", "--chdir", "/workspace",
                            "--setenv", "HOME", "/home/agent", "--setenv", "CODEX_HOME", "/codex-home",
                            "--setenv", "PATH", "/runtime/node/bin:/usr/bin:/bin", "--setenv", "EXPERIMENT_CONTROLLER",
                            "/usr/bin/python3 /experiment/controller-client.py /experiment-state/controller.sock"]
            else:
                command = self._docker_command(workspace, prompt_file, socket_dir,
                                               shim, state, request, container_name)
            codex = ["/runtime/node/bin/codex", "exec", "--json", "--ignore-user-config", "--skip-git-repo-check",
                     "--dangerously-bypass-approvals-and-sandbox", "-m", request["model"]["name"],
                     "-c", f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"', "-C", "/workspace", "-"]
            milestones.append({"name": "sandbox_ready", "elapsed_seconds": time.monotonic() - started})
            with Controller(socket_path, self.client, request, placement, self.assets,
                            snapshot_dir, started + timeout) as controller:
                previous_handlers: dict[signal.Signals, Any] = {}
                if (self.sandbox_backend == "docker"
                        and threading.current_thread() is threading.main_thread()):
                    def cancelled(_signum: int, _frame: Any) -> None:
                        raise KeyboardInterrupt
                    for watched in (signal.SIGTERM, signal.SIGHUP):
                        previous_handlers[watched] = signal.signal(watched, cancelled)
                runs: list[subprocess.CompletedProcess] = []
                try:
                    first = _run_group([*command, *codex], prompt.decode(), timeout)
                    runs.append(first)
                    if (two_shot and first.returncode == 0 and controller.used == 1
                            and controller.last_result is not None
                            and controller.last_result.get("status") != "infrastructure_error"):
                        session_id = self._session_id(first.stdout)
                        resume = ["/runtime/node/bin/codex", "exec", "resume", "--json",
                                  "--ignore-user-config", "--dangerously-bypass-approvals-and-sandbox",
                                  "-m", request["model"]["name"],
                                  "-c", f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"',
                                  session_id, "-"]
                        second_prompt = (
                            "Continue the same smoke test. Inspect the Round 1 diagnostics, modify "
                            "candidate.py, run exactly `check --scope development --round 2`, then "
                            "stop. Do not profile or pursue broad optimization.\n")
                        remaining = max(1, int(started + timeout - time.monotonic()))
                        runs.append(_run_group([*command, *resume], second_prompt, remaining))
                    run = runs[-1]
                except subprocess.TimeoutExpired as error:
                    if self.sandbox_backend == "docker":
                        self._remove_container(container_name)
                    controller.finalize_outputs()
                    if (controller.last_result
                            and controller.last_result.get("status") in {
                                "infrastructure_error", "submission_error"}):
                        result = {"status": controller.last_result["status"],
                                "diagnostics": controller.last_result.get("diagnostics", ""),
                                "controller_result": controller.last_result,
                                "agent_timeout": _bounded(error.stderr), "milestones": milestones,
                                "controller_usage": {"billed": controller.used,
                                                     "calls": controller.calls}}
                        if result["status"] == "infrastructure_error":
                            result.update(failure_type=controller.last_result.get(
                                              "failure_type", "controller_infrastructure"),
                                          handle=controller.last_result.get("handle"))
                        result["sandbox"] = self._sandbox_evidence()
                        result["controller_results"] = controller.results
                        return result
                    return {"status": "timeout", "diagnostics": _bounded(error.stderr), "milestones": milestones,
                            "controller_usage": {"billed": controller.used, "calls": controller.calls},
                            "controller_results": controller.results,
                            "sandbox": self._sandbox_evidence()}
                except KeyboardInterrupt:
                    if self.sandbox_backend == "docker":
                        self._remove_container(container_name)
                    raise
                finally:
                    for watched, previous in previous_handlers.items():
                        signal.signal(watched, previous)
                controller.finalize_outputs()
            milestones.append({"name": "agent_finished", "elapsed_seconds": time.monotonic() - started})
            output = (run.stdout + run.stderr).lower()
            if (controller.last_result
                    and controller.last_result.get("status") in {
                        "infrastructure_error", "submission_error"}):
                status = controller.last_result["status"]
            elif run.returncode == 0:
                status = "ok"
            elif any(x in output for x in ("rate limit", "service unavailable", "connection", "api error")):
                status = "model_service_error"
            elif any(x in output for x in ("bwrap:", "bubblewrap", "namespace", "docker:")):
                status = "setup_error"
            else:
                status = "protocol_error"
            result = {"status": status, "codex_exit_code": run.returncode,
                    "stdout": _bounded(run.stdout),
                    "stderr": _bounded(run.stderr), "milestones": milestones,
                    "sandbox": self._sandbox_evidence(),
                    "rounds_completed": sum(item.returncode == 0 for item in runs),
                    "turns": [{"round": index, "exit_code": item.returncode,
                               "stdout": _bounded(item.stdout), "stderr": _bounded(item.stderr)}
                              for index, item in enumerate(runs, 1)],
                    "controller_usage": {"limit": controller.limit, "billed": controller.used, "calls": controller.calls,
                                         "free_calls": controller.free_calls, "invalid": controller.invalid,
                                         "over_budget": controller.over_budget}}
            candidate_hashes = {}
            for round_number in range(1, controller.limit + 1):
                directory = (f"development-check-{round_number}"
                             if controller.limit == 2 else "development-check")
                checked = snapshot_dir / directory / "candidate.py"
                if checked.is_file():
                    candidate_hashes[str(round_number)] = _digest(checked)
            result["candidate_sha256"] = candidate_hashes
            result["controller_results"] = controller.results
            if status == "infrastructure_error":
                result.update(failure_type=controller.last_result.get("failure_type", "controller_infrastructure"),
                              diagnostics=controller.last_result.get("diagnostics", ""),
                              handle=controller.last_result.get("handle"),
                              controller_result=controller.last_result)
            elif status == "submission_error":
                result.update(diagnostics=controller.last_result.get("diagnostics", ""),
                              controller_result=controller.last_result)
            return result
        except (RunnerError, OSError, KeyError, TypeError, ValueError) as error:
            return {"status": "setup_error", "diagnostics": _bounded(error), "milestones": milestones,
                    "sandbox": self._sandbox_evidence(),
                    "controller_usage": {"limit": (2 if request.get("operation") == "two_shot" else 1),
                                         "billed": 0, "calls": []}}
        finally:
            for temporary in (state, socket_dir, snapshot_dir, shim_dir):
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise RunnerError(f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    client_parser = sub.add_parser("controller-client")
    client_parser.add_argument("socket", type=Path); client_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parser.add_argument("--skill-sources", type=Path); parser.add_argument("--assets", type=Path)
    parser.add_argument("--placements", type=Path); parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--remote-json", default='["cpl-remote"]')
    parser.add_argument("--sandbox-backend", choices=("bubblewrap", "docker"), default="bubblewrap")
    parser.add_argument("--docker", default="docker"); parser.add_argument("--docker-image")
    parser.add_argument("--docker-image-id")
    args = parser.parse_args()
    if args.command == "controller-client": return controller_client(args.socket, args.arguments)
    try:
        if not all((args.skill_sources, args.assets, args.placements, args.state_dir)):
            raise RunnerError("runner configuration is incomplete")
        remote = json.loads(args.remote_json)
        from bz_a3_diagnostic_client import RemoteTransport, BzA3DiagnosticClient
        runner = OneShotRunner(skill_sources=_load(args.skill_sources), assets=_load(args.assets),
            placements=_load(args.placements), client=BzA3DiagnosticClient(
                RemoteTransport(remote), args.state_dir),
            sandbox_backend=args.sandbox_backend, docker=args.docker,
            docker_image=args.docker_image, docker_image_id=args.docker_image_id)
        request = json.load(sys.stdin)
        result = runner.run(
            request, timeout=(INNER_TURN_TIMEOUT * 2 if request.get("operation") == "two_shot"
                              else INNER_TURN_TIMEOUT))
    except (RunnerError, OSError, ValueError, json.JSONDecodeError) as error:
        result = {"status": "setup_error", "diagnostics": _bounded(error),
                  "controller_usage": {"limit": 1, "billed": 0, "calls": []}}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
