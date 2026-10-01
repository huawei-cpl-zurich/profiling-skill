#!/usr/bin/env python3
"""Run one frozen diagnostic Codex turn in an explicit outer sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


CHECK = ["check", "--scope", "development", "--round", "1"]
HELP = "usage:\n  $EXPERIMENT_CONTROLLER help\n  $EXPERIMENT_CONTROLLER budget\n  $EXPERIMENT_CONTROLLER check --scope development --round 1\n"
MAX_DIAGNOSTIC = 64 * 1024


class RunnerError(RuntimeError):
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
                 placement: dict, assets: dict):
        self.socket_path, self.client, self.request = socket_path, client, request
        self.placement, self.assets = placement, assets
        self.used = self.invalid = self.over_budget = 0
        self.calls: list[dict] = []
        self.free_calls: list[dict] = []
        self.results: list[dict] = []
        self.lock = threading.Lock()
        self.server: socketserver.UnixStreamServer | None = None

    @staticmethod
    def _wire(document: dict, code: int = 0) -> dict:
        return {"exit_code": code, "stdout": json.dumps(document, sort_keys=True) + "\n", "stderr": ""}

    def call(self, arguments: list[str]) -> dict:
        with self.lock:
            if arguments == ["help"]:
                self.free_calls.append({"arguments": arguments})
                return self._wire({"status": "ok", "operation": "help", "usage": HELP, "billed": False})
            if arguments == ["budget"]:
                self.free_calls.append({"arguments": arguments})
                return self._wire({"status": "ok", "operation": "budget", "limit": 1,
                                   "used": self.used, "remaining": 1 - self.used, "billed": False})
            if arguments != CHECK:
                self.invalid += 1
                return self._wire({"status": "config_error", "diagnostics": "unsupported controller command",
                                   "usage": HELP, "billed": False}, 4)
            if self.used:
                self.over_budget += 1
                return {"exit_code": 75, "stdout": "", "stderr": "remote request budget exhausted\n"}
            self.used = 1
            self.calls.append({"arguments": arguments})
            workspace = Path(self.request["workspace"])
            campaign = "agent-" + hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
            result = self.client.run({
                "campaign": campaign, "wave": self.request["wave"], "cell": self.request["treatment"],
                **self.placement, "logical_device": 0, "timeout": 240,
                "candidate": str(workspace / "candidate.py"),
                "candidate_manifest": str(workspace / "candidate.manifest.json"),
                "baseline": self.assets["baseline"], "case_spec": self.assets["case_spec"],
                "runner": self.assets["runner"], "cases": [1],
            })
            result["diagnostics"] = _bounded(result.get("diagnostics"))
            self.results.append(result.copy())
            code = 0 if result.get("status") == "ok" else 2
            return self._wire(result, code)

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
        if not required.issubset(request) or request["protocol_version"] != 1 or request["operation"] != "one_shot":
            raise RunnerError("invalid one-shot request")
        prompt = Path(request["prompt"])
        model_hash = hashlib.sha256(json.dumps(request["model"], sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        if not prompt.is_file() or _digest(prompt) != request["prompt_sha256"] or model_hash != request["model_sha256"]:
            raise RunnerError("frozen prompt or model changed")
        if request["controller_contract"] != {"billed_limit": 1, "command": CHECK}:
            raise RunnerError("controller contract changed")
        if set(request["skills"]) != set(request["skill_sha256"]):
            raise RunnerError("skill inventory and hashes differ")
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
        return workspace, choices[min(int(request["attempt"]) - 1, 1)]

    @staticmethod
    def _mount_source(path: Path, *, kind: str, within: Path | None = None) -> Path:
        """Validate a bind source without silently following a caller-controlled link."""
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
                        request: dict, container_name: str) -> list[str]:
        """Construct the complete allowlisted Docker boundary for one cell."""
        workspace = self._mount_source(workspace, kind="workspace")
        prompt = self._mount_source(prompt, kind="prompt")
        auth = self._mount_source(self.auth_home / "auth.json", kind="Codex auth")
        runtime = self._mount_source(self._runtime(), kind="Codex runtime")
        script = self._mount_source(Path(__file__), kind="controller client")
        socket_dir = self._mount_source(socket_dir, kind="controller socket parent")
        uid, gid = os.getuid(), os.getgid()
        command = [
            self.docker, "run", "--rm", "--interactive", "--name", container_name,
            "--user", f"{uid}:{gid}", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--memory", "8g", "--tmpfs", f"/tmp:rw,nosuid,nodev,mode=1777,uid={uid},gid={gid}",
            "--tmpfs", f"/codex-home:rw,nosuid,nodev,mode=700,uid={uid},gid={gid}",
            "--tmpfs", f"/home/agent:rw,nosuid,nodev,mode=700,uid={uid},gid={gid}",
            "--workdir", "/workspace", "--env", "HOME=/home/agent",
            "--env", "CODEX_HOME=/codex-home",
            "--env", "PATH=/runtime/node/bin:/usr/local/bin:/usr/bin:/bin",
            "--env", ("EXPERIMENT_CONTROLLER=/usr/local/bin/python3 "
                      "/experiment/runner.py controller-client /experiment-state/controller.sock"),
            *self._docker_mount(workspace, "/workspace", readonly=False),
            *self._docker_mount(prompt, "/experiment/PROMPT.md", readonly=True),
            *self._docker_mount(auth, "/codex-home/auth.json", readonly=True),
            *self._docker_mount(runtime, "/runtime/node", readonly=True),
            *self._docker_mount(script, "/experiment/runner.py", readonly=True),
            *self._docker_mount(script, "/usr/local/bin/controller", readonly=True),
            *self._docker_mount(socket_dir, "/experiment-state", readonly=False),
        ]
        skill_root = workspace / ".agents" / "skills"
        for name in request["skills"]:
            source = self._mount_source(self.skill_sources[name], kind=f"skill {name}")
            target = self._mount_source(skill_root / name, kind=f"skill target {name}",
                                        within=workspace)
            command += self._docker_mount(source, f"/workspace/.agents/skills/{name}", readonly=True)
            # The empty target is deliberately shadowed by the read-only source.
            assert target.is_dir()
        command.append(str(self.docker_image))
        return command

    def _remove_container(self, name: str) -> None:
        subprocess.run([self.docker, "rm", "-f", name], text=True,
                       capture_output=True, check=False, timeout=30)

    def run(self, request: dict, timeout: int = 360) -> dict:
        started = time.monotonic(); milestones = [{"name": "request_validating", "elapsed_seconds": 0.0}]
        state = socket_dir = None
        try:
            workspace, placement = self._validate(request)
            for name, source in (("baseline.py", self.assets["baseline"]),
                                 ("cases.jsonl", self.assets["case_spec"])):
                shutil.copy2(source, workspace / name)
            (workspace / "AGENTS.md").write_text(
                "Write candidate.py and candidate.manifest.json. Use only declared local skills. "
                "You have one turn and exactly one billed controller check. Do not profile.\n"
            )
            skill_mounts: list[str] = []
            skill_root = workspace / ".agents/skills"; skill_root.mkdir(parents=True)
            for name in request["skills"]:
                target = skill_root / name; target.mkdir()
                skill_mounts += ["--ro-bind", str(self.skill_sources[name]), f"/workspace/.agents/skills/{name}"]
            state = Path(tempfile.mkdtemp(prefix="oneshot-codex-")); (state / "auth.json").touch(mode=0o600)
            socket_dir = Path(tempfile.mkdtemp(prefix="oneshot-ctl-")); socket_path = socket_dir / "controller.sock"
            script = Path(__file__).resolve(); runtime = self._runtime()
            container_name = f"triton-one-shot-{uuid.uuid4().hex}"
            if self.sandbox_backend == "bubblewrap":
                command = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
                           "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
                for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
                    if Path(source).exists(): command += ["--ro-bind", source, source]
                command += self._resolver_mounts()
                command += ["--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime", "--dir", "/experiment",
                            "--bind", str(workspace), "/workspace", *skill_mounts,
                            "--bind", str(state), "/codex-home", "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
                            "--ro-bind", str(runtime), "/runtime/node", "--ro-bind", str(script), "/experiment/runner.py",
                            "--bind", str(socket_dir), "/experiment-state", "--chdir", "/workspace",
                            "--setenv", "HOME", "/home/agent", "--setenv", "CODEX_HOME", "/codex-home",
                            "--setenv", "PATH", "/runtime/node/bin:/usr/bin:/bin", "--setenv", "EXPERIMENT_CONTROLLER",
                            "/usr/bin/python3 /experiment/runner.py controller-client /experiment-state/controller.sock"]
            else:
                command = self._docker_command(workspace, Path(request["prompt"]), socket_dir,
                                               request, container_name)
            codex = ["/runtime/node/bin/codex", "exec", "--json", "--ignore-user-config", "--skip-git-repo-check",
                     "--dangerously-bypass-approvals-and-sandbox", "-m", request["model"]["name"],
                     "-c", f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"', "-C", "/workspace", "-"]
            milestones.append({"name": "sandbox_ready", "elapsed_seconds": time.monotonic() - started})
            with Controller(socket_path, self.client, request, placement, self.assets) as controller:
                try:
                    run = subprocess.run([*command, *codex], input=Path(request["prompt"]).read_text(), text=True,
                                         capture_output=True, timeout=timeout, check=False)
                except subprocess.TimeoutExpired as error:
                    if self.sandbox_backend == "docker":
                        self._remove_container(container_name)
                    return {"status": "timeout", "diagnostics": _bounded(error.stderr), "milestones": milestones,
                            "controller_usage": {"billed": controller.used, "calls": controller.calls},
                            "sandbox": self._sandbox_evidence()}
                except KeyboardInterrupt:
                    if self.sandbox_backend == "docker":
                        self._remove_container(container_name)
                    raise
            milestones.append({"name": "agent_finished", "elapsed_seconds": time.monotonic() - started})
            output = (run.stdout + run.stderr).lower()
            if run.returncode == 0:
                status = "ok"
            elif any(x in output for x in ("rate limit", "service unavailable", "connection", "api error")):
                status = "model_service_error"
            elif any(x in output for x in ("bwrap:", "bubblewrap", "namespace", "docker:")):
                status = "setup_error"
            else:
                status = "protocol_error"
            return {"status": status, "codex_exit_code": run.returncode, "stdout": _bounded(run.stdout),
                    "stderr": _bounded(run.stderr), "milestones": milestones,
                    "sandbox": self._sandbox_evidence(),
                    "controller_evidence": controller.results,
                    "controller_usage": {"limit": 1, "billed": controller.used, "calls": controller.calls,
                                         "free_calls": controller.free_calls, "invalid": controller.invalid,
                                         "over_budget": controller.over_budget}}
        except (RunnerError, OSError, KeyError, TypeError, ValueError) as error:
            return {"status": "setup_error", "diagnostics": _bounded(error), "milestones": milestones,
                    "controller_usage": {"limit": 1, "billed": 0, "calls": []}}
        finally:
            for temporary in (state, socket_dir):
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)

    def _sandbox_evidence(self) -> dict:
        evidence = {"backend": self.sandbox_backend}
        if self.sandbox_backend == "docker":
            evidence.update(image=self.docker_image, image_id=self.docker_image_id)
        return evidence


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise RunnerError(f"{path} must contain an object")
    return value


def main() -> int:
    if Path(sys.argv[0]).name == "controller":
        return controller_client(Path("/experiment-state/controller.sock"), sys.argv[1:])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    client_parser = sub.add_parser("controller-client")
    client_parser.add_argument("socket", type=Path); client_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parser.add_argument("--skill-sources", type=Path); parser.add_argument("--assets", type=Path)
    parser.add_argument("--placements", type=Path); parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--remote-json", default='["cpl-remote"]'); parser.add_argument("--adapter-json")
    parser.add_argument("--sandbox-backend", choices=("bubblewrap", "docker"), default="bubblewrap")
    parser.add_argument("--docker", default="docker"); parser.add_argument("--docker-image")
    parser.add_argument("--docker-image-id")
    args = parser.parse_args()
    if args.command == "controller-client": return controller_client(args.socket, args.arguments)
    try:
        if not all((args.skill_sources, args.assets, args.placements, args.state_dir, args.adapter_json)):
            raise RunnerError("runner configuration is incomplete")
        remote, adapter = json.loads(args.remote_json), json.loads(args.adapter_json)
        from bz_a3_diagnostic_client import AdapterTransport, BzA3DiagnosticClient
        runner = OneShotRunner(skill_sources=_load(args.skill_sources), assets=_load(args.assets),
            placements=_load(args.placements), client=BzA3DiagnosticClient(
                AdapterTransport(remote, adapter), args.state_dir),
            sandbox_backend=args.sandbox_backend, docker=args.docker,
            docker_image=args.docker_image, docker_image_id=args.docker_image_id)
        request = json.load(sys.stdin); result = runner.run(request)
    except (RunnerError, OSError, ValueError, json.JSONDecodeError) as error:
        result = {"status": "setup_error", "diagnostics": _bounded(error),
                  "controller_usage": {"limit": 1, "billed": 0, "calls": []}}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
