#!/usr/bin/env python3
"""Run one frozen diagnostic Codex turn in an outer Bubblewrap sandbox."""

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


def _run_group(command: list[str], prompt: str, timeout: int) -> subprocess.CompletedProcess:
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as error:
        _kill_group(process.pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.05)
            try: os.killpg(process.pid, 0)
            except ProcessLookupError: break
        else: _kill_group(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout,
                                        stderr=stderr) from error
    finally:
        _kill_group(process.pid, signal.SIGKILL)


def _kill_group(pid: int, sig: signal.Signals) -> None:
    try: os.killpg(pid, sig)
    except ProcessLookupError: pass


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
                 placement: dict, assets: dict, snapshot_root: Path):
        self.socket_path, self.client, self.request = socket_path, client, request
        self.placement, self.assets = placement, assets
        self.snapshot_root = snapshot_root
        self.used = self.invalid = self.over_budget = 0
        self.calls: list[dict] = []
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
            snapshot = self.snapshot_root / "development-check"
            snapshot.mkdir(parents=True, exist_ok=False)
            for name in ("candidate.py", "candidate.manifest.json"):
                source = workspace / name
                if not source.is_file():
                    return self._wire({"status": "submission_error",
                        "diagnostics": f"missing {name}", "billed": True}, 2)
                destination = snapshot / name
                shutil.copyfile(source, destination); destination.chmod(0o444)
            campaign = "agent-" + hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
            result = self.client.run({
                "campaign": campaign, "wave": self.request["wave"], "cell": self.request["treatment"],
                **self.placement, "logical_device": 0, "timeout": 240,
                "candidate": str(snapshot / "candidate.py"),
                "candidate_manifest": str(snapshot / "candidate.manifest.json"),
                "baseline": self.assets["baseline"], "case_spec": self.assets["case_spec"],
                "runner": self.assets["runner"], "cases": [1],
            })
            result["diagnostics"] = _bounded(result.get("diagnostics"))
            self.last_result = result
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
                 auth_home: Path | None = None):
        self.skill_sources = {name: Path(path).resolve() for name, path in skill_sources.items()}
        self.assets, self.placements, self.client = assets, placements, client
        self.codex = Path(shutil.which(codex) or codex).resolve()
        self.bwrap = shutil.which(bwrap) or bwrap
        self.auth_home = (auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))).resolve()
        if not self.codex.is_file() or not Path(self.bwrap).is_file() or not (self.auth_home / "auth.json").is_file():
            raise RunnerError("Bubblewrap, Codex, and authenticated state are required")

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
        if not required.issubset(request) or request["protocol_version"] != 2 or request["operation"] != "one_shot":
            raise RunnerError("invalid one-shot request")
        prompt = _prompt_bytes(request)
        model_hash = hashlib.sha256(json.dumps(request["model"], sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        if model_hash != request["model_sha256"]:
            raise RunnerError("frozen model changed")
        if request["controller_contract"] != {"billed_limit": 1, "command": CHECK}:
            raise RunnerError("controller contract changed")
        if set(request["skills"]) != set(request["skill_sha256"]):
            raise RunnerError("skill inventory and hashes differ")
        from campaign import TREATMENT_SKILLS
        if request["skills"] != list(TREATMENT_SKILLS.get(request["treatment"], ())):
            raise RunnerError("noncanonical treatment skill inventory")
        for name in request["skills"]:
            source = self.skill_sources.get(name)
            if source is None or not source.is_dir() or _digest(source) != request["skill_sha256"][name]:
                raise RunnerError(f"frozen skill changed: {name}")
        workspace = Path(request["workspace"]).resolve()
        if not workspace.is_dir() or any(workspace.iterdir()):
            raise RunnerError("a fresh empty workspace is required")
        choices = self.placements.get(request["treatment"])
        if not isinstance(choices, list) or len(choices) != 2:
            raise RunnerError("treatment placement is unavailable")
        return workspace, choices[min(int(request["attempt"]) - 1, 1)], prompt

    def run(self, request: dict, timeout: int = 360) -> dict:
        started = time.monotonic(); milestones = [{"name": "request_validating", "elapsed_seconds": 0.0}]
        state = socket_dir = None
        try:
            workspace, placement, prompt = self._validate(request)
            for name, source in (("baseline.py", self.assets["baseline"]),
                                 ("cases.jsonl", self.assets["case_spec"])):
                shutil.copy2(source, workspace / name)
            shutil.copy2(self.assets["case_spec"], workspace / "baseline.json")
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
            codex = ["/runtime/node/bin/codex", "exec", "--json", "--ignore-user-config", "--skip-git-repo-check",
                     "--dangerously-bypass-approvals-and-sandbox", "-m", request["model"]["name"],
                     "-c", f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"', "-C", "/workspace", "-"]
            milestones.append({"name": "sandbox_ready", "elapsed_seconds": time.monotonic() - started})
            with Controller(socket_path, self.client, request, placement, self.assets,
                            socket_dir / "snapshots") as controller:
                try:
                    run = _run_group([*command, *codex], prompt.decode(), timeout)
                except subprocess.TimeoutExpired as error:
                    if (controller.last_result
                            and controller.last_result.get("status") == "infrastructure_error"):
                        return {"status": "infrastructure_error",
                                "failure_type": controller.last_result.get(
                                    "failure_type", "controller_infrastructure"),
                                "diagnostics": controller.last_result.get("diagnostics", ""),
                                "handle": controller.last_result.get("handle"),
                                "controller_result": controller.last_result,
                                "agent_timeout": _bounded(error.stderr), "milestones": milestones,
                                "controller_usage": {"billed": controller.used,
                                                     "calls": controller.calls}}
                    return {"status": "timeout", "diagnostics": _bounded(error.stderr), "milestones": milestones,
                            "controller_usage": {"billed": controller.used, "calls": controller.calls}}
            milestones.append({"name": "agent_finished", "elapsed_seconds": time.monotonic() - started})
            output = (run.stdout + run.stderr).lower()
            if controller.last_result and controller.last_result.get("status") == "infrastructure_error":
                status = "infrastructure_error"
            elif run.returncode == 0:
                status = "ok"
            elif any(x in output for x in ("rate limit", "service unavailable", "connection", "api error")):
                status = "model_service_error"
            elif any(x in output for x in ("bwrap:", "bubblewrap", "namespace")):
                status = "setup_error"
            else:
                status = "protocol_error"
            result = {"status": status, "codex_exit_code": run.returncode,
                    "stdout": _bounded(run.stdout),
                    "stderr": _bounded(run.stderr), "milestones": milestones,
                    "controller_usage": {"limit": 1, "billed": controller.used, "calls": controller.calls,
                                         "free_calls": controller.free_calls, "invalid": controller.invalid,
                                         "over_budget": controller.over_budget}}
            if status == "infrastructure_error":
                result.update(failure_type=controller.last_result.get("failure_type", "controller_infrastructure"),
                              diagnostics=controller.last_result.get("diagnostics", ""),
                              handle=controller.last_result.get("handle"),
                              controller_result=controller.last_result)
            return result
        except (RunnerError, OSError, KeyError, TypeError, ValueError) as error:
            return {"status": "setup_error", "diagnostics": _bounded(error), "milestones": milestones,
                    "controller_usage": {"limit": 1, "billed": 0, "calls": []}}
        finally:
            for temporary in (state, socket_dir):
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
    parser.add_argument("--remote-json", default='["cpl-remote"]'); parser.add_argument("--adapter-json")
    args = parser.parse_args()
    if args.command == "controller-client": return controller_client(args.socket, args.arguments)
    try:
        if not all((args.skill_sources, args.assets, args.placements, args.state_dir, args.adapter_json)):
            raise RunnerError("runner configuration is incomplete")
        remote, adapter = json.loads(args.remote_json), json.loads(args.adapter_json)
        from bz_a3_diagnostic_client import AdapterTransport, BzA3DiagnosticClient
        runner = OneShotRunner(skill_sources=_load(args.skill_sources), assets=_load(args.assets),
            placements=_load(args.placements), client=BzA3DiagnosticClient(
                AdapterTransport(remote, adapter), args.state_dir))
        request = json.load(sys.stdin); result = runner.run(request)
    except (RunnerError, OSError, ValueError, json.JSONDecodeError) as error:
        result = {"status": "setup_error", "diagnostics": _bounded(error),
                  "controller_usage": {"limit": 1, "billed": 0, "calls": []}}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
