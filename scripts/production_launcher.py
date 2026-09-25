#!/usr/bin/env python3
"""Outer-isolated, persistent Codex launcher for profiling campaigns."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import socket
import socketserver
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence


class LaunchError(RuntimeError):
    pass


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        owner: BudgetController = self.server.owner  # type: ignore[attr-defined]
        try:
            request = json.loads(self.rfile.readline())
            arguments = request["arguments"]
            if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
                raise ValueError("arguments must be a string array")
            with owner.lock:
                if owner.used >= owner.limit:
                    response = {"exit_code": 75, "stdout": "", "stderr": "remote request budget exhausted\n"}
                else:
                    owner.used += 1
                    mapped = owner.map_arguments(arguments)
                    run = subprocess.run(
                        [*owner.command, *mapped], text=True, capture_output=True,
                        timeout=owner.timeout, check=False, cwd=owner.workspace,
                    )
                    response = {
                        "exit_code": run.returncode, "stdout": run.stdout,
                        "stderr": run.stderr, "request": owner.used,
                    }
        except (KeyError, TypeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            response = {"exit_code": 74, "stdout": "", "stderr": f"controller transport failed: {error}\n"}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class BudgetController:
    """Host-side opaque controller endpoint with an atomic request limit."""

    def __init__(self, socket_path: Path, command: Sequence[str], limit: int,
                 workspace: Path, cell: dict, timeout: float = 900):
        if not command or limit < 1:
            raise LaunchError("controller command and positive request limit are required")
        self.socket_path = socket_path
        self.workspace = workspace.resolve()
        fields = {
            "cell_id": str(cell["cell_id"]), "device": str(cell["device"]),
            "workspace": str(self.workspace),
        }
        self.command = tuple(part.format_map(fields) for part in command)
        self.limit = limit
        self.timeout = timeout
        self.used = 0
        self.lock = threading.Lock()
        self.server: socketserver.UnixStreamServer | None = None
        self.thread: threading.Thread | None = None

    def map_arguments(self, arguments: Sequence[str]) -> list[str]:
        """Translate only sandbox-workspace paths into this cell's host tree."""
        mapped = []
        for argument in arguments:
            prefix, separator, value = argument.partition("=")
            candidate = value if separator else argument
            if candidate == "/workspace" or candidate.startswith("/workspace/"):
                relative = Path(candidate).relative_to("/workspace")
                host = (self.workspace / relative).resolve()
                if not host.is_relative_to(self.workspace):
                    raise ValueError("workspace path escaped its cell")
                replacement = str(host)
                mapped.append(f"{prefix}={replacement}" if separator else replacement)
            elif candidate.startswith("/"):
                raise ValueError("absolute paths outside /workspace are forbidden")
            else:
                mapped.append(argument)
        return mapped

    def __enter__(self) -> "BudgetController":
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.server = socketserver.UnixStreamServer(str(self.socket_path), _RequestHandler)
        self.server.owner = self  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()
        self.socket_path.unlink(missing_ok=True)
        assert self.thread is not None
        self.thread.join()


def controller_client(socket_path: Path, arguments: Sequence[str]) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps({"arguments": list(arguments)}).encode() + b"\n")
        stream = client.makefile("rb")
        response = json.loads(stream.readline())
    print(response.get("stdout", ""), end="")
    print(response.get("stderr", ""), end="", file=__import__("sys").stderr)
    return int(response["exit_code"])


class ProductionLauncher:
    """Run three turns of one Codex session in a minimal Bubblewrap root."""

    def __init__(self, controller_command: Sequence[str], *, codex: str = "codex",
                 bwrap: str = "bwrap", auth_home: Path | None = None,
                 timeout_seconds: int = 3600, forbidden_paths: Sequence[Path] = (),
                 dry_run: bool = False):
        self.controller_command = tuple(controller_command)
        self.codex = Path(shutil.which(codex) or codex).resolve()
        self.bwrap = shutil.which(bwrap) or bwrap
        self.auth_home = (auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))).resolve()
        self.timeout_seconds = timeout_seconds
        self.forbidden_paths = tuple(Path(path).resolve() for path in forbidden_paths)
        self.dry_run = dry_run
        if not Path(self.bwrap).exists() or not self.codex.exists():
            raise LaunchError("Bubblewrap and Codex executables are required")
        if not (self.auth_home / "auth.json").is_file():
            raise LaunchError("authenticated Codex state is unavailable")

    def _runtime_mount(self) -> Path:
        # Official npm installs are a symlink into the versioned Node prefix.
        for parent in self.codex.parents:
            if (parent / "bin" / "node").is_file() and (parent / "lib" / "node_modules").is_dir():
                return parent
        return self.codex.parent

    def _base_command(self, sandbox: Path, attempt: Path) -> list[str]:
        workspace = (sandbox / "workspace").resolve()
        state = (attempt / "codex-state").resolve()
        state.mkdir(mode=0o700, parents=True)
        (state / "auth.json").touch(mode=0o600)
        runtime = self._runtime_mount()
        script = Path(__file__).resolve()
        command = [
            self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
            "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        ]
        for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(source).exists():
                command += ["--ro-bind", source, source]
        command += [
            "--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime",
            "--dir", "/experiment", "--bind", str(workspace), "/workspace",
            "--bind", str(state), "/codex-home",
            "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
            "--ro-bind", str(runtime), "/runtime/node",
            "--ro-bind", str(script), "/experiment/request_gate.py",
            "--chdir", "/workspace", "--setenv", "HOME", "/home/agent",
            "--setenv", "CODEX_HOME", "/codex-home",
            "--setenv", "PATH", "/runtime/node/bin:/usr/bin:/bin",
            "--setenv", "EXPERIMENT_CONTROLLER",
            "/usr/bin/python3 /experiment/request_gate.py controller-client /experiment-state/controller.sock",
        ]
        return command

    def _preflight(self, base: list[str], socket_dir: Path) -> None:
        checks = ["test -r /codex-home/auth.json", "test -d /workspace/.agents/skills"]
        for path in self.forbidden_paths:
            checks.append(f"test ! -e {shlex.quote(str(path))}")
        checks += ["test ! -e /codex-home/skills", "test ! -e /codex-home/plugins"]
        run = subprocess.run(
            [*base, "--bind", str(socket_dir), "/experiment-state", "sh", "-ceu", ";".join(checks)],
            text=True, capture_output=True, check=False,
        )
        if run.returncode:
            raise LaunchError(f"outer isolation preflight failed: {run.stderr.strip()}")

    @staticmethod
    def _session_id(output: str) -> str:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                return event["thread_id"]
        raise LaunchError("Codex did not report a persistent thread id")

    def launch(self, sandbox: Path, cell: dict) -> dict:
        if cell.get("rounds") != 3 or cell.get("request_budget") != 12:
            raise LaunchError("production cells require three rounds and a 12-request budget")
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        attempt = sandbox / ".launcher-attempts" / attempt_id
        base = self._base_command(sandbox, attempt)
        if self.dry_run:
            return {
                "exit_code": 0, "dry_run": True, "rounds": 3, "rounds_completed": 3,
                "request_budget": 12, "outer_command": base,
                "attempt_id": attempt_id,
                "prompt_sha256": __import__("hashlib").sha256(
                    (sandbox / "PROMPT.md").read_bytes()
                ).hexdigest(),
            }
        # AF_UNIX paths are limited to roughly 108 bytes; campaign roots can be
        # deeply nested, so use a short unique host directory per attempt.
        socket_dir = Path("/tmp") / f"profctl-{uuid.uuid4().hex[:12]}"
        socket_dir.mkdir(mode=0o700)
        socket_path = socket_dir / "controller.sock"
        outputs: list[dict] = []
        with BudgetController(
            socket_path, self.controller_command, 12, sandbox / "workspace", cell,
        ) as controller:
            self._preflight(base, socket_dir)
            session_id = ""
            for round_number in range(1, 4):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LaunchError("60-minute cell limit exhausted")
                if round_number == 1:
                    codex_args = [
                        "/runtime/node/bin/codex", "exec", "--json", "--ignore-user-config",
                        "--skip-git-repo-check",
                        "--dangerously-bypass-approvals-and-sandbox", "-m", "gpt-5.6-sol",
                        "-c", 'model_reasoning_effort="low"', "-C", "/workspace", "-",
                    ]
                    prompt = (sandbox / "PROMPT.md").read_text()
                else:
                    codex_args = [
                        "/runtime/node/bin/codex", "exec", "resume", "--json",
                        "--ignore-user-config",
                        "--dangerously-bypass-approvals-and-sandbox", "-m", "gpt-5.6-sol",
                        "-c", 'model_reasoning_effort="low"', session_id, "-",
                    ]
                    prompt = f"Continue optimization round {round_number} using the same experiment contract.\n"
                run = subprocess.run(
                    [*base, "--bind", str(socket_dir), "/experiment-state", *codex_args],
                    input=prompt, text=True, capture_output=True, check=False,
                    timeout=remaining,
                )
                outputs.append({"round": round_number, "exit_code": run.returncode,
                                "stdout": run.stdout, "stderr": run.stderr})
                if run.returncode:
                    break
                if round_number == 1:
                    session_id = self._session_id(run.stdout)
        socket_dir.rmdir()
        exit_code = outputs[-1]["exit_code"]
        rounds_completed = sum(item["exit_code"] == 0 for item in outputs)
        return {
            "exit_code": exit_code, "session_id": session_id,
            "rounds_completed": rounds_completed,
            "status": "complete" if exit_code == 0 and rounds_completed == 3 else "infrastructure_error",
            "controller_requests": controller.used, "turns": outputs,
            "elapsed_seconds": time.monotonic() - started, "attempt_id": attempt_id,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    client = sub.add_parser("controller-client")
    client.add_argument("socket", type=Path)
    client.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return controller_client(args.socket, args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
