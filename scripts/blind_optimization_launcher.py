#!/usr/bin/env python3
"""One fresh optimization agent with optional guidance and an opaque tool endpoint."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_launcher import LaunchError, ProductionLauncher


PROMPT_V1 = """Optimize the supplied kernel while preserving its public interface, supported inputs, numerical correctness and single-kernel execution contract. Measure the starting implementation and your final candidate using the supplied benchmark interface. Use available tools and guidance as you find useful. Record experiments, commands, results, and the evidence supporting your decisions. Explain what improved, what did not, and what remains uncertain. Commit your work on the assigned branch; do not merge it.
"""


def build_prompt(budget_seconds: int = 600) -> str:
    if type(budget_seconds) is not int or budget_seconds <= 0:
        raise ValueError("budget_seconds must be a positive integer")
    return PROMPT_V1 + f"""
You have a {budget_seconds}-second wall-clock budget, including all tool waiting. Read /workspace/execution.json and note its UTC start and deadline before beginning. Reserve approximately the final 120 seconds for notes and commits; do not start new tool calls near the cutoff. Commit validated work early and update your experiment log incrementally so partial progress survives interruption. Prefer a small completed evidence-backed experiment to an unfinished search. Use experiment-tool help to learn the neutral tool interface. Every operational tool call already performs correctness checks; avoid redundant checks unless investigating a specific failure.
"""


PROMPT = build_prompt()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def execute(command: list[str], cwd: Path, timeout: float,
            output_files: tuple[Path, Path] | None = None, *, input_text: str | None = None) -> dict:
    """Retain partial diagnostics and terminate the complete local process group."""
    with ExitStack() as stack:
        sinks = [stack.enter_context(path.open("w")) for path in output_files] if output_files else []
        process = subprocess.Popen(command, cwd=cwd, text=True,
                                   stdin=subprocess.PIPE if input_text is not None else None,
                                   stdout=sinks[0] if sinks else subprocess.PIPE,
                                   stderr=sinks[1] if sinks else subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
    if output_files:
        stdout, stderr = (path.read_text(errors="replace") for path in output_files)
    return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out}


class ToolHandler(socketserver.StreamRequestHandler):
    def peer_disconnected(self) -> bool:
        try:
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except BlockingIOError:
            return False
        except ConnectionError:
            return True

    def handle(self) -> None:
        owner = self.server.owner
        args = None
        command = None
        try:
            args = json.loads(self.rfile.readline())["arguments"]
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError("arguments must be a string array")
            managed = {"--workspace", "--bundle", "--target", "--runtime"}
            overrides = sorted({arg.split("=", 1)[0] for arg in args} & managed)
            if overrides:
                raise ValueError("Managed configuration " + ", ".join(overrides) +
                                 " is assigned by the harness; omit these flags. Use experiment-tool help.")
            if any(arg in {"-h", "--help"} for arg in args):
                args = ["help"]
            with owner.tool_lock:
                remaining = getattr(owner, "deadline_monotonic", time.monotonic() + owner.tool_timeout) - time.monotonic()
                limit = getattr(owner, "max_tool_calls", None)
                exhausted = (limit is not None and args != ["help"]
                             and getattr(owner, "tool_calls", 0) >= limit)
                if exhausted:
                    result = {"returncode": 75, "stdout": "", "stderr": "Tool-call budget exhausted\n"}
                    entry = {"arguments": args, "dispatched": False, **result}
                elif remaining <= 0 or self.peer_disconnected():
                    reason = "deadline-expired" if remaining <= 0 else "client-disconnected"
                    result = {"returncode": 124 if remaining <= 0 else 125,
                              "stdout": "", "stderr": f"Request not dispatched: {reason}\n",
                              "reason": reason}
                    entry = {"arguments": args, "dispatched": False, **result}
                else:
                    # A domain helper receives candidate files; it is not a transport adapter.
                    if limit is not None and args != ["help"]:
                        owner.tool_calls = getattr(owner, "tool_calls", 0) + 1
                    command = [*owner.tool_command, "--workspace", str(owner.workspace), *args]
                    result = execute(command, owner.workspace, min(owner.tool_timeout, remaining))
                    entry = {"command": command, "dispatched": True, **result}
                with owner.journal.open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
        except (OSError, ValueError, KeyError) as error:
            result = {"returncode": 74 if isinstance(error, OSError) else 2,
                      "stdout": "", "stderr": str(error) + "\n"}
            if command is None:
                with owner.journal.open("a") as stream:
                    stream.write(json.dumps({"arguments": args, "dispatched": False, **result}) + "\n")
        try:
            self.wfile.write(json.dumps(result).encode() + b"\n")
        except BrokenPipeError:
            pass  # Caller reached its wall deadline; full result remains in the journal.


def tool_client(path: Path, args: list[str]) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(path))
        connection.sendall(json.dumps({"arguments": args}).encode() + b"\n")
        result = json.loads(connection.makefile("rb").readline())
    print(result["stdout"], end="")
    print(result["stderr"], end="", file=sys.stderr)
    return result["returncode"] if result["returncode"] >= 0 else 1


class BlindLauncher(ProductionLauncher):
    """Reuse the audited mount boundary without its legacy campaign assumptions."""

    def __init__(self, output: Path, bundle: Path, prompt: Path, model: dict,
                 tool_command: list[str], *, skill: Path | None = None,
                 codex: str = "codex", auth_home: Path | None = None,
                 forbidden: tuple[Path, ...] = (), timeout: int = 600,
                 finish_reserve_seconds: int = 120):
        self.root = output.resolve()
        self.bundle = bundle.resolve()
        self.prompt_source = prompt.resolve()
        self.skill_source = skill.resolve() if skill else None
        self.model = model
        self.tool_command = tool_command
        if type(timeout) is not int or timeout <= 0:
            raise LaunchError("budget_seconds must be a positive integer")
        self.timeout = timeout
        if type(finish_reserve_seconds) is not int or finish_reserve_seconds < 0:
            raise LaunchError("finish reserve must be a nonnegative integer")
        self.finish_reserve_seconds = finish_reserve_seconds
        self.tool_timeout = timeout
        self.tool_lock = threading.Lock()
        self.codex = Path(shutil.which(codex) or codex).resolve()
        self.bwrap = shutil.which("bwrap") or "bwrap"
        self.auth_home = (auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))).resolve()
        self.resolv_conf = Path("/etc/resolv.conf")
        self.forbidden = forbidden
        if not all(isinstance(model.get(k), str) and model[k] for k in ("name", "reasoning_effort")):
            raise LaunchError("explicit campaign model name and reasoning_effort are required")
        if not tool_command or not (self.auth_home / "auth.json").is_file():
            raise LaunchError("tool command and authenticated Codex state are required")
        for parent in Path(__file__).resolve().parents:
            if (parent / "canonical" / "profiling-skill").is_dir():
                self.forbidden += (parent,)
                break
        if any(self.root.is_relative_to(p.resolve()) for p in self.forbidden):
            raise LaunchError("evaluation output must be outside forbidden workspaces")

    def prepare(self) -> None:
        if self.root.exists():
            raise LaunchError("output already exists; inspect its receipt rather than restarting")
        # The exporter explicitly declares the public boundary. Never copy repository history.
        try:
            manifest = json.loads((self.bundle / "task.json").read_text())
            files = manifest["public_files"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise LaunchError("invalid task manifest: " + str(error)) from error
        if not isinstance(files, list) or not files or not all(isinstance(name, str) for name in files):
            raise LaunchError("task.json must declare public_files")
        for name in files:
            relative = Path(name)
            source = self.bundle / relative
            if relative.is_absolute() or ".." in relative.parts or source.is_symlink() or not source.is_file():
                raise LaunchError("invalid public bundle file")
        if not self.prompt_source.is_file():
            raise LaunchError("prompt file is unavailable")
        if self.skill_source:
            if not self.skill_source.is_dir() or not (self.skill_source / "SKILL.md").is_file():
                raise LaunchError("skill snapshot is unavailable")
            if any(p.is_symlink() for p in self.skill_source.rglob("*")):
                raise LaunchError("skill snapshot must not contain symlinks")
        # Ordinary bad inputs must not reserve an output path that cannot be retried.
        self.root.mkdir(parents=True, mode=0o700)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for name in files:
            relative = Path(name)
            source = self.bundle / relative
            target = self.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.frozen = self.root / "frozen"
        self.frozen.mkdir()
        shutil.copy2(self.prompt_source, self.frozen / "prompt.md")
        if self.skill_source:
            shutil.copytree(self.skill_source, self.frozen / "skill")
        self.state = self.root / "codex-state"
        self.state.mkdir()
        (self.state / "auth.json").touch(mode=0o600)
        # sockaddr_un has a small platform limit independent of evidence path length.
        self.socket_temporary = tempfile.TemporaryDirectory(prefix="blind-tool-")
        self.socket_dir = Path(self.socket_temporary.name)
        self.journal = self.root / "commands.jsonl"
        self.wrapper = self.root / "experiment-tool"
        self.wrapper.write_text('#!/bin/sh\nexec python3 /experiment/launcher.py client /experiment-state/tool.sock "$@"\n')
        self.wrapper.chmod(0o755)
        git = ["git", "-C", str(self.workspace)]
        for args in (["init", "-b", "experiment"], ["config", "user.name", "Experiment Agent"],
                     ["config", "user.email", "experiment@localhost"], ["add", "--force", "--", *files],
                     ["commit", "-m", "Starting implementation"]):
            subprocess.run([*git, *args], capture_output=True, text=True, check=True)
        (self.workspace / ".git" / "info" / "exclude").write_text(".agents/\nexecution.json\n")

    def base_command(self) -> list[str]:
        command = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
                   "--clearenv", "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(source).exists():
                command += ["--ro-bind", source, source]
        command += self._resolver_mounts()
        command += ["--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime", "--dir", "/experiment",
                    "--dir", "/tools", "--bind", str(self.workspace), "/workspace",
                    "--bind", str(self.state), "/codex-home",
                    "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
                    "--ro-bind", str(self._runtime_mount()), "/runtime/node",
                    "--ro-bind", str(Path(__file__).resolve()), "/experiment/launcher.py",
                    "--ro-bind", str(Path(__file__).with_name("production_launcher.py")), "/experiment/production_launcher.py",
                    "--ro-bind", str(self.wrapper), "/tools/experiment-tool",
                    "--bind", str(self.socket_dir), "/experiment-state"]
        for item in sorted(self.workspace.rglob("*")):
            if item.is_file() and ".git" not in item.relative_to(self.workspace).parts \
                    and not item.name.startswith("candidate."):
                command += ["--ro-bind", str(item), "/workspace/" + item.relative_to(self.workspace).as_posix()]
        if self.skill_source:
            command += ["--ro-bind", str(self.frozen / "skill"), "/workspace/.agents/skills/ascend-profiling"]
        return [*command, "--chdir", "/workspace", "--setenv", "HOME", "/home/agent",
                "--setenv", "CODEX_HOME", "/codex-home", "--setenv", "PATH", "/tools:/runtime/node/bin:/usr/bin:/bin"]

    def preflight(self) -> dict:
        checks = ["test -r /codex-home/auth.json", "test ! -e /codex-home/skills",
                  "test ! -e /codex-home/plugins", "test ! -e /home/agent/.agents/skills",
                  "command -v experiment-tool"]
        checks.append("test " + ("-r" if self.skill_source else "! -e") + " /workspace/.agents/skills/ascend-profiling/SKILL.md")
        checks += ["test ! -e " + shlex.quote(str(p.resolve())) for p in self.forbidden]
        result = execute([*self.base_command(), "sh", "-ceu", ";".join(checks)], self.root, 30)
        write_json(self.root / "isolation.json", result)
        if result["returncode"]:
            raise LaunchError("isolation preflight failed: " + result["stderr"])
        return result

    def agent_command(self) -> list[str]:
        return [*self.base_command(), "/runtime/node/bin/codex", "exec", "--json",
                "--ignore-user-config", "--dangerously-bypass-approvals-and-sandbox",
                "-m", self.model["name"], "-c", f'model_reasoning_effort="{self.model["reasoning_effort"]}"',
                (self.frozen / "prompt.md").read_text()]

    def before_agent(self) -> None:
        """Optional trusted work charged to the existing execution deadline."""

    def after_agent(self, result: dict) -> dict:
        """Optional report handling inside the original role clock/server lifetime."""
        return result

    def launch(self, preflight_only: bool = False) -> dict:
        self.prepare()
        self.preflight()
        server = socketserver.UnixStreamServer(str(self.socket_dir / "tool.sock"), ToolHandler)
        server.owner = self
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        utc_now = datetime.now(timezone.utc)
        original_path = self.root / "role-execution.json"
        if getattr(self, "resume_before_agent", False):
            original = json.loads(original_path.read_text())
            utc_start = datetime.fromisoformat(original["utc_start"])
            remaining = (datetime.fromisoformat(original["utc_deadline"]) - utc_now).total_seconds()
            started = time.monotonic() - max(0., (utc_now - utc_start).total_seconds())
            self.deadline_monotonic = time.monotonic() + max(0., remaining)
        else:
            started = time.monotonic()
            self.deadline_monotonic = started + self.timeout
            original = {"utc_start": utc_now.isoformat(),
                "utc_deadline": (utc_now + timedelta(seconds=self.timeout)).isoformat(),
                "wall_budget_seconds": self.timeout,
                "finish_reserve_seconds": min(self.finish_reserve_seconds, self.timeout)}
            write_json(original_path, original)
        write_json(self.workspace / "execution.json", original)
        try:
            if preflight_only:
                command = [*self.base_command(), "experiment-tool", "help"]
            else:
                self.before_agent()
                command = self.agent_command()
            remaining = self.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                result = dict(returncode=124, timed_out=True, stdout="", stderr="Original role deadline expired")
            else:
                if not preflight_only:
                    write_json(self.root / "model-dispatch.json", {"dispatch_intent": True,
                        "utc_deadline": original["utc_deadline"]})
                result = execute(command, self.root, remaining,
                                 (self.root / "agent.jsonl", self.root / "agent.stderr"))
            if not preflight_only:
                result = self.after_agent(result)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            self.socket_temporary.cleanup()
        result.pop("stdout")
        result.pop("stderr")
        git = subprocess.run(["git", "-C", str(self.workspace), "log", "--format=%H %s"],
                             text=True, capture_output=True, check=True)
        dirty = subprocess.check_output(["git", "-C", str(self.workspace), "status", "--porcelain"], text=True)
        result.update(elapsed_seconds=time.monotonic() - started, commits=git.stdout.splitlines(),
                      candidate_commits=max(0, len(git.stdout.splitlines()) - 1),
                      worktree_dirty=dirty.splitlines(), model=self.model, preflight_only=preflight_only)
        result["exit_code"] = completion_exit_code(result)
        write_json(self.root / "receipt.json", result)
        return result


def completion_exit_code(result: dict) -> int:
    if result["timed_out"]:
        return 124
    if result["returncode"]:
        return result["returncode"] if result["returncode"] > 0 else 1
    if not result["preflight_only"] and not result["candidate_commits"]:
        return 3
    return 0


def positive_budget(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("budget must be a positive integer")
    return result


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        return tool_client(Path(sys.argv[2]), sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "prompt", "output", "model-json", "tool-command-json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--forbidden", type=Path, action="append", default=[])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--budget-seconds", type=positive_budget, default=600)
    args = parser.parse_args()
    instance = BlindLauncher(args.output, args.bundle, args.prompt, json.loads(args.model_json.read_text()),
                             json.loads(args.tool_command_json.read_text()), skill=args.skill,
                             forbidden=tuple(args.forbidden), timeout=args.budget_seconds)
    result = instance.launch(args.preflight_only)
    print(json.dumps(result))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
