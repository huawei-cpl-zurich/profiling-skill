#!/usr/bin/env python3
"""Small Git-backed runner for isolated, resumable kernel experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable


GLOBAL_CPL_REMOTE = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
TERMINAL = {"completed", "failed", "cancelled"}


class RunnerError(RuntimeError):
    pass


class ObservationLost(RunnerError):
    pass


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RunnerError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _receipt(stdout: str) -> dict:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError("cpl-remote returned no JSON receipt") from exc
    if not isinstance(value, dict):
        raise RunnerError("cpl-remote receipt is not an object")
    return value


class CplRemote:
    """The only production remote adapter: the user-wide cpl-remote client."""

    def __init__(self, invoke: Callable | None = None, executable: Path | None = None):
        self.invoke = invoke or subprocess.run
        self.executable = executable or GLOBAL_CPL_REMOTE
        if invoke is None and (not self.executable.is_file() or not os.access(self.executable, os.X_OK)):
            raise RunnerError(f"global cpl-remote is unavailable: {self.executable}")

    def _call(self, arguments: list[str]) -> dict:
        command = [str(self.executable), "--json", *arguments]
        try:
            result = self.invoke(command, text=True, capture_output=True, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ObservationLost(str(exc)) from exc
        # Nonzero is valid for a terminal failed remote workload. Its receipt is
        # still authoritative, so parse stdout before considering stderr.
        try:
            return _receipt(result.stdout)
        except RunnerError:
            raise ObservationLost(result.stderr.strip() or "cpl-remote transport failed")

    def submit(self, request: dict, script: Path) -> str:
        receipt = self._call([
            "run", request["target"], "--runtime", request["runtime"],
            "--file", str(script), "--cwd", request["remote_workspace"],
        ])
        handle = receipt.get("handle") or receipt.get("REMOTE_HANDLE")
        if not isinstance(handle, str) or not handle.startswith("remote:"):
            raise ObservationLost("cpl-remote dispatch returned no durable handle")
        return handle

    def _logs(self, handle: str, stream: str) -> str:
        receipt = self._call(["logs", handle, "--stream", stream, "--tail", "10000"])
        return str(receipt.get("content") or receipt.get("REMOTE_CONTENT") or "")

    def observe(self, handle: str) -> dict:
        receipt = self._call(["observe", handle, "--wait"])
        if (receipt.get("handle") or receipt.get("REMOTE_HANDLE")) != handle:
            raise ObservationLost("cpl-remote observation changed the retained handle")
        state = receipt.get("state") or receipt.get("REMOTE_STATE")
        if state not in TERMINAL:
            raise ObservationLost(f"remote observation is nonterminal: {state}")
        stdout, stderr = self._logs(handle, "stdout"), self._logs(handle, "stderr")
        parsed = None
        for line in reversed(stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("outcome"):
                parsed = candidate
                break
        if parsed is not None:
            return {**parsed, "stdout": stdout, "stderr": stderr}
        exit_code = receipt.get("exit", receipt.get("REMOTE_EXIT"))
        return {
            "outcome": "success" if state == "completed" and exit_code in (0, None) else "candidate_failure",
            "diagnostics": stderr or stdout,
            "stdout": stdout,
            "stderr": stderr,
        }


class SubprocessAgent:
    """JSON stdin/stdout adapter for the experiment's chosen agent launcher."""

    def __init__(self, command: list[str]):
        if not command or not all(isinstance(part, str) and part for part in command):
            raise RunnerError("agent_command must be a non-empty string list")
        self.command = command

    def prepare(self, request: dict) -> dict:
        result = subprocess.run(
            self.command, input=json.dumps(request), text=True, capture_output=True, check=False
        )
        if result.returncode:
            raise RunnerError(result.stderr.strip() or "agent adapter failed")
        value = _receipt(result.stdout)
        required = {"candidate", "remote_script", "commands", "reasoning", "postmortem"}
        if not required <= value.keys():
            raise RunnerError(f"agent result is missing: {sorted(required - value.keys())}")
        return value


class ExperimentCell:
    def __init__(self, root: Path, config: dict, agent, remote):
        self.root, self.config, self.agent, self.remote = Path(root), config, agent, remote
        for name in ("campaign", "product", "task", "treatment"):
            if not isinstance(config.get(name), str) or not config[name]:
                raise RunnerError(f"missing configuration field: {name}")
        self.name = f"{config['product']}-{config['task']}-{config['treatment']}"
        self.path = self.root / config["campaign"] / self.name
        self.repo = self.path / "repo"
        self.state_path = self.path / "state.json"
        self.branch = f"experiment/{config['campaign']}/{self.name}"

    def _state(self) -> dict:
        if not self.state_path.is_file():
            raise RunnerError("cell is not initialized")
        return json.loads(self.state_path.read_text())

    def _save(self, state: dict) -> dict:
        _write_json(self.state_path, state)
        return state

    def init(self) -> dict:
        if self.path.exists():
            raise RunnerError(f"cell already exists: {self.path}")
        self.path.mkdir(parents=True)
        source = Path(self.config["source_repo"]).resolve()
        result = subprocess.run(
            ["git", "clone", "--no-hardlinks", str(source), str(self.repo)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            shutil.rmtree(self.path)
            raise RunnerError(result.stderr.strip() or "git clone failed")
        _git(self.repo, "checkout", "-b", self.branch, self.config.get("base_ref", "main"))
        _git(self.repo, "config", "user.name", "Experiment Runner")
        _git(self.repo, "config", "user.email", "experiment-runner@example.invalid")
        (self.path / "attempts").mkdir()
        return self._save({"status": "ready", "round": 1, "attempts": 0,
                           "infrastructure_failures": 0, "candidate_repairs": 0})

    def _request(self, state: dict) -> dict:
        return {
            "campaign": self.config["campaign"], "cell": self.name,
            "round": state["round"], "repo": str(self.repo),
            "target": self.config["target"], "runtime": self.config["runtime"],
            "remote_workspace": self.config["remote_workspace"],
            "candidate_diagnostics": state.get("candidate_diagnostics"),
        }

    def _attempt(self, state: dict, prepared: dict) -> tuple[Path, dict]:
        state["attempts"] += 1
        attempt = self.path / "attempts" / str(state["attempts"])
        attempt.mkdir()
        request = self._request(state)
        _write_json(attempt / "request.json", request)
        _write_json(attempt / "agent-result.json", prepared)
        script = attempt / "run.sh"
        script.write_text(prepared["remote_script"])
        script.chmod(0o700)
        try:
            handle = self.remote.submit(request, script)
        except ObservationLost as exc:
            _write_json(attempt / "infrastructure.json", {"diagnostics": str(exc), "phase": "dispatch"})
            state.update(status="infrastructure_unknown", pending_attempt=state["attempts"])
            return attempt, self._save(state)
        # This write is deliberately the first action after submit returns.
        (attempt / "remote-job-id.txt").write_text(handle + "\n")
        state.update(status="observing", pending_attempt=state["attempts"])
        self._save(state)
        return attempt, self._observe(state, attempt, handle, prepared)

    def _observe(self, state: dict, attempt: Path, handle: str, prepared: dict) -> dict:
        try:
            result = self.remote.observe(handle)
        except ObservationLost as exc:
            _write_json(attempt / "infrastructure.json", {"diagnostics": str(exc), "phase": "observe", "handle": handle})
            state["status"] = "infrastructure_unknown"
            return self._save(state)
        _write_json(attempt / "result.json", result)
        (attempt / "stdout.log").write_text(str(result.get("stdout", "")))
        (attempt / "stderr.log").write_text(str(result.get("stderr", "")))
        outcome = result.get("outcome")
        if outcome == "success":
            self._commit_round(state, prepared, result)
            state["round"] += 1
            state["status"] = "complete" if state["round"] > int(self.config["rounds"]) else "ready"
            state.pop("candidate_diagnostics", None)
            state.pop("pending_attempt", None)
            state["infrastructure_failures"] = state["candidate_repairs"] = 0
            return self._save(state)
        if outcome == "candidate_failure":
            state["candidate_repairs"] += 1
            state["candidate_diagnostics"] = str(result.get("diagnostics", "candidate failed"))
            state["status"] = "candidate_failure"
        elif outcome == "infrastructure_failure":
            state["infrastructure_failures"] += 1
            state["status"] = "infrastructure_failure"
        else:
            state["infrastructure_failures"] += 1
            state["status"] = "infrastructure_failure"
            result["diagnostics"] = f"invalid terminal outcome: {outcome!r}"
            _write_json(attempt / "result.json", result)
        state.pop("pending_attempt", None)
        return self._save(state)

    def _commit_round(self, state: dict, prepared: dict, result: dict) -> None:
        folder = self.repo / "rounds" / f"{state['round']:02d}"
        folder.mkdir(parents=True)
        (folder / "candidate.py").write_text(prepared["candidate"])
        with (folder / "commands.jsonl").open("w") as stream:
            for command in prepared["commands"]:
                stream.write(json.dumps({"command": command}, sort_keys=True) + "\n")
        _write_json(folder / "profile-summary.json", result.get("profile_summary", {}))
        _write_json(folder / "retrieved-knowledge.json", prepared.get("retrieved_knowledge", []))
        _write_json(folder / "result.json", result)
        (folder / "reasoning.md").write_text(prepared["reasoning"].rstrip() + "\n")
        (folder / "postmortem.md").write_text(prepared["postmortem"].rstrip() + "\n")
        _git(self.repo, "add", str(folder.relative_to(self.repo)))
        _git(self.repo, "commit", "-m", f"experiment: complete round {state['round']}")

    def run(self) -> dict:
        state = self._state()
        if state["status"] == "infrastructure_unknown":
            raise RunnerError("observation is uncertain; resume the retained handle")
        if state["status"] == "complete":
            return state
        prepared = None
        while True:
            if prepared is None:
                prepared = self.agent.prepare(self._request(state))
            _, state = self._attempt(state, prepared)
            if state["status"] in {"ready", "complete", "infrastructure_unknown"}:
                return state
            if state["status"] == "infrastructure_failure":
                if state["infrastructure_failures"] > int(self.config.get("infrastructure_retries", 2)):
                    return state
                continue
            if state["candidate_repairs"] > int(self.config.get("candidate_repairs", 2)):
                return state
            prepared = None

    def resume(self) -> dict:
        state = self._state()
        if state["status"] != "infrastructure_unknown" or not state.get("pending_attempt"):
            raise RunnerError("cell has no uncertain retained job to resume")
        attempt = self.path / "attempts" / str(state["pending_attempt"])
        handle_path = attempt / "remote-job-id.txt"
        if not handle_path.is_file():
            raise RunnerError("dispatch outcome is unknown; no handle exists to observe")
        prepared = json.loads((attempt / "agent-result.json").read_text())
        return self._observe(state, attempt, handle_path.read_text().strip(), prepared)

    def report(self) -> dict:
        state = self._state()
        outcomes = Counter()
        for result in (self.path / "attempts").glob("*/result.json"):
            outcomes[json.loads(result.read_text()).get("outcome", "unknown")] += 1
        return {"cell": self.name, "status": state["status"], "round": state["round"],
                "rounds_completed": state["round"] - 1, "attempts": state["attempts"],
                "outcomes": dict(sorted(outcomes.items())), "branch": self.branch}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "run", "resume", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    agent = SubprocessAgent(config["agent_command"]) if args.command in {"run", "resume"} else None
    remote = CplRemote() if args.command in {"run", "resume"} else None
    cell = ExperimentCell(args.root, config, agent, remote)
    value = getattr(cell, args.command)()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
