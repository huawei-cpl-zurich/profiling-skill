#!/usr/bin/env python3
"""Small Git-backed runner for isolated, resumable kernel experiments."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

GLOBAL_CPL_REMOTE = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
RESULT_SCHEMA = "simple-experiment-result/v1"
MAX_CANDIDATE_BYTES = 1_048_576


class RunnerError(RuntimeError):
    pass


class ObservationLost(RunnerError):
    pass


class AgentInterrupted(RunnerError):
    pass


class PreflightFailure(ObservationLost):
    def __init__(self, phase: str, diagnostics: str):
        super().__init__(diagnostics)
        self.phase = phase


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


def _object(stdout: str, label: str) -> dict:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{label} returned no JSON object") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{label} returned no JSON object")
    return value


class CplRemote:
    """Production transport backed only by the user-wide cpl-remote client."""

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
        try:
            return _object(result.stdout, "cpl-remote")
        except RunnerError:
            raise ObservationLost(result.stderr.strip() or "cpl-remote transport failed")

    @staticmethod
    def _identity(receipt: dict, handle: str, operation: str) -> None:
        target = handle.split(":", 3)[1] if handle.count(":") >= 3 else ""
        actual_handle = receipt.get("handle")
        actual_target = receipt.get("target")
        if actual_handle != handle or actual_target != target:
            raise ObservationLost(f"cpl-remote {operation} receipt identity mismatch")

    def preflight(self, request: dict) -> dict:
        target = request["target"]
        try:
            capabilities = self._call(["capabilities", target])
        except ObservationLost as exc:
            raise PreflightFailure("capabilities", str(exc)) from exc
        if capabilities.get("target") != target:
            raise PreflightFailure("capabilities", "cpl-remote capabilities target mismatch")
        raw = capabilities.get("capabilities")
        if not isinstance(raw, dict):
            raise PreflightFailure("capabilities", "capabilities must be a JSON object")
        available = {key for key, enabled in raw.items() if enabled}
        missing = {"run", "observe", "logs"} - available
        if missing:
            raise PreflightFailure("capabilities", f"missing capabilities: {sorted(missing)}")
        try:
            preflight = self._call(["preflight", target])
        except ObservationLost as exc:
            raise PreflightFailure("preflight", str(exc)) from exc
        state = preflight.get("state")
        if preflight.get("target") != target or state != "completed":
            raise PreflightFailure("preflight", f"remote preflight failed: state={state}")
        return {"capabilities": capabilities, "preflight": preflight}

    def submit(self, request: dict, script: Path) -> str:
        receipt = self._call([
            "run", request["target"], "--runtime", request["runtime"],
            "--file", str(script), "--cwd", request["remote_workspace"],
        ])
        handle = receipt.get("handle")
        if not isinstance(handle, str) or not handle.startswith(f"remote:{request['target']}:job:"):
            raise ObservationLost("cpl-remote dispatch returned no matching durable handle")
        return handle

    def _logs(self, handle: str, stream: str) -> str:
        receipt = self._call(["logs", handle, "--stream", stream, "--tail", "10000"])
        self._identity(receipt, handle, "logs")
        return str(receipt.get("content") or "")

    def observe(self, handle: str) -> dict:
        receipt = self._call(["observe", handle, "--wait"])
        self._identity(receipt, handle, "observation")
        state = receipt.get("state")
        if state not in {"completed", "failed", "cancelled", "missing"}:
            raise ObservationLost(f"remote observation is nonterminal: {state}")
        stdout, stderr = self._logs(handle, "stdout"), self._logs(handle, "stderr")
        final = stdout.rstrip().splitlines()[-1] if stdout.rstrip() else ""
        try:
            envelope = json.loads(final)
        except json.JSONDecodeError:
            envelope = None
        valid = isinstance(envelope, dict) and envelope.get("schema") == RESULT_SCHEMA
        exit_code = receipt.get("exit")
        if valid and envelope.get("outcome") == "success" and state == "completed" and exit_code == 0:
            return {**envelope, "stdout": stdout, "stderr": stderr}
        if valid and envelope.get("outcome") in {"candidate_failure", "infrastructure_failure"}:
            outcome = envelope["outcome"]
        elif state == "missing" or exit_code in {125, 200}:
            outcome = "infrastructure_failure"
        else:
            outcome = "candidate_failure"
        diagnostics = stderr or (envelope or {}).get("diagnostics") or stdout
        if valid and envelope.get("outcome") == "success":
            diagnostics = f"contradictory success envelope: state={state}, exit={exit_code}; {diagnostics}"
        elif not valid:
            diagnostics = f"missing final {RESULT_SCHEMA} envelope; {diagnostics}"
        return {"outcome": outcome, "diagnostics": diagnostics,
                "stdout": stdout, "stderr": stderr, "remote_state": state,
                "remote_exit": exit_code}


class SubprocessAgent:
    """JSON stdin/stdout agent command with a cell-private environment."""

    def __init__(self, command: list[str], cwd: Path, home: Path):
        if not command or not all(isinstance(part, str) and part for part in command):
            raise RunnerError("agent_command must be a non-empty string list")
        self.command, self.cwd, self.home = command, Path(cwd), Path(home)

    def _invoke(self, request: dict) -> dict:
        environment = os.environ.copy()
        environment.update(HOME=str(self.home), CODEX_HOME=str(self.home / ".codex"))
        try:
            result = subprocess.run(
                self.command, input=json.dumps(request), text=True, capture_output=True,
                check=False, cwd=self.cwd, env=environment,
            )
        except OSError as exc:
            raise AgentInterrupted(str(exc)) from exc
        if result.returncode:
            raise AgentInterrupted(result.stderr.strip() or "agent command was interrupted")
        try:
            return _object(result.stdout, "agent")
        except RunnerError as exc:
            raise AgentInterrupted(str(exc)) from exc

    def prepare(self, request: dict) -> dict:
        value = self._invoke(request)
        required = {"candidate", "script_fragment", "commands"}
        if not required <= value.keys() or not isinstance(value.get("hypothesis"), str):
            raise RunnerError("agent prepare result is incomplete")
        if (not isinstance(value["candidate"], str)
                or not isinstance(value["script_fragment"], str)
                or not isinstance(value["commands"], list)
                or not all(isinstance(command, str) for command in value["commands"])):
            raise RunnerError("agent prepare result has invalid field types")
        return value

    def analyze(self, request: dict) -> dict:
        value = self._invoke(request)
        if not isinstance(value.get("reasoning"), str) or not isinstance(value.get("postmortem"), str):
            raise RunnerError("agent analysis requires reasoning and postmortem")
        return value


class ExperimentCell:
    def __init__(self, root: Path, config: dict, agent, remote):
        self.root, self.config, self.agent, self.remote = Path(root), config, agent, remote
        for name in ("campaign", "product", "task", "treatment"):
            if not isinstance(config.get(name), str) or not config[name]:
                raise RunnerError(f"missing configuration field: {name}")
        self.name = f"{config['product']}-{config['task']}-{config['treatment']}"
        self.path = self.root / config["campaign"] / self.name
        self.repo, self.state_path = self.path / "repo", self.path / "state.json"
        self.agent_home = self.path / "agent-home"
        self.branch = f"experiment/{config['campaign']}/{self.name}"

    def _state(self) -> dict:
        if not self.state_path.is_file():
            raise RunnerError("cell is not initialized")
        return json.loads(self.state_path.read_text())

    def _save(self, state: dict) -> dict:
        _write_json(self.state_path, state)
        return state

    def _skill_sources(self) -> list[Path]:
        if not isinstance(self.config.get("prompt"), str) or not self.config["prompt"]:
            raise RunnerError("configuration requires an exact non-empty prompt")
        paths = self.config.get("skill_paths")
        if not isinstance(paths, list):
            raise RunnerError("configuration requires explicit skill_paths")
        sources = [Path(value).resolve() for value in paths]
        if any(not source.is_dir() for source in sources) or len({p.name for p in sources}) != len(sources):
            raise RunnerError("skill_paths must be existing, uniquely named directories")
        return sources

    def init(self) -> dict:
        if self.path.exists():
            raise RunnerError(f"cell already exists: {self.path}")
        sources = self._skill_sources()
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.name}-", dir=parent))
        try:
            repo = temporary / "repo"
            result = subprocess.run(
                ["git", "clone", "--no-hardlinks", str(Path(self.config["source_repo"]).resolve()), str(repo)],
                text=True, capture_output=True, check=False,
            )
            if result.returncode:
                raise RunnerError(result.stderr.strip() or "git clone failed")
            _git(repo, "checkout", "-b", self.branch, self.config.get("base_ref", "main"))
            _git(repo, "config", "user.name", "Experiment Runner")
            _git(repo, "config", "user.email", "experiment-runner@example.invalid")
            skill_root = temporary / "agent-home/.agents/skills"
            (temporary / "agent-home/.codex").mkdir(parents=True)
            for source in sources:
                shutil.copytree(source, skill_root / source.name)
            (temporary / "attempts").mkdir()
            (temporary / "prompt.txt").write_text(self.config["prompt"])
            _write_json(temporary / "state.json", {
                "status": "ready", "round": 1, "attempts": 0,
                "infrastructure_failures": 0, "candidate_repairs": 0,
            })
            temporary.replace(self.path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._state()

    @property
    def skill_names(self) -> list[str]:
        return [Path(value).name for value in self.config["skill_paths"]]

    def _request(self, state: dict, phase: str) -> dict:
        request = {
            "phase": phase, "campaign": self.config["campaign"], "cell": self.name,
            "round": state["round"], "repo": str(self.repo),
            "target": self.config["target"], "runtime": self.config["runtime"],
            "remote_workspace": self.config["remote_workspace"],
            "prompt": self.config["prompt"], "treatment": self.config["treatment"],
            "allowed_skills": self.skill_names, "session_id": state.get("session_id"),
        }
        if state.get("prior_result") is not None:
            request.update(
                candidate_diagnostics=state.get("candidate_diagnostics"),
                prior_candidate=state.get("prior_candidate"),
                prior_result=state["prior_result"], prior_attempt=state["prior_attempt"],
            )
        return request

    def _preflight(self) -> None:
        folder = self.path / "preflights"
        sequence = len(list(folder.glob("*.json"))) + 1 if folder.exists() else 1
        path = folder / f"{sequence}.json"
        try:
            result = self.remote.preflight({"target": self.config["target"]})
        except ObservationLost as exc:
            _write_json(path, {"outcome": "infrastructure_failure",
                               "phase": getattr(exc, "phase", "preflight"),
                               "diagnostics": str(exc)})
            raise
        _write_json(path, result)

    @staticmethod
    def _script(candidate: str, fragment: str) -> str:
        if not isinstance(candidate, str) or not isinstance(fragment, str):
            raise RunnerError("candidate and script_fragment must be strings")
        encoded = candidate.encode()
        if len(encoded) > MAX_CANDIDATE_BYTES:
            raise RunnerError(f"candidate is too large ({len(encoded)} bytes)")
        payload = base64.b64encode(encoded).decode()
        return (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "experiment_dir=$(mktemp -d \"${TMPDIR:-/tmp}/simple-experiment.XXXXXX\")\n"
            "trap 'rm -rf -- \"$experiment_dir\"' EXIT\n"
            "base64 -d >\"$experiment_dir/candidate.py\" <<'SIMPLE_CANDIDATE'\n"
            f"{payload}\nSIMPLE_CANDIDATE\n"
            "export CANDIDATE_PATH=\"$experiment_dir/candidate.py\"\n"
            "cd \"$experiment_dir\"\n" + fragment.rstrip() + "\n"
        )

    def _attempt(self, state: dict, prepared: dict) -> dict:
        if not isinstance(prepared.get("commands"), list) or not all(
                isinstance(command, str) for command in prepared["commands"]):
            raise RunnerError("candidate commands must be a string list")
        script_text = self._script(prepared.get("candidate"), prepared.get("script_fragment"))
        request = self._request(state, "prepare")
        try:
            json.dumps(prepared)
            json.dumps(request)
        except (TypeError, ValueError) as exc:
            raise RunnerError(f"candidate evidence is not JSON serializable: {exc}") from exc
        number = state["attempts"] + 1
        attempt = self.path / "attempts" / str(number)
        attempt.mkdir()
        _write_json(attempt / "request.json", request)
        _write_json(attempt / "agent-result.json", prepared)
        script = attempt / "run.sh"
        script.write_text(script_text)
        script.chmod(0o700)
        state["attempts"] = number
        try:
            handle = self.remote.submit(request, script)
        except ObservationLost as exc:
            _write_json(attempt / "infrastructure.json", {"diagnostics": str(exc), "phase": "dispatch"})
            state.update(status="infrastructure_unknown", pending_attempt=number)
            return self._save(state)
        (attempt / "remote-job-id.txt").write_text(handle + "\n")
        state.update(status="observing", pending_attempt=number,
                     session_id=prepared.get("session_id", state.get("session_id")))
        self._save(state)
        return self._observe(state, attempt, handle, prepared)

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
        return self._accept_result(state, attempt, prepared, result)

    def _accept_result(self, state: dict, attempt: Path, prepared: dict, result: dict) -> dict:
        outcome = result.get("outcome")
        if outcome == "success":
            state["status"] = "analyzing"
            self._save(state)
            return self._analyze(state, attempt, prepared, result)
        state.pop("pending_attempt", None)
        if outcome == "candidate_failure":
            state["candidate_repairs"] += 1
            state.update(
                status="candidate_failure",
                candidate_diagnostics=str(result.get("diagnostics", "candidate failed")),
                prior_candidate=prepared["candidate"], prior_result=result,
                prior_attempt=str(attempt),
                session_id=prepared.get("session_id", state.get("session_id")),
            )
        else:
            state["infrastructure_failures"] += 1
            state["status"] = "infrastructure_failure"
        return self._save(state)

    def _analyze(self, state: dict, attempt: Path, prepared: dict, result: dict) -> dict:
        analysis_path = attempt / "analysis-result.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
        else:
            request = {**self._request(state, "analyze"), "candidate": prepared["candidate"],
                       "terminal_result": result, "attempt": str(attempt)}
            try:
                analysis = self.agent.analyze(request)
            except AgentInterrupted as exc:
                state["status"] = "analysis_pending"
                state["analysis_diagnostics"] = str(exc)
                return self._save(state)
            _write_json(analysis_path, analysis)
        self._finalize(state, attempt, prepared, result, analysis)
        state["round"] += 1
        state["status"] = "complete" if state["round"] > int(self.config["rounds"]) else "ready"
        for key in ("pending_attempt", "candidate_diagnostics", "prior_candidate",
                    "prior_result", "prior_attempt", "analysis_diagnostics"):
            state.pop(key, None)
        state["infrastructure_failures"] = state["candidate_repairs"] = 0
        return self._save(state)

    def _finalize(self, state: dict, attempt: Path, prepared: dict,
                  result: dict, analysis: dict) -> None:
        inputs = {"prompt": self.config["prompt"], "treatment": self.config["treatment"],
                  "allowed_skills": self.skill_names, "session_id": prepared.get("session_id")}
        commands = "".join(json.dumps({"command": item}, sort_keys=True) + "\n"
                           for item in prepared["commands"])
        files = {
            "candidate.py": prepared["candidate"], "run.sh": (attempt / "run.sh").read_text(),
            "commands.jsonl": commands,
            "profile-summary.json": json.dumps(result.get("profile_summary", {}), indent=2, sort_keys=True) + "\n",
            "retrieved-knowledge.json": json.dumps(prepared.get("retrieved_knowledge", []), indent=2, sort_keys=True) + "\n",
            "result.json": json.dumps(result, indent=2, sort_keys=True) + "\n",
            "inputs.json": json.dumps(inputs, indent=2, sort_keys=True) + "\n",
            "hypothesis.md": prepared["hypothesis"].rstrip() + "\n",
            "reasoning.md": analysis["reasoning"].rstrip() + "\n",
            "postmortem.md": analysis["postmortem"].rstrip() + "\n",
        }
        folder = self.repo / "rounds" / f"{state['round']:02d}"
        if folder.exists():
            actual = {path.name for path in folder.iterdir() if path.is_file()}
            if actual != set(files) or any((folder / name).read_text() != content for name, content in files.items()):
                raise RunnerError(f"existing round {state['round']} evidence does not match")
        else:
            folder.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
            try:
                for name, content in files.items():
                    (temporary / name).write_text(content)
                temporary.replace(folder)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        relative = str(folder.relative_to(self.repo))
        tracked = subprocess.run(
            ["git", "-C", str(self.repo), "cat-file", "-e", f"HEAD:{relative}"], capture_output=True
        ).returncode == 0
        if tracked and not _git(self.repo, "status", "--porcelain", "--", relative):
            return
        _git(self.repo, "add", relative)
        _git(self.repo, "commit", "--only", "-m",
             f"experiment: complete round {state['round']}", "--", relative)

    def _budget_exhausted(self, state: dict) -> bool:
        return ((state["status"] == "infrastructure_failure" and state["infrastructure_failures"]
                 > int(self.config.get("infrastructure_retries", 2)))
                or (state["status"] == "candidate_failure" and state["candidate_repairs"]
                    > int(self.config.get("candidate_repairs", 2))))

    def run(self) -> dict:
        state = self._state()
        if state.get("pending_attempt"):
            raise RunnerError("cell has pending work; use resume")
        if state["status"] == "complete" or self._budget_exhausted(state):
            return state
        try:
            self._preflight()
        except ObservationLost:
            state["infrastructure_failures"] += 1
            state["status"] = "infrastructure_failure"
            return self._save(state)
        last_agent = self.path / "attempts" / str(state["attempts"]) / "agent-result.json"
        prepared = (json.loads(last_agent.read_text())
                    if state["status"] == "infrastructure_failure" and last_agent.is_file()
                    else None)
        while True:
            if prepared is None:
                prepared = self.agent.prepare(self._request(state, "prepare"))
            state = self._attempt(state, prepared)
            if state["status"] in {"ready", "complete", "infrastructure_unknown", "analysis_pending"}:
                return state
            if state["status"] == "infrastructure_failure":
                if self._budget_exhausted(state):
                    return state
                continue
            if self._budget_exhausted(state):
                return state
            prepared = None

    def resume(self) -> dict:
        state = self._state()
        pending = state.get("pending_attempt")
        if not pending:
            raise RunnerError("cell has no pending retained work to resume")
        attempt = self.path / "attempts" / str(pending)
        prepared = json.loads((attempt / "agent-result.json").read_text())
        if (attempt / "result.json").exists():
            result = json.loads((attempt / "result.json").read_text())
            if result.get("outcome") == "success":
                return self._analyze(state, attempt, prepared, result)
            return self._accept_result(state, attempt, prepared, result)
        handle_path = attempt / "remote-job-id.txt"
        if not handle_path.is_file():
            raise RunnerError("dispatch outcome is unknown; no handle exists to observe")
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
    cell = ExperimentCell(args.root, config, None, None)
    if args.command in {"run", "resume"}:
        cell.agent = SubprocessAgent(config["agent_command"], cell.repo, cell.agent_home)
        cell.remote = CplRemote()
    value = getattr(cell, args.command)()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
