#!/usr/bin/env python3
"""Small Git-backed runner for isolated, resumable kernel experiments."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

GLOBAL_CPL_REMOTE = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
RESULT_SCHEMA = "simple-experiment-result/v1"
MAX_CANDIDATE_BYTES = 1_048_576
MAX_GATE_BUNDLE_BYTES = 8 * 1_048_576
MAX_MANIFEST_BYTES = 64 * 1024
FUSION_DECLARATION = {
    "schema_version": 1, "mode": "single-logical-launch", "complete_operator": True,
}
EXPORTED_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:/-]*")
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
AGENT_ENV_ALLOWLIST = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
}


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


def _validate_candidate_manifest(value: object) -> None:
    required = {"schema", "kernel_name", "entrypoint", "fusion"}
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") != "profiling-skill/candidate-kernel/v2"):
        raise RunnerError("candidate manifest must be an exact v2 declaration")
    kernel, entrypoint = value.get("kernel_name"), value.get("entrypoint")
    if (not isinstance(kernel, str) or EXPORTED_NAME.fullmatch(kernel) is None
            or not isinstance(entrypoint, str) or EXPORTED_NAME.fullmatch(entrypoint) is None
            or kernel not in {entrypoint, f"{entrypoint}_mix_aic", f"{entrypoint}_mix_aiv"}
            or value.get("fusion") != FUSION_DECLARATION):
        raise RunnerError("candidate manifest has an invalid fully-fused contract")


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


def _tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise RunnerError(f"skill tree is unavailable: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        mode = path.lstat().st_mode
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(mode)):
            raise RunnerError(f"skill tree contains unsupported entry: {relative}")
        if path.is_file():
            digest.update(relative.encode() + b"\0")
            digest.update(oct(stat.S_IMODE(mode)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
        if (receipt.get("target") != request["target"] or not isinstance(handle, str)
                or not handle.startswith(f"remote:{request['target']}:job:")):
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
        environment = {key: value for key, value in os.environ.items()
                       if key in AGENT_ENV_ALLOWLIST}
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
        required = {"candidate", "commands"}
        if (not required <= value.keys() or not isinstance(value.get("hypothesis"), str)
                or not value["hypothesis"]):
            raise RunnerError("agent prepare result is incomplete")
        if (not isinstance(value["candidate"], str)
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

    def _skill_sources(self) -> list[tuple[str, Path]]:
        if not isinstance(self.config.get("prompt"), str) or not self.config["prompt"]:
            raise RunnerError("configuration requires an exact non-empty prompt")
        bindings = self.config.get("skill_bindings")
        pinned = bindings is not None
        if not pinned:
            paths = self.config.get("skill_paths")
            if not isinstance(paths, list):
                raise RunnerError("configuration requires explicit skill bindings")
            bindings = [{"name": Path(value).name, "path": value} for value in paths]
        if not isinstance(bindings, list):
            raise RunnerError("skill_bindings must be a list")
        sources = []
        for binding in bindings:
            if not isinstance(binding, dict) or not SKILL_NAME.fullmatch(str(binding.get("name", ""))):
                raise RunnerError("skill binding has an invalid canonical name")
            configured_source = Path(str(binding.get("path", "")))
            if configured_source.is_symlink():
                raise RunnerError("skill binding source is unavailable")
            source = configured_source.resolve()
            if not source.is_dir():
                raise RunnerError("skill binding source is unavailable")
            actual = _tree_sha256(source)
            expected = binding.get("sha256")
            if pinned and not isinstance(expected, str):
                raise RunnerError(f"skill binding requires sha256: {binding['name']}")
            if expected is not None and actual != expected:
                raise RunnerError(f"skill binding hash mismatch: {binding['name']}")
            if pinned:
                skill_file = source / "SKILL.md"
                if not skill_file.is_file() or not re.search(
                        rf"(?m)^name:\s*{re.escape(binding['name'])}\s*$",
                        skill_file.read_text(errors="replace")):
                    raise RunnerError(f"skill binding name mismatch: {binding['name']}")
            sources.append((binding["name"], source))
        if len({name for name, _ in sources}) != len(sources):
            raise RunnerError("skill binding canonical names must be unique")
        return sources

    def _auth_content(self) -> bytes | None:
        configured = self.config.get("agent_auth_path")
        if configured is None:
            return None
        configured_auth = Path(str(configured))
        if configured_auth.is_symlink():
            raise RunnerError("agent_auth_path must be a private regular file")
        auth = configured_auth.resolve()
        if not auth.is_file() or stat.S_IMODE(auth.stat().st_mode) & 0o077:
            raise RunnerError("agent_auth_path must be a private regular file")
        content = auth.read_bytes()
        if not content:
            raise RunnerError("agent_auth_path must not be empty")
        return content

    def _immutable_inputs(
        self,
    ) -> tuple[Path, int, bytes | None, bytes | None, bytes | None]:
        gate = Path(self.config.get("terminal_gate_path", "")).resolve()
        if not gate.is_file():
            raise RunnerError("terminal_gate_path must name an existing file")
        billed = self.config.get("billed_checks")
        if isinstance(billed, bool) or not isinstance(billed, int) or billed <= 0:
            raise RunnerError("billed_checks must be a positive integer")
        bundle_value = self.config.get("terminal_gate_bundle_path")
        manifest_value = self.config.get("candidate_manifest_path")
        campaign_gate = bundle_value is not None or manifest_value is not None
        bundle_content = manifest_content = None
        if campaign_gate:
            if not isinstance(bundle_value, str) or not isinstance(manifest_value, str):
                raise RunnerError(
                    "campaign gates require terminal_gate_bundle_path and candidate_manifest_path"
                )
            bundle, manifest = Path(bundle_value).resolve(), Path(manifest_value).resolve()
            if not bundle.is_file() or bundle.is_symlink():
                raise RunnerError("terminal_gate_bundle_path must name a regular file")
            if not manifest.is_file() or manifest.is_symlink():
                raise RunnerError("candidate_manifest_path must name a regular file")
            bundle_content, manifest_content = bundle.read_bytes(), manifest.read_bytes()
            if not bundle_content or len(bundle_content) > MAX_GATE_BUNDLE_BYTES:
                raise RunnerError("terminal gate bundle is empty or too large")
            if not manifest_content or len(manifest_content) > MAX_MANIFEST_BYTES:
                raise RunnerError("candidate manifest is empty or too large")
            try:
                candidate_manifest = json.loads(manifest_content)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RunnerError("candidate manifest must be valid UTF-8 JSON") from exc
            _validate_candidate_manifest(candidate_manifest)
        configured_task = self.config.get("task_path")
        if configured_task is None:
            return gate, billed, None, bundle_content, manifest_content
        if not isinstance(configured_task, str) or not configured_task:
            raise RunnerError("task_path must name an existing UTF-8 file")
        task = Path(configured_task).resolve()
        if not task.is_file():
            raise RunnerError("task_path must name an existing UTF-8 file")
        task_content = task.read_bytes()
        try:
            task_content.decode()
        except UnicodeDecodeError as exc:
            raise RunnerError("task_path must name an existing UTF-8 file") from exc
        return gate, billed, task_content, bundle_content, manifest_content

    def _launch_device(self, state: dict) -> int | None:
        has_campaign_gate = state.get("gate_bundle_sha256") is not None
        physical_device = self.config.get("physical_device")
        if not has_campaign_gate:
            if physical_device is not None:
                raise RunnerError("physical_device requires a snapshotted campaign gate")
            return None
        if (isinstance(physical_device, bool) or not isinstance(physical_device, int)
                or physical_device < 0):
            raise RunnerError(
                "physical_device must be supplied at run time as a non-negative integer"
            )
        return physical_device

    def _verify_task_snapshot(self, state: dict) -> None:
        expected = state.get("task_sha256")
        if expected is None:
            self._verify_gate_snapshots(state)
            return
        task = self.repo / "TASK.md"
        if task.is_symlink() or not task.is_file():
            raise RunnerError("snapshotted TASK.md is missing or replaced")
        actual = hashlib.sha256(task.read_bytes()).hexdigest()
        if actual != expected:
            raise RunnerError("snapshotted TASK.md changed after initialization")
        self._verify_gate_snapshots(state)

    def _verify_gate_snapshots(self, state: dict) -> None:
        for state_name, filename in (
            ("gate_bundle_sha256", "terminal-gate-bundle.tar"),
            ("candidate_manifest_sha256", "candidate-manifest.json"),
        ):
            expected = state.get(state_name)
            if expected is None:
                continue
            retained = self.path / filename
            if retained.is_symlink() or not retained.is_file():
                raise RunnerError(f"snapshotted {filename} is missing or replaced")
            if hashlib.sha256(retained.read_bytes()).hexdigest() != expected:
                raise RunnerError(f"snapshotted {filename} changed after initialization")

    def init(self) -> dict:
        if self.path.exists():
            raise RunnerError(f"cell already exists: {self.path}")
        sources = self._skill_sources()
        auth_content = self._auth_content()
        (gate, billed_checks, task_content, bundle_content,
         manifest_content) = self._immutable_inputs()
        gate_content = gate.read_bytes()
        if not gate_content:
            raise RunnerError("terminal gate must not be empty")
        try:
            gate_content.decode()
        except UnicodeDecodeError as exc:
            raise RunnerError("terminal gate must be UTF-8 text") from exc
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
            task_sha256 = None
            if task_content is not None:
                (repo / "TASK.md").write_bytes(task_content)
                task_sha256 = hashlib.sha256(task_content).hexdigest()
            skill_root = temporary / "agent-home/.agents/skills"
            codex_home = temporary / "agent-home/.codex"
            codex_home.mkdir(parents=True)
            if auth_content is not None:
                private_auth = codex_home / "auth.json"
                private_auth.write_bytes(auth_content)
                private_auth.chmod(0o600)
            for name, source in sources:
                shutil.copytree(source, skill_root / name)
            (temporary / "attempts").mkdir()
            (temporary / "prompt.txt").write_text(self.config["prompt"])
            retained_gate = temporary / "terminal-gate.sh"
            retained_gate.write_bytes(gate_content)
            retained_gate.chmod(0o500)
            bundle_sha256 = manifest_sha256 = None
            if bundle_content is not None and manifest_content is not None:
                retained_bundle = temporary / "terminal-gate-bundle.tar"
                retained_bundle.write_bytes(bundle_content)
                retained_bundle.chmod(0o400)
                retained_manifest = temporary / "candidate-manifest.json"
                retained_manifest.write_bytes(manifest_content)
                retained_manifest.chmod(0o400)
                bundle_sha256 = hashlib.sha256(bundle_content).hexdigest()
                manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
            _write_json(temporary / "state.json", {
                "status": "ready", "round": 1, "attempts": 0,
                "infrastructure_failures": 0, "candidate_repairs": 0,
                "billed_checks_limit": billed_checks, "billed_checks_used": 0,
                "task_sha256": task_sha256,
                "gate_bundle_sha256": bundle_sha256,
                "candidate_manifest_sha256": manifest_sha256,
                "skill_names": [name for name, _ in sources],
                "skill_sha256": {name: _tree_sha256(source) for name, source in sources},
            })
            temporary.replace(self.path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._state()

    @property
    def skill_names(self) -> list[str]:
        if self.state_path.is_file():
            names = self._state().get("skill_names")
            if isinstance(names, list) and all(isinstance(name, str) for name in names):
                return names
        return [name for name, _ in self._skill_sources()]

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
    def _script(candidate: str, terminal_gate: str, *, bundle: bytes | None = None,
                manifest: bytes | None = None, physical_device: int | None = None,
                round_number: int = 1) -> str:
        if not isinstance(candidate, str) or not isinstance(terminal_gate, str):
            raise RunnerError("candidate and terminal gate must be strings")
        encoded = candidate.encode()
        if len(encoded) > MAX_CANDIDATE_BYTES:
            raise RunnerError(f"candidate is too large ({len(encoded)} bytes)")
        payload = base64.b64encode(encoded).decode()
        gate_suffix = terminal_gate if terminal_gate.endswith("\n") else terminal_gate + "\n"
        materialize = ""
        if bundle is not None or manifest is not None or physical_device is not None:
            if bundle is None or manifest is None or physical_device is None:
                raise RunnerError("campaign gate inputs must be complete")
            if (isinstance(physical_device, bool) or not isinstance(physical_device, int)
                    or physical_device < 0):
                raise RunnerError("physical_device must be a non-negative integer")
            bundle_payload = base64.b64encode(bundle).decode()
            manifest_payload = base64.b64encode(manifest).decode()
            bundle_sha = hashlib.sha256(bundle).hexdigest()
            manifest_sha = hashlib.sha256(manifest).hexdigest()
            materialize = (
                "base64 -d >\"$experiment_dir/terminal-gate-bundle.tar\" <<'SIMPLE_GATE_BUNDLE'\n"
                f"{bundle_payload}\nSIMPLE_GATE_BUNDLE\n"
                "base64 -d >\"$experiment_dir/candidate-manifest.json\" <<'SIMPLE_MANIFEST'\n"
                f"{manifest_payload}\nSIMPLE_MANIFEST\n"
                f"test \"$(sha256sum \"$experiment_dir/terminal-gate-bundle.tar\" | cut -d' ' -f1)\" = {bundle_sha}\n"
                f"test \"$(sha256sum \"$experiment_dir/candidate-manifest.json\" | cut -d' ' -f1)\" = {manifest_sha}\n"
                "export TERMINAL_GATE_BUNDLE_PATH=\"$experiment_dir/terminal-gate-bundle.tar\"\n"
                "export CANDIDATE_MANIFEST_PATH=\"$experiment_dir/candidate-manifest.json\"\n"
                f"export ASCEND_RT_VISIBLE_DEVICES={physical_device}\n"
                "export ASCEND_DEVICE_ID=0\n"
                f"export EXPERIMENT_ROUND={round_number}\n"
            )
        return (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "experiment_dir=$(mktemp -d \"${TMPDIR:-/tmp}/simple-experiment.XXXXXX\")\n"
            "trap 'rm -rf -- \"$experiment_dir\"' EXIT\n"
            "base64 -d >\"$experiment_dir/candidate.py\" <<'SIMPLE_CANDIDATE'\n"
            f"{payload}\nSIMPLE_CANDIDATE\n"
            "export CANDIDATE_PATH=\"$experiment_dir/candidate.py\"\n"
            + materialize + "cd \"$experiment_dir\"\n" + gate_suffix
        )

    @staticmethod
    def _checks_exhausted(state: dict) -> bool:
        return state["billed_checks_used"] >= state["billed_checks_limit"]

    def _finish_billed_checks(self, state: dict, outcome: str | None = None) -> dict:
        state["status"] = "billed_checks_exhausted"
        if outcome:
            state["last_outcome"] = outcome
        return self._save(state)

    def _attempt(self, state: dict, prepared: dict) -> dict:
        if not isinstance(prepared.get("commands"), list) or not all(
                isinstance(command, str) for command in prepared["commands"]):
            raise RunnerError("candidate commands must be a string list")
        if self._checks_exhausted(state):
            return self._finish_billed_checks(state)
        terminal_gate = (self.path / "terminal-gate.sh").read_text()
        bundle_path = self.path / "terminal-gate-bundle.tar"
        manifest_path = self.path / "candidate-manifest.json"
        physical_device = self._launch_device(state)
        script_text = self._script(
            prepared.get("candidate"), terminal_gate,
            bundle=bundle_path.read_bytes() if bundle_path.is_file() else None,
            manifest=manifest_path.read_bytes() if manifest_path.is_file() else None,
            physical_device=physical_device, round_number=state["round"],
        )
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
        _write_json(attempt / "launch.json", {"physical_device": physical_device})
        script = attempt / "run.sh"
        script.write_text(script_text)
        script.chmod(0o700)
        state.update(attempts=number, billed_checks_used=state["billed_checks_used"] + 1,
                     status="infrastructure_unknown", pending_attempt=number)
        self._save(state)
        try:
            handle = self.remote.submit(request, script)
        except ObservationLost as exc:
            _write_json(attempt / "infrastructure.json", {"diagnostics": str(exc), "phase": "dispatch"})
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
        if self._checks_exhausted(state):
            state["last_outcome"] = outcome
            state["status"] = "billed_checks_exhausted"
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
            self._verify_task_snapshot(state)
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
        launch_path = attempt / "launch.json"
        launch = (json.loads(launch_path.read_text()) if launch_path.is_file()
                  else {"physical_device": state.get("physical_device")})
        inputs = {"prompt": self.config["prompt"], "treatment": self.config["treatment"],
                  "allowed_skills": self.skill_names, "session_id": prepared.get("session_id"),
                  "billed_checks": state["billed_checks_limit"],
                  "task_sha256": state.get("task_sha256"),
                  "gate_bundle_sha256": state.get("gate_bundle_sha256"),
                  "candidate_manifest_sha256": state.get("candidate_manifest_sha256"),
                  "physical_device": launch.get("physical_device")}
        commands = "".join(json.dumps({"command": item}, sort_keys=True) + "\n"
                           for item in prepared["commands"])
        files = {
            "candidate.py": prepared["candidate"], "run.sh": (attempt / "run.sh").read_text(),
            "terminal-gate.sh": (self.path / "terminal-gate.sh").read_text(),
            "commands.jsonl": commands,
            "profile-summary.json": json.dumps(result.get("profile_summary", {}), indent=2, sort_keys=True) + "\n",
            "retrieved-knowledge.json": json.dumps(prepared.get("retrieved_knowledge", []), indent=2, sort_keys=True) + "\n",
            "result.json": json.dumps(result, indent=2, sort_keys=True) + "\n",
            "inputs.json": json.dumps(inputs, indent=2, sort_keys=True) + "\n",
            "hypothesis.md": prepared["hypothesis"].rstrip() + "\n",
            "reasoning.md": analysis["reasoning"].rstrip() + "\n",
            "postmortem.md": analysis["postmortem"].rstrip() + "\n",
        }
        if state.get("task_sha256") is not None:
            files["TASK.md"] = (self.repo / "TASK.md").read_bytes()
        if state.get("candidate_manifest_sha256") is not None:
            files["candidate-manifest.json"] = (self.path / "candidate-manifest.json").read_bytes()
            files["terminal-gate-bundle.sha256"] = state["gate_bundle_sha256"] + "\n"
        folder = self.repo / "rounds" / f"{state['round']:02d}"
        if folder.exists():
            actual = {path.name for path in folder.iterdir() if path.is_file()}
            expected = {
                name: content if isinstance(content, bytes) else content.encode()
                for name, content in files.items()
            }
            if actual != set(files) or any(
                (folder / name).read_bytes() != content
                for name, content in expected.items()
            ):
                raise RunnerError(f"existing round {state['round']} evidence does not match")
        else:
            folder.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
            try:
                for name, content in files.items():
                    if isinstance(content, bytes):
                        (temporary / name).write_bytes(content)
                    else:
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
        self._verify_task_snapshot(state)
        if state.get("pending_attempt"):
            raise RunnerError("cell has pending work; use resume")
        if state["status"] in {"complete", "billed_checks_exhausted"} or self._budget_exhausted(state):
            return state
        if self._checks_exhausted(state):
            return self._finish_billed_checks(state)
        self._launch_device(state)
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
            if self._checks_exhausted(state):
                return self._finish_billed_checks(state, state.get("last_outcome"))
            if prepared is None:
                prepared = self.agent.prepare(self._request(state, "prepare"))
                self._verify_task_snapshot(state)
            state = self._attempt(state, prepared)
            if state["status"] in {"ready", "complete", "billed_checks_exhausted",
                                   "infrastructure_unknown", "analysis_pending"}:
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
        self._verify_task_snapshot(state)
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
                "billed_checks_limit": state["billed_checks_limit"],
                "billed_checks_used": state["billed_checks_used"],
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
