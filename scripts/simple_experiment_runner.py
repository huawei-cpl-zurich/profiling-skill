#!/usr/bin/env python3
"""Small Git-backed runner for isolated, resumable kernel experiments."""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import math
import re
import shlex
import shutil
import stat
import statistics
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
MAX_AUTOTUNE_CONFIGURATIONS = 128
MAX_AUTOTUNE_FINALISTS = 3
MIN_AUTOTUNE_PROFILE_REPEATS = 3
AUTOTUNE_BENCHMARK_CONTRACTS = {
    "matmul": {
        "correctness_cases": 10,
        "profile_cases": [7, 8, 9],
        "entrypoint": "streaming_matmul_add_kernel",
        "kernel_name": "streaming_matmul_add_kernel_mix_aic",
    },
    "gdn": {
        "correctness_cases": 50,
        "profile_cases": [40, 49, 47, 46, 45],
        "entrypoint": "chunk_gated_delta_rule_kernel",
        "kernel_name": "chunk_gated_delta_rule_kernel",
    },
    "bsa": {
        "correctness_cases": 50,
        "profile_cases": [47, 46, 49, 44, 43],
        "entrypoint": "block_sparse_attention_kernel",
        "kernel_name": "block_sparse_attention_kernel",
    },
}
MAX_AUTOTUNE_BUNDLE_BYTES = 16 * 1_048_576
AUTOTUNE_REQUEST_SCHEMA = "simple-experiment-autotune-request/v1"
AUTOTUNE_RESULT_SCHEMA = "simple-experiment-autotune-result/v1"
AUTOTUNE_GATE_CAPABILITY = "# simple-experiment-capability: managed-autotune/v1"
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
AUTOTUNE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
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


class TuneRejected(RunnerError):
    pass


class PreflightFailure(ObservationLost):
    def __init__(self, phase: str, diagnostics: str):
        super().__init__(diagnostics)
        self.phase = phase


class PreDispatchFailure(ObservationLost):
    """Failure proven to have occurred before a remote job was submitted."""


def _load_fusion_contract():
    path = Path(__file__).with_name("fully_fused_contract.py")
    spec = importlib.util.spec_from_file_location("simple_runner_fusion_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fusion contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FUSION_CONTRACT = _load_fusion_contract()


def _validate_candidate_manifest(value: object) -> None:
    try:
        FUSION_CONTRACT.validate_manifest(value)
    except FUSION_CONTRACT.FusionContractError as exc:
        raise RunnerError(f"candidate manifest is invalid: {exc}") from exc


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


def _json_scalar(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise RunnerError(f"{label} must be a finite JSON scalar")


def _python_scalar(value: object) -> str:
    _json_scalar(value, "autotune value")
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def _autotune_assignment(candidate: str) -> tuple[ast.AST, dict[str, object]]:
    try:
        tree = ast.parse(candidate)
    except SyntaxError as exc:
        raise RunnerError(f"candidate is not valid Python: {exc}") from exc
    matches = []
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == "AUTOTUNE_CONFIG":
            matches.append(node)
    if len(matches) != 1:
        raise RunnerError("candidate must declare exactly one top-level AUTOTUNE_CONFIG")
    try:
        value = ast.literal_eval(matches[0].value)
    except (ValueError, TypeError) as exc:
        raise RunnerError("AUTOTUNE_CONFIG must be a literal mapping") from exc
    if not isinstance(value, dict) or not value:
        raise RunnerError("AUTOTUNE_CONFIG must be a non-empty literal mapping")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise RunnerError("AUTOTUNE_CONFIG keys must be non-empty strings")
        _json_scalar(item, f"AUTOTUNE_CONFIG[{key!r}]")
    return matches[0], value


def _replace_autotune_assignment(candidate: str, node: ast.AST,
                                 values: dict[str, object]) -> str:
    lines = candidate.splitlines(keepends=True)

    def offset(line: int, byte_column: int) -> int:
        prefix = "".join(lines[:line - 1])
        raw = lines[line - 1].encode()
        try:
            column = len(raw[:byte_column].decode())
        except UnicodeDecodeError as exc:
            raise RunnerError("AUTOTUNE_CONFIG has an invalid source offset") from exc
        return len(prefix) + column

    start = offset(node.lineno, node.col_offset)
    end = offset(node.end_lineno, node.end_col_offset)
    literal = "{" + ", ".join(
        f"{json.dumps(key, ensure_ascii=False)}: {_python_scalar(values[key])}"
        for key in sorted(values)
    ) + "}"
    return candidate[:start] + f"AUTOTUNE_CONFIG = {literal}" + candidate[end:]


def materialize_autotune(candidate: str, proposed: object) -> dict:
    """Validate and deterministically materialize an agent-proposed tuning space."""
    node, incumbent = _autotune_assignment(candidate)
    if not isinstance(proposed, list) or len(proposed) > MAX_AUTOTUNE_CONFIGURATIONS:
        raise RunnerError(
            f"tune phase must return zero through {MAX_AUTOTUNE_CONFIGURATIONS} configurations"
        )
    options = [{
        "id": "incumbent", "parameters": dict(sorted(incumbent.items())),
        "hypothesis": "declared structural-candidate configuration",
    }]
    seen = {"incumbent"}
    for index, configuration in enumerate(proposed):
        if not isinstance(configuration, dict) or set(configuration) != {
                "id", "parameters", "hypothesis"}:
            raise RunnerError(f"autotune configuration {index} has invalid fields")
        identifier = configuration["id"]
        parameters = configuration["parameters"]
        hypothesis = configuration["hypothesis"]
        if (not isinstance(identifier, str) or not AUTOTUNE_ID.fullmatch(identifier)
                or identifier in seen):
            raise RunnerError(f"autotune configuration {index} has an invalid or duplicate id")
        if not isinstance(parameters, dict):
            raise RunnerError(f"autotune configuration {identifier} parameters must be an object")
        unknown = set(parameters) - set(incumbent)
        if unknown:
            raise RunnerError(
                f"autotune configuration {identifier} overrides undeclared keys: {sorted(unknown)}"
            )
        for key, value in parameters.items():
            _json_scalar(value, f"autotune configuration {identifier}.{key}")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise RunnerError(f"autotune configuration {identifier} requires a hypothesis")
        merged = {**incumbent, **parameters}
        options.append({
            "id": identifier, "parameters": dict(sorted(merged.items())),
            "hypothesis": hypothesis,
        })
        seen.add(identifier)
    materialized_sources = {}
    for option in options:
        source = _replace_autotune_assignment(candidate, node, option["parameters"])
        encoded = source.encode()
        if len(encoded) > MAX_CANDIDATE_BYTES:
            raise RunnerError(f"autotune configuration {option['id']} is too large")
        duplicate = materialized_sources.get(source)
        if duplicate is not None:
            raise RunnerError(
                f"autotune configurations {duplicate} and {option['id']} "
                "materialize identical source"
            )
        materialized_sources[source] = option["id"]
        option["source"] = source
        option["source_sha256"] = hashlib.sha256(encoded).hexdigest()
    return {
        "schema": AUTOTUNE_REQUEST_SCHEMA,
        "policy": {
            "screen": ["compile", "representative-correctness", "fusion", "cheap-device-timing"],
            "max_agent_configurations": MAX_AUTOTUNE_CONFIGURATIONS,
            "max_finalists": MAX_AUTOTUNE_FINALISTS,
            "finalist_selection": "lowest-cheap-timing-then-request-order",
            "finalist_timing": "msprof-op",
            "min_finalist_repeats": MIN_AUTOTUNE_PROFILE_REPEATS,
            "winner_selection": "lowest-msprof-geomean-then-finalist-order",
            "winner_gate": "complete-correctness-fusion-and-pipe-evidence",
        },
        "incumbent_id": "incumbent",
        "options": options,
    }


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
        missing = {"run", "observe", "logs", "upload"} - available
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
        target = request["target"]
        workspace = request["remote_workspace"]
        if (not isinstance(workspace, str) or not workspace
                or "\x00" in workspace or "\n" in workspace):
            raise PreDispatchFailure("remote_workspace must be a safe non-empty path")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        remote_name = f".simple-experiment-{digest}.sh"
        remote_script = f"{workspace.rstrip('/')}/{remote_name}"
        try:
            uploaded = self._call(["upload", target, str(script), remote_script])
        except ObservationLost as exc:
            raise PreDispatchFailure(f"payload upload failed: {exc}") from exc
        upload_exit = uploaded.get("exit")
        if (uploaded.get("target") != target or uploaded.get("state") != "completed"
                or upload_exit is not None and upload_exit not in {0, "0"}):
            raise PreDispatchFailure(
                f"payload upload was not confirmed: state={uploaded.get('state')}, "
                f"exit={uploaded.get('exit')}"
            )
        launcher = script.with_name("dispatch.sh")
        launcher_script = remote_script if workspace.startswith("/") else f"./{remote_name}"
        quoted = shlex.quote(launcher_script)
        launcher.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "infrastructure_failure() {\n"
            "  printf '%s\\n' '{\"schema\":\"simple-experiment-result/v1\","
            "\"outcome\":\"infrastructure_failure\","
            "\"diagnostics\":\"staged payload verification failed\"}'\n"
            "  exit 125\n"
            "}\n"
            f"actual_sha=$(sha256sum {quoted} 2>/dev/null | cut -d' ' -f1) || infrastructure_failure\n"
            f"test \"$actual_sha\" = {digest} || infrastructure_failure\n"
            f"exec bash {quoted}\n"
        )
        launcher.chmod(0o700)
        receipt = self._call([
            "run", request["target"], "--runtime", request["runtime"],
            "--file", str(launcher), "--cwd", workspace,
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

    def tune(self, request: dict) -> dict:
        value = self._invoke(request)
        configurations = value.get("configurations")
        if not isinstance(configurations, list) or len(configurations) > MAX_AUTOTUNE_CONFIGURATIONS:
            raise RunnerError("agent tune result requires at most 128 configurations")
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
        autotune_enabled = self.config.get("autotune", False)
        if not isinstance(autotune_enabled, bool):
            raise RunnerError("autotune must be a boolean")
        auth_content = self._auth_content()
        (gate, billed_checks, task_content, bundle_content,
         manifest_content) = self._immutable_inputs()
        gate_content = gate.read_bytes()
        if not gate_content:
            raise RunnerError("terminal gate must not be empty")
        try:
            gate_text = gate_content.decode()
        except UnicodeDecodeError as exc:
            raise RunnerError("terminal gate must be UTF-8 text") from exc
        if (autotune_enabled
                and AUTOTUNE_GATE_CAPABILITY not in gate_text.splitlines()):
            raise RunnerError(
                "autotune requires a terminal gate declaring managed-autotune/v1"
            )
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
                "autotune_enabled": autotune_enabled,
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
        if state.get("candidate_manifest_sha256") is not None:
            manifest = self.path / "candidate-manifest.json"
            try:
                value = json.loads(manifest.read_text())
                request["candidate_contract"] = FUSION_CONTRACT.validate_manifest(value)
            except (OSError, json.JSONDecodeError, FUSION_CONTRACT.FusionContractError) as exc:
                raise RunnerError(f"snapshotted candidate manifest is invalid: {exc}") from exc
        if state.get("prior_result") is not None:
            request.update(
                candidate_diagnostics=state.get("candidate_diagnostics"),
                prior_candidate=state.get("prior_candidate"),
                prior_result=state["prior_result"], prior_attempt=state["prior_attempt"],
            )
        return request

    def _previous_profile_summary(self, state: dict) -> dict | None:
        if state["round"] <= 1:
            return None
        path = self.repo / "rounds" / f"{state['round'] - 1:02d}" / "profile-summary.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise RunnerError("previous profile summary is not an object")
        return value

    def _tune(self, state: dict, prepared: dict) -> dict | None:
        if not state.get("autotune_enabled"):
            return None
        node, declared = _autotune_assignment(prepared["candidate"])
        del node
        request = {
            **self._request(state, "tune"),
            "session_id": prepared.get("session_id", state.get("session_id")),
            "candidate": prepared["candidate"],
            "declared_autotune_config": dict(sorted(declared.items())),
            "structural_hypothesis": prepared["hypothesis"],
            "prior_profile_summary": self._previous_profile_summary(state),
            "max_configurations": MAX_AUTOTUNE_CONFIGURATIONS,
            "max_profiled_finalists": MAX_AUTOTUNE_FINALISTS,
        }
        try:
            tuned = self.agent.tune(request)
        except AgentInterrupted:
            raise
        except RunnerError as exc:
            raise TuneRejected(str(exc)) from exc
        self._verify_task_snapshot(state)
        try:
            plan = materialize_autotune(prepared["candidate"], tuned.get("configurations"))
        except RunnerError as exc:
            raise TuneRejected(str(exc)) from exc
        benchmark = self.config["task"]
        benchmark_contract = AUTOTUNE_BENCHMARK_CONTRACTS.get(benchmark)
        if benchmark_contract is None:
            raise TuneRejected(f"autotune has no correctness contract for {benchmark}")
        declared = request.get("candidate_contract")
        if declared is not None and (
                declared.get("entrypoint") != benchmark_contract["entrypoint"]
                or declared.get("kernel_name") != benchmark_contract["kernel_name"]):
            raise TuneRejected("candidate manifest does not match the benchmark contract")
        plan.update(
            optimization_round=state["round"], benchmark=benchmark,
            required_correctness_cases=benchmark_contract["correctness_cases"],
            expected_profile_cases=list(benchmark_contract["profile_cases"]),
            expected_entrypoint=benchmark_contract["entrypoint"],
            expected_declared_kernel_name=benchmark_contract["kernel_name"],
        )
        raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if len(gzip.compress(raw, mtime=0)) > MAX_AUTOTUNE_BUNDLE_BYTES:
            raise TuneRejected("compressed autotune request is too large")
        return plan

    def _checkpoint_prepared(self, state: dict, prepared: dict) -> dict:
        candidate = prepared.get("candidate")
        if not isinstance(candidate, str):
            raise RunnerError("autotune prepare result requires candidate source")
        if len(candidate.encode()) > MAX_CANDIDATE_BYTES:
            raise RunnerError(f"candidate is too large ({len(candidate.encode())} bytes)")
        commands = prepared.get("commands")
        if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
            raise RunnerError("candidate commands must be a string list")
        _autotune_assignment(candidate)
        session = prepared.get("session_id")
        if not isinstance(session, str) or not session:
            raise RunnerError("autotune prepare result requires a durable session_id")
        try:
            json.dumps(prepared)
        except (TypeError, ValueError) as exc:
            raise RunnerError(f"prepared candidate is not JSON serializable: {exc}") from exc
        state.update(
            status="tuning", session_id=session,
            prepared_checkpoint={"round": state["round"], "prepared": prepared},
        )
        return self._save(state)

    def _load_prepared_checkpoint(self, state: dict) -> dict:
        value = state.get("prepared_checkpoint")
        if (not isinstance(value, dict) or value.get("round") != state["round"]
                or not isinstance(value.get("prepared"), dict)):
            raise RunnerError("retained prepared candidate is invalid")
        prepared = value["prepared"]
        if prepared.get("session_id") != state.get("session_id"):
            raise RunnerError("retained prepared session does not match state")
        return prepared

    def _load_autotune_checkpoint(self, state: dict, prepared: dict) -> dict:
        plan = state.get("autotune_checkpoint")
        if (not isinstance(plan, dict) or plan.get("schema") != AUTOTUNE_REQUEST_SCHEMA
                or plan.get("optimization_round") != state["round"]):
            raise RunnerError("retained autotune plan is invalid")
        options = plan.get("options")
        if not isinstance(options, list) or not options:
            raise RunnerError("retained autotune plan is invalid")
        proposed = [
            {key: option.get(key) for key in ("id", "parameters", "hypothesis")}
            for option in options[1:]
            if isinstance(option, dict)
        ]
        expected = materialize_autotune(prepared["candidate"], proposed)
        benchmark = self.config["task"]
        benchmark_contract = AUTOTUNE_BENCHMARK_CONTRACTS.get(benchmark, {})
        expected.update(
            optimization_round=state["round"], benchmark=benchmark,
            required_correctness_cases=benchmark_contract.get("correctness_cases"),
            expected_profile_cases=list(benchmark_contract.get("profile_cases", [])),
            expected_entrypoint=benchmark_contract.get("entrypoint"),
            expected_declared_kernel_name=benchmark_contract.get("kernel_name"),
        )
        if plan != expected:
            raise RunnerError("retained autotune plan does not match prepared source")
        return plan

    def _pending_tune(self, state: dict, prepared: dict) -> tuple[dict, dict | None]:
        try:
            plan = self._tune(state, prepared)
        except AgentInterrupted as exc:
            state.update(status="tune_pending", tune_diagnostics=str(exc))
            return self._save(state), None
        except TuneRejected as exc:
            state["candidate_repairs"] += 1
            state.update(
                status="candidate_failure", candidate_diagnostics=str(exc),
                prior_candidate=prepared["candidate"],
                prior_result={
                    "outcome": "candidate_failure", "phase": "tune",
                    "diagnostics": str(exc),
                },
                prior_attempt=None,
            )
            state.pop("prepared_checkpoint", None)
            state.pop("autotune_checkpoint", None)
            state.pop("tune_diagnostics", None)
            return self._save(state), None
        except RunnerError as exc:
            state.update(status="input_integrity_failure", diagnostics=str(exc))
            state.pop("prepared_checkpoint", None)
            state.pop("autotune_checkpoint", None)
            state.pop("tune_diagnostics", None)
            return self._save(state), None
        state.update(status="tuned", autotune_checkpoint=plan)
        state.pop("tune_diagnostics", None)
        return self._save(state), plan

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
                round_number: int = 1, autotune_request: dict | None = None) -> str:
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
        autotune_materialize = ""
        if autotune_request is not None:
            if autotune_request.get("schema") != AUTOTUNE_REQUEST_SCHEMA:
                raise RunnerError("autotune request has an invalid schema")
            raw = json.dumps(
                autotune_request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            compressed = gzip.compress(raw, mtime=0)
            if len(compressed) > MAX_AUTOTUNE_BUNDLE_BYTES:
                raise RunnerError("compressed autotune request is too large")
            tune_payload = base64.b64encode(compressed).decode()
            autotune_materialize = (
                "base64 -d >\"$experiment_dir/autotune-request.json.gz\" <<'SIMPLE_AUTOTUNE'\n"
                f"{tune_payload}\nSIMPLE_AUTOTUNE\n"
                "gzip -dc \"$experiment_dir/autotune-request.json.gz\" "
                ">\"$experiment_dir/autotune-request.json\"\n"
                "export AUTOTUNE_REQUEST_PATH=\"$experiment_dir/autotune-request.json\"\n"
            )
        return (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "experiment_dir=$(mktemp -d \"${TMPDIR:-/tmp}/simple-experiment.XXXXXX\")\n"
            "trap 'rm -rf -- \"$experiment_dir\"' EXIT\n"
            "base64 -d >\"$experiment_dir/candidate.py\" <<'SIMPLE_CANDIDATE'\n"
            f"{payload}\nSIMPLE_CANDIDATE\n"
            "export CANDIDATE_PATH=\"$experiment_dir/candidate.py\"\n"
            + materialize + autotune_materialize + "cd \"$experiment_dir\"\n" + gate_suffix
        )

    @staticmethod
    def _checks_exhausted(state: dict) -> bool:
        return state["billed_checks_used"] >= state["billed_checks_limit"]

    def _finish_billed_checks(self, state: dict, outcome: str | None = None) -> dict:
        state["status"] = "billed_checks_exhausted"
        if outcome:
            state["last_outcome"] = outcome
        return self._save(state)

    def _attempt(self, state: dict, prepared: dict, autotune_request: dict | None = None) -> dict:
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
            autotune_request=autotune_request,
        )
        request = self._request(state, "prepare")
        try:
            json.dumps(prepared)
            json.dumps(request)
        except (TypeError, ValueError) as exc:
            raise RunnerError(f"candidate evidence is not JSON serializable: {exc}") from exc
        number = state["attempts"] + 1
        attempt = self.path / "attempts" / str(number)
        if attempt.exists():
            dispatch_evidence = {
                "remote-job-id.txt", "result.json", "infrastructure.json",
                "stdout.log", "stderr.log",
            }
            if (state.get("status") != "tuned"
                    or any((attempt / name).exists() for name in dispatch_evidence)):
                raise RunnerError("attempt path already contains retained dispatch evidence")
            shutil.rmtree(attempt)
        attempt.mkdir()
        _write_json(attempt / "request.json", request)
        _write_json(attempt / "agent-result.json", prepared)
        if autotune_request is not None:
            _write_json(attempt / "autotune-request.json", autotune_request)
        _write_json(attempt / "launch.json", {"physical_device": physical_device})
        script = attempt / "run.sh"
        script.write_text(script_text)
        script.chmod(0o700)
        state.update(attempts=number, billed_checks_used=state["billed_checks_used"] + 1,
                     status="infrastructure_unknown", pending_attempt=number)
        state.pop("prepared_checkpoint", None)
        state.pop("autotune_checkpoint", None)
        self._save(state)
        try:
            handle = self.remote.submit(request, script)
        except PreDispatchFailure as exc:
            _write_json(attempt / "infrastructure.json", {
                "diagnostics": str(exc), "phase": "stage",
                "dispatch_state": "not_dispatched",
            })
            return self._rollback_pre_dispatch(state, attempt)
        except ObservationLost as exc:
            _write_json(attempt / "infrastructure.json", {"diagnostics": str(exc), "phase": "dispatch"})
            return self._save(state)
        (attempt / "remote-job-id.txt").write_text(handle + "\n")
        state.pop("retry_prepared", None)
        state.update(status="observing", pending_attempt=number,
                     session_id=prepared.get("session_id", state.get("session_id")))
        self._save(state)
        return self._observe(state, attempt, handle, prepared)

    def _rollback_pre_dispatch(self, state: dict, attempt: Path) -> dict:
        archive_root = self.path / "pre-dispatch"
        archive_root.mkdir(exist_ok=True)
        sequence = 1
        while (archive_root / str(sequence)).exists():
            sequence += 1
        archived = archive_root / str(sequence)
        attempt.replace(archived)
        state["attempts"] -= 1
        state["billed_checks_used"] -= 1
        state["infrastructure_failures"] += 1
        state.update(status="infrastructure_failure",
                     retry_prepared=str(archived / "agent-result.json"))
        for key in ("pending_attempt", "session_id"):
            state.pop(key, None)
        return self._save(state)

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

    @staticmethod
    def _attempt_autotune_request(attempt: Path) -> dict | None:
        path = attempt / "autotune-request.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("schema") != AUTOTUNE_REQUEST_SCHEMA:
            raise RunnerError("retained autotune request is invalid")
        return value

    @staticmethod
    def _validate_winner_evidence(request: dict, result: dict, summary: dict,
                                  expected: dict, winner: str) -> None:
        profile = result.get("profile_summary")
        if (not isinstance(profile, dict)
                or profile.get("schema") != "profiling-skill/a3-round-profile-summary/v1"):
            raise RunnerError("autotune winner lacks an A3 full-gate profile summary")
        benchmark = request.get("benchmark")
        required_cases = request.get("required_correctness_cases")
        expected_profile_cases = request.get("expected_profile_cases")
        expected_entrypoint = request.get("expected_entrypoint")
        expected_declared = request.get("expected_declared_kernel_name")
        contract = AUTOTUNE_BENCHMARK_CONTRACTS.get(benchmark)
        if (contract is None
                or required_cases != contract["correctness_cases"]
                or expected_profile_cases != contract["profile_cases"]
                or expected_entrypoint != contract["entrypoint"]
                or expected_declared != contract["kernel_name"]):
            raise RunnerError("retained autotune benchmark contract is invalid")
        if (profile.get("benchmark") != benchmark
                or profile.get("correctness_cases") != required_cases):
            raise RunnerError("autotune winner correctness evidence is incomplete")

        binding = summary.get("winner_full_gate")
        required_binding = {
            "winner_id", "source_sha256", "candidate_artifact_sha256",
            "profile_summary_sha256",
        }
        if not isinstance(binding, dict) or not required_binding <= set(binding):
            raise RunnerError("autotune winner lacks bound full-gate identity")
        source_sha = expected[winner]["source_sha256"]
        digest = hashlib.sha256(json.dumps(
            profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (binding.get("winner_id") != winner
                or binding.get("source_sha256") != source_sha
                or binding.get("candidate_artifact_sha256") != source_sha
                or binding.get("profile_summary_sha256") != digest):
            raise RunnerError("autotune winner full-gate identity does not match its evidence")

        fusion = profile.get("fusion")
        fusion_cases = fusion.get("cases") if isinstance(fusion, dict) else None
        if (not isinstance(fusion_cases, list) or len(fusion_cases) != required_cases
                or any(isinstance(case, bool) or not isinstance(case, int)
                       for case in fusion_cases)
                or set(fusion_cases) != set(range(required_cases))
                or fusion.get("logical_launches_per_case") != 1
                or fusion.get("entrypoint") != expected_entrypoint
                or fusion.get("kernel_name") != expected_declared):
            raise RunnerError("autotune winner lacks complete fusion evidence")

        timing = profile.get("timing")
        if not isinstance(timing, dict):
            raise RunnerError("autotune winner lacks official full-gate timing")
        repeats = timing.get("repeats")
        geomean = timing.get("geomean_us")
        declared = timing.get("declared_kernel_name")
        resolved = timing.get("resolved_kernel_name")
        timing_cases = timing.get("cases")
        if (timing.get("profiler") != "msprof-op" or timing.get("metric") != "BasicInfo"
                or isinstance(repeats, bool) or not isinstance(repeats, int)
                or repeats < MIN_AUTOTUNE_PROFILE_REPEATS
                or isinstance(geomean, bool) or not isinstance(geomean, (int, float))
                or not math.isfinite(geomean) or geomean <= 0
                or not isinstance(declared, str) or not declared
                or not isinstance(resolved, str) or not resolved
                or timing.get("kernel_name") != resolved
                or declared != expected_declared
                or not isinstance(timing_cases, list) or not timing_cases):
            raise RunnerError("autotune winner lacks official full-gate timing")
        case_ids = []
        medians = []
        for case in timing_cases:
            samples = case.get("samples_us") if isinstance(case, dict) else None
            case_id = case.get("case") if isinstance(case, dict) else None
            median = case.get("median_us") if isinstance(case, dict) else None
            if (isinstance(case_id, bool) or not isinstance(case_id, int)
                    or not 0 <= case_id < required_cases
                    or not isinstance(samples, list) or len(samples) != repeats
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(item) or item <= 0 for item in samples)
                    or isinstance(median, bool) or not isinstance(median, (int, float))
                    or not math.isfinite(median) or median <= 0
                    or not math.isclose(median, statistics.median(samples),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise RunnerError("autotune winner timing cases lack repeated samples")
            case_ids.append(case_id)
            medians.append(median)
        if case_ids != expected_profile_cases:
            raise RunnerError("autotune winner timing cases do not match ranked inventory")
        expected_geomean = math.exp(sum(math.log(value) for value in medians) / len(medians))
        if not math.isclose(geomean, expected_geomean, rel_tol=1e-12, abs_tol=1e-12):
            raise RunnerError("autotune winner full-gate geomean does not match case medians")

        captures = timing.get("captures")
        expected_captures = [
            (case["case"], iteration, sample)
            for case in timing_cases
            for iteration, sample in enumerate(case["samples_us"])
        ]
        digest_pattern = re.compile(r"^[0-9a-f]{64}$")
        if (not isinstance(captures, list) or not captures
                or len(captures) != len(expected_captures)):
            raise RunnerError("autotune winner lacks complete msprof capture inventory")
        for capture, (case_id, iteration, sample) in zip(captures, expected_captures):
            duration = capture.get("duration_us") if isinstance(capture, dict) else None
            if (not isinstance(capture, dict)
                    or (capture.get("case"), capture.get("iteration")) != (case_id, iteration)
                    or capture.get("kernel_name") != resolved
                    or capture.get("declared_kernel_name") != expected_declared
                    or capture.get("resolved_kernel_name") != resolved
                    or not isinstance(duration, (int, float)) or isinstance(duration, bool)
                    or not math.isfinite(duration) or duration <= 0
                    or not math.isclose(duration, sample, rel_tol=1e-12, abs_tol=1e-12)
                    or not isinstance(capture.get("evidence_sha256"), str)
                    or not digest_pattern.fullmatch(capture["evidence_sha256"])
                    or not isinstance(capture.get("msprof_log_sha256"), str)
                    or not digest_pattern.fullmatch(capture["msprof_log_sha256"])):
                raise RunnerError("autotune winner has invalid msprof capture evidence")

        pipes = profile.get("pipe_activity")
        observations = pipes.get("observations") if isinstance(pipes, dict) else None
        provenance = pipes.get("provenance") if isinstance(pipes, dict) else None
        if (not isinstance(pipes, dict) or pipes.get("product") != "a3"
                or pipes.get("evidence_kind") != "per-block-pipe-activity"
                or not isinstance(observations, list) or not observations
                or not isinstance(provenance, dict)
                or provenance.get("kernel_selector") != resolved):
            raise RunnerError("autotune winner lacks exact-selector pipe evidence")
        for observation in observations:
            activity = observation.get("activity") if isinstance(observation, dict) else None
            saturation = observation.get("saturation") if isinstance(observation, dict) else None
            if (not isinstance(activity, dict) or not activity
                    or not isinstance(saturation, dict)
                    or saturation.get("state") != "unknown"
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(item) or not 0 <= item <= 1
                           for item in activity.values())):
                raise RunnerError("autotune winner has empty or invalid pipe observations")

    @staticmethod
    def _validate_autotune_success(request: dict, result: dict) -> tuple[str, dict]:
        summary = result.get("autotune_summary")
        if not isinstance(summary, dict) or summary.get("schema") != AUTOTUNE_RESULT_SCHEMA:
            raise RunnerError("successful autotune gate omitted its result summary")
        if summary.get("optimization_round") != request.get("optimization_round"):
            raise RunnerError("autotune result does not match its optimization round")
        expected = {option["id"]: option for option in request["options"]}
        rows = summary.get("options")
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise RunnerError("autotune result must report every requested option exactly once")
        actual = {}
        for row in rows:
            if not isinstance(row, dict):
                raise RunnerError("autotune option result must be an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or identifier in actual or identifier not in expected:
                raise RunnerError("autotune result contains an unknown or duplicate option")
            if row.get("source_sha256") != expected[identifier]["source_sha256"]:
                raise RunnerError(f"autotune result source mismatch for {identifier}")
            outcome = row.get("outcome")
            if outcome not in {"success", "candidate_failure"}:
                raise RunnerError(f"autotune option {identifier} has invalid outcome")
            timing = row.get("cheap_device_timing_us")
            if outcome == "success" and (
                    isinstance(timing, bool) or not isinstance(timing, (int, float))
                    or not math.isfinite(timing) or timing <= 0):
                raise RunnerError(f"autotune option {identifier} lacks valid cheap timing")
            if outcome == "candidate_failure" and not isinstance(row.get("diagnostics"), str):
                raise RunnerError(f"failed autotune option {identifier} lacks diagnostics")
            actual[identifier] = row
        if set(actual) != set(expected):
            raise RunnerError("autotune result option inventory does not match the request")
        finalists = summary.get("finalists")
        profiled = summary.get("profiled_finalists")
        if (not isinstance(finalists, list) or not finalists
                or len(finalists) > MAX_AUTOTUNE_FINALISTS
                or len(finalists) != len(set(finalists))
                or any(identifier not in actual or actual[identifier]["outcome"] != "success"
                       for identifier in finalists)):
            raise RunnerError("autotune finalists must be one to three successful options")
        request_order = {identifier: index for index, identifier in enumerate(expected)}
        expected_finalists = sorted(
            (identifier for identifier, row in actual.items() if row["outcome"] == "success"),
            key=lambda identifier: (
                actual[identifier]["cheap_device_timing_us"], request_order[identifier]
            ),
        )[:MAX_AUTOTUNE_FINALISTS]
        if finalists != expected_finalists:
            raise RunnerError("autotune finalists are not the fastest successful options")
        if profiled != finalists:
            raise RunnerError("only and all finalists must receive official msprof timing")
        profiles = summary.get("finalist_profiles")
        if not isinstance(profiles, dict) or set(profiles) != set(finalists):
            raise RunnerError("autotune result must retain official evidence for every finalist")
        official_timings = {}
        for identifier, profile in profiles.items():
            timing = profile.get("geomean_us") if isinstance(profile, dict) else None
            if (isinstance(timing, bool) or not isinstance(timing, (int, float))
                    or not math.isfinite(timing) or timing <= 0):
                raise RunnerError(f"finalist {identifier} lacks valid msprof timing")
            repeats = profile.get("repeats")
            if (isinstance(repeats, bool) or not isinstance(repeats, int)
                    or repeats < MIN_AUTOTUNE_PROFILE_REPEATS):
                raise RunnerError(
                    f"finalist {identifier} requires at least "
                    f"{MIN_AUTOTUNE_PROFILE_REPEATS} msprof repeats"
                )
            official_timings[identifier] = timing
        winner = summary.get("winner_id")
        expected_winner = min(
            finalists, key=lambda identifier: (
                official_timings[identifier], finalists.index(identifier)
            )
        )
        if winner != expected_winner:
            raise RunnerError("autotune winner is not the fastest official finalist")
        if summary.get("winner_source_sha256") != expected[winner]["source_sha256"]:
            raise RunnerError("autotune winner source identity does not match the request")
        ExperimentCell._validate_winner_evidence(
            request, result, summary, expected, winner,
        )
        commands = summary.get("commands")
        if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
            raise RunnerError("autotune result commands must be a string list")
        return expected[winner]["source"], summary

    @staticmethod
    def _public_autotune_request(request: dict) -> dict:
        return {
            **request,
            "options": [
                {key: value for key, value in option.items() if key != "source"}
                for option in request["options"]
            ],
        }

    def _accept_result(self, state: dict, attempt: Path, prepared: dict, result: dict) -> dict:
        outcome = result.get("outcome")
        if outcome == "success":
            autotune = self._attempt_autotune_request(attempt)
            if autotune is not None:
                try:
                    self._validate_autotune_success(autotune, result)
                except RunnerError as exc:
                    state.pop("pending_attempt", None)
                    state.update(
                        status="gate_contract_failure", last_outcome=outcome,
                        gate_diagnostics=str(exc),
                    )
                    self._save(state)
                    raise
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
            autotune = self._attempt_autotune_request(attempt)
            selected = prepared["candidate"]
            if autotune is not None:
                selected, _ = self._validate_autotune_success(autotune, result)
            request = {**self._request(state, "analyze"), "candidate": selected,
                       "structural_candidate": prepared["candidate"],
                       "autotune_request": (
                           self._public_autotune_request(autotune)
                           if autotune is not None else None
                       ),
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
        autotune = self._attempt_autotune_request(attempt)
        selected = prepared["candidate"]
        if autotune is not None:
            selected, autotune_summary = self._validate_autotune_success(autotune, result)
        files = {
            "candidate.py": selected, "run.sh": (attempt / "run.sh").read_text(),
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
        if autotune is not None:
            public_request = self._public_autotune_request(autotune)
            files.update({
                "structural-candidate.py": prepared["candidate"],
                "autotune-request.json": json.dumps(public_request, indent=2, sort_keys=True) + "\n",
                "autotune-result.json": json.dumps(autotune_summary, indent=2, sort_keys=True) + "\n",
                "autotune-commands.jsonl": "".join(
                    json.dumps({"command": item}, sort_keys=True) + "\n"
                    for item in autotune_summary["commands"]
                ),
            })
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
        if state.get("status") in {"tuning", "tune_pending", "tuned"}:
            raise RunnerError("cell has pending tune work; use resume")
        if state.get("pending_attempt"):
            raise RunnerError("cell has pending work; use resume")
        if state["status"] in {
                "complete", "billed_checks_exhausted", "gate_contract_failure",
                "input_integrity_failure"} or self._budget_exhausted(state):
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
        retained = state.get("retry_prepared")
        last_agent = (Path(retained) if retained else
                      self.path / "attempts" / str(state["attempts"]) / "agent-result.json")
        prepared = (json.loads(last_agent.read_text())
                    if state["status"] == "infrastructure_failure" and last_agent.is_file()
                    else None)
        autotune_request = None
        if prepared is not None:
            retained_tune = last_agent.with_name("autotune-request.json")
            if retained_tune.is_file():
                autotune_request = json.loads(retained_tune.read_text())
        while True:
            if self._checks_exhausted(state):
                return self._finish_billed_checks(state, state.get("last_outcome"))
            if prepared is None:
                prepared = self.agent.prepare(self._request(state, "prepare"))
                self._verify_task_snapshot(state)
                if state.get("autotune_enabled"):
                    state = self._checkpoint_prepared(state, prepared)
                    state, autotune_request = self._pending_tune(state, prepared)
                    if state["status"] == "tune_pending":
                        return state
                    if state["status"] == "candidate_failure":
                        if self._budget_exhausted(state):
                            return state
                        prepared = None
                        continue
                    if state["status"] == "input_integrity_failure":
                        return state
                else:
                    autotune_request = None
            state = self._attempt(state, prepared, autotune_request)
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
            autotune_request = None

    def resume(self) -> dict:
        state = self._state()
        self._verify_task_snapshot(state)
        pending = state.get("pending_attempt")
        if not pending and state.get("status") in {"tuning", "tune_pending", "tuned"}:
            retained_status = state["status"]
            self._launch_device(state)
            try:
                self._preflight()
            except ObservationLost as exc:
                state["status"] = "tuned" if retained_status == "tuned" else "tune_pending"
                state["tune_diagnostics"] = str(exc)
                return self._save(state)
            prepared = self._load_prepared_checkpoint(state)
            if retained_status == "tuned":
                autotune_request = self._load_autotune_checkpoint(state, prepared)
            else:
                state, autotune_request = self._pending_tune(state, prepared)
                if state["status"] == "tune_pending":
                    return state
                if state["status"] == "candidate_failure":
                    return state
                if state["status"] == "input_integrity_failure":
                    return state
            state.pop("tune_diagnostics", None)
            return self._attempt(state, prepared, autotune_request)
        if not pending:
            raise RunnerError("cell has no pending retained work to resume")
        attempt = self.path / "attempts" / str(pending)
        prepared = json.loads((attempt / "agent-result.json").read_text())
        if (attempt / "result.json").exists():
            result = json.loads((attempt / "result.json").read_text())
            return self._accept_result(state, attempt, prepared, result)
        handle_path = attempt / "remote-job-id.txt"
        if not handle_path.is_file():
            raise RunnerError("dispatch outcome is unknown; no handle exists to observe")
        return self._observe(state, attempt, handle_path.read_text().strip(), prepared)

    def recover(self) -> dict:
        """Repair only a checkpoint proven to have failed before dispatch."""
        state = self._state()
        self._verify_task_snapshot(state)
        pending = state.get("pending_attempt")
        if state.get("status") != "infrastructure_unknown" or not pending:
            raise RunnerError("cell has no unknown dispatch checkpoint to recover")
        attempt = self.path / "attempts" / str(pending)
        if (attempt / "remote-job-id.txt").exists():
            raise RunnerError("retained handle exists; use resume")
        evidence_path = attempt / "infrastructure.json"
        if not evidence_path.is_file():
            raise RunnerError("dispatch is not proven to be pre-dispatch")
        evidence = json.loads(evidence_path.read_text())
        proven = evidence.get("dispatch_state") == "not_dispatched"
        legacy_e2big = (evidence.get("phase") == "dispatch"
                        and "[Errno 7] Argument list too long: 'ssh'"
                        in str(evidence.get("diagnostics", "")))
        if not (proven or legacy_e2big):
            raise RunnerError("dispatch is not proven to be pre-dispatch")
        return self._rollback_pre_dispatch(state, attempt)

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
    parser.add_argument("command", choices=("init", "run", "resume", "recover", "report"))
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
