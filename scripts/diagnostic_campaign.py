#!/usr/bin/env python3
"""Short, transport-neutral one-shot Triton diagnostic campaigns."""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout, as_completed
from pathlib import Path
from typing import Protocol

try:
    from scripts.campaign import TREATMENT_SKILLS
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from campaign import TREATMENT_SKILLS

TREATMENTS = ("cannbot", "project-cannbot", "project-guarded")
COUNTED = {
    "no_submission", "protocol_error", "source_error", "compile_error",
    "runtime_error", "correctness_error", "candidate_timeout", "success",
}
OBSERVED = {"diagnostic_retrieval_error", "request_budget_exhausted", "agent_timeout"}
INFRASTRUCTURE = {
    "setup_error", "pre_dispatch_infra", "transport_or_observer_error",
    "device_or_runtime_infra", "model_service_error",
}


class DiagnosticError(RuntimeError):
    pass


class Launcher(Protocol):
    def launch(self, request: dict, timeout_seconds: int) -> dict: ...


class TerminalHook(Protocol):
    def check(self, request: dict, timeout_seconds: int) -> dict: ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_prompt_bytes(request: dict) -> bytes:
    """Decode and verify the in-band prompt consumed by launcher protocols."""
    if request.get("protocol_version") != 2:
        raise DiagnosticError("in-band prompt requests require protocol version 2")
    prompt = request.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("encoding") != "base64":
        raise DiagnosticError("request prompt must use the base64 in-band contract")
    try:
        content = base64.b64decode(prompt.get("data", ""), validate=True)
    except (ValueError, TypeError) as error:
        raise DiagnosticError("request prompt is not valid base64") from error
    digest = hashlib.sha256(content).hexdigest()
    if prompt.get("sha256") != digest or request.get("prompt_sha256") != digest:
        raise DiagnosticError("request prompt hash mismatch")
    return content


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _submit_daemon(function, *args) -> Future:
    """Run a cell in a daemon thread so interrupted CLI shutdown cannot join it."""
    future = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = function(*args)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    threading.Thread(target=run, daemon=True, name="diagnostic-cell").start()
    return future


class _ProcessRegistry:
    """Track command process groups so an interrupted campaign can stop them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self._cancelled = False

    def run(self, command: list[str], request: dict,
            timeout_seconds: int) -> subprocess.CompletedProcess:
        with self._lock:
            if self._cancelled:
                raise OSError("command hook was cancelled")
            # Spawn and registration share the cancellation lock: cancel()
            # either sees this process or prevents it from starting.
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=True,
            )
            self._processes.add(process)
        try:
            stdout, stderr = process.communicate(json.dumps(request), timeout=timeout_seconds)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            self._terminate(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command, timeout_seconds, output=stdout, stderr=stderr,
            )
        finally:
            # A successful protocol response ends the one-shot command's whole
            # lifetime. Do not allow detached same-group writers to survive the
            # leader and mutate evidence after the launcher returns.
            self._terminate(process)
            with self._lock:
                self._processes.discard(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        # The group can outlive its leader and keep stdout/stderr pipes open.
        # Always target the process group created at spawn, even after poll()
        # reports that the leader itself has exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(5):
            time.sleep(0.01)
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)


def _invoke(command: list[str], request: dict, timeout_seconds: int,
            registry: _ProcessRegistry | None = None) -> dict:
    started = time.time()
    try:
        if registry is None:
            run = subprocess.run(command, input=json.dumps(request), text=True,
                                 capture_output=True, timeout=timeout_seconds, check=False)
        else:
            run = registry.run(command, request, timeout_seconds)
    except subprocess.TimeoutExpired as error:
        def timeout_text(value: str | bytes | None) -> str:
            if isinstance(value, bytes):
                return value.decode(errors="replace")
            return value or ""

        return {"status": "timeout", "invocation_timeout": True,
                "stdout": timeout_text(error.stdout),
                "stderr": timeout_text(error.stderr),
                "elapsed_seconds": time.time() - started}
    except OSError as error:
        return {"status": "transport_or_observer_error", "diagnostics": str(error),
                "elapsed_seconds": time.time() - started}
    try:
        document = json.loads(run.stdout)
        if not isinstance(document, dict):
            raise ValueError("response is not an object")
    except (json.JSONDecodeError, ValueError) as error:
        return {"status": "transport_or_observer_error",
                "diagnostics": f"invalid response: {error}", "stdout": run.stdout,
                "stderr": run.stderr, "exit_code": run.returncode,
                "elapsed_seconds": time.time() - started}
    if run.returncode and document.get("status") == "ok":
        return {"status": "transport_or_observer_error",
                "diagnostics": f"process claimed ok but exited {run.returncode}",
                "stdout": run.stdout, "stderr": run.stderr,
                "exit_code": run.returncode, "elapsed_seconds": time.time() - started}
    return {**document, "stdout": run.stdout, "stderr": run.stderr,
            "exit_code": run.returncode, "elapsed_seconds": time.time() - started}


class CommandLauncher:
    """Invoke one externally isolated agent turn through a JSON protocol."""

    def __init__(self, command: list[str]):
        if not command:
            raise DiagnosticError("agent command is required")
        self.command = command
        self._processes = _ProcessRegistry()

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        request_prompt_bytes(request)
        return _invoke(self.command, request, timeout_seconds, self._processes)

    def cancel(self) -> None:
        self._processes.cancel()


class CommandTerminalHook:
    """Invoke the host-owned correctness check independently of the agent."""

    def __init__(self, command: list[str]):
        if not command:
            raise DiagnosticError("terminal command is required")
        self.command = command
        self._processes = _ProcessRegistry()

    def check(self, request: dict, timeout_seconds: int) -> dict:
        result = _invoke(self.command, request, timeout_seconds, self._processes)
        if result.get("status") == "timeout" and result.get("invocation_timeout") is True:
            return {**result, "status": "transport_or_observer_error"}
        return result

    def resume(self, request: dict, handle: str, timeout_seconds: int) -> dict:
        """Observe a retained job, distinguishing local and workload timeouts."""
        result = _invoke(
            self.command,
            {**request, "operation": "terminal_observe", "handle": handle},
            timeout_seconds,
            self._processes,
        )
        if result.get("status") == "timeout" and result.get("invocation_timeout") is True:
            return {**result, "status": "transport_or_observer_error", "handle": handle}
        if (result.get("status") == "infrastructure_error"
                and result.get("failure_type") in {"observer_error", "transport_error"}):
            return {**result, "handle": handle}
        if result.get("status") in INFRASTRUCTURE:
            return {**result, "handle": handle}
        return result

    def cancel(self) -> None:
        self._processes.cancel()


def validate_manifest(manifest: dict) -> None:
    required = {"prompt", "prompt_sha256", "model", "model_sha256", "treatments"}
    if not required.issubset(manifest):
        raise DiagnosticError("diagnostic manifest is incomplete")
    prompt = Path(manifest["prompt"])
    if not prompt.is_file() or sha256_file(prompt) != manifest["prompt_sha256"]:
        raise DiagnosticError("prompt is missing or changed")
    model_digest = hashlib.sha256(json.dumps(manifest["model"], sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    if model_digest != manifest["model_sha256"]:
        raise DiagnosticError("model configuration changed")
    if set(manifest["treatments"]) != set(TREATMENTS):
        raise DiagnosticError("exactly the three diagnostic treatments are required")
    for name in TREATMENTS:
        treatment = manifest["treatments"][name]
        skills = treatment.get("skills")
        if not isinstance(skills, list) or not skills or len(skills) != len(set(skills)):
            raise DiagnosticError(f"invalid isolated skill inventory for {name}")
        if skills != list(TREATMENT_SKILLS[name]):
            raise DiagnosticError(f"noncanonical isolated skill inventory for {name}")
        if not isinstance(treatment.get("skill_sha256"), dict):
            raise DiagnosticError(f"skill hashes are required for {name}")
        if set(skills) != set(treatment["skill_sha256"]):
            raise DiagnosticError(f"skill inventory and hashes differ for {name}")


def classify(agent: dict, terminal: dict | None, workspace: Path) -> tuple[str, str]:
    status = agent.get("status")
    if status == "timeout":
        return "agent_timeout", "observed"
    if status == "infrastructure_error":
        return str(agent.get("failure_type") or status), "infrastructure"
    if status in INFRASTRUCTURE:
        return str(status), "infrastructure"
    if status in OBSERVED:
        return str(status), "observed"
    if status == "submission_error":
        return str(status), "counted"
    if status != "ok":
        return "protocol_error", "counted"
    usage = agent.get("controller_usage")
    expected = ["check", "--scope", "development", "--round", "1"]
    calls = usage.get("calls") if isinstance(usage, dict) else None
    if (not isinstance(usage, dict) or usage.get("billed") != 1
            or calls != [{"arguments": expected}]):
        return "protocol_error", "counted"
    candidate = workspace / "candidate.py"
    candidate_manifest = workspace / "candidate.manifest.json"
    if not candidate.is_file() or not candidate_manifest.is_file():
        return "no_submission", "counted"
    if terminal is None:
        return "protocol_error", "counted"
    terminal_status = terminal.get("status")
    if terminal_status == "infrastructure_error":
        return str(terminal.get("failure_type") or terminal_status), "infrastructure"
    if terminal_status in INFRASTRUCTURE:
        return str(terminal_status), "infrastructure"
    if terminal_status == "timeout":
        return "candidate_timeout", "counted"
    if terminal_status in {"submission_error", "source_error", "compile_error",
                           "runtime_error", "correctness_error"}:
        return str(terminal_status), "counted"
    if terminal_status == "ok" and terminal.get("passed") is True:
        return "success", "counted"
    return "diagnostic_retrieval_error", "observed"


class DiagnosticCampaign:
    """Run four sequential waves with three concurrent treatment cells each."""

    def __init__(self, manifest: dict, root: Path, launcher: Launcher,
                 terminal: TerminalHook, *, waves: int = 4,
                 agent_timeout: int = 360, cell_timeout: int = 600,
                 wave_timeout: int = 600, ledger_metadata: dict | None = None,
                 campaign_id: str | None = None,
                 campaign_identity: dict | None = None):
        validate_manifest(manifest)
        if waves != 4 or agent_timeout <= 0 or cell_timeout <= 0 or wave_timeout <= 0:
            raise DiagnosticError("four waves and positive timeouts are required")
        self.manifest, self.root = manifest, root
        self.launcher, self.terminal = launcher, terminal
        self.waves, self.agent_timeout, self.cell_timeout = waves, agent_timeout, cell_timeout
        self.wave_timeout = wave_timeout
        self.ledger_metadata = copy.deepcopy(ledger_metadata or {})
        self.campaign_id = campaign_id or str(uuid.uuid4())
        self.campaign_identity = copy.deepcopy(campaign_identity)
        self.ledger_path = root / "ledger.json"

    def _cell(self, wave: int, treatment: str, attempt: int,
              available_seconds: float | None = None) -> dict:
        cell_id = f"wave-{wave}-{treatment}"
        workspace = self.root / "cells" / cell_id / f"attempt-{attempt}" / "workspace"
        replaying = any(workspace.parent.glob("terminal-result-*.json"))
        workspace.mkdir(parents=True, exist_ok=replaying)
        started = time.time()
        monotonic_started = time.monotonic()
        treatment_record = copy.deepcopy(self.manifest["treatments"][treatment])
        request = {
            "protocol_version": 2, "operation": "one_shot", "cell_id": cell_id,
            "wave": wave, "attempt": attempt, "treatment": treatment,
            "workspace": str(workspace), "prompt": {
                "encoding": "base64",
                "data": base64.b64encode(self._prompt_bytes).decode("ascii"),
                "sha256": self.manifest["prompt_sha256"],
            },
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model": copy.deepcopy(self.manifest["model"]),
            "model_sha256": self.manifest["model_sha256"],
            "skills": list(treatment_record["skills"]),
            "skill_sha256": copy.deepcopy(treatment_record["skill_sha256"]),
            "benchmark": "streaming-matmul-add", "development_cases": [1],
            "controller_contract": {"billed_limit": 1,
                "command": ["check", "--scope", "development", "--round", "1"]},
        }
        cap = min(self.cell_timeout, available_seconds or self.cell_timeout)
        agent = self._durable_agent_launch(
            request, max(1, int(min(self.agent_timeout, cap))))
        elapsed = time.monotonic() - monotonic_started
        terminal = None
        candidate = workspace / "candidate.py"
        candidate_manifest = workspace / "candidate.manifest.json"
        submission_hashes = agent.get("candidate_sha256")
        if not isinstance(submission_hashes, dict):
            submission_hashes = {}
            for name in ("candidate.py", "candidate.manifest.json"):
                path = workspace / name
                if path.is_file():
                    submission_hashes[name] = sha256_file(path)
        if elapsed >= cap:
            agent = {"status": "infrastructure_error",
                     "failure_type": "wave_budget_exhausted",
                     "diagnostics": "agent consumed the remaining cell budget",
                     "reported_agent": agent}
        elif candidate.is_file() and candidate_manifest.is_file():
            terminal_request = {
                "protocol_version": 1, "operation": "terminal_check", "cell_id": cell_id,
                "workspace": str(workspace), "benchmark": "streaming-matmul-add",
                "cases": list(range(7)), "terminal_attempt": attempt,
                "candidate_sha256": copy.deepcopy(submission_hashes),
            }
            if (workspace.parent / "frozen-submission").exists():
                terminal_request["candidate_source"] = "frozen-submission"
            try:
                terminal = self._durable_terminal_check(
                    terminal_request, max(1, int(cap - elapsed)))
            except Exception as error:
                terminal = {"status": "transport_or_observer_error",
                            "diagnostics":
                                f"terminal hook raised {type(error).__name__}: {error}"}
        outcome, category = classify(agent, terminal, workspace)
        return {
            "cell_id": cell_id, "wave": wave, "attempt": attempt, "treatment": treatment,
            "outcome": outcome, "category": category, "agent": agent, "terminal": terminal,
            "candidate_sha256": submission_hashes, "started_at_epoch": started,
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"],
            "skill_sha256": copy.deepcopy(treatment_record["skill_sha256"]),
            "elapsed_seconds": time.time() - started,
        }

    def _durable_agent_launch(self, request: dict, timeout_seconds: int) -> dict:
        workspace = Path(request["workspace"])
        receipt = workspace.parent / "controller-agent-result.json"
        digest = hashlib.sha256(json.dumps(
            request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if receipt.is_file():
            record = json.loads(receipt.read_text())
            if (record.get("request_sha256") != digest
                    or record.get("state") != "completed"
                    or not isinstance(record.get("result"), dict)):
                raise DiagnosticError("invalid controller agent result receipt")
            return copy.deepcopy(record["result"])
        try:
            result = copy.deepcopy(self.launcher.launch(request, timeout_seconds))
        except Exception as error:
            result = {"status": "transport_or_observer_error",
                      "diagnostics": f"launcher raised {type(error).__name__}: {error}"}
        _atomic_json(receipt, {"protocol_version": 1, "state": "completed",
                               "request_sha256": digest, "result": result})
        return result

    def _durable_terminal_check(self, request: dict, timeout_seconds: int) -> dict:
        """Replay a completed terminal result across a wave-checkpoint crash."""
        workspace = Path(request["workspace"])
        digest = hashlib.sha256(json.dumps(
            request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt_request = copy.deepcopy(request)
        receipt_timeout = timeout_seconds
        prior_result = None
        terminal_attempt = request.get("terminal_attempt", workspace.parent.name)
        receipt = workspace.parent / f"terminal-result-{terminal_attempt}-{digest[:16]}.json"
        if receipt.is_file():
            record = json.loads(receipt.read_text())
            if record.get("request_sha256") != digest:
                raise DiagnosticError("terminal result receipt request mismatch")
            if record.get("state") == "completed" and isinstance(record.get("result"), dict):
                return copy.deepcopy(record["result"])
            if (record.get("state") == "uncertain"
                    and isinstance(record.get("result"), dict)
                    and (not record["result"].get("handle")
                         or record["result"].get(
                             "manual_reconciliation_required") is True)):
                return copy.deepcopy(record["result"])
            if record.get("state") not in {"started", "uncertain"}:
                raise DiagnosticError("invalid terminal result receipt")
            persisted_timeout = record.get("timeout_seconds")
            if (isinstance(persisted_timeout, bool)
                    or not isinstance(persisted_timeout, int) or persisted_timeout < 1):
                raise DiagnosticError("invalid started terminal timeout")
            persisted_request = record.get("request")
            if (not isinstance(persisted_request, dict)
                    or hashlib.sha256(json.dumps(
                        persisted_request, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest() != digest):
                raise DiagnosticError("invalid started terminal request")
            receipt_request = copy.deepcopy(persisted_request)
            receipt_timeout = persisted_timeout
            prior_result = record.get("result")
            retained_request = (prior_result.get("retained_terminal_request")
                                if isinstance(prior_result, dict) else None)
            if retained_request is not None:
                if not isinstance(retained_request, dict):
                    raise DiagnosticError("invalid retained terminal request")
                request = {**copy.deepcopy(persisted_request),
                           "retained_terminal_request": copy.deepcopy(retained_request),
                           "terminal_attempt": prior_result.get(
                               "terminal_attempt", persisted_request.get("terminal_attempt"))}
            else:
                request = copy.deepcopy(persisted_request)
            timeout_seconds = min(timeout_seconds, persisted_timeout)
        else:
            _atomic_json(receipt, {"protocol_version": 1, "state": "started",
                                   "request_sha256": digest,
                                   "request": receipt_request,
                                   "timeout_seconds": timeout_seconds})
        resume = getattr(self.terminal, "resume", None)
        prior_handle = (prior_result.get("handle")
                        if isinstance(prior_result, dict) else None)
        if callable(resume) and isinstance(prior_handle, str) and prior_handle:
            resume_request = copy.deepcopy(receipt_request)
            if isinstance(prior_result.get("terminal_request_timeout"), int):
                resume_request["terminal_request_timeout"] = prior_result[
                    "terminal_request_timeout"]
            result = copy.deepcopy(resume(resume_request, prior_handle, timeout_seconds))
        else:
            result = copy.deepcopy(self.terminal.check(request, timeout_seconds))
        uncertain_dispatch = (
            result.get("status") == "transport_or_observer_error"
            and result.get("invocation_timeout") is True
            and not result.get("handle"))
        manual_failure = result.get("manual_reconciliation_required") is True
        retriable_observation = bool(result.get("handle")) and (
                result.get("status") == "transport_or_observer_error"
                or (result.get("status") == "infrastructure_error"
                    and result.get("failure_type") in {
                        "observer_error", "transport_error"}))
        if uncertain_dispatch or manual_failure:
            if uncertain_dispatch:
                result = {**result, "manual_reconciliation_required": True,
                          "diagnostics": "terminal outcome is uncertain; supply a durable "
                                         "handle or terminal result before resuming"}
            _atomic_json(receipt, {"protocol_version": 1, "state": "uncertain",
                                   "request_sha256": digest,
                                   "request": receipt_request,
                                   "timeout_seconds": receipt_timeout,
                                   "result": result})
        elif not retriable_observation:
            _atomic_json(receipt, {"protocol_version": 1, "state": "completed",
                                   "request_sha256": digest, "result": result})
        else:
            _atomic_json(receipt, {"protocol_version": 1, "state": "started",
                                   "request_sha256": digest,
                                   "request": receipt_request,
                                   "timeout_seconds": receipt_timeout,
                                   "result": result})
        return result

    def _guarded_cell(self, wave: int, treatment: str, attempt: int,
                      available_seconds: float | None = None) -> dict:
        """Keep an unexpected cell implementation error inside durable evidence."""
        try:
            return self._cell(wave, treatment, attempt, available_seconds)
        except Exception as error:
            treatment_record = copy.deepcopy(self.manifest["treatments"][treatment])
            return {
                "cell_id": f"wave-{wave}-{treatment}", "wave": wave,
                "attempt": attempt, "treatment": treatment,
                "outcome": "transport_or_observer_error", "category": "infrastructure",
                "agent": {"status": "transport_or_observer_error",
                          "diagnostics":
                              f"cell hook raised {type(error).__name__}: {error}"},
                "terminal": None, "candidate_sha256": {},
                "started_at_epoch": time.time(),
                "prompt_sha256": self.manifest["prompt_sha256"],
                "model_sha256": self.manifest["model_sha256"],
                "skill_sha256": copy.deepcopy(treatment_record["skill_sha256"]),
                "elapsed_seconds": 0,
            }

    def _freeze_prompt(self, manifest: dict, name: str = "prompt.md") -> None:
        source = Path(manifest["prompt"])
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["prompt_sha256"]:
            raise DiagnosticError("prompt changed while campaign inputs were frozen")
        destination = self.root / "inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != manifest["prompt_sha256"]:
                raise DiagnosticError(f"existing frozen prompt changed: {destination}")
            self._prompt_bytes = destination.read_bytes()
            manifest["prompt"] = str(destination)
            return
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        self._prompt_bytes = content
        manifest["prompt"] = str(destination)

    def _adaptive_identity(self) -> dict:
        return {
            "campaign": self.campaign_identity,
            "model_sha256": self.manifest["model_sha256"],
            "treatments": copy.deepcopy(self.manifest["treatments"]),
        }

    def _load_adaptive_ledger(self) -> dict:
        try:
            ledger = json.loads(self.ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DiagnosticError(f"cannot read adaptive ledger: {error}") from error
        if not isinstance(ledger, dict) or ledger.get("adaptive") is not True:
            raise DiagnosticError("ledger is not an adaptive campaign")
        if ledger.get("campaign_identity") != self._adaptive_identity():
            raise DiagnosticError("model, treatment, skill, or campaign configuration drift")
        receipts = ledger.get("curation_receipts")
        waves = ledger.get("waves")
        if (not isinstance(receipts, list) or not isinstance(waves, list)
                or len(receipts) > len(waves)):
            raise DiagnosticError("adaptive ledger has invalid curation history")
        for index, receipt in enumerate(receipts, 1):
            self._validate_curation_receipt(
                receipt, {**ledger, "waves": waves[:index]})
        return ledger

    def _load_fixed_ledger(self) -> dict:
        try:
            ledger = json.loads(self.ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DiagnosticError(f"cannot read fixed campaign ledger: {error}") from error
        if (not isinstance(ledger, dict) or ledger.get("adaptive") is True
                or ledger.get("protocol_version") != 1
                or ledger.get("campaign_id") != self.campaign_id
                or ledger.get("fixed_campaign_identity") != self._adaptive_identity()
                or ledger.get("status") != "reschedule_pending"
                or not isinstance(ledger.get("waves"), list)
                or not isinstance(ledger.get("reschedule"), list)):
            raise DiagnosticError("invalid or drifted fixed campaign ledger")
        valid_cells = {f"wave-{wave}-{name}"
                       for wave in range(1, self.waves + 1) for name in TREATMENTS}
        if (len(ledger["reschedule"]) != len(set(ledger["reschedule"]))
                or not set(ledger["reschedule"]).issubset(valid_cells)):
            raise DiagnosticError("fixed campaign has invalid pending cells")
        return ledger

    def resume_fixed(self) -> dict:
        """Resume only pending cells in a validated fixed-run ledger."""
        ledger = self._load_fixed_ledger()
        for wave in range(1, self.waves + 1):
            if not any(cell.startswith(f"wave-{wave}-")
                       for cell in ledger["reschedule"]):
                continue
            records = [item for item in ledger["waves"] if item.get("wave") == wave]
            if len(records) != 1:
                raise DiagnosticError("fixed campaign wave record is missing or ambiguous")
            prompt = Path(records[0]["prompt"])
            if (not prompt.is_file()
                    or sha256_file(prompt) != records[0].get("prompt_sha256")):
                raise DiagnosticError("fixed campaign prompt is missing or changed")
            self._prompt_bytes = prompt.read_bytes()
            self.manifest["prompt"] = str(prompt)
            self._resume_wave(ledger, wave)
        ledger["status"] = "reschedule_pending" if ledger["reschedule"] else "complete"
        _atomic_json(self.ledger_path, ledger)
        return ledger

    @staticmethod
    def _wave_sha256(wave_record: dict) -> str:
        return hashlib.sha256(json.dumps(
            wave_record, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()

    @classmethod
    def _validate_curation_receipt(cls, receipt: dict, ledger: dict) -> None:
        wave = len(ledger["waves"])
        citations = receipt.get("stable_ref_citations") if isinstance(receipt, dict) else None
        queries = receipt.get("librarian_query_ids") if isinstance(receipt, dict) else None
        if (not isinstance(receipt, dict) or receipt.get("wave") != wave
                or receipt.get("campaign_id") != ledger["campaign_id"]
                or receipt.get("wave_sha256") != cls._wave_sha256(ledger["waves"][-1])
                or not isinstance(receipt.get("curator_operation_id"), str)
                or not receipt["curator_operation_id"].strip()
                or receipt.get("accepted") is not True
                or not isinstance(citations, list) or not citations
                or not all(isinstance(item, str) and item.startswith("ref://")
                           and bool(item.removeprefix("ref://").strip())
                           for item in citations)
                or not isinstance(queries, list) or not queries
                or not all(isinstance(item, str) and item.strip() for item in queries)):
            raise DiagnosticError("invalid curation receipt")

    def acknowledge_curation(self, receipt: dict) -> dict:
        ledger = self._load_adaptive_ledger()
        if ledger["status"] != "awaiting_curation":
            raise DiagnosticError("campaign is not awaiting curation")
        wave = len(ledger["waves"])
        self._validate_curation_receipt(receipt, ledger)
        ledger["curation_receipts"].append(copy.deepcopy(receipt))
        ledger["status"] = "complete" if wave == self.waves else "ready_for_next"
        _atomic_json(self.ledger_path, ledger)
        return ledger

    def reconcile_terminal(self, cell_id: str, agent_attempt: int,
                           terminal_attempt: int, *, handle: str | None = None,
                           result: dict | None = None) -> dict:
        """Resolve one uncertain terminal receipt without dispatching a check."""
        parts = cell_id.split("-", 2) if isinstance(cell_id, str) else []
        if (len(parts) != 3 or parts[0] != "wave" or not parts[1].isdigit()
                or parts[2] not in TREATMENTS):
            raise DiagnosticError("invalid reconciliation cell id")
        if (isinstance(agent_attempt, bool) or not isinstance(agent_attempt, int)
                or agent_attempt < 1 or isinstance(terminal_attempt, bool)
                or not isinstance(terminal_attempt, int) or terminal_attempt < 1):
            raise DiagnosticError("reconciliation attempts must be positive integers")
        if (handle is None) == (result is None):
            raise DiagnosticError("supply exactly one terminal handle or result")
        attempt_dir = self.root / "cells" / cell_id / f"attempt-{agent_attempt}"
        matches = []
        for path in attempt_dir.glob("terminal-result-*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            request = record.get("request")
            if (record.get("state") == "uncertain" and isinstance(request, dict)
                    and request.get("terminal_attempt") == terminal_attempt):
                matches.append((path, record))
        if len(matches) != 1:
            raise DiagnosticError("uncertain terminal receipt is missing or ambiguous")
        path, record = matches[0]
        request = record["request"]
        request_digest = hashlib.sha256(json.dumps(
            request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        expected_workspace = attempt_dir / "workspace"
        frozen_submission = attempt_dir / "frozen-submission"
        candidate_root = (frozen_submission
                          if (request.get("candidate_source") == "frozen-submission"
                              or frozen_submission.exists())
                          else expected_workspace)
        candidate_sha256 = request.get("candidate_sha256")
        if (record.get("protocol_version") != 1
                or request.get("protocol_version") != 1
                or request.get("operation") != "terminal_check"
                or request.get("benchmark") != "streaming-matmul-add"
                or request.get("cases") != list(range(7))
                or request.get("candidate_source") not in {
                    None, "frozen-submission"}
                or isinstance(record.get("timeout_seconds"), bool)
                or not isinstance(record.get("timeout_seconds"), int)
                or record["timeout_seconds"] < 1
                or record.get("request_sha256") != request_digest
                or request.get("cell_id") != cell_id
                or Path(str(request.get("workspace", ""))) != expected_workspace
                or not isinstance(candidate_sha256, dict)
                or any(not (candidate_root / name).is_file()
                       for name in ("candidate.py", "candidate.manifest.json"))
                or any(candidate_sha256.get(name) != sha256_file(
                    candidate_root / name)
                       for name in ("candidate.py", "candidate.manifest.json"))
                or not isinstance(record.get("result"), dict)
                or record["result"].get("manual_reconciliation_required") is not True):
            raise DiagnosticError("uncertain terminal receipt identity mismatch")
        if handle is not None:
            if not isinstance(handle, str) or not handle.strip():
                raise DiagnosticError("reconciliation handle must be non-empty")
            reconciled = {**record["result"], "handle": handle}
            reconciled["terminal_request_timeout"] = record["timeout_seconds"]
            reconciled.pop("manual_reconciliation_required", None)
            record.update({"state": "started", "result": reconciled})
        else:
            if not isinstance(result, dict):
                raise DiagnosticError("reconciliation result must be an object")
            status = result.get("status")
            if (status not in {"timeout", "submission_error", "source_error",
                               "compile_error", "runtime_error", "correctness_error", "ok"}
                    or (status == "ok" and result.get("passed") is not True)
                    or result.get("campaign_id") != self.campaign_id
                    or result.get("request_sha256") != request_digest
                    or result.get("cell_id") != cell_id
                    or result.get("terminal_attempt") != terminal_attempt
                    or not isinstance(result.get("handle"), str)
                    or not result["handle"].strip()
                    or result.get("candidate_sha256") != candidate_sha256):
                raise DiagnosticError("reconciliation result is not terminal")
            record.update({"state": "completed", "result": copy.deepcopy(result)})
        _atomic_json(path, record)
        return copy.deepcopy(record)

    def run_wave(self, wave: int) -> dict:
        """Run exactly the next adaptive wave and pause for curation."""
        if isinstance(wave, bool) or not isinstance(wave, int) or not 1 <= wave <= self.waves:
            raise DiagnosticError("wave must be an integer from 1 through 4")
        manifest = copy.deepcopy(self.manifest)
        validate_manifest(manifest)
        if wave == 1 and not self.ledger_path.exists():
            if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
                raise DiagnosticError(f"diagnostic output root is not fresh: {self.root}")
            ledger = {
                **self.ledger_metadata, "protocol_version": 1,
                "campaign_id": self.campaign_id, "adaptive": True,
                "campaign_identity": self._adaptive_identity(),
                "status": "ready_for_next", "waves": [], "reschedule": [],
                "curation_receipts": [],
            }
            _atomic_json(self.ledger_path, ledger)
        else:
            ledger = self._load_adaptive_ledger()
            self.campaign_id = ledger["campaign_id"]
        if ledger["status"] == "running":
            if wave == len(ledger["waves"]) + 1:
                ledger["status"] = "ready_for_next"
            elif wave == len(ledger["waves"]):
                current = ledger["waves"][-1]
                completed = set()
                for cell in current["cells"]:
                    attempts = cell.get("reschedule_attempts", [])
                    latest = attempts[-1] if attempts else cell.get("retry", cell)
                    if latest["category"] == "counted":
                        completed.add(cell["treatment"])
                completed_ids = {f"wave-{wave}-{name}" for name in completed}
                ledger["reschedule"] = sorted(
                    (set(ledger["reschedule"]) - completed_ids) |
                    {f"wave-{wave}-{name}" for name in TREATMENTS if name not in completed}
                )
                ledger["status"] = "reschedule_pending"
            else:
                raise DiagnosticError("running adaptive ledger has inconsistent wave state")
            _atomic_json(self.ledger_path, ledger)
        if ledger["status"] == "reschedule_pending":
            if wave != len(ledger["waves"]):
                raise DiagnosticError(f"campaign must reschedule wave {len(ledger['waves'])}")
            if self.manifest["prompt_sha256"] != ledger["waves"][-1]["prompt_sha256"]:
                raise DiagnosticError("prompt cannot change while a wave is reschedule pending")
            frozen_prompt = Path(ledger["waves"][-1]["prompt"])
            if (not frozen_prompt.is_file()
                    or sha256_file(frozen_prompt) != ledger["waves"][-1]["prompt_sha256"]):
                raise DiagnosticError("frozen wave prompt is missing or changed")
            self._prompt_bytes = frozen_prompt.read_bytes()
            self.manifest["prompt"] = ledger["waves"][-1]["prompt"]
            self._resume_wave(ledger, wave)
            ledger["status"] = (
                "reschedule_pending" if ledger["reschedule"] else "awaiting_curation"
            )
            _atomic_json(self.ledger_path, ledger)
            return ledger
        next_wave = len(ledger["waves"]) + 1
        if ledger["status"] != "ready_for_next" or wave != next_wave:
            raise DiagnosticError(f"campaign is not ready for wave {wave}; next wave is {next_wave}")
        self._freeze_prompt(manifest, f"wave-{wave}-prompt.md")
        self.manifest = manifest
        ledger["status"] = "running"
        _atomic_json(self.ledger_path, ledger)
        try:
            self._run_one_wave(ledger, wave)
        except BaseException:
            completed = set()
            for cell in ledger["waves"][-1]["cells"]:
                attempts = cell.get("reschedule_attempts", [])
                latest = attempts[-1] if attempts else cell.get("retry", cell)
                if latest["category"] == "counted":
                    completed.add(cell["treatment"])
            pending = set(ledger["reschedule"])
            pending.update(f"wave-{wave}-{name}" for name in TREATMENTS
                           if name not in completed)
            ledger["reschedule"] = sorted(pending)
            ledger["status"] = "reschedule_pending"
            _atomic_json(self.ledger_path, ledger)
            for hook in (self.launcher, self.terminal):
                cancel = getattr(hook, "cancel", None)
                if cancel is not None:
                    try:
                        cancel()
                    except Exception:
                        pass
            raise
        ledger["status"] = (
            "reschedule_pending" if ledger["reschedule"] else "awaiting_curation"
        )
        _atomic_json(self.ledger_path, ledger)
        return ledger

    def _resume_wave(self, ledger: dict, wave: int) -> None:
        """Retry only unresolved infrastructure cells in an existing wave."""
        wave_record = ledger["waves"][-1]
        by_treatment = {cell["treatment"]: cell for cell in wave_record["cells"]}
        deadline = time.monotonic() + self.wave_timeout
        for cell_id in list(ledger["reschedule"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            treatment = cell_id.removeprefix(f"wave-{wave}-")
            prior = by_treatment.get(treatment)
            last = None
            if prior is not None:
                attempts = prior.get("reschedule_attempts", [])
                last = attempts[-1] if attempts else prior.get("retry", prior)
            durable_attempts = []
            completed_receipts = []
            started_receipts = []
            uncertain_receipts = []
            for path in (self.root / "cells" / cell_id).glob(
                    "attempt-*/terminal-result-*.json"):
                try:
                    record = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                retriable_started = (
                    record.get("state") in {"started", "uncertain"}
                    and isinstance(record.get("request"), dict)
                    and isinstance(record.get("timeout_seconds"), int)
                    and isinstance(record.get("result"), dict)
                    and bool(record["result"].get("handle"))
                    and (isinstance(record["result"].get(
                        "retained_terminal_request"), dict)
                         or callable(getattr(self.terminal, "resume", None))))
                if ((record.get("state") == "completed"
                     and isinstance(record.get("result"), dict)) or retriable_started):
                    durable_attempt = int(path.parent.name.removeprefix("attempt-"))
                    durable_attempts.append(durable_attempt)
                    if (record.get("state") == "completed"
                            and isinstance(record.get("request"), dict)):
                        terminal_attempt = record["request"].get(
                            "terminal_attempt", durable_attempt)
                        if (isinstance(terminal_attempt, int)
                                and not isinstance(terminal_attempt, bool)
                                and terminal_attempt > 0):
                            completed_receipts.append(
                                (durable_attempt, terminal_attempt, record))
                    if retriable_started:
                        terminal_attempt = record["request"].get(
                            "terminal_attempt", durable_attempt)
                        if (isinstance(terminal_attempt, int)
                                and not isinstance(terminal_attempt, bool)
                                and terminal_attempt > 0):
                            started_receipts.append(
                                (durable_attempt, terminal_attempt, record))
                if (record.get("state") == "uncertain"
                        and isinstance(record.get("request"), dict)
                        and isinstance(record.get("result"), dict)
                        and not record["result"].get("handle")
                        and record["result"].get(
                            "manual_reconciliation_required") is True):
                    terminal_attempt = record["request"].get(
                        "terminal_attempt", int(path.parent.name.removeprefix("attempt-")))
                    if (isinstance(terminal_attempt, int)
                            and not isinstance(terminal_attempt, bool)
                            and terminal_attempt > 0):
                        uncertain_receipts.append((
                            int(path.parent.name.removeprefix("attempt-")),
                            terminal_attempt, record))
            newest_durable = max(durable_attempts, default=0)
            workspace = None if last is None else (
                self.root / "cells" / cell_id / f"attempt-{last['attempt']}" / "workspace"
            )
            receipt_candidates = (
                [(a, t, "completed", record) for a, t, record in completed_receipts]
                + [(a, t, "started", record) for a, t, record in started_receipts]
                + [(a, t, "uncertain", record) for a, t, record in uncertain_receipts]
            )
            newest_receipt = max(receipt_candidates, default=None,
                                 key=lambda item: (item[0], item[1]))
            if (last is not None and newest_receipt is not None
                    and newest_receipt[0] > last["attempt"]):
                resumed = self._guarded_cell(
                    wave, treatment, newest_receipt[0], remaining)
            elif (last is not None and newest_receipt is not None
                    and newest_receipt[0] == last["attempt"]
                    and newest_receipt[2] == "started"):
                terminal_request = copy.deepcopy(newest_receipt[3]["request"])
                started_workspace = Path(terminal_request["workspace"])
                terminal = self._durable_terminal_check(
                    terminal_request, max(1, int(remaining)))
                outcome, category = classify(last["agent"], terminal, started_workspace)
                resumed = {**copy.deepcopy(last), "outcome": outcome,
                           "category": category, "terminal": terminal,
                           "rescheduled": True}
            elif (last is not None and newest_receipt is not None
                    and newest_receipt[0] == last["attempt"]):
                terminal = copy.deepcopy(newest_receipt[3]["result"])
                reconciled_workspace = Path(newest_receipt[3]["request"]["workspace"])
                outcome, category = classify(
                    last["agent"], terminal, reconciled_workspace)
                resumed = {**copy.deepcopy(last), "outcome": outcome,
                           "category": category, "terminal": terminal,
                           "rescheduled": True}
            elif last is not None and newest_durable > last["attempt"]:
                resumed = self._guarded_cell(
                    wave, treatment, newest_durable, remaining)
            elif (last is not None and last["category"] == "infrastructure"
                    and last.get("agent", {}).get("status") == "ok"
                    and workspace is not None
                    and all((workspace / name).is_file()
                            for name in ("candidate.py", "candidate.manifest.json"))):
                expected = last.get("candidate_sha256")
                frozen_submission = workspace.parent / "frozen-submission"
                candidate_root = (frozen_submission if frozen_submission.exists()
                                  else workspace)
                if (not isinstance(expected, dict)
                        or any(not (candidate_root / name).is_file()
                               for name in ("candidate.py", "candidate.manifest.json"))
                        or any(expected.get(name) != sha256_file(candidate_root / name)
                               for name in ("candidate.py", "candidate.manifest.json"))):
                    raise DiagnosticError(f"retained candidate digest mismatch for {cell_id}")
                try:
                    retained_request = (last.get("terminal") or {}).get(
                        "retained_terminal_request")
                    terminal_attempt = (last.get("terminal") or {}).get(
                        "terminal_attempt", last["attempt"])
                    terminal_request = {
                        "protocol_version": 1, "operation": "terminal_check",
                        "cell_id": cell_id, "workspace": str(workspace),
                        "benchmark": "streaming-matmul-add", "cases": list(range(7)),
                        "candidate_sha256": copy.deepcopy(expected),
                        "terminal_attempt": (terminal_attempt if retained_request
                                             else terminal_attempt + 1),
                        "retained_terminal_request": copy.deepcopy(retained_request),
                    }
                    if candidate_root == frozen_submission:
                        terminal_request["candidate_source"] = "frozen-submission"
                    terminal = self._durable_terminal_check(
                        terminal_request, max(1, int(remaining)))
                except DiagnosticError:
                    raise
                except Exception as error:
                    terminal = {"status": "transport_or_observer_error",
                                "diagnostics":
                                    f"terminal hook raised {type(error).__name__}: {error}"}
                outcome, category = classify(last["agent"], terminal, workspace)
                resumed = {**copy.deepcopy(last), "outcome": outcome,
                           "category": category, "terminal": terminal,
                           "rescheduled": True}
            else:
                attempt = 1 if last is None else last["attempt"] + 1
                if last is None:
                    durable = [path.parent for path in
                               (self.root / "cells" / cell_id).glob(
                                   "attempt-*/terminal-result-*.json")]
                    if durable:
                        attempt = max(int(path.name.removeprefix("attempt-"))
                                      for path in durable)
                if not any((self.root / "cells" / cell_id /
                            f"attempt-{attempt}").glob("terminal-result-*.json")):
                    while (self.root / "cells" / cell_id / f"attempt-{attempt}").exists():
                        attempt += 1
                resumed = self._guarded_cell(wave, treatment, attempt, remaining)
            if prior is None:
                wave_record["cells"].append(resumed)
                by_treatment[treatment] = resumed
            else:
                prior.setdefault("reschedule_attempts", []).append(resumed)
                prior["resolution"] = resumed
            if resumed["category"] == "counted":
                ledger["reschedule"].remove(cell_id)
            _atomic_json(self.ledger_path, ledger)
        wave_record["cells"].sort(key=lambda item: TREATMENTS.index(item["treatment"]))

    def run(self) -> dict:
        manifest = copy.deepcopy(self.manifest)
        validate_manifest(manifest)
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise DiagnosticError(f"diagnostic output root is not fresh: {self.root}")
        self._freeze_prompt(manifest)
        self.manifest = manifest
        ledger = {
            **self.ledger_metadata,
            "protocol_version": 1, "campaign_id": self.campaign_id, "status": "running",
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"], "waves": [], "reschedule": [],
        }
        _atomic_json(self.ledger_path, ledger)
        try:
            return self._run_waves(ledger)
        except BaseException as error:
            ledger["status"] = (
                "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            )
            ledger["failure"] = {"type": type(error).__name__, "message": str(error)}
            pending = set(ledger["reschedule"])
            waves_by_number = {record["wave"]: record for record in ledger["waves"]}
            for wave in range(1, self.waves + 1):
                wave_record = waves_by_number.get(wave, {"cells": []})
                by_treatment = {cell["treatment"]: cell for cell in wave_record["cells"]}
                for treatment in TREATMENTS:
                    cell = by_treatment.get(treatment)
                    if cell is None or (cell["category"] == "infrastructure"
                                        and cell.get("retry", {}).get("category") != "counted"):
                        pending.add(f"wave-{wave}-{treatment}")
            ledger["reschedule"] = sorted(pending)
            try:
                _atomic_json(self.ledger_path, ledger)
            except BaseException:
                # Preserve the campaign exception. Cleanup below is mandatory
                # even when the durability boundary itself is unavailable.
                pass
            finally:
                for hook in (self.launcher, self.terminal):
                    cancel = getattr(hook, "cancel", None)
                    if cancel is not None:
                        try:
                            cancel()
                        except Exception:
                            pass
            raise

    def _run_waves(self, ledger: dict) -> dict:
        for wave in range(1, self.waves + 1):
            self._run_one_wave(ledger, wave)
        ledger["status"] = "reschedule_pending" if ledger["reschedule"] else "complete"
        _atomic_json(self.ledger_path, ledger)
        return ledger

    def _run_one_wave(self, ledger: dict, wave: int) -> None:
        wave_started = time.monotonic()
        results = []
        wave_record = {
            "wave": wave, "cells": results,
            "prompt": self.manifest["prompt"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"],
            "skill_sha256": {name: copy.deepcopy(
                self.manifest["treatments"][name]["skill_sha256"])
                for name in TREATMENTS},
        }
        ledger["waves"].append(wave_record)
        _atomic_json(self.ledger_path, ledger)
        futures = {}
        try:
            for treatment in TREATMENTS:
                future = _submit_daemon(
                    self._guarded_cell, wave, treatment, 1, self.wave_timeout,
                )
                futures[future] = treatment
            for future in as_completed(futures):
                results.append(future.result())
                _atomic_json(self.ledger_path, ledger)
        except BaseException:
            for future in futures:
                future.cancel()
            if ledger.get("adaptive") is True:
                for hook in (self.launcher, self.terminal):
                    cancel = getattr(hook, "cancel", None)
                    if cancel is not None:
                        try:
                            cancel()
                        except Exception:
                            pass
            recorded = {item["treatment"] for item in results}
            for future, treatment in futures.items():
                if (treatment in recorded or future.cancelled()
                        or (ledger.get("adaptive") is not True and not future.done())):
                    continue
                try:
                    results.append(future.result(timeout=min(5, self.cell_timeout)))
                except FutureTimeout:
                    continue
                except BaseException:
                    pass
            results.sort(key=lambda item: TREATMENTS.index(item["treatment"]))
            _atomic_json(self.ledger_path, ledger)
            raise
        for result in [item for item in results if item["category"] == "infrastructure"]:
            remaining = self.wave_timeout - (time.monotonic() - wave_started)
            retry = None
            manual_reconciliation = bool(
                (result.get("terminal") or {}).get("manual_reconciliation_required"))
            if remaining > 0 and not manual_reconciliation:
                retry_attempt = 2
                attempt_dir = (self.root / "cells" / result["cell_id"]
                               / f"attempt-{result['attempt']}")
                for receipt_path in attempt_dir.glob("terminal-result-*.json"):
                    try:
                        receipt = json.loads(receipt_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (receipt.get("state") == "started"
                            and isinstance(receipt.get("request"), dict)
                            and isinstance(receipt.get("result"), dict)):
                        retry_attempt = result["attempt"]
                        break
                retry = self._guarded_cell(
                    wave, result["treatment"], retry_attempt, remaining)
                result["retry"] = retry
                _atomic_json(self.ledger_path, ledger)
            if retry is None or retry["category"] != "counted":
                ledger["reschedule"].append(result["cell_id"])
                _atomic_json(self.ledger_path, ledger)
        results.sort(key=lambda item: TREATMENTS.index(item["treatment"]))
        _atomic_json(self.ledger_path, ledger)
