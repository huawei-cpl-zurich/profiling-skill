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

        return {"status": "timeout", "stdout": timeout_text(error.stdout),
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
        return _invoke(self.command, request, timeout_seconds, self._processes)

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
        workspace.mkdir(parents=True, exist_ok=False)
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
        try:
            agent = copy.deepcopy(self.launcher.launch(
                request, max(1, int(min(self.agent_timeout, cap))),
            ))
        except Exception as error:
            agent = {"status": "transport_or_observer_error",
                     "diagnostics": f"launcher raised {type(error).__name__}: {error}"}
        elapsed = time.monotonic() - monotonic_started
        terminal = None
        candidate = workspace / "candidate.py"
        candidate_manifest = workspace / "candidate.manifest.json"
        if elapsed >= cap:
            agent = {"status": "infrastructure_error",
                     "failure_type": "wave_budget_exhausted",
                     "diagnostics": "agent consumed the remaining cell budget",
                     "reported_agent": agent}
        elif candidate.is_file() and candidate_manifest.is_file():
            terminal_request = {
                "protocol_version": 1, "operation": "terminal_check", "cell_id": cell_id,
                "workspace": str(workspace), "benchmark": "streaming-matmul-add",
                "cases": list(range(7)),
            }
            try:
                terminal = copy.deepcopy(self.terminal.check(
                    terminal_request, max(1, int(cap - elapsed)),
                ))
            except Exception as error:
                terminal = {"status": "transport_or_observer_error",
                            "diagnostics":
                                f"terminal hook raised {type(error).__name__}: {error}"}
            if time.monotonic() - monotonic_started >= cap:
                terminal = {"status": "infrastructure_error",
                            "failure_type": "wave_budget_exhausted",
                            "diagnostics": "terminal hook exceeded the remaining cell budget"}
        outcome, category = classify(agent, terminal, workspace)
        hashes = {}
        for name in ("candidate.py", "candidate.manifest.json"):
            path = workspace / name
            if path.is_file():
                hashes[name] = sha256_file(path)
        return {
            "cell_id": cell_id, "wave": wave, "attempt": attempt, "treatment": treatment,
            "outcome": outcome, "category": category, "agent": agent, "terminal": terminal,
            "candidate_sha256": hashes, "started_at_epoch": started,
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"],
            "skill_sha256": copy.deepcopy(treatment_record["skill_sha256"]),
            "elapsed_seconds": time.time() - started,
        }

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
            raise DiagnosticError(f"frozen prompt already exists: {destination}")
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
        if ledger["status"] == "reschedule_pending":
            if wave != len(ledger["waves"]):
                raise DiagnosticError(f"campaign must reschedule wave {len(ledger['waves'])}")
            if self.manifest["prompt_sha256"] != ledger["waves"][-1]["prompt_sha256"]:
                raise DiagnosticError("prompt cannot change while a wave is reschedule pending")
            self._prompt_bytes = Path(ledger["waves"][-1]["prompt"]).read_bytes()
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
            completed = {cell["treatment"] for cell in ledger["waves"][-1]["cells"]
                         if cell["category"] != "infrastructure"}
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
        for cell_id in list(ledger["reschedule"]):
            treatment = cell_id.removeprefix(f"wave-{wave}-")
            prior = by_treatment.get(treatment)
            last = None
            if prior is not None:
                attempts = prior.get("reschedule_attempts", [])
                last = attempts[-1] if attempts else prior.get("retry", prior)
            workspace = None if last is None else (
                self.root / "cells" / cell_id / f"attempt-{last['attempt']}" / "workspace"
            )
            if (last is not None and workspace is not None
                    and all((workspace / name).is_file()
                            for name in ("candidate.py", "candidate.manifest.json"))):
                expected = last.get("candidate_sha256")
                if (not isinstance(expected, dict)
                        or any(expected.get(name) != sha256_file(workspace / name)
                               for name in ("candidate.py", "candidate.manifest.json"))):
                    raise DiagnosticError(f"retained candidate digest mismatch for {cell_id}")
                try:
                    terminal = self.terminal.check({
                        "protocol_version": 1, "operation": "terminal_check",
                        "cell_id": cell_id, "workspace": str(workspace),
                        "benchmark": "streaming-matmul-add", "cases": list(range(7)),
                        "candidate_sha256": copy.deepcopy(expected),
                    }, self.cell_timeout)
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
                while (self.root / "cells" / cell_id / f"attempt-{attempt}").exists():
                    attempt += 1
                resumed = self._guarded_cell(wave, treatment, attempt, self.wave_timeout)
            if prior is None:
                wave_record["cells"].append(resumed)
                by_treatment[treatment] = resumed
            else:
                prior.setdefault("reschedule_attempts", []).append(resumed)
                prior["resolution"] = resumed
            if resumed["category"] != "infrastructure":
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
            if remaining > 0:
                retry = self._guarded_cell(wave, result["treatment"], 2, remaining)
                result["retry"] = retry
                _atomic_json(self.ledger_path, ledger)
            if retry is None or retry["category"] != "counted":
                ledger["reschedule"].append(result["cell_id"])
                _atomic_json(self.ledger_path, ledger)
        results.sort(key=lambda item: TREATMENTS.index(item["treatment"]))
        _atomic_json(self.ledger_path, ledger)
