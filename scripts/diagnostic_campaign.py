#!/usr/bin/env python3
"""Short, transport-neutral one-shot Triton diagnostic campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol


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
    def resume(self, request: dict, handle: str, timeout_seconds: int) -> dict: ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _invoke(command: list[str], request: dict, timeout_seconds: int) -> dict:
    started = time.time()
    try:
        run = subprocess.run(command, input=json.dumps(request), text=True,
                             capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        return {"status": "timeout", "invocation_timeout": True,
                "stdout": error.stdout or "",
                "stderr": error.stderr or "", "elapsed_seconds": time.time() - started}
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
    return {**document, "stdout": run.stdout, "stderr": run.stderr,
            "exit_code": run.returncode, "elapsed_seconds": time.time() - started}


class CommandLauncher:
    """Invoke one externally isolated agent turn through a JSON protocol."""

    def __init__(self, command: list[str]):
        if not command:
            raise DiagnosticError("agent command is required")
        self.command = command

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        return _invoke(self.command, request, timeout_seconds)


class CommandTerminalHook:
    """Invoke the host-owned correctness check independently of the agent."""

    def __init__(self, command: list[str]):
        if not command:
            raise DiagnosticError("terminal command is required")
        self.command = command

    def check(self, request: dict, timeout_seconds: int) -> dict:
        return _invoke(self.command, request, timeout_seconds)

    def resume(self, request: dict, handle: str, timeout_seconds: int) -> dict:
        result = _invoke(self.command, {**request, "operation": "terminal_observe",
                                        "handle": handle}, timeout_seconds)
        if result.get("status") == "timeout" and result.get("invocation_timeout") is True:
            return {**result, "status": "transport_or_observer_error",
                    "handle": handle}
        return result


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
        if not isinstance(treatment.get("skill_sha256"), dict):
            raise DiagnosticError(f"skill hashes are required for {name}")
        if set(skills) != set(treatment["skill_sha256"]):
            raise DiagnosticError(f"skill inventory and hashes differ for {name}")


def classify(agent: dict, terminal: dict | None, workspace: Path) -> tuple[str, str]:
    status = agent.get("status")
    if status == "timeout":
        return "agent_timeout", "observed"
    if status in INFRASTRUCTURE:
        return str(status), "infrastructure"
    if status in OBSERVED:
        return str(status), "observed"
    if status in COUNTED - {"success"}:
        return str(status), "counted"
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
    if terminal_status in INFRASTRUCTURE:
        return str(terminal_status), "infrastructure"
    if terminal_status == "timeout":
        return "candidate_timeout", "counted"
    if terminal_status in {"source_error", "compile_error", "runtime_error", "correctness_error"}:
        return str(terminal_status), "counted"
    if terminal_status == "ok" and terminal.get("passed") is True:
        return "success", "counted"
    return "diagnostic_retrieval_error", "observed"


class DiagnosticCampaign:
    """Run four sequential waves with three concurrent treatment cells each."""

    def __init__(self, manifest: dict, root: Path, launcher: Launcher,
                 terminal: TerminalHook, *, waves: int = 4,
                 agent_timeout: int = 360, cell_timeout: int = 600,
                 wave_timeout: int = 600, campaign_config_sha256: str | None = None):
        validate_manifest(manifest)
        if waves != 4 or agent_timeout <= 0 or cell_timeout <= 0 or wave_timeout <= 0:
            raise DiagnosticError("four waves and positive timeouts are required")
        self.manifest, self.root = manifest, root
        self.launcher, self.terminal = launcher, terminal
        self.waves, self.agent_timeout, self.cell_timeout = waves, agent_timeout, cell_timeout
        self.wave_timeout = wave_timeout
        self.campaign_config_sha256 = campaign_config_sha256
        self.ledger_path = root / "ledger.json"

    def _identity(self) -> dict:
        return {
            "model_sha256": self.manifest["model_sha256"],
            "treatments": self.manifest["treatments"],
            "campaign_config_sha256": self.campaign_config_sha256,
        }

    def _new_ledger(self, *, adaptive: bool = False) -> dict:
        ledger = {
            "protocol_version": 1, "campaign_id": str(uuid.uuid4()),
            "status": "ready_for_next" if adaptive else "running",
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"], "waves": [],
            "reschedule": [],
        }
        if adaptive:
            ledger["adaptive"] = True
            ledger["campaign_identity"] = self._identity()
            ledger["curation_receipts"] = []
        return ledger

    def _load_adaptive_ledger(self) -> dict:
        if not self.ledger_path.is_file():
            return self._new_ledger(adaptive=True)
        try:
            ledger = json.loads(self.ledger_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DiagnosticError(f"cannot read campaign ledger: {error}") from error
        if not isinstance(ledger, dict) or ledger.get("adaptive") is not True:
            raise DiagnosticError("ledger is not an adaptive campaign")
        if ledger.get("campaign_identity") != self._identity():
            raise DiagnosticError("model, treatment, skill, or campaign configuration drift")
        return ledger

    def _run_one_wave(self, ledger: dict, wave: int) -> None:
        wave_started = time.monotonic()
        results = []
        wave_record = {
            "wave": wave, "cells": results,
            "prompt": self.manifest["prompt"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model_sha256": self.manifest["model_sha256"],
            "skill_sha256": {
                name: self.manifest["treatments"][name]["skill_sha256"]
                for name in TREATMENTS
            },
        }
        ledger["waves"].append(wave_record)
        _atomic_json(self.ledger_path, ledger)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._cell, wave, treatment, 1, self.wave_timeout): treatment
                       for treatment in TREATMENTS}
            for future in as_completed(futures):
                result = future.result()
                if result["category"] == "infrastructure":
                    remaining = self.wave_timeout - (time.monotonic() - wave_started)
                    retry = None
                    if remaining > 0:
                        handle = ((result.get("terminal") or {}).get("handle"))
                        if isinstance(handle, str) and handle:
                            retry = self._resume_terminal(result, handle, remaining)
                        else:
                            retry = self._cell(wave, result["treatment"], 2, remaining)
                        result["retry"] = retry
                    if retry is None or retry["category"] == "infrastructure":
                        ledger["reschedule"].append(result["cell_id"])
                results.append(result)
                _atomic_json(self.ledger_path, ledger)
        results.sort(key=lambda item: TREATMENTS.index(item["treatment"]))
        _atomic_json(self.ledger_path, ledger)

    def _resume_terminal(self, original: dict, handle: str,
                         available_seconds: float) -> dict:
        """Observe one retained terminal job without replaying agent or dispatch."""
        started = time.time()
        workspace = (self.root / "cells" / original["cell_id"]
                     / f"attempt-{original['attempt']}" / "workspace")
        request = {
            "protocol_version": 1, "operation": "terminal_observe",
            "cell_id": original["cell_id"], "workspace": str(workspace),
            "benchmark": "streaming-matmul-add", "cases": list(range(7)),
            "profile": original["terminal"].get("profile"),
            "device": original["terminal"].get("device"),
        }
        terminal = self.terminal.resume(
            request, handle, max(1, int(min(self.cell_timeout, available_seconds))))
        outcome, category = classify(original["agent"], terminal, workspace)
        return {
            **{key: original[key] for key in (
                "cell_id", "wave", "attempt", "treatment", "candidate_sha256",
                "prompt_sha256", "model_sha256", "skill_sha256")},
            "outcome": outcome, "category": category, "agent": original["agent"],
            "terminal": terminal, "terminal_resumed": True,
            "started_at_epoch": started, "elapsed_seconds": time.time() - started,
        }

    def run_wave(self, wave: int) -> dict:
        """Run only the next adaptive wave and pause for curated evidence."""
        if isinstance(wave, bool) or not isinstance(wave, int) or not 1 <= wave <= self.waves:
            raise DiagnosticError("wave must be an integer from 1 through 4")
        ledger = self._load_adaptive_ledger()
        next_wave = len(ledger["waves"]) + 1
        if ledger["status"] != "ready_for_next" or wave != next_wave:
            raise DiagnosticError(f"campaign is not ready for wave {wave}; next wave is {next_wave}")
        ledger["status"] = "running"
        _atomic_json(self.ledger_path, ledger)
        self._run_one_wave(ledger, wave)
        ledger["status"] = "awaiting_curation"
        _atomic_json(self.ledger_path, ledger)
        return ledger

    def acknowledge_curation(self, receipt: dict) -> dict:
        """Record accepted library evidence and unlock the next adaptive wave."""
        ledger = self._load_adaptive_ledger()
        if ledger["status"] != "awaiting_curation":
            raise DiagnosticError("campaign is not awaiting curation")
        wave = len(ledger["waves"])
        required = {"wave", "accepted", "stable_ref_citations", "librarian_query_ids"}
        citations = receipt.get("stable_ref_citations") if isinstance(receipt, dict) else None
        queries = receipt.get("librarian_query_ids") if isinstance(receipt, dict) else None
        if (not isinstance(receipt, dict) or not required.issubset(receipt)
                or receipt.get("wave") != wave or receipt.get("accepted") is not True
                or not isinstance(citations, list) or not citations
                or not all(isinstance(item, str) and item.startswith("ref://") for item in citations)
                or not isinstance(queries, list) or not queries
                or not all(isinstance(item, str) and item.strip() for item in queries)):
            raise DiagnosticError("invalid curation receipt")
        ledger["curation_receipts"].append(receipt)
        ledger["status"] = "complete" if wave == self.waves else "ready_for_next"
        _atomic_json(self.ledger_path, ledger)
        return ledger

    def _cell(self, wave: int, treatment: str, attempt: int,
              available_seconds: float | None = None) -> dict:
        cell_id = f"wave-{wave}-{treatment}"
        workspace = self.root / "cells" / cell_id / f"attempt-{attempt}" / "workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        started = time.time()
        treatment_record = self.manifest["treatments"][treatment]
        request = {
            "protocol_version": 1, "operation": "one_shot", "cell_id": cell_id,
            "wave": wave, "attempt": attempt, "treatment": treatment,
            "workspace": str(workspace), "prompt": self.manifest["prompt"],
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model": self.manifest["model"], "model_sha256": self.manifest["model_sha256"],
            "skills": treatment_record["skills"],
            "skill_sha256": treatment_record["skill_sha256"],
            "benchmark": "streaming-matmul-add", "development_cases": [1],
            "controller_contract": {"billed_limit": 1,
                "command": ["check", "--scope", "development", "--round", "1"]},
        }
        cap = min(self.cell_timeout, available_seconds or self.cell_timeout)
        agent = self.launcher.launch(request, max(1, int(min(self.agent_timeout, cap))))
        elapsed = time.time() - started
        terminal = None
        candidate = workspace / "candidate.py"
        candidate_manifest = workspace / "candidate.manifest.json"
        if elapsed < cap and candidate.is_file() and candidate_manifest.is_file():
            terminal_request = {
                "protocol_version": 1, "operation": "terminal_check", "cell_id": cell_id,
                "workspace": str(workspace), "benchmark": "streaming-matmul-add",
                "cases": list(range(7)),
            }
            terminal = self.terminal.check(terminal_request, max(1, int(cap - elapsed)))
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
            "skill_sha256": treatment_record["skill_sha256"],
            "elapsed_seconds": time.time() - started,
        }

    def run(self) -> dict:
        ledger = self._new_ledger()
        _atomic_json(self.ledger_path, ledger)
        for wave in range(1, self.waves + 1):
            self._run_one_wave(ledger, wave)
        ledger["status"] = "complete"
        _atomic_json(self.ledger_path, ledger)
        return ledger
