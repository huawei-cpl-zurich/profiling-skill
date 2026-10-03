#!/usr/bin/env python3
"""Run two fresh, two-turn smoke waves across the isolated treatments."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import diagnostic_campaign

TREATMENTS = diagnostic_campaign.TREATMENTS
INFRASTRUCTURE = diagnostic_campaign.INFRASTRUCTURE
DEFAULT_SUCCESS_POLICY = {
    "cannbot": {"counted_trials": 3, "minimum_successes": 1},
    "project-cannbot": {"counted_trials": 2, "minimum_successes": 1},
    "project-guarded": {"counted_trials": 2, "minimum_successes": 1},
}


class SmokeError(RuntimeError):
    pass


class Launcher(Protocol):
    def launch(self, request: dict, timeout_seconds: int) -> dict: ...


class Terminal(Protocol):
    def check(self, request: dict, timeout_seconds: int) -> dict: ...


def check_command(round_number: int) -> list[str]:
    return ["check", "--scope", "development", "--round", str(round_number)]


def manifest_identity(manifest: dict) -> str:
    frozen = {"prompt_sha256": manifest["prompt_sha256"],
              "model_sha256": manifest["model_sha256"],
              "treatments": manifest["treatments"]}
    return hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_success_policy(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict) or set(value) != set(TREATMENTS):
        raise SmokeError("success policy must configure every treatment")
    policy = copy.deepcopy(value)
    for treatment, rule in policy.items():
        if (not isinstance(rule, dict)
                or set(rule) != {"counted_trials", "minimum_successes"}):
            raise SmokeError(f"invalid success policy for {treatment}")
        target, minimum = rule["counted_trials"], rule["minimum_successes"]
        if (isinstance(target, bool) or not isinstance(target, int) or target < 1
                or isinstance(minimum, bool) or not isinstance(minimum, int)
                or minimum < 1 or minimum > target):
            raise SmokeError(f"invalid success policy for {treatment}")
    return policy


def _gate_summary(waves: list[dict], policy: dict[str, dict[str, int]]) -> dict:
    cells = [cell for wave in waves for cell in wave.get("cells", [])]
    summary = {}
    for treatment in TREATMENTS:
        treatment_cells = [cell for cell in cells if cell.get("treatment") == treatment]
        counted = sum(cell.get("category") == "counted" for cell in treatment_cells)
        successes = sum(cell.get("category") == "counted"
                        and cell.get("outcome") == "success" for cell in treatment_cells)
        rule = policy[treatment]
        summary[treatment] = {
            "counted": counted, "successes": successes,
            "required": rule["minimum_successes"], "target": rule["counted_trials"],
            "passed": (counted == rule["counted_trials"]
                       and successes >= rule["minimum_successes"]),
        }
    return summary


def _valid_schedule_topology(waves: object,
                             policy: dict[str, dict[str, int]]) -> bool:
    if not isinstance(waves, list):
        return False
    maximum = max(rule["counted_trials"] for rule in policy.values())
    if [wave.get("wave") if isinstance(wave, dict) else None for wave in waves] != list(
            range(1, maximum + 1)):
        return False
    for wave_number, wave in enumerate(waves, 1):
        cells = wave.get("cells")
        expected = [treatment for treatment in TREATMENTS
                    if policy[treatment]["counted_trials"] >= wave_number]
        if (not isinstance(cells, list)
                or [cell.get("treatment") if isinstance(cell, dict) else None
                    for cell in cells] != expected):
            return False
    return True


def validate_matmul_gate(path: Path, *, prompt_sha256: str | None = None,
                         manifest_identity: str | None = None,
                         success_policy: dict | None = None) -> dict:
    try:
        ledger = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"matmul gate is unreadable: {error}") from error
    waves = ledger.get("waves", [])
    if success_policy is not None:
        policy = _validate_success_policy(success_policy)
        valid = (ledger.get("success_policy") == policy
                 and _valid_schedule_topology(waves, policy)
                 and ledger.get("gate") == _gate_summary(waves, policy)
                 and all(item["passed"] for item in ledger.get("gate", {}).values()))
        if (ledger.get("status") != "complete" or ledger.get("benchmark") != "matmul"
                or ledger.get("cases") != list(range(7)) or not valid
                or (prompt_sha256 is not None
                    and ledger.get("prompt_sha256") != prompt_sha256)
                or (manifest_identity is not None
                    and ledger.get("manifest_identity") != manifest_identity)):
            raise SmokeError("BSA requires complete configured success thresholds for matmul")
        return ledger
    valid_waves = (isinstance(waves, list) and len(waves) == 2
                   and [wave.get("wave") for wave in waves] == [1, 2]
                   and all([cell.get("treatment") for cell in wave.get("cells", [])]
                           == list(TREATMENTS) for wave in waves)
                   and all(cell.get("outcome") == "success"
                           for wave in waves for cell in wave["cells"]))
    if (ledger.get("status") != "complete" or ledger.get("benchmark") != "matmul"
            or ledger.get("cases") != list(range(7)) or not valid_waves
            or (prompt_sha256 is not None and ledger.get("prompt_sha256") != prompt_sha256)
            or (manifest_identity is not None
                and ledger.get("manifest_identity") != manifest_identity)):
        raise SmokeError("BSA requires six successful matmul cells across two waves")
    return ledger


def _concise_diagnostics(value: object, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _diagnostic_fallback(*documents: dict) -> str:
    for document in documents:
        if not isinstance(document, dict):
            continue
        candidates = [document.get("diagnostics")]
        results = document.get("controller_results")
        if isinstance(results, list):
            candidates.extend(item.get("diagnostics") for item in reversed(results)
                              if isinstance(item, dict))
        result = document.get("controller_result")
        if isinstance(result, dict):
            candidates.append(result.get("diagnostics"))
        turns = document.get("turns")
        if isinstance(turns, list):
            for turn in reversed(turns):
                if isinstance(turn, dict):
                    candidates.extend((turn.get("stderr"), turn.get("stdout")))
        candidates.extend((document.get("agent_timeout"), document.get("stderr"),
                           document.get("stdout")))
        for candidate in candidates:
            if _concise_diagnostics(candidate):
                return _concise_diagnostics(candidate)
    return ""


def _durable_handles(*documents: object) -> list[str]:
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"handle", "job_handle"} and isinstance(item, str) and item:
                    found.add(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for document in documents:
        visit(document)
    return sorted(found)


def _failure_summaries(benchmark: str, waves: list[dict]) -> list[dict]:
    summaries = []
    for wave in waves:
        trial = wave.get("wave")
        for cell in wave.get("cells", []):
            if cell.get("category") != "counted" or cell.get("outcome") == "success":
                continue
            agent, terminal = cell.get("agent") or {}, cell.get("terminal") or {}
            agent_outcomes = {"agent_timeout", "protocol_error", "submission_error"}
            phase = ("agent" if cell.get("outcome") in agent_outcomes
                     or agent.get("status") != "ok" else "terminal")
            evidence = ((agent, terminal) if phase == "agent" else (terminal, agent))
            summaries.append({
                "treatment": cell.get("treatment"), "benchmark": benchmark,
                "trial": trial, "wave": trial, "phase": phase,
                "failure_type": cell.get("outcome"), "outcome": cell.get("outcome"),
                "diagnostics": _diagnostic_fallback(*evidence),
                "candidate_sha256": copy.deepcopy(agent.get("candidate_sha256", {})),
                "durable_handles": _durable_handles(agent, terminal),
            })
    return summaries


def _freeze_submission(workspace: Path) -> tuple[Path, dict[str, str]]:
    snapshot = workspace.parent / "frozen-submission"
    snapshot.mkdir(mode=0o700, exist_ok=False)
    hashes = {}
    try:
        for name in ("candidate.py", "candidate.manifest.json"):
            source = workspace / name
            if source.is_symlink() or not source.is_file():
                raise SmokeError(f"invalid smoke submission: {name}")
            resolved = source.resolve(strict=True)
            if not resolved.is_relative_to(workspace.resolve(strict=True)):
                raise SmokeError(f"smoke submission escaped workspace: {name}")
            content = source.read_bytes()
            destination = snapshot / name
            destination.write_bytes(content)
            destination.chmod(0o444)
            hashes[name] = hashlib.sha256(content).hexdigest()
        snapshot.chmod(0o555)
    except BaseException:
        for path in snapshot.glob("*"):
            path.unlink(missing_ok=True)
        snapshot.rmdir()
        raise
    return snapshot, hashes


def _classification(agent: dict, terminal: dict | None, workspace: Path) -> tuple[str, str]:
    status = agent.get("status")
    if status in INFRASTRUCTURE or status == "infrastructure_error":
        return str(agent.get("failure_type") or status), "infrastructure"
    if status == "timeout":
        return "agent_timeout", "counted"
    if status in {"compile_error", "runtime_error", "correctness_error",
                  "submission_error"}:
        return str(status), "counted"
    usage = agent.get("controller_usage", {})
    calls = usage.get("calls")
    expected = [{"arguments": check_command(1)}, {"arguments": check_command(2)}]
    hashes = agent.get("candidate_sha256")
    valid_hashes = (isinstance(hashes, dict) and set(hashes) == {"1", "2"}
                    and all(isinstance(value, str)
                            and re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in hashes.values())
                    and hashes["1"] != hashes["2"])
    if (status != "ok" or agent.get("rounds_completed") != 2 or calls != expected
            or usage.get("billed") != 2 or usage.get("invalid") != 0
            or usage.get("over_budget") != 0
            or not valid_hashes
            or not all((workspace / name).is_file()
                       for name in ("candidate.py", "candidate.manifest.json"))):
        return "protocol_error", "counted"
    if terminal is None:
        return "protocol_error", "counted"
    terminal_status = terminal.get("status")
    if terminal_status in INFRASTRUCTURE or terminal_status == "infrastructure_error":
        return str(terminal.get("failure_type") or terminal_status), "infrastructure"
    if terminal_status in {"compile_error", "runtime_error", "correctness_error"}:
        return str(terminal_status), "counted"
    if terminal_status == "timeout":
        return "candidate_timeout", "counted"
    if terminal_status == "ok" and terminal.get("passed") is True:
        return "success", "counted"
    return "diagnostic_retrieval_error", "counted"


class TwoShotSmokeCampaign:
    def __init__(self, manifest: dict, root: Path, launcher: Launcher, terminal: Terminal,
                 *, benchmark: str, cases: list[int], agent_timeout: int = 720,
                 terminal_timeout: int = 300, resume: bool = False,
                 ledger_metadata: dict | None = None,
                 success_policy: dict | None = None):
        diagnostic_campaign.validate_manifest(manifest)
        if benchmark not in {"matmul", "bsa"} or not cases or any(
                isinstance(case, bool) or not isinstance(case, int) or case < 0 for case in cases):
            raise SmokeError("benchmark and non-empty nonnegative case list are required")
        self.manifest, self.root = manifest, root
        self.launcher, self.terminal = launcher, terminal
        self.benchmark, self.cases = benchmark, list(cases)
        self.agent_timeout, self.terminal_timeout = agent_timeout, terminal_timeout
        self.resume = resume
        self.success_policy = (_validate_success_policy(success_policy)
                               if success_policy is not None else None)
        self.ledger_metadata = copy.deepcopy(ledger_metadata or {})
        self.campaign_id = str(uuid.uuid4())

    def _request(self, wave: int, treatment: str, attempt: int, workspace: Path) -> dict:
        treatment_record = copy.deepcopy(self.manifest["treatments"][treatment])
        prompt = Path(self.manifest["prompt"]).read_bytes()
        return {
            "protocol_version": 2, "operation": "two_shot",
            "cell_id": f"wave-{wave}-{treatment}", "campaign_id": self.campaign_id,
            "benchmark": self.benchmark, "wave": wave, "attempt": attempt,
            "treatment": treatment, "workspace": str(workspace),
            "prompt": {"encoding": "base64",
                       "data": __import__("base64").b64encode(prompt).decode(),
                       "sha256": self.manifest["prompt_sha256"]},
            "prompt_sha256": self.manifest["prompt_sha256"],
            "model": copy.deepcopy(self.manifest["model"]),
            "model_sha256": self.manifest["model_sha256"],
            "skills": treatment_record["skills"],
            "skill_sha256": treatment_record["skill_sha256"],
            "controller_contract": {"billed_limit": 2,
                                    "commands": [check_command(1), check_command(2)]},
            "cases": self.cases,
        }

    def _cell(self, wave: int, treatment: str, previous: dict | None = None) -> dict:
        history = copy.deepcopy(previous.get("attempts", []) if previous else [])
        first_attempt = len(history) + 1
        for attempt in range(first_attempt, first_attempt + 2):
            workspace = self.root / "cells" / f"wave-{wave}-{treatment}" / f"attempt-{attempt}" / "workspace"
            workspace.mkdir(parents=True, exist_ok=False)
            request = self._request(wave, treatment, attempt, workspace)
            started = time.time()
            try:
                agent = self.launcher.launch(request, self.agent_timeout)
            except Exception as error:
                agent = {"status": "infrastructure_error", "failure_type": "launcher_error",
                         "diagnostics": f"launcher raised {type(error).__name__}: {error}"}
            terminal = None
            if agent.get("status") == "ok" and all(
                    (workspace / name).is_file()
                    for name in ("candidate.py", "candidate.manifest.json")):
                _snapshot, frozen_hashes = _freeze_submission(workspace)
                terminal_request = {**request, "cases": self.cases,
                                    "candidate_sha256": frozen_hashes}
                terminal = self.terminal.check(terminal_request, self.terminal_timeout)
            outcome, category = _classification(agent, terminal, workspace)
            record = {"attempt": attempt, "outcome": outcome, "category": category,
                      "started_at": started, "finished_at": time.time(),
                      "agent": agent, "terminal": terminal}
            if (isinstance(terminal, dict)
                    and terminal.get("manual_reconciliation_required") is True):
                record["manual_reconciliation_required"] = True
            history.append(record)
            if record.get("manual_reconciliation_required"):
                return {"treatment": treatment, **record, "attempts": history}
            if category != "infrastructure":
                return {"treatment": treatment, **record, "attempts": history}
        return {"treatment": treatment, **history[-1], "attempts": history}

    def run(self) -> dict:
        ledger_path = self.root / "ledger.json"
        if self.resume:
            if not ledger_path.is_file():
                raise SmokeError("smoke resume requires an existing ledger")
            try:
                ledger = json.loads(ledger_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise SmokeError(f"smoke ledger is unreadable: {error}") from error
            expected = {"benchmark": self.benchmark, "cases": self.cases,
                        "prompt_sha256": self.manifest["prompt_sha256"],
                        "manifest_identity": manifest_identity(self.manifest)}
            if any(ledger.get(key) != value for key, value in expected.items()):
                raise SmokeError("smoke resume inputs changed")
            if self.success_policy is not None:
                recorded = ledger.get("success_policy")
                if recorded is not None and recorded != self.success_policy:
                    raise SmokeError("smoke resume success policy changed")
                ledger["success_policy"] = copy.deepcopy(self.success_policy)
            self.campaign_id = ledger.get("campaign_id")
            if not isinstance(self.campaign_id, str) or not self.campaign_id:
                raise SmokeError("smoke ledger has no campaign identity")
        else:
            if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
                raise SmokeError(f"smoke output root is not fresh: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
            ledger = {"protocol_version": 1, "campaign_id": self.campaign_id,
                      "benchmark": self.benchmark, "cases": self.cases,
                      "prompt_sha256": self.manifest["prompt_sha256"],
                      "manifest_identity": manifest_identity(self.manifest),
                      "status": "running", "waves": []}
            if self.success_policy is not None:
                ledger["success_policy"] = copy.deepcopy(self.success_policy)
            ledger.update(self.ledger_metadata)
            diagnostic_campaign._atomic_json(ledger_path, ledger)
        schedule = (self.success_policy or {
            treatment: {"counted_trials": 2, "minimum_successes": 2}
            for treatment in TREATMENTS
        })
        for wave in range(1, max(rule["counted_trials"]
                                 for rule in schedule.values()) + 1):
            wave_record = next((item for item in ledger["waves"] if item.get("wave") == wave), None)
            if wave_record is None:
                wave_record = {"wave": wave, "cells": []}
                ledger["waves"].append(wave_record)
                diagnostic_campaign._atomic_json(ledger_path, ledger)
            by_treatment = {cell["treatment"]: cell for cell in wave_record["cells"]}
            scheduled = [name for name in TREATMENTS
                         if wave <= schedule[name]["counted_trials"]]
            pending = [name for name in scheduled
                       if name not in by_treatment
                       or (by_treatment[name].get("category") == "infrastructure"
                           and not by_treatment[name].get("manual_reconciliation_required"))]
            if pending:
                checkpoint_lock = threading.Lock()

                def run_and_checkpoint(name: str) -> dict:
                    cell = self._cell(wave, name, by_treatment.get(name))
                    with checkpoint_lock:
                        by_treatment[name] = cell
                        wave_record["cells"] = [by_treatment[item] for item in scheduled
                                                if item in by_treatment]
                        diagnostic_campaign._atomic_json(ledger_path, ledger)
                    return cell

                with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                    futures = {pool.submit(run_and_checkpoint, name): name
                               for name in pending}
                    for future in as_completed(futures):
                        future.result()
        cells = [cell for wave in ledger["waves"] for cell in wave["cells"]]
        if self.success_policy is not None:
            ledger["gate"] = _gate_summary(ledger["waves"], self.success_policy)
            ledger["failure_summaries"] = _failure_summaries(
                self.benchmark, ledger["waves"])
        if any(cell.get("manual_reconciliation_required") for cell in cells):
            ledger["status"] = "reconciliation_required"
        elif any(cell.get("category") == "infrastructure" for cell in cells):
            ledger["status"] = "infrastructure_pending"
        elif (self.success_policy is not None
              and all(item["passed"] for item in ledger["gate"].values())):
            ledger["status"] = "complete"
        elif self.success_policy is None and all(cell["outcome"] == "success" for cell in cells):
            ledger["status"] = "complete"
        else:
            ledger["status"] = "smoke_failures"
        diagnostic_campaign._atomic_json(ledger_path, ledger)
        return ledger
