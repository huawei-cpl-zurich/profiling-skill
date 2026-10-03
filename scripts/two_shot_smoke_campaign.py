#!/usr/bin/env python3
"""Run two fresh, two-turn smoke waves across the isolated treatments."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import diagnostic_campaign

TREATMENTS = diagnostic_campaign.TREATMENTS
INFRASTRUCTURE = diagnostic_campaign.INFRASTRUCTURE


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


def validate_matmul_gate(path: Path, *, prompt_sha256: str | None = None,
                         manifest_identity: str | None = None) -> dict:
    try:
        ledger = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SmokeError(f"matmul gate is unreadable: {error}") from error
    waves = ledger.get("waves", [])
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
    if (status != "ok" or agent.get("rounds_completed") != 2 or calls != expected
            or usage.get("billed") != 2 or usage.get("invalid") != 0
            or usage.get("over_budget") != 0
            or not isinstance(hashes, dict) or hashes.get("1") == hashes.get("2")
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
                 terminal_timeout: int = 300):
        diagnostic_campaign.validate_manifest(manifest)
        if benchmark not in {"matmul", "bsa"} or not cases or any(
                isinstance(case, bool) or not isinstance(case, int) or case < 0 for case in cases):
            raise SmokeError("benchmark and non-empty nonnegative case list are required")
        self.manifest, self.root = manifest, root
        self.launcher, self.terminal = launcher, terminal
        self.benchmark, self.cases = benchmark, list(cases)
        self.agent_timeout, self.terminal_timeout = agent_timeout, terminal_timeout
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

    def _cell(self, wave: int, treatment: str) -> dict:
        history = []
        for attempt in (1, 2):
            workspace = self.root / "cells" / f"wave-{wave}-{treatment}" / f"attempt-{attempt}" / "workspace"
            workspace.mkdir(parents=True, exist_ok=False)
            request = self._request(wave, treatment, attempt, workspace)
            started = time.time()
            agent = self.launcher.launch(request, self.agent_timeout)
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
            history.append(record)
            if category != "infrastructure":
                return {"treatment": treatment, **record, "attempts": history}
        return {"treatment": treatment, **history[-1], "attempts": history}

    def run(self) -> dict:
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise SmokeError(f"smoke output root is not fresh: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        ledger = {"protocol_version": 1, "campaign_id": self.campaign_id,
                  "benchmark": self.benchmark, "cases": self.cases,
                  "prompt_sha256": self.manifest["prompt_sha256"],
                  "manifest_identity": manifest_identity(self.manifest),
                  "status": "running", "waves": []}
        for wave in (1, 2):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {name: pool.submit(self._cell, wave, name) for name in TREATMENTS}
                cells = [futures[name].result() for name in TREATMENTS]
            ledger["waves"].append({"wave": wave, "cells": cells})
            diagnostic_campaign._atomic_json(self.root / "ledger.json", ledger)
        ledger["status"] = ("complete" if all(
            cell["outcome"] == "success" for wave in ledger["waves"] for cell in wave["cells"])
                            else "smoke_failures")
        diagnostic_campaign._atomic_json(self.root / "ledger.json", ledger)
        return ledger
