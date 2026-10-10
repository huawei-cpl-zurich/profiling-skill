#!/usr/bin/env python3
"""Build, schedule, resume, and report an audited treatment campaign."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Protocol


TASKS = ("matmul", "gdn", "bsa")
TREATMENTS = (
    "cannbot-all", "cannbot-new-profiler",
    "guarded-new-profiler", "guarded-old-profiler",
)
LEGACY_TREATMENTS = ("cannbot", "project-cannbot", "project-guarded")
LEGACY_REQUEST_BUDGET = 24
REPAIR_REQUEST_BUDGET = 48
SUPPORTED_REQUEST_BUDGETS = {LEGACY_REQUEST_BUDGET, REPAIR_REQUEST_BUDGET}
TREATMENT_SKILLS = {
    "cannbot-all": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ops-profiling",
    ),
    "cannbot-new-profiler": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ascend-profiling",
    ),
    "guarded-new-profiler": ("ascend-profiling", "triton-guarded-kernel"),
    "guarded-old-profiler": ("ascend-profiling", "triton-guarded-kernel"),
}
LEGACY_TREATMENT_SKILLS = {
    "cannbot": TREATMENT_SKILLS["cannbot-all"],
    "project-cannbot": TREATMENT_SKILLS["cannbot-new-profiler"],
    "project-guarded": TREATMENT_SKILLS["guarded-new-profiler"],
}


class CampaignError(RuntimeError):
    pass


class CampaignPaused(CampaignError):
    """The ledger is durable, but infrastructure must recover before resume."""


class InfrastructureFailure(RuntimeError):
    def __init__(self, message: str, durable_handle: str | None = None):
        super().__init__(message)
        self.durable_handle = durable_handle


class ResourcePool(Protocol):
    def admit(self) -> list[dict]: ...


class CellLauncher(Protocol):
    def launch(self, cell: dict, slot: dict) -> dict: ...

    def observe(self, cell: dict, placement: dict, durable_handle: str) -> dict: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hex(value: object, length: int, field: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise CampaignError(f"{field} must be a pinned {length}-character digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CampaignError(f"{field} must be hexadecimal") from error


def _validate_provenance(provenance: dict, *, require_starters: bool = True) -> None:
    if not isinstance(provenance, dict):
        raise CampaignError("provenance must be an object")
    _require_hex(provenance.get("source_revision"), 40, "source_revision")
    _require_hex(provenance.get("controller_sha256"), 64, "controller_sha256")
    image = provenance.get("runtime_image_digest")
    if not isinstance(image, str) or not image.startswith("sha256:"):
        raise CampaignError("runtime_image_digest must be pinned by sha256")
    _require_hex(image.removeprefix("sha256:"), 64, "runtime_image_digest")
    model = provenance.get("model")
    if not isinstance(model, dict) or not all(model.get(key) for key in
                                               ("name", "reasoning_effort")):
        raise CampaignError("model name and reasoning_effort must be pinned")
    baselines = provenance.get("baselines")
    if not isinstance(baselines, dict) or set(baselines) != set(TASKS):
        raise CampaignError("all task baselines must be pinned")
    for task in TASKS:
        _require_hex(baselines[task], 64, f"baselines.{task}")
    starters = provenance.get("starters")
    if starters is None and not require_starters:
        starters = None
    elif not isinstance(starters, dict) or set(starters) != set(TASKS):
        raise CampaignError("all task starters must be pinned")
    for task in TASKS if starters is not None else ():
        starter = starters[task]
        if not isinstance(starter, dict) or set(starter) != {"candidate", "manifest"}:
            raise CampaignError(f"starter.{task} must pin candidate and manifest")
        for kind in ("candidate", "manifest"):
            binding = starter[kind]
            path = Path(binding.get("path", "")) if isinstance(binding, dict) else Path("")
            if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                    or not path.is_absolute()):
                raise CampaignError(f"starter.{task}.{kind} must pin an absolute path")
            _require_hex(binding.get("sha256"), 64, f"starter.{task}.{kind}.sha256")
    skills = provenance.get("skills")
    # Composite upstream bundles may pin their internal skills with one digest.
    accepted = (
        {"cannbot", "profiler-new", "profiler-old", "guarded"},
        {"cannbot", "ascend-profiling", "triton-guarded-kernel"},
    )
    if (not isinstance(skills, dict)
            or not any(required.issubset(skills) for required in accepted)):
        raise CampaignError("all treatment skill freezes must be pinned")
    for name, digest in skills.items():
        _require_hex(digest, 64, f"skills.{name}")


def _balanced_order(
    seed: str, treatments: tuple[str, ...] = TREATMENTS,
) -> list[tuple[str, str]]:
    """Return task-complete blocks balanced across the selected treatments."""
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    tasks = list(TASKS)
    treatments = list(treatments)
    rng.shuffle(tasks)
    rng.shuffle(treatments)
    direction = -1 if rng.randrange(2) else 1
    return [
        (tasks[index], treatments[(index + direction * offset) % len(treatments)])
        for offset in range(len(treatments))
        for index in range(len(tasks))
    ]


def _validate_treatment_subset(treatments: tuple[str, ...]) -> None:
    if not treatments:
        raise CampaignError("treatment subset must be nonempty")
    if len(treatments) != len(set(treatments)):
        raise CampaignError("treatment subset must contain unique names")
    if any(treatment not in TREATMENT_SKILLS for treatment in treatments):
        raise CampaignError("treatment subset must contain only known treatments")


def build_manifest(
    run_id: str,
    prompt: Path,
    task_files: dict[str, Path],
    provenance: dict,
    ordering_seed: str,
    *,
    rounds: int = 4,
    request_budget: int = REPAIR_REQUEST_BUDGET,
    treatments: tuple[str, ...] | None = None,
) -> dict:
    if not run_id or "/" in run_id:
        raise CampaignError("run_id must be a nonempty branch-safe component")
    if rounds != 4:
        raise CampaignError("this campaign requires exactly four rounds")
    if request_budget not in SUPPORTED_REQUEST_BUDGETS:
        raise CampaignError("this campaign requires a 24 or 48-request budget")
    if set(task_files) != set(TASKS):
        raise CampaignError("task files must cover exactly matmul, gdn, and bsa")
    if not prompt.is_file() or any(not task_files[name].is_file() for name in TASKS):
        raise CampaignError("prompt and task files must be regular files")
    _validate_provenance(provenance)
    selected = TREATMENTS if treatments is None else tuple(treatments)
    _validate_treatment_subset(selected)
    order = _balanced_order(ordering_seed, selected)
    prompt_digest = _sha256(prompt)
    task_digests = {name: _sha256(task_files[name]) for name in TASKS}
    cells = []
    for task, treatment in order:
        cell_id = f"{task}-{treatment}"
        cells.append({
            "cell_id": cell_id,
            "task": task,
            "treatment": treatment,
            "branch": f"experiment/{run_id}/{cell_id}",
            "round_count": rounds,
            "request_budget": request_budget,
            "task_sha256": task_digests[task],
            "skills": list(TREATMENT_SKILLS[treatment]),
            "prompt_contract": {
                "invariant_sha256": prompt_digest,
                "task_sha256": task_digests[task],
            },
        })
    document = {
        "schema_version": 4,
        "run_id": run_id,
        "dimensions": {"tasks": list(TASKS), "treatments": list(selected),
                       "round_count": rounds},
        "request_budget": request_budget,
        "prompt": {"path": str(prompt.resolve()), "sha256": prompt_digest},
        "tasks": {name: {"path": str(task_files[name].resolve()),
                         "sha256": task_digests[name]} for name in TASKS},
        "provenance": provenance,
        "ordering": {"algorithm": "balanced-latin-v1", "seed": ordering_seed},
        "order": [f"{task}-{treatment}" for task, treatment in order],
        "cells": cells,
    }
    document["manifest_sha256"] = _document_digest(document)
    return document


def _document_digest(document: dict) -> str:
    payload = {key: value for key, value in document.items()
               if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(document: dict) -> None:
    version = document.get("schema_version")
    if version not in {1, 2, 3, 4}:
        raise CampaignError("manifest schema_version must be 1, 2, 3, or 4")
    if document.get("manifest_sha256") != _document_digest(document):
        raise CampaignError("manifest hash mismatch")
    if document.get("order") != [cell.get("cell_id") for cell in document.get("cells", [])]:
        raise CampaignError("manifest order and cells disagree")
    dimensions = document.get("dimensions")
    selected = tuple(dimensions.get("treatments", [])) if isinstance(dimensions, dict) else ()
    if version in {1, 2}:
        if selected != LEGACY_TREATMENTS:
            raise CampaignError("legacy manifests require all three treatments")
    elif version == 3:
        if (not selected or len(selected) != len(set(selected))
                or any(name not in LEGACY_TREATMENT_SKILLS for name in selected)):
            raise CampaignError("schema v3 contains an unknown legacy treatment")
    else:
        _validate_treatment_subset(selected)
    if len(document["cells"]) != len(TASKS) * len(selected):
        raise CampaignError("manifest cell count does not match its treatment matrix")
    actual = {(cell["task"], cell["treatment"]) for cell in document["cells"]}
    expected = {(task, treatment) for task in TASKS for treatment in selected}
    if actual != expected:
        raise CampaignError("manifest must contain the exact declared treatment matrix")
    budget = document.get("request_budget")
    if budget not in SUPPORTED_REQUEST_BUDGETS:
        raise CampaignError("campaign requires a 24 or 48-request budget")
    for cell in document["cells"]:
        if cell["round_count"] != 4 or cell["request_budget"] != budget:
            raise CampaignError(
                "every cell requires four rounds and the campaign request budget"
            )
        policies = TREATMENT_SKILLS if version == 4 else LEGACY_TREATMENT_SKILLS
        if cell["skills"] != list(policies[cell["treatment"]]):
            raise CampaignError(f"treatment skill isolation mismatch: {cell['cell_id']}")
        if "device" in cell or "target" in cell:
            raise CampaignError("device placement belongs only in the runtime ledger")
    _validate_provenance(document["provenance"], require_starters=version in {2, 3, 4})


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _ledger_lock(path: Path):
    """Exclude reconciliation for the full ledger read/modify/write lifetime."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _new_ledger(manifest: dict) -> dict:
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "status": "running",
        "order": manifest["order"],
        "cells": {
            cell_id: {"status": "queued", "attempts": []}
            for cell_id in manifest["order"]
        },
    }


def _validate_ledger(manifest: dict, ledger: dict) -> None:
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        raise CampaignError("ledger schema_version must be 1")
    if ledger.get("run_id") != manifest["run_id"]:
        raise CampaignError("ledger run_id does not match manifest")
    if ledger.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise CampaignError("ledger manifest_sha256 does not match manifest")
    if ledger.get("order") != manifest["order"]:
        raise CampaignError("ledger cell order does not match manifest")
    cells = ledger.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(manifest["order"]):
        raise CampaignError("ledger cells do not match manifest")
    if any(not isinstance(cells[cell_id], dict)
           or not isinstance(cells[cell_id].get("attempts"), list)
           for cell_id in manifest["order"]):
        raise CampaignError("ledger cell state is malformed")


def _admitted_slots(pool: ResourcePool) -> list[dict]:
    admitted = []
    seen = set()
    for slot in pool.admit():
        if not slot.get("healthy") or not slot.get("idle"):
            continue
        identity = (slot.get("target"), slot.get("device"))
        if not isinstance(identity[0], str) or identity[1] is None or identity in seen:
            continue
        seen.add(identity)
        admitted.append({"target": identity[0], "device": identity[1]})
    return admitted


def _validate_receipt(cell: dict, receipt: dict) -> None:
    status = receipt.get("status")
    if status not in {"complete", "candidate_failed"}:
        raise InfrastructureFailure("launcher returned a nonterminal receipt",
                                    receipt.get("durable_handle"))
    if not receipt.get("durable_handle"):
        raise InfrastructureFailure("terminal receipt lacks durable handle")
    count = cell["round_count"]
    if receipt.get("rounds_completed") != count:
        raise InfrastructureFailure("terminal receipt has incomplete round count")
    rounds = receipt.get("rounds")
    commits = receipt.get("commits")
    if (not isinstance(rounds, list) or len(rounds) != count
            or not isinstance(commits, list) or len(commits) != count
            or any(not isinstance(commit, str) or not commit for commit in commits)
            or len(set(commits)) != count):
        raise InfrastructureFailure("terminal receipt lacks four commits and receipts")
    expected_rounds = list(range(1, count + 1))
    if (any(not isinstance(item, dict) for item in rounds)
            or [item.get("round") for item in rounds] != expected_rounds
            or any(item.get("status") not in {"ok", "candidate_error"}
                   or not isinstance(item.get("handle"), str) or not item["handle"]
                   for item in rounds)):
        raise InfrastructureFailure("terminal receipt has malformed round evidence")
    failures = [item for item in rounds if item["status"] == "candidate_error"]
    if any(not isinstance(item.get(field), str) or not item[field]
           for item in failures for field in ("failure_type", "reason")):
        raise InfrastructureFailure("candidate_error round lacks failure evidence")
    if status == "candidate_failed" and not failures:
        raise InfrastructureFailure("candidate_failed lacks a candidate_error round")
    if status == "complete" and failures:
        raise InfrastructureFailure("complete receipt contains a candidate_error round")
    history = receipt.get("attempt_history")
    if history is not None:
        if (not isinstance(history, list) or len(history) != count
                or [item.get("round") if isinstance(item, dict) else None
                    for item in history] != expected_rounds
                or any(not isinstance(item, dict)
                       or set(item) != {"round", "statuses"}
                       or not isinstance(item["statuses"], list)
                       or not 1 <= len(item["statuses"]) <= 3
                       or any(not isinstance(value, str) or not value
                              for value in item["statuses"])
                       for item in history)):
            raise InfrastructureFailure("terminal receipt has malformed attempt history")
        for item, round_receipt in zip(history, rounds):
            expected = (round_receipt.get("failure_type", "candidate_error")
                        if round_receipt["status"] == "candidate_error" else "ok")
            if item["statuses"][-1] != expected:
                raise InfrastructureFailure(
                    "terminal receipt attempt history contradicts round status"
                )


def _run_campaign_locked(
    manifest: dict,
    ledger_path: Path,
    pool: ResourcePool,
    launcher: CellLauncher,
    *,
    resume: bool = False,
) -> dict:
    verify_manifest(manifest)
    if ledger_path.exists():
        if not resume:
            raise CampaignError("ledger exists; use resume to continue")
        ledger = json.loads(ledger_path.read_text())
        _validate_ledger(manifest, ledger)
    else:
        if resume:
            raise CampaignError("cannot resume without a ledger")
        ledger = _new_ledger(manifest)
        _atomic_json(ledger_path, ledger)
    cell_by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    deferred = set()
    occupied = set()
    running = {}
    admission_error = None

    def finish(future) -> None:
        cell_id, attempt, slot = running.pop(future)
        occupied.discard((slot["target"], slot["device"]))
        state = ledger["cells"][cell_id]
        try:
            receipt = future.result()
            _validate_receipt(cell_by_id[cell_id], receipt)
        except InfrastructureFailure as error:
            attempt.update({"status": "infrastructure_error", "error": str(error)})
            if error.durable_handle:
                attempt["durable_handle"] = error.durable_handle
            else:
                attempt.pop("durable_handle", None)
            state["status"] = "infrastructure_pending"
            deferred.add(cell_id)
        except Exception as error:  # launcher crashes are infrastructure, not kernels
            attempt.update({"status": "infrastructure_error", "error": str(error)})
            state["status"] = "infrastructure_pending"
            deferred.add(cell_id)
        else:
            attempt.update({"status": receipt["status"],
                            "durable_handle": receipt["durable_handle"],
                            "receipt": receipt})
            state["status"] = receipt["status"]
        _atomic_json(ledger_path, ledger)

    with ThreadPoolExecutor(max_workers=len(manifest["cells"])) as executor:
        # Retained handles are already-running work. Observe them concurrently,
        # reserve their placements, and never convert observation loss into a
        # fresh launch. Known handleless failures are retryable; an interrupted
        # dispatch without a handle remains pending because relaunch could
        # duplicate remote work.
        for cell_id in manifest["order"]:
            state = ledger["cells"][cell_id]
            if state["status"] not in {"infrastructure_pending", "running"}:
                continue
            attempt = state["attempts"][-1]
            handle = attempt.get("durable_handle")
            if not handle:
                if state["status"] == "running":
                    attempt.update({
                        "status": "infrastructure_error",
                        "error": "interrupted dispatch has no retained handle",
                        "retryable": False,
                    })
                    state["status"] = "infrastructure_pending"
                    deferred.add(cell_id)
                elif attempt.get("retryable", True):
                    state["status"] = "queued"
                else:
                    deferred.add(cell_id)
                continue
            observe = getattr(launcher, "observe", None)
            if not callable(observe):
                attempt["error"] = "launcher cannot observe its retained handle"
                deferred.add(cell_id)
                continue
            slot = {"target": attempt["target"], "device": attempt["device"]}
            observation = {
                "target": slot["target"], "device": slot["device"],
                "status": "running", "durable_handle": handle,
                "kind": "observe",
            }
            state["attempts"].append(observation)
            occupied.add((slot["target"], slot["device"]))
            state["status"] = "running"
            future = executor.submit(
                observe, cell_by_id[cell_id], slot, handle
            )
            running[future] = (cell_id, observation, slot)
        _atomic_json(ledger_path, ledger)

        while True:
            # Refresh admission for every assignment. This continuously fills
            # newly free slots rather than waiting for an earlier batch.
            while True:
                cell_id = next((candidate for candidate in manifest["order"]
                                if ledger["cells"][candidate]["status"] == "queued"
                                and candidate not in deferred), None)
                if cell_id is None:
                    break
                try:
                    slots = [slot for slot in _admitted_slots(pool)
                             if (slot["target"], slot["device"]) not in occupied]
                    admission_error = None
                except Exception as error:
                    slots = []
                    admission_error = str(error)
                if not slots:
                    break
                slot = slots[0]
                state = ledger["cells"][cell_id]
                state["status"] = "running"
                attempt = {"target": slot["target"], "device": slot["device"],
                           "status": "running"}
                state["attempts"].append(attempt)
                occupied.add((slot["target"], slot["device"]))
                future = executor.submit(
                    launcher.launch, cell_by_id[cell_id], slot
                )
                running[future] = (cell_id, attempt, slot)
                _atomic_json(ledger_path, ledger)

            if running:
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in completed:
                    finish(future)
                continue

            queued = [cell_id for cell_id in manifest["order"]
                      if ledger["cells"][cell_id]["status"] == "queued"]
            pending_infra = [cell_id for cell_id in manifest["order"]
                             if ledger["cells"][cell_id]["status"] ==
                             "infrastructure_pending"]
            if queued or pending_infra:
                ledger["status"] = "infrastructure_pending"
                _atomic_json(ledger_path, ledger)
                reason = ("infrastructure failed; independent cells completed"
                          if pending_infra else admission_error
                          or "no healthy idle BZ-A3 devices were admitted")
                raise CampaignPaused(reason)
            break
    ledger["status"] = "complete"
    _atomic_json(ledger_path, ledger)
    return ledger


def run_campaign(
    manifest: dict,
    ledger_path: Path,
    pool: ResourcePool,
    launcher: CellLauncher,
    *,
    resume: bool = False,
) -> dict:
    with _ledger_lock(ledger_path):
        return _run_campaign_locked(
            manifest, ledger_path, pool, launcher, resume=resume,
        )


def build_report(manifest: dict, ledger: dict) -> dict:
    verify_manifest(manifest)
    _validate_ledger(manifest, ledger)
    rows = []
    discarded = []
    summary = {"complete": 0, "candidate_failed": 0,
               "infrastructure_pending": 0}
    aggregate_attempts = {"attempts": 0, "successful_attempts": 0,
                          "successful_rounds": 0, "repair_attempted_rounds": 0,
                          "repaired_rounds": 0, "repair_count": 0}
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    for cell_id in manifest["order"]:
        state = ledger["cells"][cell_id]
        status = state["status"]
        summary[status if status in summary else "infrastructure_pending"] += 1
        terminal = next((attempt.get("receipt") for attempt in reversed(state["attempts"])
                         if attempt.get("receipt")), {})
        evolution = terminal.get("rounds", [])
        retained_history = terminal.get("attempt_history")
        history_by_round = {
            item["round"]: item["statuses"] for item in retained_history
        } if isinstance(retained_history, list) else {}
        attempt_history = [
            {"round": item.get("round"), "statuses": history_by_round.get(
                item.get("round"),
                [item.get("failure_type", item.get("status"))],
            )}
            for item in evolution
        ]
        candidate_errors = [
            {"round": item.get("round"), "failure_type": item.get("failure_type"),
             "reason": item.get("reason")}
            for item in evolution if item.get("status") == "candidate_error"
        ]
        raw = [item for item in evolution
               if isinstance(item.get("median_us"), (int, float))
               and math.isfinite(item["median_us"]) and item["median_us"] > 0]
        def normalized_value(item: dict) -> float | None:
            value = item.get("normalized_median_us")
            if not isinstance(value, (int, float)):
                normalization = item.get("normalization")
                value = (normalization.get("normalized_latency_us")
                         if isinstance(normalization, dict) else None)
            return (float(value) if isinstance(value, (int, float))
                    and math.isfinite(value) and value > 0 else None)

        normalized = [(item, normalized_value(item)) for item in evolution]
        normalized = [(item, value) for item, value in normalized if value is not None]
        best_raw = min(raw, key=lambda item: item["median_us"]) if raw else None
        best_normalized = min(normalized, key=lambda pair: pair[1]) if normalized else None
        best = best_normalized[0] if best_normalized else best_raw
        baseline = terminal.get("baseline") or (best.get("baseline") if best else None)
        baseline_us = terminal.get("baseline_median_us")
        if baseline_us is None and best:
            baseline_us = best.get("baseline_median_us")
        if isinstance(baseline, dict):
            baseline_us = baseline.get("median_us", baseline_us)
        comparison_us = best_normalized[1] if best_normalized else (
            best_raw["median_us"] if best_raw else None
        )
        controller_speedup = best.get("speedup_vs_baseline") if best else None
        speedup = (controller_speedup
                   if best_normalized and isinstance(controller_speedup, (int, float))
                   and math.isfinite(controller_speedup) and controller_speedup > 0
                   else baseline_us / comparison_us
                   if comparison_us and isinstance(baseline_us, (int, float))
                   and math.isfinite(baseline_us) and baseline_us > 0 else None)
        for number, attempt in enumerate(state["attempts"], 1):
            if attempt.get("status") == "infrastructure_error":
                discarded.append({
                    "cell_id": cell_id, "attempt": number,
                    "target": attempt.get("target"), "device": attempt.get("device"),
                    "durable_handle": attempt.get("durable_handle"),
                    "error": attempt.get("error"),
                })
        cell = by_id[cell_id]
        attempts = 0
        successful_attempts = 0
        repair_attempted_rounds = 0
        repaired_rounds = 0
        transitions = []
        for item, history_item in zip(evolution, attempt_history):
            statuses = history_item["statuses"]
            attempts += len(statuses)
            successful_attempts += sum(status == "ok" for status in statuses)
            repair_attempted_rounds += int(len(statuses) > 1)
            repaired_rounds += int(len(statuses) > 1 and statuses[-1] == "ok")
            transitions.extend(
                {"round": item.get("round"), "from": before, "to": after}
                for before, after in zip(statuses, statuses[1:])
            )
        successful_rounds = sum(item.get("status") == "ok" for item in evolution)
        terminal_round = evolution[-1] if evolution else {}
        final_raw = terminal_round.get("median_us")
        final_normalized = normalized_value(terminal_round)
        valid_final_raw = (isinstance(final_raw, (int, float))
                           and math.isfinite(final_raw) and final_raw > 0)
        final_timing = (
            {"round": terminal_round.get("round"), "median_us": final_raw,
             "normalized_median_us": final_normalized}
            if (valid_final_raw or final_normalized is not None) else None
        )
        attempt_summary = {
            "attempts": attempts,
            "successful_attempts": successful_attempts,
            "raw_attempt_success_rate": (
                successful_attempts / attempts if attempts else None
            ),
            "successful_rounds": successful_rounds,
            "repair_attempted_rounds": repair_attempted_rounds,
            "repaired_rounds": repaired_rounds,
            "repaired_round_success_rate": (
                repaired_rounds / repair_attempted_rounds
                if repair_attempted_rounds else None
            ),
            "repair_count": attempts - len(evolution),
            "failure_transitions": transitions,
            "final_timing": final_timing,
        }
        for key in aggregate_attempts:
            aggregate_attempts[key] += attempt_summary[key]
        rows.append({
            "cell_id": cell_id, "task": cell["task"],
            "treatment": cell["treatment"], "branch": cell["branch"],
            "status": status, "attempt_count": len(state["attempts"]),
            "raw_evolution": evolution,
            "attempt_history": attempt_history,
            "normalized_evolution": [
                {"round": item["round"],
                 "normalized_samples_us": item.get("normalized_samples_us"),
                 "normalized_median_us": value,
                 "speedup_vs_baseline": item.get("speedup_vs_baseline"),
                 "baseline_median_us": item.get("baseline_median_us"),
                 "calibration": item.get("calibration")}
                for item, value in normalized
            ],
            "per_case_evidence": [
                {"round": item.get("round"), "case_results": item["case_results"]}
                for item in evolution if isinstance(item.get("case_results"), list)
            ],
            "controls": [
                {"round": item.get("round"), "controls": item.get("controls"),
                 "calibration": item.get("calibration"), "policy": item.get("policy")}
                for item in evolution
                if any(key in item for key in ("controls", "calibration", "policy"))
            ],
            "best_round": best["round"] if best else None,
            "best_median_us": best_raw["median_us"] if best_raw else None,
            "best_normalized_median_us": (
                best_normalized[1] if best_normalized else None
            ),
            "comparison_basis": ("calibration_normalized_median_us"
                                 if best_normalized else "raw_median_us"),
            "comparison_median_us": comparison_us,
            "baseline": baseline,
            "baseline_median_us": baseline_us,
            "speedup_vs_baseline": speedup,
            "candidate_errors": candidate_errors,
            "attempt_summary": attempt_summary,
            "failure": candidate_errors or (
                terminal.get("failure") if terminal else None
            ),
        })
    aggregate_attempts["raw_attempt_success_rate"] = (
        aggregate_attempts["successful_attempts"] / aggregate_attempts["attempts"]
        if aggregate_attempts["attempts"] else None
    )
    aggregate_attempts["repaired_round_success_rate"] = (
        aggregate_attempts["repaired_rounds"]
        / aggregate_attempts["repair_attempted_rounds"]
        if aggregate_attempts["repair_attempted_rounds"] else None
    )
    return {"schema_version": 2, "run_id": manifest["run_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": summary, "cells": rows,
            "attempt_summary": aggregate_attempts,
            "discarded_infrastructure_attempts": discarded}


class _FakePool:
    def __init__(self, slots: int):
        self._slots = slots

    def admit(self) -> list[dict]:
        return [{"target": f"fake-bz-{index % 2 + 1}", "device": index,
                 "healthy": True, "idle": True} for index in range(self._slots)]


class _FakeLauncher:
    def launch(self, cell: dict, slot: dict) -> dict:
        del slot
        return {"status": "complete", "durable_handle": f"fake:{cell['cell_id']}",
                "rounds_completed": 4,
                "rounds": [{"round": number, "status": "ok",
                            "handle": f"fake:round-{number}",
                            "median_us": 10.0 - number}
                           for number in range(1, 5)],
                "commits": [f"fake-commit-{number}" for number in range(1, 5)]}

    def observe(self, cell: dict, placement: dict, durable_handle: str) -> dict:
        del placement
        receipt = self.launch(cell, {})
        receipt["durable_handle"] = durable_handle
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--prompt", type=Path, required=True)
    for task in TASKS:
        generate.add_argument(f"--{task}-task", type=Path, required=True)
    generate.add_argument("--provenance", type=Path, required=True)
    generate.add_argument("--ordering-seed", required=True)
    generate.add_argument("--request-budget", type=int, choices=(24, 48), default=48)
    generate.add_argument(
        "--treatment", action="append", choices=TREATMENTS,
        help="repeat to select a schema-v4 treatment subset; defaults to all treatments",
    )
    generate.add_argument("--output", type=Path, required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--manifest", type=Path, required=True)
    simulate.add_argument("--ledger", type=Path, required=True)
    simulate.add_argument("--slots", type=int, default=4)
    simulate.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--ledger", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        document = build_manifest(
            args.run_id, args.prompt,
            {task: getattr(args, f"{task}_task") for task in TASKS},
            json.loads(args.provenance.read_text()), args.ordering_seed,
            request_budget=args.request_budget,
            treatments=(tuple(args.treatment) if args.treatment is not None else None),
        )
        _atomic_json(args.output, document)
    elif args.command == "simulate":
        document = json.loads(args.manifest.read_text())
        run_campaign(document, args.ledger, _FakePool(args.slots), _FakeLauncher(),
                     resume=args.resume)
    else:
        document = json.loads(args.manifest.read_text())
        ledger = json.loads(args.ledger.read_text())
        _atomic_json(args.output, build_report(document, ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
