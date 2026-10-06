#!/usr/bin/env python3
"""Build, schedule, resume, and report an audited nine-branch campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol


TASKS = ("matmul", "gdn", "bsa")
TREATMENTS = ("cannbot", "project-cannbot", "project-guarded")
TREATMENT_SKILLS = {
    "cannbot": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ops-profiling",
    ),
    "project-cannbot": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ascend-profiling",
    ),
    "project-guarded": ("ascend-profiling", "triton-guarded-kernel"),
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


def _validate_provenance(provenance: dict) -> None:
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
    skills = provenance.get("skills")
    # Composite upstream bundles may pin their internal skills with one digest.
    accepted = {"cannbot", "ascend-profiling", "triton-guarded-kernel"}
    if not isinstance(skills, dict) or not accepted.issubset(skills):
        raise CampaignError("cannbot and project skill bundles must be pinned")
    for name, digest in skills.items():
        _require_hex(digest, 64, f"skills.{name}")


def _balanced_order(seed: str) -> list[tuple[str, str]]:
    """Return three balanced blocks; every block covers every axis once."""
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
    tasks = list(TASKS)
    treatments = list(TREATMENTS)
    rng.shuffle(tasks)
    rng.shuffle(treatments)
    direction = -1 if rng.randrange(2) else 1
    return [
        (tasks[index], treatments[(index + direction * offset) % 3])
        for offset in range(3)
        for index in range(3)
    ]


def build_manifest(
    run_id: str,
    prompt: Path,
    task_files: dict[str, Path],
    provenance: dict,
    ordering_seed: str,
    *,
    rounds: int = 4,
    request_budget: int = 24,
) -> dict:
    if not run_id or "/" in run_id:
        raise CampaignError("run_id must be a nonempty branch-safe component")
    if rounds != 4:
        raise CampaignError("this campaign requires exactly four rounds")
    if request_budget != 24:
        raise CampaignError("this campaign requires a 24-request budget")
    if set(task_files) != set(TASKS):
        raise CampaignError("task files must cover exactly matmul, gdn, and bsa")
    if not prompt.is_file() or any(not task_files[name].is_file() for name in TASKS):
        raise CampaignError("prompt and task files must be regular files")
    _validate_provenance(provenance)
    order = _balanced_order(ordering_seed)
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
        "schema_version": 1,
        "run_id": run_id,
        "dimensions": {"tasks": list(TASKS), "treatments": list(TREATMENTS),
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
    if document.get("manifest_sha256") != _document_digest(document):
        raise CampaignError("manifest hash mismatch")
    if document.get("order") != [cell.get("cell_id") for cell in document.get("cells", [])]:
        raise CampaignError("manifest order and cells disagree")
    if len(document["cells"]) != 9:
        raise CampaignError("manifest must contain exactly nine cells")
    actual = {(cell["task"], cell["treatment"]) for cell in document["cells"]}
    expected = {(task, treatment) for task in TASKS for treatment in TREATMENTS}
    if actual != expected:
        raise CampaignError("manifest must contain the exact 3x3 treatment matrix")
    for cell in document["cells"]:
        if cell["round_count"] != 4 or cell["request_budget"] != 24:
            raise CampaignError("every cell requires four rounds and 24 requests")
        if cell["skills"] != list(TREATMENT_SKILLS[cell["treatment"]]):
            raise CampaignError(f"treatment skill isolation mismatch: {cell['cell_id']}")
        if "device" in cell or "target" in cell:
            raise CampaignError("device placement belongs only in the runtime ledger")
    _validate_provenance(document["provenance"])


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
    if receipt.get("status") not in {"complete", "candidate_failed"}:
        raise InfrastructureFailure("launcher returned a nonterminal receipt",
                                    receipt.get("durable_handle"))
    if not receipt.get("durable_handle"):
        raise InfrastructureFailure("terminal receipt lacks durable handle")
    if receipt["status"] == "complete":
        if receipt.get("rounds_completed") != cell["round_count"]:
            raise InfrastructureFailure("terminal receipt has incomplete round count",
                                        receipt["durable_handle"])
        rounds = receipt.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != cell["round_count"]:
            raise InfrastructureFailure("terminal receipt lacks round evidence",
                                        receipt["durable_handle"])


def run_campaign(
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
        if ledger.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise CampaignError("resume manifest does not match ledger")
    else:
        if resume:
            raise CampaignError("cannot resume without a ledger")
        ledger = _new_ledger(manifest)
        _atomic_json(ledger_path, ledger)
    cell_by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    # A retained handle is an already-submitted workload. Recover it before
    # admitting new work, and never turn observer loss into duplicate dispatch.
    for cell_id in manifest["order"]:
        state = ledger["cells"][cell_id]
        if state["status"] != "infrastructure_pending" or not state["attempts"]:
            continue
        attempt = state["attempts"][-1]
        handle = attempt.get("durable_handle")
        if not handle:
            continue
        observe = getattr(launcher, "observe", None)
        if not callable(observe):
            ledger["status"] = "infrastructure_pending"
            _atomic_json(ledger_path, ledger)
            raise CampaignPaused(
                f"{cell_id} has retained handle {handle}; launcher cannot observe it"
            )
        placement = {"target": attempt["target"], "device": attempt["device"]}
        try:
            receipt = observe(cell_by_id[cell_id], placement, handle)
            _validate_receipt(cell_by_id[cell_id], receipt)
        except InfrastructureFailure as error:
            attempt["error"] = str(error)
            _atomic_json(ledger_path, ledger)
            raise CampaignPaused("retained handle is still pending") from error
        except Exception as error:
            attempt["error"] = str(error)
            _atomic_json(ledger_path, ledger)
            raise CampaignPaused("retained handle observation failed") from error
        attempt.update({"status": receipt["status"], "receipt": receipt})
        state["status"] = receipt["status"]
        _atomic_json(ledger_path, ledger)
    pending = [cell_id for cell_id in manifest["order"]
               if ledger["cells"][cell_id]["status"] not in
               {"complete", "candidate_failed"}]
    while pending:
        slots = _admitted_slots(pool)
        if not slots:
            ledger["status"] = "infrastructure_pending"
            _atomic_json(ledger_path, ledger)
            raise CampaignPaused("no healthy idle BZ-A3 devices were admitted")
        batch = pending[:len(slots)]
        failures = False
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for cell_id, slot in zip(batch, slots):
                state = ledger["cells"][cell_id]
                state["status"] = "running"
                attempt = {"target": slot["target"], "device": slot["device"],
                           "status": "running"}
                state["attempts"].append(attempt)
                _atomic_json(ledger_path, ledger)
                futures[executor.submit(launcher.launch, cell_by_id[cell_id], slot)] = (
                    cell_id, attempt
                )
            for future in as_completed(futures):
                cell_id, attempt = futures[future]
                state = ledger["cells"][cell_id]
                try:
                    receipt = future.result()
                    _validate_receipt(cell_by_id[cell_id], receipt)
                except InfrastructureFailure as error:
                    attempt.update({"status": "infrastructure_error", "error": str(error)})
                    if error.durable_handle:
                        attempt["durable_handle"] = error.durable_handle
                    state["status"] = "infrastructure_pending"
                    failures = True
                except Exception as error:  # launcher crashes are infrastructure, not kernels
                    attempt.update({"status": "infrastructure_error", "error": str(error)})
                    state["status"] = "infrastructure_pending"
                    failures = True
                else:
                    attempt.update({"status": receipt["status"],
                                    "durable_handle": receipt["durable_handle"],
                                    "receipt": receipt})
                    state["status"] = receipt["status"]
                _atomic_json(ledger_path, ledger)
        if failures:
            ledger["status"] = "infrastructure_pending"
            _atomic_json(ledger_path, ledger)
            raise CampaignPaused("infrastructure failed; resume from the durable ledger")
        pending = [cell_id for cell_id in manifest["order"]
                   if ledger["cells"][cell_id]["status"] not in
                   {"complete", "candidate_failed"}]
    ledger["status"] = "complete"
    _atomic_json(ledger_path, ledger)
    return ledger


def build_report(manifest: dict, ledger: dict) -> dict:
    verify_manifest(manifest)
    rows = []
    summary = {"complete": 0, "candidate_failed": 0,
               "infrastructure_pending": 0}
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    for cell_id in manifest["order"]:
        state = ledger["cells"][cell_id]
        status = state["status"]
        summary[status if status in summary else "infrastructure_pending"] += 1
        terminal = next((attempt.get("receipt") for attempt in reversed(state["attempts"])
                         if attempt.get("receipt")), {})
        evolution = terminal.get("rounds", [])
        valid = [item for item in evolution
                 if isinstance(item.get("median_us"), (int, float))
                 and math.isfinite(item["median_us"]) and item["median_us"] > 0]
        best = min(valid, key=lambda item: item["median_us"]) if valid else None
        cell = by_id[cell_id]
        rows.append({
            "cell_id": cell_id, "task": cell["task"],
            "treatment": cell["treatment"], "branch": cell["branch"],
            "status": status, "attempt_count": len(state["attempts"]),
            "evolution": evolution,
            "best_round": best["round"] if best else None,
            "best_median_us": best["median_us"] if best else None,
            "baseline_median_us": terminal.get("baseline_median_us"),
            "speedup_vs_baseline": (
                terminal["baseline_median_us"] / best["median_us"]
                if best and isinstance(terminal.get("baseline_median_us"), (int, float))
                and math.isfinite(terminal["baseline_median_us"])
                and terminal["baseline_median_us"] > 0 else None
            ),
            "failure": terminal.get("failure") if terminal else None,
        })
    return {"schema_version": 1, "run_id": manifest["run_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": summary, "cells": rows}


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
                "rounds": [{"round": number, "median_us": 10.0 - number}
                           for number in range(1, 5)]}

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
