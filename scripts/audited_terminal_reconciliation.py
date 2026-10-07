#!/usr/bin/env python3
"""Seal and apply one terminal ledger reconciliation without rerunning a cell."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from audited_campaign import InfrastructureFailure, _validate_receipt
from audited_contract import AuditError
from audited_verifier import validate_branch

PLAN_SCHEMA = "profiling-skill/audited-terminal-reconciliation-plan/v1"
ATTESTATION_SCHEMA = "profiling-skill/audited-terminal-reconciliation/v1"
ARCHIVE_SCHEMA = "profiling-skill/audited-terminal-reconciliation-archive/v1"
JOURNAL_SCHEMA = "profiling-skill/audited-terminal-reconciliation-journal/v1"
PROOF_SCHEMA = "profiling-skill/audited-runtime-migration-trust/v1"
_PLAN_FIELDS = {
    "schema", "ledger_path", "ledger_sha256", "cell_id", "repo_path", "branch",
    "base", "failure_reason", "failure_attempt_sha256", "migration_proofs",
}
_OLD_VERIFIER_FAILURE = re.compile(
    r"^(?:existing branch has no valid blocked checkpoint and is not complete: )?"
    r"independent audited verifier rejected branch: "
    r"usage: validate_audited_experiment\.py \[-h\] \[--base BASE\] repo\n"
    r"validate_audited_experiment\.py: error: experiment [1-9][0-9]* "
    r"evidence provenance diverges$"
)


class ReconciliationError(RuntimeError):
    pass


class InterruptedForTest(RuntimeError):
    """Test-only crash injection after a durable transaction boundary."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _document_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReconciliationError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ReconciliationError(f"{label} must be a JSON object")
    return value


def _load_plan(path: Path) -> dict:
    plan = _read_object(path, "reconciliation plan")
    if set(plan) != _PLAN_FIELDS or plan.get("schema") != PLAN_SCHEMA:
        raise ReconciliationError("reconciliation plan schema is invalid")
    for field in ("ledger_path", "repo_path"):
        value = plan.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ReconciliationError(f"{field} must be an absolute path")
    for field in ("ledger_sha256", "failure_attempt_sha256"):
        value = plan.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ReconciliationError(f"{field} must be a SHA-256 digest")
    if not all(isinstance(plan.get(field), str) and plan[field]
               for field in ("cell_id", "branch", "base", "failure_reason")):
        raise ReconciliationError("plan identity fields are invalid")
    proofs = plan.get("migration_proofs")
    expected = {"schema", "attestation_path", "attestation_file_sha256",
                "attestation_sha256"}
    if (not isinstance(proofs, list) or not proofs
            or any(not isinstance(proof, dict) or set(proof) != expected
                   or proof.get("schema") != PROOF_SCHEMA for proof in proofs)):
        raise ReconciliationError("ordered migration proofs are invalid")
    return plan


def _verified_receipt(plan: dict) -> tuple[dict, dict]:
    repo = Path(plan["repo_path"])
    try:
        verified = validate_branch(
            repo, plan["base"], migration_proofs=plan["migration_proofs"],
        )
    except AuditError as error:
        raise ReconciliationError(f"completed branch verification failed: {error}") from error
    if verified.get("branch") != plan["branch"]:
        raise ReconciliationError("verified branch does not match the plan")
    if verified.get("round_count") != 4:
        raise ReconciliationError("reconciliation requires a completed four-round branch")
    rounds = []
    for number in range(1, verified.get("round_count", 0) + 1):
        result = _read_object(
            repo / "experiments" / f"{number:02d}" / "results.json",
            f"round {number} compact receipt",
        )
        rounds.append({"round": number, **result})
    commits = [item.get("commit") for item in verified.get("experiments", [])]
    if not rounds or len(commits) != len(rounds) or any(
            not isinstance(commit, str) or not commit for commit in commits):
        raise ReconciliationError("verified branch has incomplete terminal evidence")
    terminal = rounds[-1]
    status = ("candidate_failed" if any(
        item.get("status") == "candidate_error" for item in rounds
    ) else "complete")
    receipt = {
        "status": status, "durable_handle": terminal.get("handle"),
        "rounds_completed": len(rounds), "rounds": rounds,
        "branch": verified.get("branch"), "session_id": verified.get("session_id"),
        "seed_commit": verified.get("seed_commit"), "commits": commits,
    }
    for field in ("baseline_median_us", "baseline", "calibration"):
        if field in terminal:
            receipt[field] = terminal[field]
    try:
        _validate_receipt({"round_count": len(rounds)}, receipt)
    except InfrastructureFailure as error:
        raise ReconciliationError(f"reconstructed terminal receipt is invalid: {error}") from error
    return verified, receipt


def _old_ledger(plan: dict) -> tuple[dict, dict, dict]:
    path = Path(plan["ledger_path"])
    if not path.is_file() or _file_sha(path) != plan["ledger_sha256"]:
        raise ReconciliationError("campaign ledger drifted from the plan")
    ledger = _read_object(path, "campaign ledger")
    cells = ledger.get("cells")
    cell = cells.get(plan["cell_id"]) if isinstance(cells, dict) else None
    attempts = cell.get("attempts") if isinstance(cell, dict) else None
    attempt = attempts[-1] if isinstance(attempts, list) and attempts else None
    if (cell is None or cell.get("status") != "infrastructure_pending"
            or not isinstance(attempt, dict)
            or attempt.get("status") != "infrastructure_error"
            or attempt.get("error") != plan["failure_reason"]
            or _document_sha(attempt) != plan["failure_attempt_sha256"]):
        raise ReconciliationError("target cell failure does not match the plan")
    if not _OLD_VERIFIER_FAILURE.fullmatch(plan["failure_reason"]):
        raise ReconciliationError("target is not the old migration-verifier failure")
    return ledger, cell, attempt


def _replacement(ledger: dict, plan: dict, receipt: dict) -> dict:
    updated = copy.deepcopy(ledger)
    cell = updated["cells"][plan["cell_id"]]
    attempt = cell["attempts"][-1]
    attempt.pop("error", None)
    attempt.pop("retryable", None)
    attempt.update({"status": receipt["status"],
                    "durable_handle": receipt["durable_handle"], "receipt": receipt})
    cell["status"] = receipt["status"]
    terminal = {"complete", "candidate_failed"}
    updated["status"] = ("complete" if all(
        item.get("status") in terminal for item in updated["cells"].values()
    ) else "infrastructure_pending")
    return updated


def preflight(plan_path: Path, attestation_path: Path) -> dict:
    plan_path, attestation_path = Path(plan_path).resolve(), Path(attestation_path).resolve()
    plan = _load_plan(plan_path)
    ledger, cell, attempt = _old_ledger(plan)
    verified, receipt = _verified_receipt(plan)
    replacement = _replacement(ledger, plan, receipt)
    archive = {
        "schema": ARCHIVE_SCHEMA, "plan_sha256": _file_sha(plan_path),
        "ledger_sha256": plan["ledger_sha256"], "cell_id": plan["cell_id"],
        "target_cell": cell, "failure_attempt": attempt,
    }
    non_targets = {
        name: _document_sha(value) for name, value in ledger["cells"].items()
        if name != plan["cell_id"]
    }
    body = {
        "schema": ATTESTATION_SCHEMA, "plan_sha256": _file_sha(plan_path),
        "ledger_before_sha256": plan["ledger_sha256"],
        "ledger_after_sha256": hashlib.sha256(_json_bytes(replacement)).hexdigest(),
        "cell_id": plan["cell_id"], "repo_head": verified["experiments"][-1]["commit"],
        "receipt": receipt, "receipt_sha256": _document_sha(receipt),
        "target_before_sha256": _document_sha(cell),
        "target_after_sha256": _document_sha(replacement["cells"][plan["cell_id"]]),
        "non_target_cells_sha256": non_targets,
        "archive_sha256": hashlib.sha256(_json_bytes(archive)).hexdigest(),
    }
    result = {**body, "attestation_sha256": _document_sha(body)}
    _atomic_json(attestation_path, result)
    return result


def _load_attestation(path: Path, expected_file_sha: str, plan_path: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha or ""):
        raise ReconciliationError("attestation file SHA-256 is invalid")
    if not path.is_file() or _file_sha(path) != expected_file_sha:
        raise ReconciliationError("reconciliation attestation drifted")
    value = _read_object(path, "reconciliation attestation")
    seal = value.get("attestation_sha256")
    body = {key: item for key, item in value.items() if key != "attestation_sha256"}
    if (value.get("schema") != ATTESTATION_SCHEMA or seal != _document_sha(body)
            or value.get("plan_sha256") != _file_sha(plan_path)):
        raise ReconciliationError("reconciliation attestation seal is invalid")
    return value


def apply(plan_path: Path, attestation_path: Path, attestation_file_sha256: str,
          journal_dir: Path, *, interrupt_after: str | None = None) -> dict:
    plan_path, attestation_path = Path(plan_path).resolve(), Path(attestation_path).resolve()
    journal_dir = Path(journal_dir).resolve()
    plan = _load_plan(plan_path)
    attestation = _load_attestation(attestation_path, attestation_file_sha256, plan_path)
    _, receipt = _verified_receipt(plan)
    if (_document_sha(receipt) != attestation.get("receipt_sha256")
            or receipt != attestation.get("receipt")):
        raise ReconciliationError("terminal receipt drifted from the attestation")

    ledger_path = Path(plan["ledger_path"])
    current_sha = _file_sha(ledger_path) if ledger_path.is_file() else ""
    before_sha, after_sha = (attestation["ledger_before_sha256"],
                             attestation["ledger_after_sha256"])
    journal_path, archive_path = journal_dir / "journal.json", journal_dir / "archive.json"
    journal = _read_object(journal_path, "transaction journal") if journal_path.exists() else None
    identity = {"schema": JOURNAL_SCHEMA, "attestation_sha256":
                attestation["attestation_sha256"], "plan_sha256": attestation["plan_sha256"],
                "ledger_before_sha256": before_sha, "ledger_after_sha256": after_sha,
                "archive_sha256": attestation["archive_sha256"]}
    if journal is not None and (
            set(journal) != set(identity) | {"phase"}
            or journal.get("phase") not in {"archived", "applied"}
            or any(journal.get(key) != value for key, value in identity.items())):
        raise ReconciliationError("transaction journal drifted from its inputs")
    if current_sha == after_sha:
        if journal is None or not archive_path.is_file() \
                or _file_sha(archive_path) != attestation["archive_sha256"]:
            raise ReconciliationError("applied ledger lacks its authenticated archive")
        _atomic_json(journal_path, {**identity, "phase": "applied"})
        return {"status": "already_applied" if journal.get("phase") == "applied" else "applied",
                "ledger_sha256": after_sha, "archive_sha256": attestation["archive_sha256"]}
    if current_sha != before_sha:
        raise ReconciliationError("campaign ledger drifted from the attested transaction")
    if journal is not None and journal["phase"] == "applied":
        raise ReconciliationError("applied transaction ledger was rolled back")

    ledger, cell, attempt = _old_ledger(plan)
    if ({name: _document_sha(value) for name, value in ledger["cells"].items()
         if name != plan["cell_id"]} != attestation["non_target_cells_sha256"]
            or _document_sha(cell) != attestation["target_before_sha256"]):
        raise ReconciliationError("ledger cells drifted from the attestation")
    archive = {"schema": ARCHIVE_SCHEMA, "plan_sha256": _file_sha(plan_path),
               "ledger_sha256": before_sha, "cell_id": plan["cell_id"],
               "target_cell": cell, "failure_attempt": attempt}
    if archive_path.exists() and _file_sha(archive_path) != attestation["archive_sha256"]:
        raise ReconciliationError("transaction archive drifted")
    if not archive_path.exists():
        _atomic_json(archive_path, archive)
    _atomic_json(journal_path, {**identity, "phase": "archived"})
    if interrupt_after == "archive":
        raise InterruptedForTest("interrupted after archive")

    replacement = _replacement(ledger, plan, receipt)
    if (hashlib.sha256(_json_bytes(replacement)).hexdigest() != after_sha
            or _document_sha(replacement["cells"][plan["cell_id"]]) !=
            attestation["target_after_sha256"]):
        raise ReconciliationError("replacement ledger does not match the attestation")
    _atomic_json(ledger_path, replacement)
    if interrupt_after == "ledger":
        raise InterruptedForTest("interrupted after ledger")
    _atomic_json(journal_path, {**identity, "phase": "applied"})
    return {"status": "applied", "ledger_sha256": after_sha,
            "archive_sha256": attestation["archive_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--attestation", type=Path, required=True)
    execute = subparsers.add_parser("apply")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--attestation", type=Path, required=True)
    execute.add_argument("--attestation-file-sha256", required=True)
    execute.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (preflight(args.plan, args.attestation) if args.command == "preflight"
                  else apply(args.plan, args.attestation, args.attestation_file_sha256,
                             args.journal_dir))
    except ReconciliationError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
