from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audited_terminal_reconciliation as reconciliation  # noqa: E402
from test_audited_verifier import git, migrated_branch  # noqa: E402


OLD_FAILURE = (
    "independent audited verifier rejected branch: "
    "usage: validate_audited_experiment.py [-h] [--base BASE] repo\n"
    "validate_audited_experiment.py: error: experiment 3 evidence provenance diverges"
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@pytest.fixture
def transaction(tmp_path: Path) -> dict:
    repo, proof, _ = migrated_branch(tmp_path, 3)
    target = {
        "status": "infrastructure_pending",
        "attempts": [{
            "target": "bz-a3-1", "device": 2,
            "status": "infrastructure_error", "error": OLD_FAILURE,
            "durable_handle": "remote:bz-a3-1:job:terminal",
            "kind": "observe",
        }],
    }
    ledger = {
        "schema_version": 1, "run_id": "validation",
        "manifest_sha256": "a" * 64, "status": "infrastructure_pending",
        "order": ["agent-a", "other"],
        "cells": {"agent-a": target, "other": {
            "status": "complete", "attempts": [{"receipt": {"opaque": True}}],
        }},
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    plan = {
        "schema": reconciliation.PLAN_SCHEMA,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": file_sha(ledger_path),
        "cell_id": "agent-a", "repo_path": str(repo.resolve()),
        "branch": "experiment/validation/agent-a", "base": "main",
        "failure_reason": OLD_FAILURE,
        "failure_attempt_sha256": document_sha(target["attempts"][-1]),
        "migration_proofs": [proof],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"repo": repo, "proof": proof, "ledger": ledger,
            "ledger_path": ledger_path, "plan": plan, "plan_path": plan_path,
            "attestation": tmp_path / "attestation.json",
            "journal": tmp_path / "journal"}


def attest(tx: dict) -> tuple[dict, str]:
    result = reconciliation.preflight(tx["plan_path"], tx["attestation"])
    return result, file_sha(tx["attestation"])


def repin(tx: dict, ledger: dict) -> None:
    tx["ledger_path"].write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    tx["plan"]["ledger_sha256"] = file_sha(tx["ledger_path"])
    tx["plan_path"].write_text(json.dumps(tx["plan"], indent=2, sort_keys=True) + "\n")


def test_preflight_and_apply_archive_only_target_failure(transaction):
    tx = transaction
    before_other = copy.deepcopy(tx["ledger"]["cells"]["other"])
    attestation, attestation_file_sha = attest(tx)

    result = reconciliation.apply(
        tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"],
    )

    ledger = json.loads(tx["ledger_path"].read_text())
    target = ledger["cells"]["agent-a"]
    assert result["status"] == "applied"
    assert target["status"] == "complete"
    assert target["attempts"][-1]["receipt"] == attestation["receipt"]
    assert "error" not in target["attempts"][-1]
    assert ledger["cells"]["other"] == before_other
    assert attestation["non_target_cells_sha256"] == {"other": document_sha(before_other)}
    archive = json.loads((tx["journal"] / "archive.json").read_text())
    assert archive["failure_attempt"] == tx["ledger"]["cells"]["agent-a"]["attempts"][-1]
    journal = json.loads((tx["journal"] / "journal.json").read_text())
    assert journal["phase"] == "applied"
    assert journal["archive_sha256"] == file_sha(tx["journal"] / "archive.json")


@pytest.mark.parametrize("fault", ["incomplete", "dirty", "reason", "ledger", "proof"])
def test_preflight_rejects_untrusted_inputs(transaction, fault):
    tx = transaction
    if fault == "incomplete":
        git(tx["repo"], "reset", "--hard", "HEAD^")
    elif fault == "dirty":
        (tx["repo"] / "candidate.py").write_text("DIRTY = True\n")
    elif fault == "reason":
        ledger = json.loads(tx["ledger_path"].read_text())
        ledger["cells"]["agent-a"]["attempts"][-1]["error"] = "network stopped"
        tx["ledger_path"].write_text(json.dumps(ledger, sort_keys=True) + "\n")
        tx["plan"]["ledger_sha256"] = file_sha(tx["ledger_path"])
        tx["plan"]["failure_attempt_sha256"] = document_sha(
            ledger["cells"]["agent-a"]["attempts"][-1])
        tx["plan"]["failure_reason"] = "network stopped"
        tx["plan_path"].write_text(json.dumps(tx["plan"], sort_keys=True) + "\n")
    elif fault == "ledger":
        tx["ledger_path"].write_text(tx["ledger_path"].read_text() + " ")
    else:
        Path(tx["proof"]["attestation_path"]).write_text("{}\n")

    with pytest.raises(reconciliation.ReconciliationError):
        reconciliation.preflight(tx["plan_path"], tx["attestation"])


@pytest.mark.parametrize("fault", ["run", "cell", "repo"])
def test_preflight_joins_repo_to_ledger_identity(transaction, fault):
    tx = transaction
    if fault == "run":
        ledger = copy.deepcopy(tx["ledger"])
        ledger["run_id"] = "another-run"
        repin(tx, ledger)
    elif fault == "cell":
        ledger = copy.deepcopy(tx["ledger"])
        ledger["cells"]["other-agent"] = ledger["cells"].pop("agent-a")
        ledger["order"] = ["other-agent" if item == "agent-a" else item
                           for item in ledger["order"]]
        tx["plan"]["cell_id"] = "other-agent"
        repin(tx, ledger)
    else:
        other_repo, _, _ = migrated_branch(tx["plan_path"].parent / "swapped", 3)
        git(other_repo, "branch", "-m", "experiment/another-run/agent-a")
        tx["plan"]["repo_path"] = str(other_repo.resolve())
        tx["plan_path"].write_text(json.dumps(tx["plan"], indent=2, sort_keys=True) + "\n")

    with pytest.raises(reconciliation.ReconciliationError, match="identity|branch"):
        reconciliation.preflight(tx["plan_path"], tx["attestation"])


def test_apply_rejects_ledger_attestation_and_receipt_drift(transaction, monkeypatch):
    tx = transaction
    attestation, attestation_file_sha = attest(tx)
    tx["attestation"].write_text(tx["attestation"].read_text() + " ")
    with pytest.raises(reconciliation.ReconciliationError, match="attestation drift"):
        reconciliation.apply(
            tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"])
    tx["attestation"].write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    ledger = json.loads(tx["ledger_path"].read_text())
    ledger["cells"]["other"]["new"] = "mutation"
    tx["ledger_path"].write_text(json.dumps(ledger, sort_keys=True) + "\n")
    with pytest.raises(reconciliation.ReconciliationError, match="ledger drift"):
        reconciliation.apply(
            tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"])

    tx["ledger_path"].write_text(json.dumps(tx["ledger"], indent=2, sort_keys=True) + "\n")
    original = reconciliation._verified_receipt
    def drifted_receipt(*args, **kwargs):
        verified, receipt = original(*args, **kwargs)
        return verified, {**receipt, "session_id": "different"}
    monkeypatch.setattr(reconciliation, "_verified_receipt", drifted_receipt)
    with pytest.raises(reconciliation.ReconciliationError, match="receipt drift"):
        reconciliation.apply(
            tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"])


@pytest.mark.parametrize("interrupt_after", ["archive", "ledger"])
def test_interruption_recovery_and_idempotent_reentry(transaction, interrupt_after):
    tx = transaction
    _, attestation_file_sha = attest(tx)
    with pytest.raises(reconciliation.InterruptedForTest):
        reconciliation.apply(
            tx["plan_path"], tx["attestation"], attestation_file_sha,
            tx["journal"], interrupt_after=interrupt_after,
        )

    recovered = reconciliation.apply(
        tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"],
    )
    again = reconciliation.apply(
        tx["plan_path"], tx["attestation"], attestation_file_sha, tx["journal"],
    )

    assert recovered["status"] == "applied"
    assert again == {**recovered, "status": "already_applied"}
    assert len(json.loads(tx["ledger_path"].read_text())
               ["cells"]["agent-a"]["attempts"]) == 1


def test_reentry_rejects_journal_drift(transaction):
    tx = transaction
    _, attestation_file_sha = attest(tx)
    with pytest.raises(reconciliation.InterruptedForTest):
        reconciliation.apply(tx["plan_path"], tx["attestation"], attestation_file_sha,
                             tx["journal"], interrupt_after="archive")
    journal_path = tx["journal"] / "journal.json"
    journal = json.loads(journal_path.read_text())
    journal["unexpected"] = True
    journal_path.write_text(json.dumps(journal, sort_keys=True) + "\n")
    with pytest.raises(reconciliation.ReconciliationError, match="journal drift"):
        reconciliation.apply(tx["plan_path"], tx["attestation"], attestation_file_sha,
                             tx["journal"])


def test_apply_cas_preserves_concurrent_ledger_update(transaction, monkeypatch):
    tx = transaction
    _, attestation_file_sha = attest(tx)

    def mutate(path):
        ledger = json.loads(path.read_text())
        ledger["cells"]["other"]["concurrent"] = "preserve-me"
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    monkeypatch.setattr(reconciliation, "_before_ledger_commit", mutate)
    with pytest.raises(reconciliation.ReconciliationError, match="changed before commit"):
        reconciliation.apply(tx["plan_path"], tx["attestation"], attestation_file_sha,
                             tx["journal"])

    ledger = json.loads(tx["ledger_path"].read_text())
    assert ledger["cells"]["other"]["concurrent"] == "preserve-me"
    assert ledger["cells"]["agent-a"]["status"] == "infrastructure_pending"


@pytest.mark.parametrize("output", ["repo", "plan", "ledger", "ledger-lock", "proof"])
def test_preflight_rejects_output_inside_repo_or_colliding_with_inputs(transaction, output):
    tx = transaction
    paths = {
        "repo": tx["repo"] / "attestation.json",
        "plan": tx["plan_path"],
        "ledger": tx["ledger_path"],
        "ledger-lock": tx["ledger_path"].with_name(f".{tx['ledger_path'].name}.lock"),
        "proof": Path(tx["proof"]["attestation_path"]),
    }
    with pytest.raises(reconciliation.ReconciliationError, match="output|collid"):
        reconciliation.preflight(tx["plan_path"], paths[output])


@pytest.mark.parametrize("journal_kind", ["inside-repo", "not-dedicated", "input-parent"])
def test_apply_requires_external_dedicated_journal(transaction, journal_kind):
    tx = transaction
    _, attestation_file_sha = attest(tx)
    if journal_kind == "inside-repo":
        journal = tx["repo"] / "journal"
    elif journal_kind == "input-parent":
        journal = tx["plan_path"].parent
    else:
        journal = tx["journal"]
        journal.mkdir()
        (journal / "unrelated.txt").write_text("do not overwrite\n")
    with pytest.raises(reconciliation.ReconciliationError, match="journal|external|dedicated"):
        reconciliation.apply(tx["plan_path"], tx["attestation"], attestation_file_sha,
                             journal)
