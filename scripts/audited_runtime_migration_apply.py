#!/usr/bin/env python3
"""Apply a validated audited-runtime migration as a recoverable transaction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import audited_runtime_migration as preflight
from audited_lifecycle import MIGRATION_RETRY_REASON


JOURNAL_SCHEMA = "profiling-skill/audited-runtime-migration-journal/v1"
TRUST_SCHEMA = "profiling-skill/audited-runtime-migration-trust/v1"
EVIDENCE_SCHEMA = "profiling-skill/excluded-harness-operation/v1"


class ApplyError(RuntimeError):
    pass


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ApplyError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ApplyError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise ApplyError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _trust(attestation_path: Path, trusted: dict) -> tuple[dict, bytes]:
    try:
        raw = attestation_path.resolve().read_bytes()
        attestation = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ApplyError("trusted attestation is missing or unreadable") from error
    if not isinstance(attestation, dict):
        raise ApplyError("trusted attestation must be a JSON object")
    seal = attestation.get("attestation_sha256")
    unsealed = {key: value for key, value in attestation.items()
                if key != "attestation_sha256"}
    expected = {
        "schema": TRUST_SCHEMA,
        "attestation_path": str(attestation_path.resolve()),
        "attestation_file_sha256": preflight.file_sha256(attestation_path.resolve()),
        "attestation_sha256": seal,
    }
    if (trusted != expected or attestation.get("schema") != preflight.ATTESTATION_SCHEMA
            or preflight.document_sha256(unsealed) != seal):
        raise ApplyError("trusted attestation binding does not match")
    return attestation, raw


def _archive(root: Path, cells: list[dict], ledger_sha256: str,
             transaction: Path) -> dict:
    archive = transaction / "archive"
    manifest_path = archive / "manifest.json"
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        archived_files = manifest.get("files")
        if not isinstance(archived_files, dict):
            raise ApplyError("migration archive manifest is malformed")
        for relative, digest in archived_files.items():
            path = archive / relative
            if not path.is_file() or preflight.file_sha256(path) != digest:
                raise ApplyError("migration archive integrity check failed")
        return manifest
    ledger_path = root / "ledger.json"
    if preflight.file_sha256(ledger_path) != ledger_sha256:
        raise ApplyError("campaign ledger drifted before archival")
    files = {"ledger.json": ledger_path.read_bytes()}
    commits = {}
    for cell in cells:
        cell_id = cell["cell_id"]
        repo = root / cell_id / "repo"
        blocked = repo / ".experiment/blocked.json"
        state = preflight._state_path(root, cell)
        if (preflight.file_sha256(blocked) != cell["blocked_sha256"]
                or preflight.file_sha256(state) != cell["controller_state_sha256"]
                or _git(repo, "rev-parse", "HEAD") != cell["checkpoint_commit"]):
            raise ApplyError(f"{cell_id} drifted before archival")
        files[f"{cell_id}/blocked.json"] = blocked.read_bytes()
        files[f"{cell_id}/controller-state.json"] = state.read_bytes()
        files[f"{cell_id}/checkpoint.commit"] = (
            _git(repo, "cat-file", "commit", cell["checkpoint_commit"]) + "\n"
        ).encode()
        commits[cell_id] = {
            "commit": cell["checkpoint_commit"],
            "parent": _git(repo, "rev-parse", f"{cell['checkpoint_commit']}^"),
            "candidate_sha256": cell["candidate_sha256"],
            "manifest_sha256": cell["manifest_sha256"],
        }
    for relative, data in files.items():
        _atomic(archive / relative, data)
    manifest = {
        "schema": "profiling-skill/audited-runtime-migration-archive/v1",
        "files": {name: preflight.file_sha256(archive / name) for name in files},
        "commits": commits,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _controller_after(original: dict, migration_id: str) -> dict:
    updated = json.loads(json.dumps(original))
    pending, operations = updated.get("pending"), updated.get("operations")
    if not isinstance(pending, dict) or not isinstance(operations, list) or not operations:
        raise ApplyError("controller state lacks the attested pending operation")
    operation = operations[-1]
    if (not isinstance(operation, dict) or operation.get("request") != pending.get("request")
            or operation.get("handle") != pending.get("handle")):
        raise ApplyError("controller pending operation does not match its ledger")
    updated["operations"] = operations[:-1]
    updated["pending"] = None
    updated["measurement_generation"] += 1
    updated["infrastructure_attempts"] = 0
    updated["infrastructure_retries"] = 0
    evidence = updated.setdefault("excluded_harness_evidence", [])
    if not isinstance(evidence, list):
        raise ApplyError("excluded harness evidence is malformed")
    evidence.append({"schema": EVIDENCE_SCHEMA, "migration_id": migration_id,
                     "pending": pending, "operation": operation})
    return updated


def _blocked_after(original: dict, cell: dict, attestation: dict,
                   attestation_path: Path, attestation_raw: bytes) -> dict:
    updated = json.loads(json.dumps(original))
    updated.update({
        "reason": MIGRATION_RETRY_REASON,
        "receipt": {}, "controller_submissions": 0, "measurement_attempts": 0,
        "runtime_migration": {
            "schema": "profiling-skill/audited-runtime-migration-citation/v1",
            "attestation_path": str(attestation_path.resolve()),
            "attestation_file_sha256": preflight.file_sha256(attestation_path.resolve()),
            "attestation_sha256": attestation["attestation_sha256"],
            "plan_sha256": attestation["plan_sha256"],
            "cell_id": cell["cell_id"], "experiment": cell["experiment"],
            "checkpoint_commit": cell["checkpoint_commit"],
            "original_blocked_json": attestation_raw.decode(),
            "old_controller_identity": cell["old_controller_identity"],
            "new_controller_identity": cell["new_controller_identity"],
        },
    })
    return updated


def _write_if_original(path: Path, original: dict, updated: dict, label: str) -> None:
    current = _json(path)
    if current == updated:
        return
    if current != original:
        raise ApplyError(f"{label} drifted from the archived state")
    _atomic_json(path, updated)


def _step(journal_path: Path, journal: dict, name: str, mutation,
          interrupt_after: str | None) -> None:
    if name in journal["completed"]:
        mutation()
        return
    journal["active"] = {"phase": name, "status": "before"}
    journal["events"].append(journal["active"])
    _atomic_json(journal_path, journal)
    mutation()
    if interrupt_after == name:
        raise ApplyError(f"injected interruption after {name}")
    journal["completed"].append(name)
    journal["active"] = {"phase": name, "status": "after"}
    journal["events"].append(journal["active"])
    _atomic_json(journal_path, journal)


def apply(plan_path: Path, attestation_path: Path, trusted: dict,
          transaction: Path, *, interrupt_after: str | None = None) -> dict:
    plan_path, attestation_path = plan_path.resolve(), attestation_path.resolve()
    transaction, journal_path = transaction.resolve(), transaction.resolve() / "journal.json"
    attestation, attestation_raw = _trust(attestation_path, trusted)
    plan = _json(plan_path)
    identity = {"plan_sha256": preflight.file_sha256(plan_path),
                "attestation_file_sha256": preflight.file_sha256(attestation_path),
                "trusted": trusted}
    if identity["plan_sha256"] != attestation.get("plan_sha256"):
        raise ApplyError("plan does not match the trusted attestation")
    if journal_path.is_file():
        journal = _json(journal_path)
        if journal.get("schema") != JOURNAL_SCHEMA or journal.get("identity") != identity:
            raise ApplyError("transaction journal belongs to different inputs")
    else:
        if transaction.exists() and any(transaction.iterdir()):
            raise ApplyError("transaction directory is not empty")
        probe = transaction / ".preflight.json"
        try:
            preflight.run(plan_path, probe)
            if probe.read_bytes() != attestation_raw:
                raise ApplyError("live preflight does not reproduce the trusted attestation")
        finally:
            if probe.is_file():
                probe.unlink()
        journal = {"schema": JOURNAL_SCHEMA, "identity": identity,
                   "completed": [], "events": [], "active": None,
                   "status": "applying"}
        _atomic_json(journal_path, journal)
    root, cells = Path(plan["run_root"]).resolve(), plan["cells"]
    archive_manifest = None

    def archive_phase() -> None:
        nonlocal archive_manifest
        archive_manifest = _archive(root, cells, plan["ledger_sha256"], transaction)
        journal["archive_manifest_sha256"] = preflight.file_sha256(
            transaction / "archive/manifest.json"
        )

    _step(journal_path, journal, "archive", archive_phase, interrupt_after)
    archive_manifest = archive_manifest or _archive(
        root, cells, plan["ledger_sha256"], transaction,
    )
    archive = transaction / "archive"
    attested = {cell["cell_id"]: cell for cell in attestation["cells"]}
    for cell in cells:
        cell_id = cell["cell_id"]
        attested_cell = attested.get(cell_id)
        if (not isinstance(attested_cell, dict)
                or any(attested_cell.get(key) != value for key, value in cell.items())
                or cell_id not in preflight.ALLOWED_CELLS):
            raise ApplyError("attestation contains an unapproved migration cell")
        state_path = preflight._state_path(root, cell)
        original_state = _json(archive / cell_id / "controller-state.json")
        updated_state = _controller_after(original_state, plan["migration_id"])
        _step(journal_path, journal, f"controller:{cell_id}",
              lambda p=state_path, old=original_state, new=updated_state, name=cell_id:
              _write_if_original(p, old, new, f"{name} controller state"), interrupt_after)

        repo = root / cell_id / "repo"
        blocked_path = repo / ".experiment/blocked.json"
        original_bytes = (archive / cell_id / "blocked.json").read_bytes()
        original_blocked = json.loads(original_bytes)
        updated_blocked = _blocked_after(
            original_blocked, attested_cell, attestation, attestation_path, original_bytes,
        )

        def amend(repo=repo, path=blocked_path, old=original_blocked,
                  new=updated_blocked, metadata=archive_manifest["commits"][cell_id],
                  name=cell_id) -> None:
            current = _json(path)
            checkpoint_phase = f"checkpoint:{name}"
            original_head = _git(repo, "rev-parse", "HEAD") == metadata["commit"]
            unstaged = set(_git(repo, "diff", "--name-only").splitlines())
            staged = set(_git(repo, "diff", "--cached", "--name-only").splitlines())
            blocked_only = {".experiment/blocked.json"}
            if current == new:
                if original_head and (unstaged, staged) == (blocked_only, set()):
                    pass
                elif original_head and (unstaged, staged) == (set(), blocked_only):
                    pass
                elif not original_head and not unstaged and not staged:
                    if (_git(repo, "rev-parse", "HEAD^") != metadata["parent"]
                            or preflight.file_sha256(repo / "candidate.py")
                            != metadata["candidate_sha256"]
                            or preflight.file_sha256(repo / "candidate.manifest.json")
                            != metadata["manifest_sha256"]):
                        raise ApplyError(f"{name} amended checkpoint drifted")
                    return
                else:
                    raise ApplyError(f"{name} partial checkpoint amendment drifted")
            elif (current == old and original_head and not unstaged and not staged):
                _atomic_json(path, new)
                if interrupt_after == f"{checkpoint_phase}:write":
                    raise ApplyError(f"injected interruption after {checkpoint_phase}:write")
            else:
                raise ApplyError(f"{name} checkpoint drifted from the archive")
            if not staged:
                _git(repo, "add", ".experiment/blocked.json")
                if interrupt_after == f"{checkpoint_phase}:add":
                    raise ApplyError(f"injected interruption after {checkpoint_phase}:add")
            _git(repo, "commit", "--amend", "--no-edit", "--no-verify")
            if interrupt_after == f"{checkpoint_phase}:commit":
                raise ApplyError(f"injected interruption after {checkpoint_phase}:commit")
            if (_git(repo, "rev-parse", "HEAD^") != metadata["parent"]
                    or preflight.file_sha256(repo / "candidate.py") != metadata["candidate_sha256"]
                    or preflight.file_sha256(repo / "candidate.manifest.json")
                    != metadata["manifest_sha256"]
                    or _git(repo, "status", "--porcelain")):
                raise ApplyError(f"{name} checkpoint amendment changed immutable history")

        _step(journal_path, journal, f"checkpoint:{cell_id}", amend, interrupt_after)

    original_ledger = _json(archive / "ledger.json")
    updated_ledger = json.loads(json.dumps(original_ledger))
    for cell in cells:
        latest = updated_ledger["cells"][cell["cell_id"]]["attempts"][-1]
        latest.pop("durable_handle", None)
        latest["retryable"] = True
    _step(journal_path, journal, "ledger",
          lambda: _write_if_original(root / "ledger.json", original_ledger,
                                     updated_ledger, "campaign ledger"), interrupt_after)
    journal["status"], journal["active"] = "complete", None
    _atomic_json(journal_path, journal)
    return {"status": "complete", "migration_id": plan["migration_id"],
            "journal": str(journal_path), "archive_manifest": archive_manifest,
            "trusted_runtime_migration": trusted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--attestation-file-sha256", required=True)
    parser.add_argument("--attestation-sha256", required=True)
    parser.add_argument("--transaction", type=Path, required=True)
    args = parser.parse_args(argv)
    trusted = {"schema": TRUST_SCHEMA, "attestation_path": str(args.attestation.resolve()),
               "attestation_file_sha256": args.attestation_file_sha256,
               "attestation_sha256": args.attestation_sha256}
    try:
        result = apply(args.plan, args.attestation, trusted, args.transaction)
    except (ApplyError, preflight.MigrationError) as error:
        parser.exit(2, f"migration apply rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
