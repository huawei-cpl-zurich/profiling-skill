#!/usr/bin/env python3
"""Fail-closed preflight for the audited A3 selector-runtime migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path


PLAN_SCHEMA = "profiling-skill/audited-runtime-migration-plan/v1"
RECORD_SCHEMA = "profiling-skill/audited-runtime-migration/v1"
ATTESTATION_SCHEMA = "profiling-skill/audited-runtime-preflight/v1"
ALLOWED_CELLS = {
    "bsa-project-guarded": 3,
    "gdn-cannbot": 1,
    "bsa-project-cannbot": 1,
}
_FAILURE = re.compile(r"case-(?:40|47)-repeat-0 msprof evidence is invalid: 'kernels'")


class MigrationError(RuntimeError):
    pass

def document_sha256(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_tree(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise MigrationError(f"tree is unavailable: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        mode = path.lstat().st_mode
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(mode)):
            raise MigrationError(f"tree contains unsupported entry: {relative}")
        if path.is_file():
            digest.update(relative.encode() + b"\0")
            digest.update(oct(stat.S_IMODE(mode)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise MigrationError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise MigrationError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_blob(repo: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=repo,
                            capture_output=True, check=False)
    if result.returncode:
        raise MigrationError(f"checkpoint commit does not contain {path}")
    return result.stdout


def _identity(value: object, label: str) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("identity_sha256"), str):
        raise MigrationError(f"{label} controller identity is invalid")
    unsealed = {key: item for key, item in value.items()
                if key != "identity_sha256"}
    if document_sha256(unsealed) != value["identity_sha256"]:
        raise MigrationError(f"{label} controller identity hash is invalid")
    return value


def _runtime(binding: object, label: str) -> tuple[dict, Path]:
    if not isinstance(binding, dict):
        raise MigrationError(f"{label} runtime binding is invalid")
    path = Path(binding.get("config_path", "")).resolve()
    expected = binding.get("config_sha256")
    if not path.is_file() or file_sha256(path) != expected:
        raise MigrationError(f"{label} runtime config hash does not match")
    config = _json(path, f"{label} runtime config")
    runtime = config.get("runtime_scripts")
    tree = (Path(runtime.get("path", "")).resolve()
            if isinstance(runtime, dict) else Path(""))
    closure = digest_tree(tree)
    if (runtime.get("sha256") != closure
            or binding.get("closure_sha256") != closure):
        raise MigrationError(f"{label} runtime closure hash does not match")
    return config, tree


def _state_path(root: Path, cell: dict) -> Path:
    key = hashlib.sha256(
        f"{cell['experiment']}:{cell['candidate_sha256']}:{cell['manifest_sha256']}".encode()
    ).hexdigest()
    return root / cell["cell_id"] / "state" / "controller" / key / "state.json"


def _validate_controller_identity(identity: dict, runtime: Path, root: Path,
                                  cell_id: str, label: str) -> None:
    identity = _identity(identity, label)
    script = (runtime / "audited_bz_controller.py").resolve()
    config = (root / cell_id / "state/controller.json").resolve()
    state = (root / cell_id / "state/controller").resolve()
    argv, files, mutable = (identity.get(name) for name in
                            ("argv", "file_arguments", "mutable_directories"))
    indexed = ({item.get("argument_index"): item for item in files
                if isinstance(item, dict)} if isinstance(files, list) else {})
    directories = ({item.get("argument_index"): item for item in mutable
                    if isinstance(item, dict)} if isinstance(mutable, list) else {})
    if (not isinstance(argv, list) or len(argv) != 6
            or not isinstance(argv[0], str) or not argv[0]
            or argv[1:] != [str(script), "--config", str(config), "--state-dir", str(state)]
            or len(files) != 2 or len(mutable) != 1
            or set(indexed) != {1, 3} or set(directories) != {5}
            or Path(indexed[1].get("path", "")).resolve() != script
            or indexed[1].get("sha256") != file_sha256(script)
            or Path(indexed[3].get("path", "")).resolve() != config
            or not re.fullmatch(r"[0-9a-f]{64}", str(indexed[3].get("sha256", "")))
            or Path(directories[5].get("path", "")).resolve() != state
            or (label == "new" and indexed[3]["sha256"] != file_sha256(config))):
        raise MigrationError(f"{label} controller identity does not match runtime")


def _load_plan(path: Path) -> tuple[dict, Path, Path, Path]:
    plan = _json(path.resolve(), "migration plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise MigrationError(f"migration plan requires schema {PLAN_SCHEMA}")
    migration_id = plan.get("migration_id")
    if (not isinstance(migration_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", migration_id) is None):
        raise MigrationError("migration id is invalid")
    root = Path(plan.get("run_root", "")).resolve()
    if not root.is_dir():
        raise MigrationError("run root is unavailable")
    cells = plan.get("cells")
    actual = ({item.get("cell_id"): item.get("experiment") for item in cells
               if isinstance(item, dict)} if isinstance(cells, list) else {})
    if (not isinstance(cells, list) or actual != ALLOWED_CELLS
            or len(cells) != len(ALLOWED_CELLS)):
        raise MigrationError("migration cell allowlist must exactly match the approved cells")
    _, old_runtime = _runtime(plan.get("old_runtime"), "old")
    _, new_runtime = _runtime(plan.get("new_runtime"), "new")
    old, new = plan["old_runtime"], plan["new_runtime"]
    if (old.get("config_sha256") == new.get("config_sha256")
            or old.get("closure_sha256") == new.get("closure_sha256")):
        raise MigrationError("old and new runtime config/closure require distinct digests")
    return plan, root, old_runtime, new_runtime


def _validate_resume_topology(repo: Path, ledger: dict, cell: dict,
                              blocked: dict) -> None:
    cell_id, experiment = cell["cell_id"], cell["experiment"]
    expected_branch = f"experiment/{ledger.get('run_id')}/{cell_id}"
    required = ("reason", "seed_commit", "seed_hash", "resume_parent", "prior_candidate_sha256", "session_id")
    if (blocked.get("schema") != "profiling-skill/audited-blocked/v2"
            or blocked.get("round_count") != 4
            or blocked.get("experiment") != experiment
            or blocked.get("stage") != "controller"
            or blocked.get("branch") != expected_branch
            or blocked.get("pre_session") is not False
            or not isinstance(blocked.get("commands"), list)
            or any(not isinstance(blocked.get(name), str) or not blocked[name]
                   for name in required)
            or type(blocked.get("controller_submissions")) is not int
            or blocked["controller_submissions"] < 0
            or type(blocked.get("measurement_attempts")) is not int
            or blocked["measurement_attempts"] < 0
            or _git(repo, "branch", "--show-current") != expected_branch
            or _git(repo, "rev-parse", "HEAD") != cell["checkpoint_commit"]
            or _git(repo, "rev-parse", "HEAD^") != blocked["resume_parent"]):
        raise MigrationError(f"{cell_id} checkpoint resume topology does not match")
    revision_range = f"{blocked['seed_commit']}..HEAD^"
    completed = _git(repo, "rev-list", "--first-parent", "--reverse",
                     revision_range).splitlines()
    all_commits = _git(repo, "rev-list", revision_range).splitlines()
    try:
        seed_bytes = _git_blob(repo, blocked["seed_commit"], ".experiment/seed.json")
        seed = json.loads(seed_bytes)
    except json.JSONDecodeError as error:
        raise MigrationError(f"{cell_id} experiment seed is invalid") from error
    if (len(completed) != experiment - 1 or len(all_commits) != len(completed)
            or hashlib.sha256(seed_bytes).hexdigest() != blocked["seed_hash"]
            or hashlib.sha256(_git_blob(repo, "HEAD^", "candidate.py")).hexdigest() != blocked["prior_candidate_sha256"]
            or seed.get("schema") != "profiling-skill/audited-seed/v2"
            or seed.get("round_count") != 4 or seed.get("run_id") != ledger.get("run_id")
            or seed.get("agent_id") != cell_id
            or hashlib.sha256((repo / "PROMPT.md").read_bytes()).hexdigest()
            != seed.get("prompt_sha256")
            or hashlib.sha256((repo / "TASK.md").read_bytes()).hexdigest()
            != seed.get("task_sha256")):
        raise MigrationError(f"{cell_id} checkpoint resume topology does not match")


def _validate_cell(root: Path, ledger: dict, cell: dict, old_runtime: Path,
                   new_runtime: Path) -> dict:
    cell_id, experiment = cell["cell_id"], cell["experiment"]
    repo = root / cell_id / "repo"
    blocked_path = repo / ".experiment" / "blocked.json"
    for name, path in (("candidate_sha256", repo / "candidate.py"),
                       ("manifest_sha256", repo / "candidate.manifest.json")):
        if not path.is_file() or cell.get(name) != file_sha256(path):
            raise MigrationError(f"{cell_id} {name.replace('_', ' ')} does not match")
    state_path = _state_path(root, cell)
    blocked = _json(blocked_path, f"{cell_id} blocked checkpoint")
    state = _json(state_path, f"{cell_id} controller state")
    _validate_resume_topology(repo, ledger, cell, blocked)
    checks = {
        "blocked_sha256": file_sha256(blocked_path),
        "controller_state_sha256": file_sha256(state_path),
        "checkpoint_commit": _git(repo, "rev-parse", "HEAD"),
        "candidate_sha256": file_sha256(repo / "candidate.py"),
        "manifest_sha256": file_sha256(repo / "candidate.manifest.json"),
    }
    for name, actual in checks.items():
        if cell.get(name) != actual:
            raise MigrationError(f"{cell_id} {name.replace('_', ' ')} does not match")
    if (_git(repo, "status", "--porcelain") or blocked.get("experiment") != experiment
            or blocked.get("stage") != "controller"
            or blocked.get("candidate_sha256") != cell["candidate_sha256"]
            or blocked.get("manifest_sha256") != cell["manifest_sha256"]):
        raise MigrationError(f"{cell_id} checkpoint fingerprint does not match")
    reason, receipt = blocked.get("reason"), blocked.get("receipt")
    pending, operations = state.get("pending"), state.get("operations")
    expected_cases = ([40, 49, 47, 46, 45] if cell_id == "gdn-cannbot"
                      else [47, 46, 49, 44, 43])
    request = pending.get("request") if isinstance(pending, dict) else None
    generation = state.get("measurement_generation")
    expected_attempt = f"experiment-{experiment}-measurement-{generation}-primary"
    if (state.get("schema") != "profiling-skill/audited-bz-controller-state/v1"
            or state.get("experiment") != experiment
            or state.get("candidate_sha256") != cell["candidate_sha256"]
            or state.get("manifest_sha256") != cell["manifest_sha256"]
            or state.get("stage") != "profile" or type(generation) is not int
            or generation < 0 or not isinstance(request, dict)
            or request.get("protocol_version") != 1
            or request.get("benchmark") != ("gdn" if cell_id == "gdn-cannbot" else "bsa")
            or request.get("action") != "profile" or request.get("round") != experiment
            or request.get("cases") != expected_cases or request.get("repeats") != 3
            or request.get("attempt_id") != expected_attempt
            or type(request.get("device")) is not int or request["device"] < 0):
        raise MigrationError(f"{cell_id} controller state semantics do not match")
    if (reason != cell.get("failure_reason") or not isinstance(reason, str)
            or _FAILURE.fullmatch(reason) is None
            or not isinstance(receipt, dict) or receipt.get("status") != "infrastructure_error"
            or receipt.get("terminal") is not False
            or receipt.get("handle") != cell.get("durable_handle")
            or any(name in receipt and receipt[name] != cell[name]
                   for name in ("candidate_sha256", "manifest_sha256"))
            or not isinstance(pending, dict)
            or pending.get("handle") != cell.get("durable_handle")
            or receipt.get("reason") != reason
            or document_sha256(request) != cell.get("request_sha256")
            or not isinstance(operations, list) or not operations):
        raise MigrationError(f"{cell_id} selector failure fingerprint does not match")
    operation = operations[-1]
    if (not isinstance(operation, dict)
            or operation.get("request_sha256") != cell.get("request_sha256")
            or operation.get("failure_type") != "profile_tool_error"
            or operation.get("diagnostics") != reason
            or operation.get("handle") != cell.get("durable_handle")
            or operation.get("request") != request
            or operation.get("status") != "infrastructure_error"
            or operation.get("terminal") is not False
            or operation.get("mode") not in {"submit", "retry_submit", "observe"}):
        raise MigrationError(f"{cell_id} request sha256 fingerprint does not match")
    ledger_state = ledger.get("cells", {}).get(cell_id)
    attempts = ledger_state.get("attempts") if isinstance(ledger_state, dict) else None
    latest = attempts[-1] if isinstance(attempts, list) and attempts else None
    if (not isinstance(ledger_state, dict)
            or ledger_state.get("status") != "infrastructure_pending"
            or not isinstance(latest, dict)
            or latest.get("status") != "infrastructure_error"
            or latest.get("durable_handle") != cell["durable_handle"]
            or not isinstance(latest.get("error"), str) or not latest["error"]
            or document_sha256(latest) != cell.get("ledger_attempt_sha256")):
        raise MigrationError(f"{cell_id} ledger attempt does not match checkpoint")
    seed = _json(repo / ".experiment" / "seed.json", f"{cell_id} seed")
    old_identity = seed.get("reproducibility", {}).get("controller")
    if old_identity != cell.get("old_controller_identity"):
        raise MigrationError(f"{cell_id} old controller identity does not match seed")
    _validate_controller_identity(cell["old_controller_identity"], old_runtime,
                                  root, cell_id, "old")
    _validate_controller_identity(cell["new_controller_identity"], new_runtime,
                                  root, cell_id, "new")
    return {key: cell[key] for key in (
        "cell_id", "experiment", "checkpoint_commit", "candidate_sha256",
        "manifest_sha256", "blocked_sha256", "controller_state_sha256",
        "ledger_attempt_sha256", "durable_handle", "request_sha256",
        "failure_reason", "old_controller_identity", "new_controller_identity",
    )} | {key: blocked[key] for key in (
        "branch", "session_id", "seed_commit", "seed_hash", "resume_parent",
        "prior_candidate_sha256",
    )}


def _run(plan_path: Path, attestation_path: Path) -> dict:
    plan_path = plan_path.resolve()
    plan, root, old_runtime, new_runtime = _load_plan(plan_path)
    if attestation_path.resolve().is_relative_to(root):
        raise MigrationError("attestation output must be outside the campaign run root")
    ledger_path = root / "ledger.json"
    if file_sha256(ledger_path) != plan.get("ledger_sha256"):
        raise MigrationError("ledger sha256 does not match")
    ledger = _json(ledger_path, "campaign ledger")
    cells = [_validate_cell(root, ledger, cell, old_runtime, new_runtime)
             for cell in plan["cells"]]
    runtime_bindings = {}
    for label, tree in (("old", old_runtime), ("new", new_runtime)):
        binding = plan[f"{label}_runtime"]
        runtime_bindings[label] = {
            **binding, "config_path": str(Path(binding["config_path"]).resolve()),
            "runtime_path": str(tree.resolve()),
        }
    attestation = {
        "schema": ATTESTATION_SCHEMA, "migration_id": plan["migration_id"],
        "plan_sha256": file_sha256(plan_path),
        "ledger_sha256": plan["ledger_sha256"], "runtimes": runtime_bindings,
        "cells": cells,
    }
    attestation["attestation_sha256"] = document_sha256(attestation)
    _atomic_json(attestation_path.resolve(), attestation)
    return {"status": "ready", "cells": sorted(ALLOWED_CELLS),
            "attestation": str(attestation_path.resolve()),
            "attestation_sha256": file_sha256(attestation_path.resolve())}


def run(plan_path: Path, attestation_path: Path) -> dict:
    try:
        return _run(plan_path, attestation_path)
    except MigrationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        raise MigrationError(f"migration input is malformed: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan, args.attestation)
    except MigrationError as error:
        parser.exit(2, f"migration rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
