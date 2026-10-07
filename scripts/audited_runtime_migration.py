#!/usr/bin/env python3
"""Fail-closed preflight for the audited A3 selector-runtime migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path


PLAN_SCHEMA = "profiling-skill/audited-runtime-migration-plan/v1"
RECORD_SCHEMA = "profiling-skill/audited-runtime-migration/v1"
ALLOWED_CELLS = {
    "bsa-project-guarded": 3,
    "gdn-cannbot": 1,
    "bsa-project-cannbot": 1,
}
_FAILURE = re.compile(r"case-(?:40|47)-repeat-0 msprof evidence is invalid: 'kernels'")


class MigrationError(RuntimeError):
    pass


def document_sha256(document: dict) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


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


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise MigrationError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _identity(value: object, label: str) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("identity_sha256"), str):
        raise MigrationError(f"{label} controller identity is invalid")
    unsealed = {key: item for key, item in value.items() if key != "identity_sha256"}
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
    tree = Path(runtime.get("path", "")).resolve() if isinstance(runtime, dict) else Path("")
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


def _validate_controller_identity(identity: dict, runtime: Path, label: str) -> None:
    identity = _identity(identity, label)
    scripts = [item for item in identity.get("file_arguments", [])
               if isinstance(item, dict)
               and Path(item.get("path", "")).name == "audited_bz_controller.py"]
    expected = (runtime / "audited_bz_controller.py").resolve()
    if (len(scripts) != 1 or Path(scripts[0].get("path", "")).resolve() != expected
            or scripts[0].get("sha256") != file_sha256(expected)):
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
    if (not isinstance(cells, list)
            or {item.get("cell_id"): item.get("experiment")
                for item in cells if isinstance(item, dict)} != ALLOWED_CELLS
            or len(cells) != len(ALLOWED_CELLS)):
        raise MigrationError("migration cell allowlist must exactly match the approved cells")
    _, old_runtime = _runtime(plan.get("old_runtime"), "old")
    _, new_runtime = _runtime(plan.get("new_runtime"), "new")
    if old_runtime == new_runtime:
        raise MigrationError("old and new runtime closures must be distinct")
    return plan, root, old_runtime, new_runtime


def _validate_cell(root: Path, cell: dict, old_runtime: Path,
                   new_runtime: Path) -> None:
    cell_id, experiment = cell["cell_id"], cell["experiment"]
    repo = root / cell_id / "repo"
    blocked_path = repo / ".experiment" / "blocked.json"
    for name, path in (("candidate_sha256", repo / "candidate.py"),
                       ("manifest_sha256", repo / "candidate.manifest.json")):
        if not path.is_file() or cell.get(name) != file_sha256(path):
            raise MigrationError(f"{cell_id} {name.replace('_', ' ')} does not match")
    state_path = _state_path(root, cell)
    blocked, state = _json(blocked_path, f"{cell_id} blocked checkpoint"), _json(
        state_path, f"{cell_id} controller state",
    )
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
    if (reason != cell.get("failure_reason") or not isinstance(reason, str)
            or _FAILURE.fullmatch(reason) is None
            or not isinstance(receipt, dict) or receipt.get("status") != "infrastructure_error"
            or receipt.get("terminal") is not False
            or receipt.get("handle") != cell.get("durable_handle")
            or not isinstance(pending, dict)
            or pending.get("handle") != cell.get("durable_handle")
            or not isinstance(pending.get("request"), dict)
            or pending["request"].get("action") != "profile"
            or document_sha256(pending["request"]) != cell.get("request_sha256")
            or not isinstance(operations, list) or not operations):
        raise MigrationError(f"{cell_id} selector failure fingerprint does not match")
    operation = operations[-1]
    if (not isinstance(operation, dict)
            or operation.get("request_sha256") != cell.get("request_sha256")
            or operation.get("failure_type") != "profile_tool_error"
            or operation.get("diagnostics") != reason
            or operation.get("handle") != cell.get("durable_handle")):
        raise MigrationError(f"{cell_id} request sha256 fingerprint does not match")
    seed = _json(repo / ".experiment" / "seed.json", f"{cell_id} seed")
    old_identity = seed.get("reproducibility", {}).get("controller")
    if old_identity != cell.get("old_controller_identity"):
        raise MigrationError(f"{cell_id} old controller identity does not match seed")
    _validate_controller_identity(cell["old_controller_identity"], old_runtime, "old")
    _validate_controller_identity(cell["new_controller_identity"], new_runtime, "new")
def run(plan_path: Path) -> dict:
    plan_path = plan_path.resolve()
    plan, root, old_runtime, new_runtime = _load_plan(plan_path)
    ledger_path = root / "ledger.json"
    if file_sha256(ledger_path) != plan.get("ledger_sha256"):
        raise MigrationError("ledger sha256 does not match")
    for cell in plan["cells"]:
        _validate_cell(root, cell, old_runtime, new_runtime)
    return {"status": "ready", "cells": sorted(ALLOWED_CELLS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan)
    except MigrationError as error:
        parser.exit(2, f"migration rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
