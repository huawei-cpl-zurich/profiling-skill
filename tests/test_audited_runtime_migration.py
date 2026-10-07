from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audited_runtime_migration", ROOT / "scripts" / "audited_runtime_migration.py"
)
migration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.strip()


def controller_identity(script: Path, config: Path, state: Path) -> dict:
    value = {
        "adapter": "CommandController",
        "argv": [sys.executable, str(script), "--config", str(config),
                 "--state-dir", str(state)],
        "file_arguments": [
            {"argument_index": 1, "path": str(script), "sha256": sha(script)},
            {"argument_index": 3, "path": str(config), "sha256": sha(config)},
        ],
        "mutable_directories": [{"argument_index": 5, "path": str(state)}],
    }
    value["identity_sha256"] = migration.document_sha256(value)
    return value


def make_runtime(tmp_path: Path, name: str) -> tuple[Path, dict]:
    runtime = tmp_path / name / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "audited_bz_controller.py").write_text(f"# {name}\n")
    (runtime / "batch_profile_a3.py").write_text(f"SELECTOR = {name!r}\n")
    document = {
        "schema": "profiling-skill/audited-campaign-runtime/v2",
        "runtime_scripts": {
            "path": str(runtime), "sha256": migration.digest_tree(runtime),
        },
    }
    config = tmp_path / name / "runtime.json"
    write_json(config, document)
    return config, document


def make_repo(root: Path, cell: str, experiment: int, old_runtime: Path,
              reason: str, candidate_hash: str, manifest_hash: str,
              handle: str) -> tuple[dict, dict, str]:
    repo = root / cell / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    candidate = b"candidate\n" + candidate_hash.encode() + b"\n"
    manifest = b'{"kernel_name":"kernel_mix_aiv"}\n'
    (repo / "candidate.py").write_bytes(candidate)
    (repo / "candidate.manifest.json").write_bytes(manifest)
    candidate_hash, manifest_hash = sha(repo / "candidate.py"), sha(repo / "candidate.manifest.json")
    old_controller_config = root / cell / "state" / "controller.json"
    write_json(old_controller_config, {"runtime": "old", "cell": cell})
    controller_dir = root / cell / "state" / "controller"
    controller_dir.mkdir(parents=True)
    old_identity = controller_identity(
        old_runtime / "audited_bz_controller.py", old_controller_config, controller_dir,
    )
    agent = {"adapter": "agent", "identity_sha256": "a" * 64}
    seed = {"reproducibility": {"agent": agent, "controller": old_identity}}
    write_json(repo / ".experiment" / "seed.json", seed)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "seed")
    seed_commit = git(repo, "rev-parse", "HEAD")
    for number in range(1, experiment):
        (repo / f"round-{number}").write_text("done\n")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", f"round {number}")
    parent = git(repo, "rev-parse", "HEAD")
    request = {
        "protocol_version": 1, "benchmark": "gdn" if cell.startswith("gdn") else "bsa",
        "device": 0, "action": "profile", "cases": [40 if cell.startswith("gdn") else 47],
        "repeats": 3, "round": experiment,
        "attempt_id": f"experiment-{experiment}-measurement-1-primary",
    }
    failed = {
        "request": request, "request_sha256": migration.document_sha256(request),
        "mode": "retry_submit", "status": "infrastructure_error", "terminal": False,
        "handle": handle, "failure_type": "profile_tool_error", "diagnostics": reason,
    }
    state = {
        "schema": "profiling-skill/audited-bz-controller-state/v1",
        "experiment": experiment, "candidate_sha256": candidate_hash,
        "manifest_sha256": manifest_hash, "stage": "profile",
        "measurement_generation": 1, "infrastructure_attempts": 2,
        "infrastructure_retries": 1, "operations_consumed": 4,
        "operations": [{"status": "ok", "request": {"action": "check"}}, failed],
        "pending": {"request": request, "handle": handle},
    }
    key = sha_bytes(f"{experiment}:{candidate_hash}:{manifest_hash}".encode())
    state_path = controller_dir / key / "state.json"
    write_json(state_path, state)
    blocked = {
        "schema": "profiling-skill/audited-blocked/v2", "round_count": 4,
        "experiment": experiment, "stage": "controller", "reason": reason,
        "branch": f"experiment/run/{cell}", "session_id": f"session-{cell}",
        "seed_commit": seed_commit, "resume_parent": parent,
        "candidate_sha256": candidate_hash, "manifest_sha256": manifest_hash,
        "controller_submissions": 2, "measurement_attempts": 0,
        "receipt": {"status": "infrastructure_error", "terminal": False,
                    "reason": reason, "handle": handle},
    }
    write_json(repo / ".experiment" / "blocked.json", blocked)
    git(repo, "add", ".experiment/blocked.json")
    git(repo, "commit", "-qm", f"checkpoint blocked experiment {experiment}")
    return blocked, state, git(repo, "rev-parse", "HEAD")


@pytest.fixture
def campaign(tmp_path: Path) -> dict:
    root = tmp_path / "production-v3"
    root.mkdir()
    old_config, old = make_runtime(tmp_path, "old")
    new_config, new = make_runtime(tmp_path, "new")
    specs = {
        "bsa-project-guarded": (3, 47),
        "gdn-cannbot": (1, 40),
        "bsa-project-cannbot": (1, 47),
    }
    plan_cells = []
    ledger_cells = {}
    for index, (cell, (experiment, case)) in enumerate(specs.items(), 1):
        reason = f"case-{case}-repeat-0 msprof evidence is invalid: 'kernels'"
        blocked, state, commit = make_repo(
            root, cell, experiment, Path(old["runtime_scripts"]["path"]), reason,
            str(index) * 64, str(index + 3) * 64, f"remote:bz-a3-1:job:{index}",
        )
        new_controller_config = root / cell / "state" / "controller-v4.json"
        write_json(new_controller_config, {"runtime": "new", "cell": cell})
        new_identity = controller_identity(
            Path(new["runtime_scripts"]["path"]) / "audited_bz_controller.py",
            new_controller_config, root / cell / "state" / "controller",
        )
        state_key = sha_bytes(
            f"{experiment}:{blocked['candidate_sha256']}:{blocked['manifest_sha256']}".encode()
        )
        state_path = root / cell / "state" / "controller" / state_key / "state.json"
        plan_cells.append({
            "cell_id": cell, "experiment": experiment,
            "blocked_sha256": sha(root / cell / "repo/.experiment/blocked.json"),
            "controller_state_sha256": sha(state_path),
            "checkpoint_commit": commit, "candidate_sha256": blocked["candidate_sha256"],
            "manifest_sha256": blocked["manifest_sha256"],
            "durable_handle": blocked["receipt"]["handle"],
            "request_sha256": state["operations"][-1]["request_sha256"],
            "failure_reason": reason,
            "old_controller_identity": json.loads(
                (root / cell / "repo/.experiment/seed.json").read_text()
            )["reproducibility"]["controller"],
            "new_controller_identity": new_identity,
        })
        ledger_cells[cell] = {"status": "infrastructure_pending", "attempts": [{
            "target": "bz-a3-1", "device": index, "status": "infrastructure_error",
            "durable_handle": blocked["receipt"]["handle"], "error": reason,
        }]}
    immutable = root / "matmul-project-cannbot"
    immutable.mkdir()
    (immutable / "result.json").write_text('{"status":"complete"}\n')
    ledger_cells[immutable.name] = {"status": "complete", "attempts": []}
    ledger = {"run_id": "run", "status": "infrastructure_pending",
              "cells": ledger_cells, "order": list(ledger_cells)}
    write_json(root / "ledger.json", ledger)
    plan = {
        "schema": migration.PLAN_SCHEMA, "migration_id": "selector-fallback-v4",
        "run_root": str(root), "ledger_sha256": sha(root / "ledger.json"),
        "old_runtime": {"config_path": str(old_config),
                        "config_sha256": sha(old_config),
                        "closure_sha256": old["runtime_scripts"]["sha256"]},
        "new_runtime": {"config_path": str(new_config),
                        "config_sha256": sha(new_config),
                        "closure_sha256": new["runtime_scripts"]["sha256"]},
        "cells": plan_cells,
    }
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    return {"root": root, "plan": plan_path, "document": plan,
            "immutable_sha": migration.digest_tree(immutable)}


def test_check_is_read_only_and_preserves_non_targets(campaign):
    checked = migration.run(campaign["plan"])
    assert checked["status"] == "ready"
    assert migration.digest_tree(campaign["root"] / "matmul-project-cannbot") == campaign["immutable_sha"]


@pytest.mark.parametrize("field", [
    "blocked_sha256", "controller_state_sha256", "candidate_sha256",
    "manifest_sha256", "durable_handle", "request_sha256", "failure_reason",
])
def test_check_rejects_each_fingerprint_mismatch(campaign, field):
    plan = copy.deepcopy(campaign["document"])
    plan["cells"][0][field] = "wrong"
    bad = campaign["plan"].with_name(f"bad-{field}.json")
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError, match=field.replace("_", " ") + "|fingerprint"):
        migration.run(bad)


def test_exact_allowlist_and_runtime_hashes_fail_closed(campaign):
    plan = copy.deepcopy(campaign["document"])
    plan["cells"].pop()
    bad = campaign["plan"].with_name("bad-allowlist.json")
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError, match="allowlist"):
        migration.run(bad)
    plan = copy.deepcopy(campaign["document"])
    plan["new_runtime"]["closure_sha256"] = "0" * 64
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError, match="runtime closure"):
        migration.run(bad)
