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
    return subprocess.run(["git", *args], cwd=repo, text=True,
                          capture_output=True, check=True).stdout.strip()
def controller_identity(script: Path, config: Path, state: Path) -> dict:
    value = {
        "adapter": "CommandController",
        "argv": [sys.executable, str(script), "--config", str(config), "--state-dir", str(state)],
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
    document = {"schema": "profiling-skill/audited-campaign-runtime/v2",
                "runtime_scripts": {"path": str(runtime),
                                    "sha256": migration.digest_tree(runtime)}}
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
    git(repo, "switch", "-qc", f"experiment/run/{cell}")
    candidate = b"candidate\n" + candidate_hash.encode() + b"\n"
    manifest = b'{"kernel_name":"kernel_mix_aiv"}\n'
    (repo / "candidate.py").write_bytes(candidate)
    (repo / "candidate.manifest.json").write_bytes(manifest)
    candidate_hash, manifest_hash = sha(repo / "candidate.py"), sha(repo / "candidate.manifest.json")
    old_controller_config = root / cell / "state" / "controller.json"
    write_json(old_controller_config, {"runtime": "old", "cell": cell})
    controller_dir = root / cell / "state" / "controller"
    controller_dir.mkdir(parents=True)
    old_identity = controller_identity(old_runtime / "audited_bz_controller.py",
                                       old_controller_config, controller_dir)
    agent = {"adapter": "agent", "identity_sha256": "a" * 64}
    (repo / "PROMPT.md").write_text("prompt\n")
    (repo / "TASK.md").write_text("task\n")
    seed = {
        "schema": "profiling-skill/audited-seed/v2", "round_count": 4,
        "run_id": "run", "agent_id": cell,
        "prompt_sha256": sha(repo / "PROMPT.md"),
        "task_sha256": sha(repo / "TASK.md"),
        "reproducibility": {"agent": agent, "controller": old_identity},
    }
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
        "device": 0, "action": "profile",
        "cases": ([40, 49, 47, 46, 45] if cell.startswith("gdn")
                  else [47, 46, 49, 44, 43]),
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
        "pre_session": False, "commands": [], "seed_commit": seed_commit,
        "seed_hash": sha(repo / ".experiment/seed.json"),
        "resume_parent": parent,
        "prior_candidate_sha256": sha_bytes(subprocess.run(
            ["git", "show", f"{parent}:candidate.py"], cwd=repo,
            capture_output=True, check=True).stdout),
        "candidate_sha256": candidate_hash, "manifest_sha256": manifest_hash,
        "controller_submissions": 2, "measurement_attempts": 0,
        "receipt": {"status": "infrastructure_error", "terminal": False,
                    "reason": reason, "handle": handle,
                    "candidate_sha256": candidate_hash, "manifest_sha256": manifest_hash},
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
    specs = {"bsa-project-guarded": (3, 47), "gdn-cannbot": (1, 40),
             "bsa-project-cannbot": (1, 47)}
    plan_cells = []
    ledger_cells = {}
    for index, (cell, (experiment, case)) in enumerate(specs.items(), 1):
        reason = f"case-{case}-repeat-0 msprof evidence is invalid: 'kernels'"
        blocked, state, commit = make_repo(
            root, cell, experiment, Path(old["runtime_scripts"]["path"]), reason,
            str(index) * 64, str(index + 3) * 64, f"remote:bz-a3-1:job:{index}",
        )
        new_controller_config = root / cell / "state" / "controller.json"
        write_json(new_controller_config, {"runtime": "new", "cell": cell})
        new_identity = controller_identity(
            Path(new["runtime_scripts"]["path"]) / "audited_bz_controller.py",
            new_controller_config, root / cell / "state" / "controller")
        state_key = sha_bytes(f"{experiment}:{blocked['candidate_sha256']}:"
                              f"{blocked['manifest_sha256']}".encode())
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
            "old_controller_identity": json.loads((
                root / cell / "repo/.experiment/seed.json").read_text()
            )["reproducibility"]["controller"],
            "new_controller_identity": new_identity,
        })
        latest_attempt = {
            "target": "bz-a3-1", "device": index, "status": "infrastructure_error",
            "durable_handle": blocked["receipt"]["handle"], "error": reason,
        }
        plan_cells[-1]["ledger_attempt_sha256"] = migration.document_sha256(latest_attempt)
        ledger_cells[cell] = {"status": "infrastructure_pending", "attempts": [latest_attempt]}
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
        "old_runtime": {"config_path": str(old_config), "config_sha256": sha(old_config),
                        "closure_sha256": old["runtime_scripts"]["sha256"]},
        "new_runtime": {"config_path": str(new_config), "config_sha256": sha(new_config),
                        "closure_sha256": new["runtime_scripts"]["sha256"]},
        "cells": plan_cells,
    }
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    return {"root": root, "plan": plan_path, "document": plan,
            "immutable_sha": migration.digest_tree(immutable)}

def test_check_is_read_only_and_preserves_non_targets(campaign):
    attestation = campaign["plan"].with_name("attestation.json")
    checked = migration.run(campaign["plan"], attestation)
    assert checked["status"] == "ready"
    assert checked["attestation_sha256"] == sha(attestation)
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
        migration.run(bad, bad.with_suffix(".attestation"))

@pytest.mark.parametrize(("name", "mutation", "message"), [
    ("allowlist", lambda plan: plan["cells"].pop(), "allowlist"),
    ("closure", lambda plan: plan["new_runtime"].update(
        {"closure_sha256": "0" * 64}), "runtime closure"),
    ("noop", lambda plan: plan.update(
        {"new_runtime": copy.deepcopy(plan["old_runtime"])}), "distinct digests"),
    ("ledger", lambda plan: plan["cells"][0].update(
        {"ledger_attempt_sha256": "0" * 64}), "ledger attempt"),
])
def test_plan_safety_boundaries_fail_closed(campaign, name, mutation, message):
    plan = copy.deepcopy(campaign["document"])
    mutation(plan)
    bad = campaign["plan"].with_name(f"bad-{name}.json")
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError, match=message):
        migration.run(bad, bad.with_suffix(".attestation"))

@pytest.mark.parametrize("mutation", [
    lambda plan: plan.update({"run_root": []}),
    lambda plan: plan.update({"old_runtime": "bad"}),
    lambda plan: plan["cells"][0].pop("candidate_sha256"),
    lambda plan: plan["cells"].insert(0, None),
])
def test_malformed_plans_are_classified_without_tracebacks(campaign, mutation):
    plan = copy.deepcopy(campaign["document"])
    mutation(plan)
    bad = campaign["plan"].with_name("malformed.json")
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError):
        migration.run(bad, bad.with_suffix(".attestation"))

def test_controller_state_and_ledger_semantics_are_validated(campaign):
    cell = campaign["document"]["cells"][0]
    root = campaign["root"]
    key = sha_bytes(f"{cell['experiment']}:{cell['candidate_sha256']}:{cell['manifest_sha256']}".encode())
    state_path = root / cell["cell_id"] / "state/controller" / key / "state.json"
    state = json.loads(state_path.read_text())
    state["stage"] = "check"
    write_json(state_path, state)
    cell["controller_state_sha256"] = sha(state_path)
    write_json(campaign["plan"], campaign["document"])
    with pytest.raises(migration.MigrationError, match="controller state semantics"):
        migration.run(campaign["plan"], campaign["plan"].with_suffix(".attestation"))

def test_controller_identity_binds_cell_config_and_state_paths(campaign):
    plan = copy.deepcopy(campaign["document"])
    identity = plan["cells"][0]["new_controller_identity"]
    identity["argv"][3] += ".other"
    identity["identity_sha256"] = migration.document_sha256({
        key: value for key, value in identity.items() if key != "identity_sha256"
    })
    bad = campaign["plan"].with_name("bad-controller-path.json")
    write_json(bad, plan)
    with pytest.raises(migration.MigrationError, match="controller identity"):
        migration.run(bad, bad.with_suffix(".attestation"))

@pytest.mark.parametrize("defect", ["branch", "parent", "extra-commit"])
def test_checkpoint_topology_must_be_resumable(campaign, defect):
    cell = campaign["document"]["cells"][0]
    repo = campaign["root"] / cell["cell_id"] / "repo"
    path = repo / ".experiment/blocked.json"
    if defect == "extra-commit":
        git(repo, "reset", "--mixed", "HEAD^")
        (repo / "unexpected-round").write_text("extra\n")
        git(repo, "add", "unexpected-round")
        git(repo, "commit", "-qm", "unexpected round")
    blocked = json.loads(path.read_text())
    blocked["branch" if defect == "branch" else "resume_parent"] = (
        "experiment/run/wrong" if defect == "branch" else
        git(repo, "rev-parse", "HEAD" if defect == "extra-commit" else "HEAD~2")
    )
    write_json(path, blocked)
    git(repo, "add", ".experiment/blocked.json")
    git(repo, "commit", "-qm" if defect == "extra-commit" else "--amend",
        "checkpoint blocked experiment 3" if defect == "extra-commit" else "--no-edit")
    cell["blocked_sha256"] = sha(path)
    cell["checkpoint_commit"] = git(repo, "rev-parse", "HEAD")
    write_json(campaign["plan"], campaign["document"])
    with pytest.raises(migration.MigrationError, match="resume topology"):
        migration.run(campaign["plan"], campaign["plan"].with_suffix(".attestation"))
