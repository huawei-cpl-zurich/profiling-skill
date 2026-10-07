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
sys.path.insert(0, str(ROOT / "scripts"))
import audited_lifecycle as lifecycle  # noqa: E402
import audited_campaign_production as production  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "audited_runtime_migration", ROOT / "scripts" / "audited_runtime_migration.py"
)
migration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)
APPLY_SPEC = importlib.util.spec_from_file_location(
    "audited_runtime_migration_apply",
    ROOT / "scripts" / "audited_runtime_migration_apply.py",
)
migration_apply = importlib.util.module_from_spec(APPLY_SPEC)
assert APPLY_SPEC.loader
sys.modules[APPLY_SPEC.name] = migration_apply
APPLY_SPEC.loader.exec_module(migration_apply)

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


def test_check_accepts_production_receipt_without_redundant_artifact_hashes(campaign):
    cell = campaign["document"]["cells"][0]
    blocked_path = campaign["root"] / cell["cell_id"] / "repo/.experiment/blocked.json"
    blocked = json.loads(blocked_path.read_text())
    blocked["receipt"].pop("candidate_sha256")
    blocked["receipt"].pop("manifest_sha256")
    write_json(blocked_path, blocked)
    git(campaign["root"] / cell["cell_id"] / "repo", "add", ".experiment/blocked.json")
    git(campaign["root"] / cell["cell_id"] / "repo", "commit", "--amend", "--no-edit", "-q")
    cell["checkpoint_commit"] = git(
        campaign["root"] / cell["cell_id"] / "repo", "rev-parse", "HEAD")
    cell["blocked_sha256"] = sha(blocked_path)
    write_json(campaign["plan"], campaign["document"])

    checked = migration.run(campaign["plan"], campaign["plan"].with_name("attestation.json"))

    assert checked["status"] == "ready"


def test_preflight_chains_from_authenticated_current_controller(campaign, tmp_path: Path):
    first_path = tmp_path / "first-attestation.json"
    first_result = migration.run(campaign["plan"], first_path)
    first = json.loads(first_path.read_text())
    first_cells = {cell["cell_id"]: cell for cell in first["cells"]}
    prior_trust = {
        "schema": migration_apply.TRUST_SCHEMA,
        "attestation_path": str(first_path.resolve()),
        "attestation_file_sha256": first_result["attestation_sha256"],
        "attestation_sha256": first["attestation_sha256"],
    }
    third_config, third = make_runtime(tmp_path, "third")
    plan = campaign["document"]
    plan["migration_id"] = "selector-fallback-v5"
    plan["prior_migrations"] = [prior_trust]
    plan["old_runtime"] = plan["new_runtime"]
    plan["new_runtime"] = {
        "config_path": str(third_config), "config_sha256": sha(third_config),
        "closure_sha256": third["runtime_scripts"]["sha256"],
    }
    for cell in plan["cells"]:
        repo = campaign["root"] / cell["cell_id"] / "repo"
        blocked_path = repo / ".experiment/blocked.json"
        blocked = json.loads(blocked_path.read_text())
        attested = first_cells[cell["cell_id"]]
        blocked["runtime_migration"] = {
            "schema": "profiling-skill/audited-runtime-migration-citation/v1",
            "attestation_path": str(first_path.resolve()),
            "attestation_file_sha256": sha(first_path),
            "attestation_sha256": first["attestation_sha256"],
            "plan_sha256": first["plan_sha256"],
            "cell_id": cell["cell_id"], "experiment": cell["experiment"],
            "checkpoint_commit": attested["checkpoint_commit"],
            "original_blocked_json": blocked_path.read_text(),
            "old_controller_identity": attested["old_controller_identity"],
            "new_controller_identity": attested["new_controller_identity"],
        }
        write_json(blocked_path, blocked)
        git(repo, "add", ".experiment/blocked.json")
        git(repo, "commit", "--amend", "--no-edit", "-q")
        cell["checkpoint_commit"] = git(repo, "rev-parse", "HEAD")
        cell["blocked_sha256"] = sha(blocked_path)
        controller_config = campaign["root"] / cell["cell_id"] / "state/controller.json"
        cell["old_controller_identity"] = attested["new_controller_identity"]
        cell["new_controller_identity"] = controller_identity(
            Path(third["runtime_scripts"]["path"]) / "audited_bz_controller.py",
            controller_config, campaign["root"] / cell["cell_id"] / "state/controller",
        )
    write_json(campaign["plan"], plan)

    second_path = tmp_path / "second-attestation.json"
    result = migration.run(campaign["plan"], second_path)

    assert result["status"] == "ready"
    second = json.loads(second_path.read_text())
    second_trust = {
        "schema": migration_apply.TRUST_SCHEMA,
        "attestation_path": str(second_path.resolve()),
        "attestation_file_sha256": result["attestation_sha256"],
        "attestation_sha256": second["attestation_sha256"],
    }
    applied = migration_apply.apply(
        campaign["plan"], second_path, second_trust, tmp_path / "second-transaction",
    )
    assert applied["trusted_runtime_migrations"] == [prior_trust, second_trust]


@pytest.mark.parametrize(("field", "value"), [
    ("candidate_sha256", "0" * 64),
    ("manifest_sha256", "0" * 64),
    ("candidate_sha256", None),
    ("manifest_sha256", None),
])
def test_check_rejects_present_conflicting_optional_receipt_hash(campaign, field, value):
    cell = campaign["document"]["cells"][0]
    blocked_path = campaign["root"] / cell["cell_id"] / "repo/.experiment/blocked.json"
    blocked = json.loads(blocked_path.read_text())
    blocked["receipt"][field] = value
    write_json(blocked_path, blocked)
    git(campaign["root"] / cell["cell_id"] / "repo", "add", ".experiment/blocked.json")
    git(campaign["root"] / cell["cell_id"] / "repo", "commit", "--amend", "--no-edit", "-q")
    cell["checkpoint_commit"] = git(
        campaign["root"] / cell["cell_id"] / "repo", "rev-parse", "HEAD")
    cell["blocked_sha256"] = sha(blocked_path)
    write_json(campaign["plan"], campaign["document"])

    with pytest.raises(migration.MigrationError, match="selector failure fingerprint"):
        migration.run(campaign["plan"], campaign["plan"].with_name("attestation.json"))


@pytest.mark.parametrize("missing", ["candidate_sha256", "manifest_sha256"])
def test_check_accepts_each_receipt_hash_independently_optional(campaign, missing):
    cell = campaign["document"]["cells"][0]
    blocked_path = campaign["root"] / cell["cell_id"] / "repo/.experiment/blocked.json"
    blocked = json.loads(blocked_path.read_text())
    blocked["receipt"].pop(missing)
    write_json(blocked_path, blocked)
    git(campaign["root"] / cell["cell_id"] / "repo", "add", ".experiment/blocked.json")
    git(campaign["root"] / cell["cell_id"] / "repo", "commit", "--amend", "--no-edit", "-q")
    cell["checkpoint_commit"] = git(
        campaign["root"] / cell["cell_id"] / "repo", "rev-parse", "HEAD")
    cell["blocked_sha256"] = sha(blocked_path)
    write_json(campaign["plan"], campaign["document"])

    assert migration.run(
        campaign["plan"], campaign["plan"].with_name("attestation.json"))["status"] == "ready"

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


class Component:
    def __init__(self, identity):
        self.identity = identity

    def reproducibility_metadata(self):
        return self.identity


def migrated_runner(campaign):
    attestation_path = campaign["plan"].with_name("attestation.json")
    migration.run(campaign["plan"], attestation_path)
    attestation = json.loads(attestation_path.read_text())
    cell = campaign["document"]["cells"][0]
    repo = campaign["root"] / cell["cell_id"] / "repo"
    blocked_path = repo / ".experiment/blocked.json"
    blocked = json.loads(blocked_path.read_text())
    original_blocked_json = blocked_path.read_text()
    blocked["runtime_migration"] = {
        "schema": "profiling-skill/audited-runtime-migration-citation/v1",
        "attestation_path": str(attestation_path),
        "attestation_file_sha256": sha(attestation_path),
        "attestation_sha256": attestation["attestation_sha256"],
        "plan_sha256": attestation["plan_sha256"],
        "cell_id": cell["cell_id"], "experiment": cell["experiment"],
        "checkpoint_commit": cell["checkpoint_commit"],
        "original_blocked_json": original_blocked_json,
        "old_controller_identity": cell["old_controller_identity"],
        "new_controller_identity": cell["new_controller_identity"],
    }
    write_json(blocked_path, blocked)
    git(repo, "add", ".experiment/blocked.json")
    git(repo, "commit", "--amend", "--no-edit")
    seed = json.loads((repo / ".experiment/seed.json").read_text())
    runner = lifecycle.AuditedExperimentRunner(
        repo, repo / "PROMPT.md", repo / "TASK.md",
        Component(seed["reproducibility"]["agent"]),
        Component(cell["new_controller_identity"]), round_count=4,
        trusted_runtime_migration={
            "schema": "profiling-skill/audited-runtime-migration-trust/v1",
            "attestation_path": str(attestation_path),
            "attestation_file_sha256": sha(attestation_path),
            "attestation_sha256": attestation["attestation_sha256"],
        },
    )
    return runner, repo, blocked, attestation_path, cell


def test_lifecycle_accepts_real_preflight_attestation(campaign):
    runner, _, _, _, cell = migrated_runner(campaign)

    resumed = runner._resume("run", cell["cell_id"])

    assert resumed["runtime_migration"]["checkpoint_commit"] == cell["checkpoint_commit"]


def migrated_recheckpoint(campaign):
    runner, repo, _, _, cell = migrated_runner(campaign)
    state = runner._resume("run", cell["cell_id"])
    runner._checkpoint(
        state["experiment"], state["session_id"], "retry failed", state["stage"],
        state["branch"], state["seed_commit"], state["seed_hash"],
        state["prior_candidate_sha256"], tuple(state["commands"]), state["receipt"],
        controller_submissions=state["controller_submissions"],
        measurement_attempts=state["measurement_attempts"],
    )
    second = lifecycle.AuditedExperimentRunner(
        repo, repo / "PROMPT.md", repo / "TASK.md", runner.invoke, runner.controller,
        round_count=4, trusted_runtime_migration=runner.trusted_runtime_migration,
    )
    return second, repo, cell


def test_migrated_resume_preserves_provenance_across_another_checkpoint(campaign):
    second, _, cell = migrated_recheckpoint(campaign)

    resumed = second._resume("run", cell["cell_id"])

    assert resumed["runtime_migration"]["continuation"]["state_sha256"]


def test_lifecycle_authenticates_ordered_sequential_migrations(campaign, tmp_path: Path):
    runner, repo, _, first_path, first_cell = migrated_runner(campaign)
    original = json.loads((repo / ".experiment/blocked.json").read_text())
    original["experiment"] = 4
    original_raw = json.dumps(original, indent=2, sort_keys=True) + "\n"
    third_config, third = make_runtime(tmp_path, "third-lifecycle")
    controller_config = campaign["root"] / first_cell["cell_id"] / "state/controller.json"
    third_identity = controller_identity(
        Path(third["runtime_scripts"]["path"]) / "audited_bz_controller.py",
        controller_config, campaign["root"] / first_cell["cell_id"] / "state/controller",
    )
    first_attestation = json.loads(first_path.read_text())
    second = {
        "schema": migration.ATTESTATION_SCHEMA, "migration_id": "runtime-v5",
        "plan_sha256": "7" * 64,
        "ledger_sha256": first_attestation["ledger_sha256"],
        "runtimes": {
            "old": first_attestation["runtimes"]["new"],
            "new": {
                "config_path": str(third_config.resolve()),
                "config_sha256": sha(third_config),
                "closure_sha256": third["runtime_scripts"]["sha256"],
                "runtime_path": third["runtime_scripts"]["path"],
            },
        },
        "cells": [{
            **first_cell, "experiment": 4, "branch": original["branch"],
            "seed_commit": original["seed_commit"],
            "blocked_sha256": sha_bytes(original_raw.encode()),
            "old_controller_identity": first_cell["new_controller_identity"],
            "new_controller_identity": third_identity,
        }],
    }
    second["attestation_sha256"] = migration.document_sha256(second)
    second_path = tmp_path / "second-lifecycle-attestation.json"
    write_json(second_path, second)
    second_trust = {
        "schema": migration_apply.TRUST_SCHEMA,
        "attestation_path": str(second_path.resolve()),
        "attestation_file_sha256": sha(second_path),
        "attestation_sha256": second["attestation_sha256"],
    }
    state = json.loads(original_raw)
    state["runtime_migration"] = {
        "schema": "profiling-skill/audited-runtime-migration-citation/v1",
        "attestation_path": str(second_path.resolve()),
        "attestation_file_sha256": sha(second_path),
        "attestation_sha256": second["attestation_sha256"],
        "plan_sha256": second["plan_sha256"], "cell_id": first_cell["cell_id"],
        "experiment": 4, "checkpoint_commit": first_cell["checkpoint_commit"],
        "original_blocked_json": original_raw,
        "old_controller_identity": first_cell["new_controller_identity"],
        "new_controller_identity": third_identity,
    }
    seed = json.loads((repo / ".experiment/seed.json").read_text())

    citation = lifecycle._validate_runtime_migration(
        repo, state, seed,
        {"agent": runner.reproducibility["agent"], "controller": third_identity},
        first_cell["cell_id"], [runner.trusted_runtime_migration, second_trust],
    )

    assert citation["attestation_sha256"] == second["attestation_sha256"]


def test_migrated_resume_rejects_tampered_carried_provenance(campaign):
    second, repo, cell = migrated_recheckpoint(campaign)
    path = repo / ".experiment/blocked.json"
    blocked = json.loads(path.read_text())
    blocked["runtime_migration"]["continuation"]["state_sha256"] = "0" * 64
    write_json(path, blocked)
    git(repo, "add", ".experiment/blocked.json")
    git(repo, "commit", "--amend", "--no-edit")

    with pytest.raises(lifecycle.AuditError, match="continuation"):
        second._resume("run", cell["cell_id"])


@pytest.mark.parametrize("defect", [
    "dummy", "untrusted", "missing", "tampered", "cell", "old-identity", "new-identity",
    "agent", "config", "closure", "original-blocked",
])
def test_lifecycle_rejects_untrusted_runtime_migrations(campaign, defect):
    runner, repo, blocked, attestation_path, cell = migrated_runner(campaign)
    citation = blocked["runtime_migration"]
    if defect == "dummy":
        citation["attestation_sha256"] = "0" * 64
    elif defect == "untrusted":
        runner.trusted_runtime_migration = None
    elif defect == "missing":
        attestation_path.unlink()
    elif defect == "tampered":
        attestation_path.write_text(attestation_path.read_text() + " ")
    elif defect == "cell":
        citation["cell_id"] = "gdn-cannbot"
    elif defect.endswith("identity"):
        citation[f"{defect.split('-')[0]}_controller_identity"] = {
            "identity_sha256": "0" * 64,
        }
    elif defect == "agent":
        runner.invoke.identity = {"identity_sha256": "0" * 64}
    elif defect == "config":
        Path(campaign["document"]["new_runtime"]["config_path"]).write_text("{}\n")
    elif defect == "original-blocked":
        citation["original_blocked_json"] = "{}\n"
    else:
        runtime = json.loads(Path(campaign["document"]["new_runtime"]["config_path"]).read_text())
        (Path(runtime["runtime_scripts"]["path"]) / "batch_profile_a3.py").write_text("changed\n")
    if defect in {"dummy", "cell", "old-identity", "new-identity", "original-blocked"}:
        write_json(repo / ".experiment/blocked.json", blocked)
        git(repo, "add", ".experiment/blocked.json")
        git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(lifecycle.AuditError, match="runtime migration|agent identity"):
        runner._resume("run", cell["cell_id"])


def apply_inputs(campaign):
    attestation_path = campaign["plan"].with_name("apply-attestation.json")
    checked = migration.run(campaign["plan"], attestation_path)
    attestation = json.loads(attestation_path.read_text())
    trusted = {
        "schema": migration_apply.TRUST_SCHEMA,
        "attestation_path": str(attestation_path),
        "attestation_file_sha256": checked["attestation_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
    }
    return attestation_path, trusted, campaign["plan"].with_name("transaction")


def test_apply_archives_and_repairs_only_attested_cells(campaign):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    root = campaign["root"]
    immutable_before = migration.digest_tree(root / "matmul-project-cannbot")
    cells_before = {}
    for cell in campaign["document"]["cells"]:
        repo = root / cell["cell_id"] / "repo"
        cells_before[cell["cell_id"]] = {
            "parent": git(repo, "rev-parse", "HEAD^"),
            "candidate": sha(repo / "candidate.py"),
            "manifest": sha(repo / "candidate.manifest.json"),
        }

    result = migration_apply.apply(campaign["plan"], attestation_path,
                                   trusted, transaction)
    repeated = migration_apply.apply(campaign["plan"], attestation_path,
                                     trusted, transaction)

    assert result["status"] == repeated["status"] == "complete"
    assert migration.digest_tree(root / "matmul-project-cannbot") == immutable_before
    journal = json.loads((transaction / "journal.json").read_text())
    assert journal["status"] == "complete" and len(journal["completed"]) == 8
    assert len(journal["events"]) == 16
    archive = transaction / "archive"
    manifest = json.loads((archive / "manifest.json").read_text())
    assert all(sha(archive / path) == digest for path, digest in manifest["files"].items())
    ledger = json.loads((root / "ledger.json").read_text())
    for cell in campaign["document"]["cells"]:
        cell_id = cell["cell_id"]
        repo = root / cell_id / "repo"
        blocked = json.loads((repo / ".experiment/blocked.json").read_text())
        state_path = migration._state_path(root, cell)
        state = json.loads(state_path.read_text())
        original = json.loads((archive / cell_id / "controller-state.json").read_text())
        assert state["pending"] is None and state["measurement_generation"] == 2
        assert state["operations"] == original["operations"][:-1]
        assert state["excluded_harness_evidence"][-1]["operation"] == original["operations"][-1]
        assert blocked["receipt"] == {} and blocked["controller_submissions"] == 0
        assert blocked["runtime_migration"]["original_blocked_json"] == (
            archive / cell_id / "blocked.json").read_text()
        assert git(repo, "rev-parse", "HEAD^") == cells_before[cell_id]["parent"]
        assert sha(repo / "candidate.py") == cells_before[cell_id]["candidate"]
        assert sha(repo / "candidate.manifest.json") == cells_before[cell_id]["manifest"]
        latest = ledger["cells"][cell_id]["attempts"][-1]
        assert latest["retryable"] is True and "durable_handle" not in latest
    first = campaign["document"]["cells"][0]
    repo = root / first["cell_id"] / "repo"
    seed = json.loads((repo / ".experiment/seed.json").read_text())
    runner = lifecycle.AuditedExperimentRunner(
        repo, repo / "PROMPT.md", repo / "TASK.md",
        Component(seed["reproducibility"]["agent"]),
        Component(first["new_controller_identity"]), round_count=4,
        trusted_runtime_migration=result["trusted_runtime_migration"],
    )
    resumed = runner._resume("run", first["cell_id"])
    assert resumed["receipt"] == {} and resumed["controller_submissions"] == 0


class RetryableLaunchFailure(RuntimeError):
    def __init__(self, message, durable_handle=None):
        super().__init__(message)
        self.durable_handle = durable_handle


class ResumeProbe(lifecycle.AuditedExperimentRunner):
    resumed = []

    def run(self, run_id, agent_id, *, resume=False):
        assert resume is True
        type(self).resumed.append(self._resume(run_id, agent_id))


class ProductionInvoker(Component):
    def __init__(self, identity, image):
        super().__init__(identity)
        self.docker_image_id = image

    def scrub_auth(self):
        pass


def applied_production_launcher(campaign, tmp_path: Path, monkeypatch):
    cell_plan = campaign["document"]["cells"][0]
    repo = campaign["root"] / cell_plan["cell_id"] / "repo"
    runtime = json.loads(Path(
        campaign["document"]["new_runtime"]["config_path"]
    ).read_text())["runtime_scripts"]
    config = {
        "schema": production.RUNTIME_SCHEMA,
        "run_id": "run", "run_root": str(campaign["root"]),
        "runtime_scripts": runtime,
        "provenance": {
            "controller_sha256": campaign["document"]["old_runtime"]["closure_sha256"],
        },
        "runtime_mode": "docker", "runtime_image_digest": "sha256:" + "e" * 64,
        "agent_turn_timeout": 20, "controller_transaction_timeout": 15,
        "verifier_timeout": 20, "backend_job_timeout": 10, "timeout_grace": 2,
        "auth_home": str(tmp_path / "auth"), "model": "model",
        "reasoning_effort": "low",
        "prompt": {"path": str(repo / "PROMPT.md"),
                   "sha256": sha(repo / "PROMPT.md")},
        "tasks": {"bsa": {"path": str(repo / "TASK.md"),
                            "sha256": sha(repo / "TASK.md")}},
    }
    remote_closure_sha256 = "f" * 64
    config["cpl_remote_closure_sha256"] = remote_closure_sha256
    config_path = tmp_path / "active-runtime-v4.json"
    write_json(config_path, config)
    campaign["document"]["new_runtime"].update({
        "config_path": str(config_path), "config_sha256": sha(config_path),
        "closure_sha256": runtime["sha256"],
    })
    write_json(campaign["plan"], campaign["document"])
    attestation_path, trusted, transaction = apply_inputs(campaign)
    result = migration_apply.apply(
        campaign["plan"], attestation_path, trusted, transaction,
    )
    seed = json.loads((repo / ".experiment/seed.json").read_text())
    monkeypatch.setattr(
        production, "cpl_remote_closure_sha256",
        lambda path: remote_closure_sha256,
    )
    monkeypatch.setattr(production.ProductionCellLauncher, "_validate_files", lambda self: None)
    launcher = production.ProductionCellLauncher(
        config, infrastructure_failure_type=RetryableLaunchFailure,
        trusted_runtime_migration=result["trusted_runtime_migration"],
        runtime_config_path=config_path, runtime_config_sha256=sha(config_path),
        invoker_factory=lambda *args, **kwargs: ProductionInvoker(
            seed["reproducibility"]["agent"], config["runtime_image_digest"],
        ),
        controller_factory=lambda *args, **kwargs: None,
        runner_factory=ResumeProbe, audit_error_type=lifecycle.AuditError,
    )
    monkeypatch.setattr(launcher, "_prepare_repo", lambda cell: (repo, True))
    monkeypatch.setattr(launcher, "_controller", lambda *args: Component(
        cell_plan["new_controller_identity"]
    ))
    monkeypatch.setattr(launcher, "_verify", lambda *args: {})
    monkeypatch.setattr(launcher, "_receipt", lambda *args: {"status": "complete"})
    cell = {
        "cell_id": cell_plan["cell_id"], "task": "bsa",
        "treatment": "project-guarded", "round_count": 4, "request_budget": 24,
        "skills": list(production.TREATMENT_SKILLS["project-guarded"]),
        "task_sha256": sha(repo / "TASK.md"),
        "prompt_contract": {
            "task_sha256": sha(repo / "TASK.md"),
            "invariant_sha256": sha(repo / "PROMPT.md"),
        },
    }
    return launcher, cell, repo


def test_production_launcher_resumes_applied_migrated_checkpoint(
        campaign, tmp_path: Path, monkeypatch):
    launcher, cell, _ = applied_production_launcher(campaign, tmp_path, monkeypatch)
    ResumeProbe.resumed.clear()

    receipt = launcher.launch(cell, {"target": "bz-a3-1", "device": 0})

    assert receipt == {"status": "complete"}
    assert ResumeProbe.resumed[-1]["runtime_migration"]["cell_id"] == cell["cell_id"]


def test_production_launcher_classifies_migration_mismatch_as_deterministic(
        campaign, tmp_path: Path, monkeypatch):
    launcher, cell, repo = applied_production_launcher(campaign, tmp_path, monkeypatch)
    blocked_path = repo / ".experiment/blocked.json"
    blocked = json.loads(blocked_path.read_text())
    blocked["runtime_migration"]["cell_id"] = "gdn-cannbot"
    write_json(blocked_path, blocked)
    git(repo, "add", ".experiment/blocked.json")
    git(repo, "commit", "--amend", "--no-edit")

    with pytest.raises(production.ProductionError, match="runtime migration"):
        launcher.launch(cell, {"target": "bz-a3-1", "device": 0})


@pytest.mark.parametrize("phase", [
    "archive", "controller:bsa-project-guarded", "checkpoint:gdn-cannbot", "ledger",
])
def test_apply_recovers_interrupted_phases(campaign, phase):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    with pytest.raises(migration_apply.ApplyError, match="injected interruption"):
        migration_apply.apply(campaign["plan"], attestation_path, trusted,
                              transaction, interrupt_after=phase)

    result = migration_apply.apply(campaign["plan"], attestation_path,
                                   trusted, transaction)

    assert result["status"] == "complete"
    assert json.loads((transaction / "journal.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("point", ["write", "add", "commit"])
def test_apply_recovers_inside_checkpoint_amend(campaign, point):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    phase = f"checkpoint:bsa-project-guarded:{point}"
    with pytest.raises(migration_apply.ApplyError, match="injected interruption"):
        migration_apply.apply(campaign["plan"], attestation_path, trusted,
                              transaction, interrupt_after=phase)

    result = migration_apply.apply(campaign["plan"], attestation_path,
                                   trusted, transaction)

    assert result["status"] == "complete"


def test_apply_rejects_clean_candidate_drift_after_amend(campaign):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    phase = "checkpoint:bsa-project-guarded:commit"
    with pytest.raises(migration_apply.ApplyError, match="injected interruption"):
        migration_apply.apply(campaign["plan"], attestation_path, trusted,
                              transaction, interrupt_after=phase)
    repo = campaign["root"] / "bsa-project-guarded/repo"
    (repo / "candidate.py").write_text("clean committed drift\n")
    git(repo, "add", "candidate.py")
    git(repo, "commit", "--amend", "--no-edit")

    with pytest.raises(migration_apply.ApplyError, match="amended checkpoint drifted"):
        migration_apply.apply(campaign["plan"], attestation_path, trusted, transaction)


def test_apply_rejects_post_attestation_drift(campaign):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    cell = campaign["document"]["cells"][0]
    (campaign["root"] / cell["cell_id"] / "repo/candidate.py").write_text("drift\n")

    with pytest.raises(migration.MigrationError):
        migration_apply.apply(campaign["plan"], attestation_path, trusted, transaction)


@pytest.mark.parametrize("mode", ["missing", "unreadable", "malformed", "nonobject"])
def test_apply_normalizes_unreadable_attestations(campaign, mode):
    attestation_path, trusted, transaction = apply_inputs(campaign)
    attestation_path.unlink()
    if mode == "unreadable":
        attestation_path.mkdir()
    elif mode != "missing":
        attestation_path.write_text("{bad" if mode == "malformed" else "[]")

    with pytest.raises(migration_apply.ApplyError, match="attestation"):
        migration_apply.apply(campaign["plan"], attestation_path, trusted, transaction)


def test_apply_cli_reports_missing_attestation_without_traceback(campaign):
    missing = campaign["plan"].with_name("missing-attestation.json")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/audited_runtime_migration_apply.py"),
        "--plan", str(campaign["plan"]), "--attestation", str(missing),
        "--attestation-file-sha256", "0" * 64,
        "--attestation-sha256", "0" * 64,
        "--transaction", str(campaign["plan"].with_name("cli-transaction")),
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "migration apply rejected" in result.stderr and "Traceback" not in result.stderr
