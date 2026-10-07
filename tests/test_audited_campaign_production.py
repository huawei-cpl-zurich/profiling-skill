from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audited_campaign_production", ROOT / "scripts" / "audited_campaign_production.py"
)
production = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = production
SPEC.loader.exec_module(production)
CAMPAIGN_SPEC = importlib.util.spec_from_file_location(
    "campaign_for_production_test", ROOT / "scripts" / "audited_campaign.py"
)
campaign = importlib.util.module_from_spec(CAMPAIGN_SPEC)
assert CAMPAIGN_SPEC.loader
sys.modules[CAMPAIGN_SPEC.name] = campaign
CAMPAIGN_SPEC.loader.exec_module(campaign)
sys.modules.setdefault("audited_campaign", campaign)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import audited_runtime
    import audited_campaign as runtime_campaign
    import audited_lifecycle
finally:
    sys.path.pop(0)


def valid_verifier(*args, **kwargs):
    del kwargs
    command = args[0]
    repo = Path(command[2])
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    return SimpleNamespace(returncode=0, stdout=json.dumps({
        "status": "valid", "branch": branch,
        "seed_commit": "seed", "session_id": "thread",
        "experiments": [{"commit": str(number)} for number in range(1, 5)],
    }), stderr="")


@pytest.fixture(autouse=True)
def isolated_global_remote(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    remote = home / ".agents/skills/remote-access/scripts/cpl-remote"
    remote.parent.mkdir(parents=True)
    remote.write_text("#!/bin/sh\n")
    remote.chmod(0o755)
    remote.with_name("cpl_remote.py").write_text("# pinned implementation\n")
    monkeypatch.setattr(Path, "home", lambda: home)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_baseline(task: str) -> dict:
    document = {
        "schema": "profiling-skill/baseline-timing/v1",
        "benchmark": task,
        "case_medians_us": [
            {"case": case, "median_us": float(index + 10)}
            for index, case in enumerate(production.DEVELOPMENT_CASES[task])
        ],
        "control_median_us": 20.0,
    }
    document["sha256"] = production.document_sha256(document)
    return document


def git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path,
                   check=True)
    (path / "candidate.py").write_text("VALUE = 1\n")
    (path / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"kernel"}\n'
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                              text=True, capture_output=True).stdout.strip()
    return path, revision


def runtime_fixture(tmp_path: Path, cell: dict):
    repo, revision = git_repo(tmp_path / "source")
    skills = {}
    for name in {skill for names in production.TREATMENT_SKILLS.values() for skill in names}:
        skill = tmp_path / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(name)
        skills[name] = {"path": str(skill), "sha256": production.digest_tree(skill)}
    scripts = tmp_path / "runtime"
    scripts.mkdir()
    for name in ("audited_bz_controller.py", "benchmark_backend.py",
                 "bz_a3_job_client.py",
                 "validate_audited_experiment.py", "audited_verifier.py",
                 "audited_contract.py", "audited_lifecycle.py", "audited_runtime.py"):
        (scripts / name).write_text("# pinned\n")
    assets = tmp_path / "benchmarks"
    assets.mkdir()
    baseline_root = tmp_path / "timing-baselines"
    baseline_root.mkdir()
    baselines = {}
    starters = {}
    for task_name in production.DEVELOPMENT_CASES:
        (assets / task_name).mkdir()
        (assets / task_name / "cases.json").write_text(json.dumps({"task": task_name}))
        baseline = baseline_root / f"{task_name}.json"
        baseline.write_text(json.dumps(timing_baseline(task_name)))
        baselines[task_name] = {"path": str(baseline), "sha256": sha(baseline)}
        starter = tmp_path / "starters" / task_name
        starter.mkdir(parents=True)
        candidate = starter / "candidate.py"
        manifest = starter / "candidate.manifest.json"
        candidate.write_text(f"TASK = {task_name!r}\nVALUE = 0\n")
        manifest.write_text(json.dumps({
            "schema": "profiling-skill/candidate-kernel/v1",
            "kernel_name": task_name,
        }) + "\n")
        starters[task_name] = {
            "candidate": {"path": str(candidate.resolve()), "sha256": sha(candidate)},
            "manifest": {"path": str(manifest.resolve()), "sha256": sha(manifest)},
        }
    cpl_remote = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
    prompt = tmp_path / "prompt.md"
    task = tmp_path / "task.md"
    prompt.write_text("prompt")
    task.write_text("task")
    cell["task_sha256"] = sha(task)
    cell["prompt_contract"] = {
        "invariant_sha256": sha(prompt), "task_sha256": sha(task),
    }
    runtime_digest = production.digest_tree(scripts)
    cannbot_names = sorted({
        name for names in production.TREATMENT_SKILLS.values() for name in names
        if name not in {"ascend-profiling", "triton-guarded-kernel"}
    })
    provenance = {
        "source_revision": revision,
        "controller_sha256": runtime_digest,
        "runtime_image_digest": "sha256:" + "b" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {name: binding["sha256"] for name, binding in baselines.items()},
        "starters": starters,
        "skills": {
            "cannbot": production.digest_skill_bundle(skills, cannbot_names),
            **({"ascend-profiling": skills["ascend-profiling"]["sha256"]}
               if "ascend-profiling" in skills else {}),
            **({"triton-guarded-kernel": skills["triton-guarded-kernel"]["sha256"]}
               if "triton-guarded-kernel" in skills else {}),
        },
    }
    resource_module = ROOT / "scripts" / "audited_resource_admission.py"
    provenance["resource_admission_sha256"] = sha(resource_module)
    provenance["cpl_remote_closure_sha256"] = (
        production.cpl_remote_closure_sha256(cpl_remote)
    )
    provenance["admission_provider_id"] = "campaign-operator"
    provenance["admission_allowlist_sha256"] = "d" * 64
    config = {
        "schema": production.RUNTIME_SCHEMA,
        "run_id": "production-e2e",
        "run_root": str(tmp_path / "runs"),
        "source_repositories": {
            name: {"path": str(repo), "revision": revision}
            for name in production.DEVELOPMENT_CASES
        },
        "skill_sources": skills,
        "runtime_scripts": {"path": str(scripts), "sha256": runtime_digest},
        "benchmark_assets": {"path": str(assets),
                             "sha256": production.digest_tree(assets)},
        "baseline_sources": baselines,
        "starter_sources": starters,
        "admission_provider_id": "campaign-operator",
        "admission_allowlist_sha256": "d" * 64,
        "cpl_remote_closure_sha256": production.cpl_remote_closure_sha256(cpl_remote),
        "resource_admission": {
            "path": str(resource_module), "sha256": sha(resource_module),
        },
        "prompt": {"path": str(prompt), "sha256": sha(prompt)},
        "tasks": {name: {"path": str(task), "sha256": sha(task)}
                  for name in production.DEVELOPMENT_CASES},
        "remote_root": "/remote/campaign",
        "auth_home": str(tmp_path / "auth"),
        "model": "gpt-5.6-sol", "reasoning_effort": "low",
        "runtime_mode": "docker", "runtime_image_digest": provenance["runtime_image_digest"],
        "agent_turn_timeout": 20,
        "controller_transaction_timeout": 15,
        "verifier_timeout": 20,
        "backend_job_timeout": 10,
        "timeout_grace": 2,
        "provenance": provenance,
    }
    return config


def production_launcher(config: dict, **kwargs):
    return production.ProductionCellLauncher(
        config,
        infrastructure_failure_type=runtime_campaign.InfrastructureFailure,
        **kwargs,
    )


def test_production_runtime_uses_starter_bound_v2_schema(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    assert production.RUNTIME_SCHEMA == "profiling-skill/audited-campaign-runtime/v2"
    config["schema"] = "profiling-skill/audited-campaign-runtime/v1"
    with pytest.raises(production.ProductionError, match="runtime config requires schema"):
        production_launcher(config)


def canary_gate_fixture(tmp_path: Path, config: dict) -> tuple[Path, str, Path, str]:
    definition = {
        "schema": "profiling-skill/audited-repair-canaries/v1",
        "benchmark": "matmul", "request_budget": 48,
        "max_candidate_repairs_per_round": 2,
        "placement": "dynamic-bz-a3-admission",
        "canaries": [
            {"id": "matmul-cannbot-repair", "treatment": "cannbot",
             "required_evidence": ["candidate-repair", "offline-verifier",
                                   "msprof-op-timing"]},
            {"id": "matmul-project-cannbot-resume", "treatment": "project-cannbot",
             "required_evidence": ["checkpoint-resume", "same-session",
                                   "offline-verifier", "msprof-op-timing"]},
            {"id": "matmul-project-guarded-repair", "treatment": "project-guarded",
             "required_evidence": ["candidate-repair", "offline-verifier",
                                   "msprof-op-timing"]},
        ],
        "gate": {"all_canaries_terminal_ok": True,
                 "all_branches_offline_valid": True,
                 "all_final_timings_positive": True,
                 "minimum_repaired_canaries": 2,
                 "resume_canary_required": True},
    }
    definition_path = tmp_path / "canaries.json"
    definition_path.write_text(json.dumps(definition, sort_keys=True) + "\n")
    definition_sha = sha(definition_path)
    records = []
    for index, item in enumerate(definition["canaries"]):
        branch = f"experiment/canary/{item['id']}"
        commits = [f"{index + 1:x}" * 40, f"{index + 4:x}" * 40,
                   f"{index + 7:x}" * 40, f"{index + 10:x}" * 40]
        verifier = {
            "status": "valid", "branch": branch, "session_id": f"thread-{index}",
            "experiments": [{"commit": commit} for commit in commits],
        }
        verifier_path = tmp_path / f"{item['id']}-verifier.json"
        verifier_path.write_text(json.dumps(verifier, sort_keys=True) + "\n")
        histories = [["candidate_error", "ok"] if index != 1 or number == 2 else ["ok"]
                     for number in range(1, 5)]
        receipt = {
            "status": "complete", "branch": branch, "commits": commits,
            "rounds_completed": 4, "durable_handle": f"bz-a3-1:{item['id']}:round-4",
            "attempt_history": [
                {"round": number, "statuses": statuses}
                for number, statuses in enumerate(histories, 1)
            ],
            "rounds": [
                {"round": number, "status": "ok",
                 "handle": f"bz-a3-1:{item['id']}:round-{number}",
                 "median_us": 10.0 + number,
                 "compact_artifacts": [f"/remote/{item['id']}/{number}.json"]}
                for number in range(1, 5)
            ],
        }
        receipt_path = tmp_path / f"{item['id']}-receipt.json"
        resume_binding = None
        if index == 1:
            handle = "remote:bz-a3-1:job:resume-pending"
            operation = {
                "request_sha256": "e" * 64, "mode": "observe", "status": "ok",
                "terminal": True, "handle": handle, "action": "check",
                "attempt_id": None,
            }
            receipt["rounds"][1]["policy"] = {"operation_history": [operation]}
            checkpoint_document = {
                "schema": "profiling-skill/audited-blocked/v2", "stage": "controller",
                "experiment": 2, "session_id": verifier["session_id"],
                "candidate_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                "receipt": {"handle": handle},
            }
            checkpoint_path = tmp_path / f"{item['id']}-checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint_document, sort_keys=True) + "\n")
            marker_document = {
                "schema": "profiling-skill/canary-observer-interrupt/v1",
                "job_request_sha256": "f" * 64,
                "controller_request_sha256": operation["request_sha256"],
                "handle": handle,
                "candidate_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                "target": "bz-a3-1", "device": 6,
                "job_identity": {"action": "profile"},
            }
            marker_path = tmp_path / f"{item['id']}-marker.json"
            marker_path.write_text(json.dumps(marker_document, sort_keys=True) + "\n")
            resume = {
                "schema": "profiling-skill/audited-repair-canary-resume/v1",
                "canary_id": item["id"],
                "checkpoint": {"path": str(checkpoint_path),
                               "sha256": sha(checkpoint_path)},
                "marker": {"path": str(marker_path), "sha256": sha(marker_path)},
                "checkpoint_sha256": sha(checkpoint_path), "experiment": 2,
                "candidate_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                "session_id_before": verifier["session_id"],
                "session_id_after": verifier["session_id"],
                "durable_handle_before": handle, "durable_handle_after": handle,
                "observe_round": 2, "observe_operation": operation,
            }
            resume_path = tmp_path / f"{item['id']}-resume.json"
            resume_path.write_text(json.dumps(resume, sort_keys=True) + "\n")
            resume_binding = {"path": str(resume_path), "sha256": sha(resume_path)}
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        records.append({
            "id": item["id"], "treatment": item["treatment"],
            "experiment_commit": commits[-1],
            "verifier_report": {"path": str(verifier_path),
                                "sha256": sha(verifier_path)},
            "cell_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
            "resume_receipt": resume_binding,
        })
    results = {
        "schema": "profiling-skill/audited-repair-canary-results/v1",
        "definition_sha256": definition_sha,
        "source_revision": config["provenance"]["source_revision"],
        "runtime_closure_sha256": config["runtime_scripts"]["sha256"],
        "results": records,
    }
    results_path = tmp_path / "canary-results.json"
    results_path.write_text(json.dumps(results, sort_keys=True) + "\n")
    return definition_path, definition_sha, results_path, sha(results_path)


def test_repair_campaign_requires_pinned_successful_canary_gate(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    definition, definition_sha, results, results_sha = canary_gate_fixture(
        tmp_path, config
    )
    config["canary_definition"] = {
        "path": str(definition), "sha256": definition_sha,
    }
    assert production.validate_canary_gate(config, results, results_sha)["status"] == "passed"

    with pytest.raises(production.ProductionError, match="required"):
        production.validate_canary_gate(config, None, None)
    with pytest.raises(production.ProductionError, match="hash"):
        production.validate_canary_gate(config, results, "0" * 64)
    definition.write_text("{}\n")
    with pytest.raises(production.ProductionError, match="definition hash"):
        production.validate_canary_gate(config, results, results_sha)


@pytest.mark.parametrize("mutation", ["missing", "malformed", "failed", "artifact-drift"])
def test_repair_campaign_rejects_unsatisfied_canary_gate(
        tmp_path: Path, mutation: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    definition, definition_sha, results, _ = canary_gate_fixture(tmp_path, config)
    config["canary_definition"] = {
        "path": str(definition), "sha256": definition_sha,
    }
    document = json.loads(results.read_text())
    if mutation == "missing":
        document["results"].pop()
    elif mutation == "malformed":
        document["results"][0]["experiment_commit"] = "not-a-commit"
    elif mutation == "failed":
        verifier = Path(document["results"][0]["verifier_report"]["path"])
        rejected = json.loads(verifier.read_text())
        rejected["status"] = "rejected"
        verifier.write_text(json.dumps(rejected, sort_keys=True) + "\n")
        document["results"][0]["verifier_report"]["sha256"] = sha(verifier)
    else:
        receipt = Path(document["results"][0]["cell_receipt"]["path"])
        receipt.write_text("{}\n")
    results.write_text(json.dumps(document, sort_keys=True) + "\n")
    with pytest.raises(production.ProductionError, match="canary"):
        production.validate_canary_gate(config, results, sha(results))


@pytest.mark.parametrize("mutation", [
    "arbitrary-checkpoint-hash", "incomplete-marker", "missing-observe-operation",
])
def test_canary_gate_authenticates_resume_proof_artifacts(
        tmp_path: Path, mutation: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    definition, definition_sha, results, _ = canary_gate_fixture(tmp_path, config)
    config["canary_definition"] = {"path": str(definition), "sha256": definition_sha}
    document = json.loads(results.read_text())
    record = next(item for item in document["results"]
                  if item["id"] == "matmul-project-cannbot-resume")
    resume_path = Path(record["resume_receipt"]["path"])
    resume = json.loads(resume_path.read_text())

    if mutation == "arbitrary-checkpoint-hash":
        resume["checkpoint_sha256"] = "0" * 64
    elif mutation == "incomplete-marker":
        marker_path = Path(resume["marker"]["path"])
        marker = json.loads(marker_path.read_text())
        marker.pop("job_identity")
        marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
        resume["marker"]["sha256"] = sha(marker_path)
    else:
        receipt_path = Path(record["cell_receipt"]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["rounds"][1]["policy"]["operation_history"] = []
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        record["cell_receipt"]["sha256"] = sha(receipt_path)
    resume_path.write_text(json.dumps(resume, sort_keys=True) + "\n")
    record["resume_receipt"]["sha256"] = sha(resume_path)
    results.write_text(json.dumps(document, sort_keys=True) + "\n")

    with pytest.raises(production.ProductionError, match="canary"):
        production.validate_canary_gate(config, results, sha(results))


def test_production_entrypoint_gates_dispatch_on_pinned_canary_results(
        tmp_path: Path, monkeypatch):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    definition, definition_sha, results, results_sha = canary_gate_fixture(
        tmp_path, config
    )
    config["canary_definition"] = {
        "path": str(definition), "sha256": definition_sha,
    }
    manifest = campaign.build_manifest(
        config["run_id"], Path(config["prompt"]["path"]),
        {name: Path(binding["path"]) for name, binding in config["tasks"].items()},
        config["provenance"], "gate-test", request_budget=48,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(config))
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    dispatched = []
    monkeypatch.setattr(production, "CplRemoteResourcePool", lambda *args, **kwargs: object())
    monkeypatch.setattr(production, "ProductionCellLauncher", lambda *args, **kwargs: object())
    monkeypatch.setattr(sys.modules["audited_campaign"], "run_campaign",
                        lambda *args, **kwargs: (
        dispatched.append(args) or {"status": "complete"}
    ))
    arguments = [
        "--manifest", str(manifest_path), "--runtime-config", str(config_path),
        "--runtime-config-sha256", sha(config_path), "--admission", str(admission),
        "--admission-sha256", sha(admission), "--ledger", str(tmp_path / "ledger.json"),
    ]
    with pytest.raises(production.ProductionError, match="required"):
        production.main(arguments)
    assert dispatched == []

    assert production.main(arguments + [
        "--canary-results", str(results),
        "--canary-results-sha256", results_sha,
    ]) == 0
    assert len(dispatched) == 1
    retained = json.loads((
        Path(config["run_root"]) / "state/canary-gate.json"
    ).read_text())
    assert retained["results_sha256"] == results_sha


def test_migration_trust_cli_requires_resume(capsys):
    with pytest.raises(SystemExit) as failure:
        production.main([
            "--manifest", "manifest.json",
            "--runtime-config", "runtime.json",
            "--runtime-config-sha256", "a" * 64,
            "--admission", "admission.json",
            "--admission-sha256", "b" * 64,
            "--ledger", "ledger.json",
            "--migration-attestation", "attestation.json",
            "--migration-attestation-file-sha256", "c" * 64,
            "--migration-attestation-sha256", "d" * 64,
        ])

    assert failure.value.code == 2
    assert "runtime migration trust requires --resume" in capsys.readouterr().err


def test_three_task_starters_share_revision_and_materialize_distinct_pairs(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    assert len({binding["revision"] for binding in
                config["source_repositories"].values()}) == 1
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )

    observed = {}
    for task in production.DEVELOPMENT_CASES:
        task_cell = {**cell, "cell_id": f"{task}-cannbot", "task": task}
        repo, existing = launcher._prepare_repo(task_cell)
        assert existing is False
        observed[task] = (repo / "candidate.py").read_text()
        assert (repo / "candidate.manifest.json").read_bytes() == Path(
            config["starter_sources"][task]["manifest"]["path"]
        ).read_bytes()
    assert len(set(observed.values())) == 3


@pytest.mark.parametrize("mutation", ["missing", "hash-drift"])
def test_starter_failure_precedes_invoker(tmp_path: Path, mutation: str):
    cell = {"cell_id": "gdn-project-guarded", "task": "gdn",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    invocations = []
    candidate = Path(config["starter_sources"]["gdn"]["candidate"]["path"])
    if mutation == "missing":
        candidate.unlink()
    else:
        candidate.write_text("drift\n")

    with pytest.raises(production.ProductionError, match="starter"):
        production_launcher(
            config, invoker_factory=lambda *args, **kwargs: invocations.append(args),
            controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
        )
    assert invocations == []


def test_seed_contains_pinned_starter_and_resume_does_not_overwrite_work(tmp_path: Path):
    cell = {"cell_id": "bsa-project-guarded", "task": "bsa",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    runner = audited_lifecycle.AuditedExperimentRunner(
        repo, Path(config["prompt"]["path"]), Path(config["tasks"]["bsa"]["path"]),
        lambda *args: "", lambda *args: {}, round_count=4,
    )
    _, seed_commit, _ = runner._initialize(config["run_id"], cell["cell_id"])
    expected = config["starter_sources"]["bsa"]
    assert subprocess.check_output(
        ["git", "show", f"{seed_commit}:candidate.py"], cwd=repo
    ) == Path(expected["candidate"]["path"]).read_bytes()
    assert subprocess.check_output(
        ["git", "show", f"{seed_commit}:candidate.manifest.json"], cwd=repo
    ) == Path(expected["manifest"]["path"]).read_bytes()

    (repo / "candidate.py").write_text("agent work in progress\n")
    resumed, existing = launcher._prepare_repo(cell)
    assert existing is True and resumed == repo
    assert (repo / "candidate.py").read_text() == "agent work in progress\n"

    subprocess.run(["git", "add", "candidate.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--amend", "--no-edit", "-q"], cwd=repo, check=True,
    )
    with pytest.raises(production.ProductionError, match="seed commit starter"):
        launcher._prepare_repo(cell)


def test_launcher_recovers_exact_seed_only_branch_without_verifier_shortcut(tmp_path: Path):
    cell = {"cell_id": "gdn-project-guarded", "task": "gdn",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    FakeRunner.calls.clear()
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
        verifier_invoke=valid_verifier,
    )
    repo, _ = launcher._prepare_repo(cell)
    audited_lifecycle.AuditedExperimentRunner(
        repo, Path(config["prompt"]["path"]), Path(config["tasks"]["gdn"]["path"]),
        lambda *args: "", lambda *args: {}, round_count=4,
    )._initialize(config["run_id"], cell["cell_id"])
    (repo / "candidate.py").write_text("VALUE = 9\n")
    (repo / "candidate.manifest.json").write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": "prepared",
    }) + "\n")

    receipt = launcher.launch(cell, {"target": "bz-a3-1", "device": 1})

    assert receipt["status"] == "complete"
    assert FakeRunner.calls[-1][-1] is True


def test_starter_manifest_contract_is_validated_before_materialization(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    binding = config["starter_sources"]["matmul"]["manifest"]
    path = Path(binding["path"])
    path.write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": "",
    }))
    binding["sha256"] = sha(path)
    config["provenance"]["starters"]["matmul"]["manifest"]["sha256"] = sha(path)

    with pytest.raises(production.ProductionError, match="manifest contract"):
        production_launcher(config)


class FakeRunner:
    calls = []

    def __init__(self, repo, prompt, task, invoker, controller, *, round_count,
                 max_candidate_repairs):
        assert round_count == 4
        assert max_candidate_repairs == 2
        self.repo = repo

    def run(self, run_id, agent_id, *, resume=False):
        self.calls.append((self.repo, run_id, agent_id, resume))
        branch = f"experiment/{run_id}/{agent_id}"
        if subprocess.run(
            ["git", "branch", "--show-current"], cwd=self.repo, text=True,
            capture_output=True, check=True,
        ).stdout.strip() != branch:
            subprocess.run(["git", "checkout", "-qb", branch], cwd=self.repo, check=True)
        experiment = self.repo / ".experiment"
        experiment.mkdir(exist_ok=True)
        seed = experiment / "seed.json"
        if not seed.is_file():
            seed.write_text("{}")
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "add", ".experiment/seed.json", "candidate.py", "candidate.manifest.json"],
                cwd=self.repo, check=True,
            )
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "experiment seed"], cwd=self.repo, check=True,
            )
        for number in range(1, 5):
            directory = self.repo / "experiments" / f"{number:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "results.json").write_text(json.dumps({
                "status": "ok", "handle": f"bz-a3-1:round-{number}",
                "median_us": 10.0 - number,
            }))
            subprocess.run(["git", "add", str(directory.relative_to(self.repo))],
                           cwd=self.repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", f"experiment {number}"], cwd=self.repo, check=True,
            )
        return SimpleNamespace(status="complete", branch=branch,
                               session_id="thread", seed_commit="seed",
                               commits=("1", "2", "3", "4"))


def migration_trust_fixture(tmp_path: Path, config: dict) -> tuple[Path, str, dict]:
    """Create the external trust inputs produced by the migration apply command."""
    old_closure = "a" * 64
    config["provenance"]["controller_sha256"] = old_closure
    config_path = tmp_path / "runtime-v4.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    config_sha256 = sha(config_path)
    attestation = {
        "schema": "profiling-skill/audited-runtime-preflight/v1",
        "plan_sha256": "b" * 64,
        "runtimes": {
            "old": {"closure_sha256": old_closure},
            "new": {
                "config_path": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "closure_sha256": config["runtime_scripts"]["sha256"],
            },
        },
        "cells": [],
    }
    attestation["attestation_sha256"] = production.document_sha256(attestation)
    attestation_path = tmp_path / "migration-attestation.json"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
    trusted = {
        "schema": "profiling-skill/audited-runtime-migration-trust/v1",
        "attestation_path": str(attestation_path.resolve()),
        "attestation_file_sha256": sha(attestation_path),
        "attestation_sha256": attestation["attestation_sha256"],
    }
    return config_path, config_sha256, trusted


class TrustRecordingRunner(FakeRunner):
    trusted = None

    def __init__(self, *args, round_count, max_candidate_repairs,
                 trusted_runtime_migration):
        super().__init__(
            *args, round_count=round_count,
            max_candidate_repairs=max_candidate_repairs,
        )
        type(self).trusted = trusted_runtime_migration


def test_launcher_authenticates_and_forwards_external_migration_trust(tmp_path: Path):
    cell = {"cell_id": "gdn-project-guarded", "task": "gdn",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    config_path, config_sha256, trusted = migration_trust_fixture(tmp_path, config)
    attestation_path = Path(trusted["attestation_path"])
    attestation = json.loads(attestation_path.read_text())
    attestation["cells"] = [{"cell_id": cell["cell_id"]}]
    attestation["attestation_sha256"] = production.document_sha256({
        key: value for key, value in attestation.items()
        if key != "attestation_sha256"
    })
    attestation_path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
    trusted["attestation_file_sha256"] = sha(attestation_path)
    trusted["attestation_sha256"] = attestation["attestation_sha256"]
    verifier_commands = []

    def recording_verifier(*args, **kwargs):
        verifier_commands.append(args[0])
        return valid_verifier(*args, **kwargs)

    TrustRecordingRunner.calls.clear()
    launcher = production_launcher(
        config, trusted_runtime_migration=trusted,
        runtime_config_path=config_path, runtime_config_sha256=config_sha256,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=TrustRecordingRunner, verifier_invoke=recording_verifier,
    )

    receipt = launcher.launch(cell, {"target": "bz-a3-1", "device": 0})

    assert receipt["status"] == "complete"
    assert TrustRecordingRunner.trusted == trusted
    proof_index = verifier_commands[-1].index("--migration-proof")
    assert verifier_commands[-1][proof_index + 1:] == [
        trusted["attestation_path"], trusted["attestation_file_sha256"],
        trusted["attestation_sha256"],
    ]


def test_launcher_forwards_ordered_sequential_migration_proofs(tmp_path: Path):
    cell = {"cell_id": "gdn-project-guarded", "task": "gdn",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    config_path, config_sha256, second = migration_trust_fixture(tmp_path, config)
    second_path = Path(second["attestation_path"])
    second_attestation = json.loads(second_path.read_text())
    intermediate = "9" * 64
    second_attestation["runtimes"]["old"]["closure_sha256"] = intermediate
    second_attestation["cells"] = [{"cell_id": cell["cell_id"]}]
    second_attestation["attestation_sha256"] = production.document_sha256({
        key: value for key, value in second_attestation.items()
        if key != "attestation_sha256"
    })
    second_path.write_text(json.dumps(second_attestation, sort_keys=True) + "\n")
    second["attestation_file_sha256"] = sha(second_path)
    second["attestation_sha256"] = second_attestation["attestation_sha256"]
    first_attestation = {
        "schema": "profiling-skill/audited-runtime-preflight/v1",
        "plan_sha256": "8" * 64,
        "runtimes": {
            "old": {"closure_sha256": config["provenance"]["controller_sha256"]},
            "new": {"closure_sha256": intermediate},
        },
        "cells": [{"cell_id": cell["cell_id"]}],
    }
    first_attestation["attestation_sha256"] = production.document_sha256(first_attestation)
    first_path = tmp_path / "migration-attestation-first.json"
    first_path.write_text(json.dumps(first_attestation, sort_keys=True) + "\n")
    first = {
        "schema": production.MIGRATION_TRUST_SCHEMA,
        "attestation_path": str(first_path.resolve()),
        "attestation_file_sha256": sha(first_path),
        "attestation_sha256": first_attestation["attestation_sha256"],
    }
    commands = []

    launcher = production_launcher(
        config, trusted_runtime_migration=[first, second],
        runtime_config_path=config_path, runtime_config_sha256=config_sha256,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=TrustRecordingRunner,
        verifier_invoke=lambda command, **kwargs: (
            commands.append(command) or valid_verifier(command, **kwargs)
        ),
    )

    assert launcher.launch(cell, {"target": "bz-a3-1", "device": 0})["status"] == "complete"
    assert TrustRecordingRunner.trusted == [first, second]
    assert [commands[-1][index + 1:index + 4]
            for index, value in enumerate(commands[-1])
            if value == "--migration-proof"] == [
        [first["attestation_path"], first["attestation_file_sha256"],
         first["attestation_sha256"]],
        [second["attestation_path"], second["attestation_file_sha256"],
         second["attestation_sha256"]],
    ]


@pytest.mark.parametrize("defect", [
    "missing-trust", "missing-config-binding", "trust-file-hash", "trust-seal",
    "attestation-seal", "old-closure", "new-closure", "new-config-path",
    "new-config-hash",
])
def test_launcher_rejects_untrusted_mixed_runtime_before_invoker(
        tmp_path: Path, defect: str):
    cell = {"cell_id": "bsa-project-guarded", "task": "bsa",
            "treatment": "project-guarded", "round_count": 4,
            "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["project-guarded"])}
    config = runtime_fixture(tmp_path, cell)
    config_path, config_sha256, trusted = migration_trust_fixture(tmp_path, config)
    kwargs = {"runtime_config_path": config_path,
              "runtime_config_sha256": config_sha256}
    if defect == "missing-trust":
        trusted = None
        kwargs = {}
    elif defect == "missing-config-binding":
        kwargs.pop("runtime_config_path")
    elif defect.startswith("trust-"):
        trusted["attestation_file_sha256" if defect == "trust-file-hash"
                else "attestation_sha256"] = "0" * 64
    else:
        attestation_path = Path(trusted["attestation_path"])
        attestation = json.loads(attestation_path.read_text())
        if defect == "attestation-seal":
            attestation["attestation_sha256"] = "0" * 64
        elif defect == "old-closure":
            attestation["runtimes"]["old"]["closure_sha256"] = "0" * 64
        elif defect == "new-closure":
            attestation["runtimes"]["new"]["closure_sha256"] = "0" * 64
        elif defect == "new-config-path":
            attestation["runtimes"]["new"]["config_path"] = str(tmp_path / "other.json")
        else:
            attestation["runtimes"]["new"]["config_sha256"] = "0" * 64
        if defect != "attestation-seal":
            attestation["attestation_sha256"] = production.document_sha256({
                key: value for key, value in attestation.items()
                if key != "attestation_sha256"
            })
            trusted["attestation_sha256"] = attestation["attestation_sha256"]
        attestation_path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
        trusted["attestation_file_sha256"] = sha(attestation_path)
    invocations = []

    with pytest.raises(production.ProductionError, match="migration|controller provenance"):
        production_launcher(
            config, trusted_runtime_migration=trusted, **kwargs,
            invoker_factory=lambda *args, **kw: invocations.append((args, kw)),
            controller_factory=lambda *args, **kw: object(), runner_factory=FakeRunner,
        )
    assert invocations == []


def test_cell_launcher_materializes_isolation_controller_and_resume(tmp_path: Path):
    cell = {"cell_id": "gdn-project-guarded", "task": "gdn",
            "treatment": "project-guarded", "round_count": 4, "request_budget": 24,
            "skills": ["ascend-profiling", "triton-guarded-kernel"]}
    config = runtime_fixture(tmp_path, cell)
    created = {}

    def invoker(repo, **kwargs):
        created["invoker"] = (repo, kwargs)
        return SimpleNamespace(
            scrub_auth=lambda: None,
            docker_image_id=config["runtime_image_digest"],
        )

    def controller(command, repo, **kwargs):
        created["controller"] = (command, repo, kwargs)
        return object()

    def verifier(*args, **kwargs):
        created["verifier"] = (args, kwargs)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({
                "status": "valid",
                "branch": "experiment/production-e2e/gdn-project-guarded",
                "seed_commit": "seed", "session_id": "thread",
                "experiments": [{"commit": str(number)} for number in range(1, 5)],
            }), stderr="",
        )

    launcher = production_launcher(
        config, invoker_factory=invoker, controller_factory=controller,
        runner_factory=FakeRunner,
        verifier_invoke=verifier,
    )
    slot = {"target": "bz-a3-2", "device": 6}
    receipt = launcher.launch(cell, slot)
    assert receipt["status"] == "complete"
    assert receipt["rounds_completed"] == 4
    assert receipt["durable_handle"] == "bz-a3-1:round-4"
    assert len(receipt["rounds"]) == 4
    repo = Path(config["run_root"]) / cell["cell_id"] / "repo"
    assert {path.name for path in (repo / ".agents" / "skills").iterdir()} == set(cell["skills"])
    controller_config = json.loads(
        (Path(config["run_root"]) / cell["cell_id"] / "state" / "controller.json").read_text()
    )
    assert controller_config["benchmark"] == "gdn"
    assert controller_config["round_count"] == 4
    assert controller_config["request_budget"] == 24
    assert controller_config["development_cases"] == [40, 49, 47, 46, 45]
    assert controller_config["all_cases"] == list(range(50))
    assert controller_config["baseline"] == json.loads(
        Path(config["baseline_sources"]["gdn"]["path"]).read_text()
    )
    assert controller_config["devices"] == [{"id": "bz-a3-2/device-6", "device": 0}]
    assert controller_config["timeout_seconds"] == 12
    nested = json.loads(controller_config["backend_command"][-1])
    assert "--device" not in nested
    assert nested[nested.index("--timeout") + 1] == "10"
    assert created["controller"][0][1].endswith("audited_bz_controller.py")
    assert created["controller"][2]["timeout"] == 15
    assert created["invoker"][1]["agent_id"] == cell["cell_id"]
    assert created["invoker"][1]["timeout"] == 20
    assert created["verifier"][1]["timeout"] == 20
    assert FakeRunner.calls[-1][-1] is False
    launcher.launch(cell, slot)
    # A ledger-loss restart reconstructs the already-complete verified branch;
    # it does not ask the lifecycle to resume without a blocked checkpoint.
    assert FakeRunner.calls[-1][-1] is False


def test_completed_cell_rejects_same_run_manifest_budget_change(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config["manifest_sha256"] = "1" * 64
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
        verifier_invoke=valid_verifier,
    )
    assert launcher.launch(cell, {"target": "bz-a3-1", "device": 0})["status"] == "complete"
    calls = len(FakeRunner.calls)

    regenerated = {**cell, "request_budget": 48}
    config["manifest_sha256"] = "2" * 64
    rejected = production_launcher(
        config,
        invoker_factory=lambda *args, **kwargs: pytest.fail("invoker must not start"),
        controller_factory=lambda *args, **kwargs: pytest.fail("controller must not start"),
        runner_factory=FakeRunner, verifier_invoke=valid_verifier,
    )
    with pytest.raises(production.ProductionError, match="different identity"):
        rejected.launch(regenerated, {"target": "bz-a3-1", "device": 0})
    assert len(FakeRunner.calls) == calls


def test_same_manifest_identity_accepts_normal_existing_cell(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config["manifest_sha256"] = "3" * 64
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, existing = launcher._prepare_repo(cell)
    assert existing is False
    resumed, existing = launcher._prepare_repo(cell)
    assert resumed == repo and existing is False
    identity = json.loads((repo.parent / "state/cell.json").read_text())
    assert identity["schema"] == "profiling-skill/cell-identity/v2"
    assert identity["request_budget"] == 48
    assert identity["manifest_sha256"] == "3" * 64


def test_legacy_identity_upgrades_only_with_matching_seed_and_controller(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    audited_lifecycle.AuditedExperimentRunner(
        repo, Path(config["prompt"]["path"]), Path(config["tasks"]["matmul"]["path"]),
        lambda *args: "", lambda *args: {}, round_count=4,
    )._initialize(config["run_id"], cell["cell_id"])
    launcher._controller(cell, {"target": "bz-a3-1", "device": 0}, repo.parent, repo)
    identity_path = repo.parent / "state/cell.json"
    identity = json.loads(identity_path.read_text())
    legacy = {key: identity[key] for key in (
        "cell_id", "task", "treatment", "source_revision", "starter",
    )}
    identity_path.write_text(json.dumps(legacy) + "\n")

    resumed, existing = launcher._prepare_repo(cell)
    assert resumed == repo and existing is True
    assert json.loads(identity_path.read_text())["schema"] == (
        "profiling-skill/cell-identity/v2"
    )


def test_repair_runtime_uses_48_operations_and_two_candidate_repairs(tmp_path: Path):
    cell = {"cell_id": "matmul-project-guarded", "task": "matmul",
            "treatment": "project-guarded", "round_count": 4, "request_budget": 48,
            "skills": ["ascend-profiling", "triton-guarded-kernel"]}
    config = runtime_fixture(tmp_path, cell)
    config["max_candidate_repairs_per_round"] = 2
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=FakeRunner, verifier_invoke=valid_verifier,
    )
    launcher.launch(cell, {"target": "bz-a3-1", "device": 2})
    controller_config = json.loads((
        Path(config["run_root"]) / cell["cell_id"] / "state/controller.json"
    ).read_text())
    assert controller_config["request_budget"] == 48


def test_declared_resume_canary_routes_only_job_client_observation_fault(tmp_path: Path):
    cell = {
        "cell_id": "matmul-project-cannbot-resume", "task": "matmul",
        "treatment": "project-cannbot", "round_count": 4, "request_budget": 48,
        "skills": list(production.TREATMENT_SKILLS["project-cannbot"]),
        "canary_fault": "interrupt-after-dispatch-once",
    }
    config = runtime_fixture(tmp_path, cell)
    definition = tmp_path / "canaries.json"
    definition.write_bytes((ROOT / "experiments/audited-repair-canaries.json").read_bytes())
    config["canary_definition"] = {"path": str(definition), "sha256": sha(definition)}
    captured = {}

    def controller(command, repo, **kwargs):
        captured["command"] = command
        return object()

    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=controller, runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    launcher._controller(cell, {"target": "bz-a3-1", "device": 3}, repo.parent, repo)

    controller_config = json.loads((repo.parent / "state/controller.json").read_text())
    backend = controller_config["backend_command"]
    client = json.loads(backend[backend.index("--job-client-json") + 1])
    flag = client.index("--canary-interrupt-after-dispatch-once")
    assert Path(client[flag + 1]) == (
        repo.parent / "state/canary-observer-interrupt.json"
    ).resolve()


@pytest.mark.parametrize("value", [-1, 0, 1, 3, True])
def test_production_rejects_nonstandard_candidate_repair_limit(
        tmp_path: Path, value):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config["max_candidate_repairs_per_round"] = value
    with pytest.raises(production.ProductionError, match="exactly two"):
        production_launcher(config)


def test_launcher_rejects_skill_or_runtime_hash_drift(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    Path(config["runtime_scripts"]["path"], "benchmark_backend.py").write_text("drift")
    with pytest.raises(production.ProductionError, match="runtime scripts hash"):
        production_launcher(config)


@pytest.mark.parametrize("failure", ["clone", "copy"])
def test_cell_bootstrap_recovers_only_owned_interrupted_staging(
        tmp_path: Path, monkeypatch, failure: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    if failure == "clone":
        original = production.subprocess.run

        def interrupted_clone(argv, **kwargs):
            if argv[:3] == ["git", "clone", "--quiet"]:
                return SimpleNamespace(returncode=1, stderr="interrupted", stdout="")
            return original(argv, **kwargs)

        monkeypatch.setattr(production.subprocess, "run", interrupted_clone)
    else:
        original = production._copy_tree
        monkeypatch.setattr(
            production, "_copy_tree",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("interrupted")),
        )

    with pytest.raises((production.ProductionError, OSError), match="interrupted"):
        launcher._prepare_repo(cell)
    staging = Path(config["run_root"]) / f".{cell['cell_id']}.initializing"
    assert json.loads((staging / "state/bootstrap.json").read_text())["identity"][
        "cell_id"] == cell["cell_id"]

    if failure == "clone":
        monkeypatch.setattr(production.subprocess, "run", original)
    else:
        monkeypatch.setattr(production, "_copy_tree", original)
    repo, existing = launcher._prepare_repo(cell)
    assert repo.is_dir() and existing is False
    assert not staging.exists()


def test_cell_bootstrap_rejects_unowned_partial_staging(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    staging = Path(config["run_root"]) / f".{cell['cell_id']}.initializing"
    (staging / "state").mkdir(parents=True)
    (staging / "state/bootstrap.json").write_text(json.dumps({"identity": {}}))
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    with pytest.raises(production.ProductionError, match="unowned partial bootstrap"):
        launcher._prepare_repo(cell)
    assert staging.exists()


@pytest.mark.parametrize("mutation", ["drift", "extra"])
def test_resume_rejects_treatment_skill_tree_drift_or_extras(
        tmp_path: Path, mutation: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config, invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    if mutation == "drift":
        (repo / ".agents/skills/triton-op-coding/SKILL.md").write_text("changed")
    else:
        (repo / ".agents/skills/unapproved").mkdir()
    with pytest.raises(production.ProductionError, match="isolated treatment skills"):
        launcher._prepare_repo(cell)


@pytest.mark.parametrize(("key", "value"), [
    ("agent_turn_timeout", 0),
    ("verifier_timeout", True),
    ("backend_job_timeout", -1),
    ("controller_transaction_timeout", 14),
])
def test_launcher_rejects_invalid_or_overlapping_timeout_contract(
        tmp_path: Path, key: str, value: object):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config[key] = value
    with pytest.raises(production.ProductionError, match="timeout"):
        production_launcher(config)


def test_launcher_rejects_direct_runtime_and_arbitrary_adapter(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config["runtime_mode"] = "direct"
    config["adapter_command"] = ["untrusted-adapter"]
    with pytest.raises(production.ProductionError, match="isolated Docker"):
        production_launcher(config)


@pytest.mark.parametrize("pin", ["source_revision", "controller_sha256", "baseline", "skill"])
def test_launcher_binds_manifest_provenance_to_runtime_inputs(tmp_path: Path, pin: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    if pin == "source_revision":
        config["provenance"][pin] = "a" * 40
    elif pin == "controller_sha256":
        config["provenance"][pin] = "a" * 64
    elif pin == "baseline":
        config["provenance"]["baselines"]["matmul"] = "a" * 64
    else:
        config["provenance"]["skills"]["cannbot"] = "a" * 64
    with pytest.raises(production.ProductionError, match="provenance"):
        production_launcher(config)


@pytest.mark.parametrize("pin", ["resource", "client"])
def test_launcher_binds_resource_module_and_client_to_manifest_provenance(
        tmp_path: Path, pin: str):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    key = ("resource_admission_sha256" if pin == "resource"
           else "cpl_remote_closure_sha256")
    config["provenance"][key] = "a" * 64
    with pytest.raises(production.ProductionError, match="resource admission provenance"):
        production_launcher(config)


def test_admission_refresh_evidence_is_durable_before_slots_are_returned(tmp_path: Path):
    receipt = SimpleNamespace(
        receipt_sha256="a" * 64, provider_id="campaign-operator",
        allowlist_sha256="d" * 64,
        generated_at="2026-10-06T09:00:00Z",
        expires_at="2026-10-06T09:03:00Z",
        as_slots=lambda: [
            {"target": "bz-a3-1", "device": 2, "healthy": True, "idle": True}
        ],
    )
    evidence = tmp_path / "admission-evidence.json"
    pool = production.DurableAdmissionPool(
        SimpleNamespace(admit_snapshot=lambda: receipt), evidence, "run-id",
    )

    assert pool.admit() == receipt.as_slots()
    assert pool.admit() == receipt.as_slots()
    document = json.loads(evidence.read_text())
    assert document == {
        "schema": "profiling-skill/admission-evidence/v1",
        "run_id": "run-id",
        "refreshes": [
            {"sequence": 1, "receipt_sha256": "a" * 64,
             "provider_id": "campaign-operator", "allowlist_sha256": "d" * 64,
             "generated_at": "2026-10-06T09:00:00Z",
             "expires_at": "2026-10-06T09:03:00Z"},
            {"sequence": 2, "receipt_sha256": "a" * 64,
             "provider_id": "campaign-operator", "allowlist_sha256": "d" * 64,
             "generated_at": "2026-10-06T09:00:00Z",
             "expires_at": "2026-10-06T09:03:00Z"},
        ],
    }


def test_expired_admission_pauses_without_launch_or_evidence(tmp_path: Path):
    prompt = tmp_path / "prompt.md"
    tasks = {}
    provenance = {
        "source_revision": "a" * 40, "controller_sha256": "b" * 64,
        "runtime_image_digest": "sha256:" + "c" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {name: "d" * 64 for name in production.DEVELOPMENT_CASES},
        "starters": {
            name: {
                "candidate": {"path": f"/frozen/{name}/candidate.py", "sha256": "2" * 64},
                "manifest": {"path": f"/frozen/{name}/candidate.manifest.json",
                             "sha256": "3" * 64},
            } for name in production.DEVELOPMENT_CASES
        },
        "skills": {"cannbot": "e" * 64, "ascend-profiling": "f" * 64,
                   "triton-guarded-kernel": "1" * 64},
    }
    prompt.write_text("prompt")
    for name in production.DEVELOPMENT_CASES:
        tasks[name] = tmp_path / f"{name}.md"
        tasks[name].write_text(name)
    manifest = campaign.build_manifest("expired", prompt, tasks, provenance, "fixed")
    launches = []

    class ExpiredPool:
        def admit_snapshot(self):
            raise production.AdmissionError("admission receipt is expired")

    class NeverLauncher:
        def launch(self, cell, slot):
            launches.append((cell, slot))

    evidence = tmp_path / "admission-evidence.json"
    pool = production.DurableAdmissionPool(ExpiredPool(), evidence, "expired")
    with pytest.raises(campaign.CampaignPaused, match="expired"):
        campaign.run_campaign(manifest, tmp_path / "ledger.json", pool, NeverLauncher())
    assert launches == []
    assert not evidence.exists()


def test_launcher_rejects_global_client_implementation_drift(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    client = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
    client.with_name("cpl_remote.py").write_text("# implementation drift\n")
    with pytest.raises(production.ProductionError, match="closure hash"):
        production_launcher(config)


def test_launcher_rejects_invalid_timing_baseline_contract(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    baseline = Path(config["baseline_sources"]["matmul"]["path"])
    document = json.loads(baseline.read_text())
    document["case_medians_us"][0]["median_us"] = 999.0
    baseline.write_text(json.dumps(document))
    pin = sha(baseline)
    config["baseline_sources"]["matmul"]["sha256"] = pin
    config["provenance"]["baselines"]["matmul"] = pin

    with pytest.raises(production.ProductionError, match="timing baseline contract"):
        production_launcher(config)


def test_launcher_rejects_nonfinite_timing_baseline_value(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    baseline = Path(config["baseline_sources"]["matmul"]["path"])
    document = json.loads(baseline.read_text())
    document["control_median_us"] = float("inf")
    document["sha256"] = production.document_sha256({
        key: value for key, value in document.items() if key != "sha256"
    })
    baseline.write_text(json.dumps(document))
    pin = sha(baseline)
    config["baseline_sources"]["matmul"]["sha256"] = pin
    config["provenance"]["baselines"]["matmul"] = pin

    with pytest.raises(production.ProductionError, match="timing baseline contract"):
        production_launcher(config)


def test_controller_uses_pinned_global_boundary_without_adapter_argv(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    created = {}
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda command, repo, **kwargs: created.setdefault("controller", command),
        runner_factory=FakeRunner,
            verifier_invoke=valid_verifier,
    )
    launcher.launch(cell, {"target": "bz-a3-1", "device": 1})
    controller_config = json.loads((
        Path(config["run_root"]) / cell["cell_id"] / "state" / "controller.json"
    ).read_text())
    client = json.loads(controller_config["backend_command"][-1])
    assert "--adapter-json" not in client
    global_client = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
    assert client[client.index("--cpl-remote-sha256") + 1] == sha(global_client)
    placements = json.loads((
        Path(config["run_root"]) / cell["cell_id"] / "state" / "placements.json"
    ).read_text())
    assert placements == {"0": {"target": "bz-a3-1", "device": 1}}


def test_runner_crash_is_infrastructure_not_candidate_failure(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)

    class CrashingRunner(FakeRunner):
        def run(self, *args, **kwargs):
            raise RuntimeError("codex transport vanished")

    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=CrashingRunner,
    )
    with pytest.raises(runtime_campaign.InfrastructureFailure,
                       match="codex transport vanished"):
        launcher.launch(cell, {"target": "bz-a3-1", "device": 1})


def test_infrastructure_exception_identity_is_injected_not_looked_up_late(
        tmp_path: Path, monkeypatch):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)

    class CrashingRunner(FakeRunner):
        def run(self, *args, **kwargs):
            raise RuntimeError("controller transport vanished")

    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=CrashingRunner,
    )
    replacement = SimpleNamespace(InfrastructureFailure=type(
        "DifferentInfrastructureFailure", (RuntimeError,), {}
    ))
    monkeypatch.setitem(sys.modules, "audited_campaign", replacement)

    with pytest.raises(runtime_campaign.InfrastructureFailure,
                       match="controller transport vanished"):
        launcher.launch(cell, {"target": "bz-a3-1", "device": 1})


def test_existing_nonseed_incomplete_branch_requires_valid_blocked_checkpoint(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    audited_lifecycle.AuditedExperimentRunner(
        repo, Path(config["prompt"]["path"]), Path(config["tasks"]["matmul"]["path"]),
        lambda *args: "", lambda *args: {}, round_count=4,
    )._initialize(config["run_id"], cell["cell_id"])
    (repo / "unexpected.txt").write_text("committed but not an experiment\n")
    subprocess.run(["git", "add", "unexpected.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "unexpected"], cwd=repo, check=True)
    with pytest.raises(runtime_campaign.InfrastructureFailure, match="blocked checkpoint"):
        launcher.launch(cell, {"target": "bz-a3-1", "device": 1})


@pytest.mark.parametrize(("schema", "round_count", "accepted"), [
    ("profiling-skill/audited-blocked/v2", 4, True),
    ("profiling-skill/audited-blocked/v2", 3, False),
    ("profiling-skill/audited-blocked/v1", None, False),
])
def test_four_round_campaign_accepts_only_v2_four_round_checkpoint(
        tmp_path: Path, schema: str, round_count: int | None, accepted: bool):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production_launcher(
        config,
        invoker_factory=lambda *args, **kwargs: object(),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    blocked = {
        "schema": schema, "experiment": 2, "stage": "controller",
        "branch": "experiment/production-e2e/matmul-cannbot",
    }
    if round_count is not None:
        blocked["round_count"] = round_count
    (repo / ".experiment").mkdir()
    (repo / ".experiment" / "blocked.json").write_text(json.dumps(blocked))
    if accepted:
        assert launcher._resume_checkpoint(repo, cell) == blocked
    else:
        with pytest.raises(production.ProductionError, match="checkpoint"):
            launcher._resume_checkpoint(repo, cell)


def test_complete_branch_is_verified_and_reconstructed_with_full_receipts(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    verifier_calls = []

    class EvidenceRunner(FakeRunner):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            for number in range(1, 5):
                path = self.repo / "experiments" / f"{number:02d}" / "results.json"
                receipt = json.loads(path.read_text())
                receipt.update({
                    "case_results": [{"case": 7, "median_us": 9.0}],
                    "baseline_median_us": 12.0,
                    "calibration": {"median_us": 1.0},
                    "policy": {"admission_controls": [{"status": "pass"}],
                               "infra_attempts": [{"status": "retry"}]},
                })
                path.write_text(json.dumps(receipt))
            return result

    def verifier(argv, **kwargs):
        verifier_calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "status": "valid", "branch": "experiment/production-e2e/matmul-cannbot",
            "seed_commit": "seed", "session_id": "thread",
            "experiments": [{"commit": str(number)} for number in range(1, 5)],
        }), stderr="")

    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=EvidenceRunner,
        verifier_invoke=verifier,
    )
    slot = {"target": "bz-a3-1", "device": 1}
    first = launcher.launch(cell, slot)
    assert first["rounds"][0]["case_results"][0]["case"] == 7
    assert first["rounds"][0]["policy"]["infra_attempts"] == [{"status": "retry"}]
    assert first["baseline_median_us"] == 12.0
    second = launcher.launch(cell, slot)
    assert second == first
    assert len(verifier_calls) == 2


def test_real_command_controller_preserves_genuine_candidate_failure(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    controller_script = Path(config["runtime_scripts"]["path"]) / "audited_bz_controller.py"
    controller_script.write_text("""\
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--candidate-sha256', required=True)
p.add_argument('--manifest-sha256', required=True)
p.add_argument('--experiment', required=True)
a, _ = p.parse_known_args()
h = f'remote:bz-a3-1:job:candidate-{a.experiment}'
print(json.dumps({
  'candidate_sha256': a.candidate_sha256,
  'manifest_sha256': a.manifest_sha256,
  'handle': h, 'status': 'candidate_error', 'device': 'bz-a3-1/device-1',
  'reason': 'Triton compilation failed',
  'policy': {
    'schema': 'profiling-skill/controller-policy/v1',
    'selected_device': 'bz-a3-1/device-1',
    'admission_controls': [{'device': 'bz-a3-1/device-1', 'status': 'pass',
      'healthy': True, 'idle': True, 'warmed': True}],
    'submission_candidate_sha256': a.candidate_sha256,
    'submitted_handles': [h], 'observed_handles': [h], 'infra_retries': 0,
    'retry_budget': 3, 'quarantined_devices': [], 'quarantine_controls': {},
    'confirmation_count': 0, 'post_control': 'not_run',
    'variability_threshold': 0.25
  }
}))
""")
    closure_hash = production.digest_tree(Path(config["runtime_scripts"]["path"]))
    config["runtime_scripts"]["sha256"] = closure_hash
    config["provenance"]["controller_sha256"] = closure_hash

    class ControllerRunner(FakeRunner):
        def __init__(self, repo, prompt, task, invoker, controller, *, round_count,
                     max_candidate_repairs):
            super().__init__(
                repo, prompt, task, invoker, controller, round_count=round_count,
                max_candidate_repairs=max_candidate_repairs,
            )
            self.controller = controller

        def run(self, run_id, agent_id, *, resume=False):
            result = super().run(run_id, agent_id, resume=resume)
            candidate_hash = sha(self.repo / "candidate.py")
            manifest_hash = sha(self.repo / "candidate.manifest.json")
            for number in range(1, 5):
                receipt = self.controller(number, candidate_hash, manifest_hash)
                path = self.repo / "experiments" / f"{number:02d}" / "results.json"
                path.write_text(json.dumps(receipt))
            return result

    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=audited_runtime.CommandController,
        runner_factory=ControllerRunner, verifier_invoke=valid_verifier,
    )
    receipt = launcher.launch(cell, {"target": "bz-a3-1", "device": 1})
    assert receipt["status"] == "candidate_failed"
    assert receipt["rounds_completed"] == 4
    assert [item["status"] for item in receipt["rounds"]] == ["candidate_error"] * 4
    assert receipt["rounds"][0]["reason"] == "Triton compilation failed"


def test_real_four_round_composition_resumes_verifies_and_reconstructs(
        tmp_path: Path, monkeypatch):
    if "round_count" not in inspect.signature(
            audited_lifecycle.AuditedExperimentRunner).parameters:
        pytest.skip(
            "requires stacked audited-configurable-rounds prerequisite; "
            "this test runs unskipped once that PR is integrated"
        )
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    runtime_root = Path(config["runtime_scripts"]["path"])
    for name in production.RUNTIME_FILES - {"audited_bz_controller.py"}:
        source = ROOT / "scripts" / name
        if source.is_file():
            shutil.copy2(source, runtime_root / name)
    fake_codex = runtime_root / "fake_codex.py"
    fake_codex.write_text(r'''\
import hashlib, json, sys
from pathlib import Path

request = json.load(sys.stdin)
repo = Path(request["repo"])
number = request["number"]
instruction = request["instruction"]
thread = "thread-composed"
events = [{"type": "thread.started", "thread_id": thread}]
if instruction is None:
    (repo / "candidate.py").write_text(f"VALUE = {number + 1}\n")
    events.append({"type": "item.completed", "item": {
        "type": "command_execution", "command": "python local-check.py",
        "exit_code": 0, "aggregated_output": "ok\n"}})
    document = {"prepared": True}
elif instruction.startswith("Repair candidate attempt"):
    value = int((repo / "candidate.py").read_text().split()[-1]) + 10
    (repo / "candidate.py").write_text(f"VALUE = {value}\n")
    events.append({"type": "item.completed", "item": {
        "type": "command_execution", "command": "python local-check.py",
        "exit_code": 0, "aggregated_output": "ok\n"}})
    document = {"repaired": True}
else:
    marker = "exact host controller receipt:\n"
    receipt_line = instruction.split(marker, 1)[1].splitlines()[0]
    receipt = json.loads(receipt_line)
    candidate = hashlib.sha256((repo / "candidate.py").read_bytes()).hexdigest()
    manifest = hashlib.sha256((repo / "candidate.manifest.json").read_bytes()).hexdigest()
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    document = {
        "hypothesis": f"round {number} changes the candidate",
        "expected_result": "the controller returns attributable evidence",
        "change": f"set VALUE for round {number}",
        "evidence": "local check plus compact controller receipt",
        "observed_result": receipt["status"],
        "decision": "revert" if receipt["status"] == "candidate_error" else "retain",
        "postmortem": "the retained receipt determines the decision",
        "next_experiment": "continue with the next host-directed round",
        "candidate_sha256": candidate, "manifest_sha256": manifest,
        "controller_handle": receipt["handle"],
        "controller_receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "sources": [], "no_sources_reason": "self-contained composition fixture",
    }
events.append({"type": "item.completed", "item": {
    "type": "agent_message", "text": json.dumps(document)}})
print("\n".join(json.dumps(event) for event in events))
''')
    controller_script = runtime_root / "audited_bz_controller.py"
    controller_script.write_text(r'''\
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--state-dir", required=True)
p.add_argument("--experiment", required=True, type=int)
p.add_argument("--candidate-sha256", required=True)
p.add_argument("--manifest-sha256", required=True)
p.add_argument("--observe-handle")
p.add_argument("--remeasure-handle")
a = p.parse_args()
cell = Path.cwd().parent.name
state_path = Path(a.config).parent / "fake-controller-state.json"
state = json.loads(state_path.read_text()) if state_path.is_file() else {}
key = f"{cell}:{a.experiment}"
state[key] = state.get(key, 0) + 1
state_path.write_text(json.dumps(state))
handle = a.observe_handle or a.remeasure_handle or f"remote:bz-a3-1:job:{cell}-{a.experiment}"

def policy(post="stable", samples=True):
    result = {
        "schema": "profiling-skill/controller-policy/v1",
        "selected_device": "bz-a3-1/device-1",
        "admission_controls": [{"device": "bz-a3-1/device-1", "status": "pass",
            "healthy": True, "idle": True, "warmed": True}],
        "submission_candidate_sha256": a.candidate_sha256,
        "submitted_handles": [handle], "observed_handles": [handle],
        "infra_retries": 0, "retry_budget": 3, "quarantined_devices": [],
        "quarantine_controls": {}, "confirmation_count": 0,
        "post_control": post, "variability_threshold": 0.25,
    }
    if samples:
        result.update({"sample_count": 3, "variability_ratio": 0.02})
    return result

base = {"candidate_sha256": a.candidate_sha256,
        "manifest_sha256": a.manifest_sha256, "handle": handle,
        "device": "bz-a3-1/device-1"}
if cell.startswith("matmul-") and a.experiment == 1 and not a.observe_handle:
    print(json.dumps({**base, "status": "infrastructure_error",
                      "reason": "observer disconnected"}))
elif cell.startswith("matmul-") and a.experiment == 2 and not a.remeasure_handle:
    print(json.dumps({**base, "status": "measurement_pending",
        "samples_us": [1.0, 2.0, 3.0], "median_us": 2.0,
        "policy": {**policy(), "variability_ratio": 1.0}}))
elif cell.startswith("gdn-") and a.experiment == 2:
    print(json.dumps({**base, "status": "candidate_error",
        "reason": "Triton compilation failed", "policy": policy("not_run", False)}))
else:
    samples = [10.0 + a.experiment, 10.1 + a.experiment, 10.2 + a.experiment]
    print(json.dumps({**base, "status": "ok", "samples_us": samples,
        "median_us": samples[1],
        "policy": {**policy(), "variability_ratio": 0.2 / samples[1]}}))
''')
    closure_hash = production.digest_tree(runtime_root)
    config["runtime_scripts"]["sha256"] = closure_hash
    config["provenance"]["controller_sha256"] = closure_hash
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Composition Host")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "host@example.invalid")

    class SubprocessCodex:
        docker_image_id = config["runtime_image_digest"]

        def __init__(self, repo: Path):
            self.repo = repo

        def __call__(self, number, session, instruction):
            run = subprocess.run(
                [sys.executable, str(fake_codex)], input=json.dumps({
                    "repo": str(self.repo), "number": number,
                    "session": session, "instruction": instruction,
                }), text=True, capture_output=True, check=True,
            )
            return run.stdout

        def scrub_auth(self):
            pass

    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SubprocessCodex(repo),
        controller_factory=audited_runtime.CommandController,
        runner_factory=audited_lifecycle.AuditedExperimentRunner,
    )
    slot = {"target": "bz-a3-1", "device": 1}
    with pytest.raises(runtime_campaign.InfrastructureFailure) as controller_block:
        launcher.launch(cell, slot)
    first_checkpoint = json.loads((
        Path(config["run_root"]) / cell["cell_id"] / "repo/.experiment/blocked.json"
    ).read_text())
    assert (first_checkpoint["schema"], first_checkpoint["round_count"],
            first_checkpoint["stage"]) == (
                "profiling-skill/audited-blocked/v2", 4, "controller")

    with pytest.raises(runtime_campaign.InfrastructureFailure) as measurement_block:
        launcher.observe(cell, slot, controller_block.value.durable_handle)
    second_checkpoint = json.loads((
        Path(config["run_root"]) / cell["cell_id"] / "repo/.experiment/blocked.json"
    ).read_text())
    assert (second_checkpoint["schema"], second_checkpoint["round_count"],
            second_checkpoint["stage"]) == (
                "profiling-skill/audited-blocked/v2", 4, "measurement")

    complete = launcher.observe(cell, slot, measurement_block.value.durable_handle)
    assert complete["status"] == "complete" and complete["rounds_completed"] == 4
    assert launcher.launch(cell, slot) == complete

    failed_cell = dict(cell, cell_id="gdn-cannbot", task="gdn")
    failed_cell["task_sha256"] = config["tasks"]["gdn"]["sha256"]
    failed_cell["prompt_contract"] = {
        "invariant_sha256": config["prompt"]["sha256"],
        "task_sha256": config["tasks"]["gdn"]["sha256"],
    }
    failed = launcher.launch(failed_cell, slot)
    assert failed["status"] == "candidate_failed"
    assert failed["rounds"][1]["status"] == "candidate_error"
    assert failed["rounds_completed"] == 4


@pytest.mark.parametrize(("task", "development", "count"), [
    ("matmul", [7, 8, 9], 10),
    ("gdn", [40, 49, 47, 46, 45], 50),
    ("bsa", [47, 46, 49, 44, 43], 50),
])
def test_production_case_contracts_cover_all_tasks(task, development, count):
    assert production.DEVELOPMENT_CASES[task] == development
    assert production.ALL_CASES[task] == list(range(count))


def test_fake_production_launcher_runs_all_nine_isolated_branches(tmp_path: Path):
    source, revision = git_repo(tmp_path / "source")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt")
    tasks = {}
    for task in production.DEVELOPMENT_CASES:
        tasks[task] = tmp_path / f"{task}.md"
        tasks[task].write_text(task)
    skills = {}
    for name in {skill for names in production.TREATMENT_SKILLS.values() for skill in names}:
        path = tmp_path / "skills" / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(name)
        skills[name] = {"path": str(path), "sha256": production.digest_tree(path)}
    scripts = tmp_path / "runtime"
    scripts.mkdir()
    for name in ("audited_bz_controller.py", "benchmark_backend.py",
                 "bz_a3_job_client.py", "validate_audited_experiment.py",
                 "audited_verifier.py", "audited_contract.py",
                 "audited_lifecycle.py", "audited_runtime.py"):
        (scripts / name).write_text("# pinned\n")
    assets = tmp_path / "benchmarks"
    assets.mkdir()
    baseline_root = tmp_path / "timing-baselines"
    baseline_root.mkdir()
    baselines = {}
    starters = {}
    for task in tasks:
        (assets / task).mkdir()
        (assets / task / "cases.json").write_text(json.dumps({"task": task}))
        path = baseline_root / f"{task}.json"
        path.write_text(json.dumps(timing_baseline(task)))
        baselines[task] = {"path": str(path), "sha256": sha(path)}
        starter = tmp_path / "starters" / task
        starter.mkdir(parents=True)
        candidate = starter / "candidate.py"
        candidate.write_text(f"TASK = {task!r}\nVALUE = 0\n")
        candidate_manifest = starter / "candidate.manifest.json"
        candidate_manifest.write_text(json.dumps({
            "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": task,
        }) + "\n")
        starters[task] = {
            "candidate": {"path": str(candidate.resolve()), "sha256": sha(candidate)},
            "manifest": {"path": str(candidate_manifest.resolve()),
                         "sha256": sha(candidate_manifest)},
        }
    provenance = {
        "source_revision": revision,
        "controller_sha256": production.digest_tree(scripts),
        "runtime_image_digest": "sha256:" + "b" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {task: binding["sha256"] for task, binding in baselines.items()},
        "starters": starters,
        "skills": {
            "cannbot": production.digest_skill_bundle(skills, production.CANNBOT_SKILLS),
            "ascend-profiling": skills["ascend-profiling"]["sha256"],
            "triton-guarded-kernel": skills["triton-guarded-kernel"]["sha256"],
        },
    }
    cpl_remote = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
    resource_module = ROOT / "scripts" / "audited_resource_admission.py"
    provenance["resource_admission_sha256"] = sha(resource_module)
    provenance["cpl_remote_closure_sha256"] = (
        production.cpl_remote_closure_sha256(cpl_remote)
    )
    provenance["admission_provider_id"] = "campaign-operator"
    provenance["admission_allowlist_sha256"] = "d" * 64
    manifest = campaign.build_manifest(
        "fake-production", prompt, tasks, provenance, "fixed"
    )
    config = {
        "schema": production.RUNTIME_SCHEMA, "run_id": manifest["run_id"],
        "run_root": str(tmp_path / "runs"),
        "source_repositories": {
            task: {"path": str(source), "revision": revision} for task in tasks
        },
        "skill_sources": skills,
        "runtime_scripts": {"path": str(scripts),
                            "sha256": production.digest_tree(scripts)},
        "benchmark_assets": {"path": str(assets),
                             "sha256": production.digest_tree(assets)},
        "baseline_sources": baselines,
        "starter_sources": starters,
        "admission_provider_id": "campaign-operator",
        "admission_allowlist_sha256": "d" * 64,
        "cpl_remote_closure_sha256": production.cpl_remote_closure_sha256(cpl_remote),
        "resource_admission": {
            "path": str(resource_module), "sha256": sha(resource_module),
        },
        "prompt": manifest["prompt"], "tasks": manifest["tasks"],
        "remote_root": "/remote/campaign", "provenance": provenance,
        "auth_home": str(tmp_path / "auth"), "model": "gpt-5.6-sol",
        "reasoning_effort": "low", "runtime_mode": "docker",
        "runtime_image_digest": provenance["runtime_image_digest"],
        "agent_turn_timeout": 20,
        "controller_transaction_timeout": 15,
        "verifier_timeout": 20,
        "backend_job_timeout": 10,
        "timeout_grace": 2,
    }
    launcher = production_launcher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=provenance["runtime_image_digest"]),
        controller_factory=lambda command, repo, **kwargs: object(),
        runner_factory=FakeRunner,
        verifier_invoke=valid_verifier,
    )
    pool = SimpleNamespace(admit=lambda: [
        {"target": "bz-a3-1", "device": 2, "healthy": True, "idle": True},
        {"target": "bz-a3-2", "device": 5, "healthy": True, "idle": True},
    ])
    ledger = campaign.run_campaign(
        manifest, tmp_path / "ledger.json", pool, launcher
    )
    assert ledger["status"] == "complete"
    assert len(list((tmp_path / "runs").glob("*/repo/.git"))) == 9
    assert all(state["status"] == "complete" for state in ledger["cells"].values())
    for cell in manifest["cells"]:
        controller_config = json.loads((
            tmp_path / "runs" / cell["cell_id"] / "state" / "controller.json"
        ).read_text())
        assert controller_config["benchmark"] == cell["task"]
        assert controller_config["development_cases"] == production.DEVELOPMENT_CASES[cell["task"]]
        assert controller_config["all_cases"] == production.ALL_CASES[cell["task"]]
        assert controller_config["baseline"] == json.loads(
            Path(baselines[cell["task"]]["path"]).read_text()
        )
