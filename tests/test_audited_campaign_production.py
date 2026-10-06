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


def test_resource_pool_uses_only_capabilities_preflight_and_pinned_admission(tmp_path: Path):
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "schema": production.ADMISSION_SCHEMA,
        "slots": [
            {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
            {"target": "bz-a3-2", "device": 4, "healthy": True, "idle": False},
        ],
    }))
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    pool = production.CplRemoteResourcePool(
        admission, sha(admission), cpl_remote="cpl-remote", invoke=invoke
    )
    assert pool.admit() == [
        {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True},
        {"target": "bz-a3-2", "device": 4, "healthy": True, "idle": False},
    ]
    assert calls == [
        ["cpl-remote", "capabilities", "bz-a3-1"],
        ["cpl-remote", "preflight", "bz-a3-1"],
        ["cpl-remote", "capabilities", "bz-a3-2"],
        ["cpl-remote", "preflight", "bz-a3-2"],
    ]


def test_resource_pool_fails_closed_without_valid_placement_provider(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(production.ProductionError, match="placement-provider"):
        production.CplRemoteResourcePool(missing, "0" * 64).admit()


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
    for task_name in production.DEVELOPMENT_CASES:
        (assets / task_name).mkdir()
        (assets / task_name / "cases.json").write_text(json.dumps({"task": task_name}))
        baseline = baseline_root / f"{task_name}.json"
        baseline.write_text(json.dumps(timing_baseline(task_name)))
        baselines[task_name] = {"path": str(baseline), "sha256": sha(baseline)}
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
        "skills": {
            "cannbot": production.digest_skill_bundle(skills, cannbot_names),
            **({"ascend-profiling": skills["ascend-profiling"]["sha256"]}
               if "ascend-profiling" in skills else {}),
            **({"triton-guarded-kernel": skills["triton-guarded-kernel"]["sha256"]}
               if "triton-guarded-kernel" in skills else {}),
        },
    }
    config = {
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
        "cpl_remote_sha256": sha(cpl_remote),
        "prompt": {"path": str(prompt), "sha256": sha(prompt)},
        "tasks": {name: {"path": str(task), "sha256": sha(task)}
                  for name in production.DEVELOPMENT_CASES},
        "remote_root": "/remote/campaign",
        "auth_home": str(tmp_path / "auth"),
        "model": "gpt-5.6-sol", "reasoning_effort": "low",
        "runtime_mode": "docker", "runtime_image_digest": provenance["runtime_image_digest"],
        "timeout": 10,
        "provenance": provenance,
    }
    return config


class FakeRunner:
    calls = []

    def __init__(self, repo, prompt, task, invoker, controller, *, round_count):
        assert round_count == 4
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
        (experiment / "seed.json").write_text("{}")
        for number in range(1, 5):
            directory = self.repo / "experiments" / f"{number:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "results.json").write_text(json.dumps({
                "status": "ok", "handle": f"bz-a3-1:round-{number}",
                "median_us": 10.0 - number,
            }))
        return SimpleNamespace(status="complete", branch=branch,
                               session_id="thread", seed_commit="seed",
                               commits=("1", "2", "3", "4"))


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

    launcher = production.ProductionCellLauncher(
        config, invoker_factory=invoker, controller_factory=controller,
        runner_factory=FakeRunner,
        verifier_invoke=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps({
                "status": "valid", "branch": "experiment/production-e2e/gdn-project-guarded",
                "seed_commit": "seed", "session_id": "thread",
                "experiments": [{"commit": str(number)} for number in range(1, 5)],
            }), stderr=""),
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
    nested = json.loads(controller_config["backend_command"][-1])
    assert "--device" not in nested
    assert created["controller"][0][1].endswith("audited_bz_controller.py")
    assert created["invoker"][1]["agent_id"] == cell["cell_id"]
    assert FakeRunner.calls[-1][-1] is False

    launcher.launch(cell, slot)
    # A ledger-loss restart reconstructs the already-complete verified branch;
    # it does not ask the lifecycle to resume without a blocked checkpoint.
    assert FakeRunner.calls[-1][-1] is False


def test_launcher_rejects_skill_or_runtime_hash_drift(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    Path(config["runtime_scripts"]["path"], "benchmark_backend.py").write_text("drift")
    with pytest.raises(production.ProductionError, match="runtime scripts hash"):
        production.ProductionCellLauncher(config)


def test_launcher_rejects_direct_runtime_and_arbitrary_adapter(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    config["runtime_mode"] = "direct"
    config["adapter_command"] = ["untrusted-adapter"]
    with pytest.raises(production.ProductionError, match="isolated Docker"):
        production.ProductionCellLauncher(config)


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
        production.ProductionCellLauncher(config)


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
        production.ProductionCellLauncher(config)


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
        production.ProductionCellLauncher(config)


def test_controller_uses_pinned_global_boundary_without_adapter_argv(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    created = {}
    launcher = production.ProductionCellLauncher(
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
    assert client[client.index("--cpl-remote-sha256") + 1] == config["cpl_remote_sha256"]
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

    launcher = production.ProductionCellLauncher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(),
        runner_factory=CrashingRunner,
    )
    with pytest.raises(runtime_campaign.InfrastructureFailure,
                       match="codex transport vanished"):
        launcher.launch(cell, {"target": "bz-a3-1", "device": 1})


def test_existing_incomplete_branch_requires_valid_blocked_checkpoint(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    launcher = production.ProductionCellLauncher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(
            scrub_auth=lambda: None, docker_image_id=config["runtime_image_digest"]),
        controller_factory=lambda *args, **kwargs: object(), runner_factory=FakeRunner,
    )
    repo, _ = launcher._prepare_repo(cell)
    (repo / ".experiment").mkdir()
    (repo / ".experiment" / "seed.json").write_text("{}")
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
    launcher = production.ProductionCellLauncher(
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

    launcher = production.ProductionCellLauncher(
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
        def __init__(self, repo, prompt, task, invoker, controller, *, round_count):
            super().__init__(repo, prompt, task, invoker, controller, round_count=round_count)
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

    launcher = production.ProductionCellLauncher(
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

    launcher = production.ProductionCellLauncher(
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
    for task in tasks:
        (assets / task).mkdir()
        (assets / task / "cases.json").write_text(json.dumps({"task": task}))
        path = baseline_root / f"{task}.json"
        path.write_text(json.dumps(timing_baseline(task)))
        baselines[task] = {"path": str(path), "sha256": sha(path)}
    provenance = {
        "source_revision": revision,
        "controller_sha256": production.digest_tree(scripts),
        "runtime_image_digest": "sha256:" + "b" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {task: binding["sha256"] for task, binding in baselines.items()},
        "skills": {
            "cannbot": production.digest_skill_bundle(skills, production.CANNBOT_SKILLS),
            "ascend-profiling": skills["ascend-profiling"]["sha256"],
            "triton-guarded-kernel": skills["triton-guarded-kernel"]["sha256"],
        },
    }
    manifest = campaign.build_manifest(
        "fake-production", prompt, tasks, provenance, "fixed"
    )
    cpl_remote = Path.home() / ".agents/skills/remote-access/scripts/cpl-remote"
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
        "cpl_remote_sha256": sha(cpl_remote),
        "prompt": manifest["prompt"], "tasks": manifest["tasks"],
        "remote_root": "/remote/campaign", "provenance": provenance,
        "auth_home": str(tmp_path / "auth"), "model": "gpt-5.6-sol",
        "reasoning_effort": "low", "runtime_mode": "docker",
        "runtime_image_digest": provenance["runtime_image_digest"], "timeout": 10,
    }
    launcher = production.ProductionCellLauncher(
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
