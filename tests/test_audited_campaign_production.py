from __future__ import annotations

import hashlib
import importlib.util
import json
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for name in cell["skills"]:
        skill = tmp_path / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(name)
        skills[name] = {"path": str(skill), "sha256": production.digest_tree(skill)}
    scripts = tmp_path / "runtime"
    scripts.mkdir()
    for name in ("audited_bz_controller.py", "benchmark_backend.py",
                 "bz_a3_job_client.py"):
        (scripts / name).write_text("# pinned\n")
    assets = tmp_path / "benchmarks"
    assets.mkdir()
    (assets / "pinned.txt").write_text("assets")
    cpl_remote = tmp_path / "cpl-remote"
    cpl_remote.write_text("#!/bin/sh\n")
    prompt = tmp_path / "prompt.md"
    task = tmp_path / "task.md"
    prompt.write_text("prompt")
    task.write_text("task")
    cell["task_sha256"] = sha(task)
    cell["prompt_contract"] = {
        "invariant_sha256": sha(prompt), "task_sha256": sha(task),
    }
    return {
        "run_id": "production-e2e",
        "run_root": str(tmp_path / "runs"),
        "source_repositories": {cell["task"]: {"path": str(repo), "revision": revision}},
        "skill_sources": skills,
        "runtime_scripts": {"path": str(scripts), "sha256": production.digest_tree(scripts)},
        "benchmark_assets": {"path": str(assets),
                             "sha256": production.digest_tree(assets)},
        "cpl_remote": str(cpl_remote), "cpl_remote_sha256": sha(cpl_remote),
        "prompt": {"path": str(prompt), "sha256": sha(prompt)},
        "tasks": {cell["task"]: {"path": str(task), "sha256": sha(task)}},
        "adapter_command": ["approved-adapter"],
        "remote_root": "/remote/campaign",
        "auth_home": str(tmp_path / "auth"),
        "model": "gpt-5.6-sol", "reasoning_effort": "low",
        "runtime_mode": "direct", "timeout": 10,
    }


class FakeRunner:
    calls = []

    def __init__(self, repo, prompt, task, invoker, controller, *, round_count):
        assert round_count == 4
        self.repo = repo

    def run(self, run_id, agent_id, *, resume=False):
        self.calls.append((self.repo, run_id, agent_id, resume))
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
        return SimpleNamespace(status="complete", branch=f"experiment/{run_id}/{agent_id}",
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
        return SimpleNamespace(scrub_auth=lambda: None)

    def controller(command, repo, **kwargs):
        created["controller"] = (command, repo, kwargs)
        return object()

    launcher = production.ProductionCellLauncher(
        config, invoker_factory=invoker, controller_factory=controller,
        runner_factory=FakeRunner,
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
    assert controller_config["devices"] == [{"id": "bz-a3-2/device-6", "device": 0}]
    nested = json.loads(controller_config["backend_command"][-1])
    assert "--device" not in nested
    assert created["controller"][0][1].endswith("audited_bz_controller.py")
    assert created["invoker"][1]["agent_id"] == cell["cell_id"]
    assert FakeRunner.calls[-1][-1] is False

    launcher.launch(cell, slot)
    assert FakeRunner.calls[-1][-1] is True


def test_launcher_rejects_skill_or_runtime_hash_drift(tmp_path: Path):
    cell = {"cell_id": "matmul-cannbot", "task": "matmul", "treatment": "cannbot",
            "round_count": 4, "request_budget": 24,
            "skills": list(production.TREATMENT_SKILLS["cannbot"])}
    config = runtime_fixture(tmp_path, cell)
    Path(config["runtime_scripts"]["path"], "benchmark_backend.py").write_text("drift")
    with pytest.raises(production.ProductionError, match="runtime scripts hash"):
        production.ProductionCellLauncher(config)


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
    provenance = {
        "source_revision": revision, "controller_sha256": "a" * 64,
        "runtime_image_digest": "sha256:" + "b" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {task: "c" * 64 for task in tasks},
        "skills": {"cannbot": "d" * 64, "ascend-profiling": "e" * 64,
                   "triton-guarded-kernel": "f" * 64},
    }
    manifest = campaign.build_manifest(
        "fake-production", prompt, tasks, provenance, "fixed"
    )
    skills = {}
    for name in {skill for cell in manifest["cells"] for skill in cell["skills"]}:
        path = tmp_path / "skills" / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(name)
        skills[name] = {"path": str(path), "sha256": production.digest_tree(path)}
    scripts = tmp_path / "runtime"
    scripts.mkdir()
    for name in ("audited_bz_controller.py", "benchmark_backend.py",
                 "bz_a3_job_client.py"):
        (scripts / name).write_text("# pinned\n")
    assets = tmp_path / "benchmarks"
    assets.mkdir()
    (assets / "pinned.txt").write_text("assets")
    cpl_remote = tmp_path / "cpl-remote"
    cpl_remote.write_text("#!/bin/sh\n")
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
        "cpl_remote": str(cpl_remote), "cpl_remote_sha256": sha(cpl_remote),
        "prompt": manifest["prompt"], "tasks": manifest["tasks"],
        "adapter_command": ["approved-adapter"], "remote_root": "/remote/campaign",
        "auth_home": str(tmp_path / "auth"), "model": "gpt-5.6-sol",
        "reasoning_effort": "low", "runtime_mode": "direct", "timeout": 10,
    }
    launcher = production.ProductionCellLauncher(
        config,
        invoker_factory=lambda repo, **kwargs: SimpleNamespace(scrub_auth=lambda: None),
        controller_factory=lambda command, repo, **kwargs: object(),
        runner_factory=FakeRunner,
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
