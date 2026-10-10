from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audited_repair_canaries", ROOT / "scripts" / "audited_repair_canaries.py"
)
canaries = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = canaries
SPEC.loader.exec_module(canaries)

PRODUCTION_SPEC = importlib.util.spec_from_file_location(
    "canary_gate_production", ROOT / "scripts" / "audited_campaign_production.py"
)
production = importlib.util.module_from_spec(PRODUCTION_SPEC)
assert PRODUCTION_SPEC.loader
sys.modules[PRODUCTION_SPEC.name] = production
PRODUCTION_SPEC.loader.exec_module(production)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import audited_contract
    import audited_lifecycle
finally:
    sys.path.pop(0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakePool:
    def __init__(self):
        self.calls = 0

    def admit(self):
        self.calls += 1
        return [{"target": "bz-a3-1", "device": 6, "id": "bz-a3-1/device-6",
                 "healthy": True, "idle": True}]


class FakeLauncher:
    calls: list[tuple[str, str]] = []

    def __init__(self, config: dict, mode: str):
        self.config, self.mode = config, mode

    def _repo(self, cell: dict) -> Path:
        repo = Path(self.config["run_root"]) / cell["cell_id"] / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        return repo

    def launch(self, cell: dict, slot: dict) -> dict:
        del slot
        assert cell["skills"] == list(production.TREATMENT_SKILLS[cell["treatment"]])
        self.calls.append((cell["cell_id"], "launch"))
        repo = self._repo(cell)
        if self.mode == "resume" and not (repo / "resumed").exists():
            checkpoint = {
                "schema": "profiling-skill/audited-blocked/v2", "round_count": 4,
                "experiment": 2, "stage": "controller", "session_id": "thread-resume",
                "candidate_sha256": "e" * 64, "manifest_sha256": "f" * 64,
                "receipt": {"handle": "remote:bz-a3-1:job:resume"},
            }
            blocked = repo / ".experiment" / "blocked.json"
            blocked.parent.mkdir()
            blocked.write_text(json.dumps(checkpoint, sort_keys=True) + "\n")
            marker = repo.parent / "state" / "canary-observer-interrupt.json"
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({
                "schema": canaries.OBSERVER_SCHEMA,
                "job_request_sha256": "8" * 64,
                "controller_request_sha256": "9" * 64,
                "handle": "remote:bz-a3-1:job:resume",
                "candidate_sha256": "e" * 64, "manifest_sha256": "f" * 64,
                "target": "bz-a3-1", "device": 6, "job_identity": {"action": "check"},
            }) + "\n")
            raise canaries.CanaryInterrupted("injected observer interruption")
        return self._receipt(cell)

    def observe(self, cell: dict, slot: dict, handle: str) -> dict:
        del slot
        assert handle == "remote:bz-a3-1:job:resume"
        self.calls.append((cell["cell_id"], "observe"))
        repo = self._repo(cell)
        (repo / "resumed").write_text(handle)
        (repo / ".experiment" / "blocked.json").unlink()
        return self._receipt(cell)

    def _receipt(self, cell: dict) -> dict:
        repaired = self.mode == "repair"
        histories = [
            {"round": number,
             "statuses": (["candidate_error", "ok"] if repaired and number == 1 else ["ok"])}
            for number in range(1, 5)
        ]
        commits = [f"{number:x}" * 40 for number in range(1, 5)]
        rounds = [{
            "round": number, "status": "ok", "median_us": 10.0 + number,
            "handle": f"remote:bz-a3-1:job:{cell['cell_id']}:{number}",
            "compact_artifacts": [f"/remote/{cell['cell_id']}/{number}.json"],
        } for number in range(1, 5)]
        if self.mode == "resume":
            rounds[1]["policy"] = {"operation_history": [{
                "request_sha256": "9" * 64, "mode": "observe", "status": "ok",
                "terminal": True, "handle": "remote:bz-a3-1:job:resume",
                "action": "check", "attempt_id": None,
            }]}
        return {
            "status": "complete", "branch": f"experiment/canary/{cell['cell_id']}",
            "session_id": "thread-resume" if self.mode == "resume" else "thread-fixed",
            "commits": commits, "rounds_completed": 4,
            "durable_handle": f"remote:bz-a3-1:job:{cell['cell_id']}:4",
            "attempt_history": histories,
            "rounds": rounds,
        }

    def verify(self, cell: dict) -> dict:
        receipt = self._receipt(cell)
        return {
            "status": "valid", "branch": receipt["branch"],
            "session_id": receipt["session_id"],
            "experiments": [{"commit": commit} for commit in receipt["commits"]],
        }


def inputs(tmp_path: Path):
    definition = tmp_path / "canaries.json"
    definition.write_bytes((ROOT / "experiments/audited-repair-canaries.json").read_bytes())
    config = {
        "run_id": "must-not-be-used", "run_root": str(tmp_path / "production"),
        "provenance": {"source_revision": "a" * 40},
        "runtime_scripts": {"sha256": "b" * 64},
        "prompt": {"sha256": "c" * 64},
        "tasks": {"matmul": {"sha256": "d" * 64}},
        "model": "gpt-test", "runtime_image_digest": "sha256:" + "e" * 64,
        "starter_sources": {"matmul": {"sha256": "f" * 64}},
        "baseline_sources": {"matmul": {"sha256": "1" * 64}},
        "canary_definition": {"path": str(definition), "sha256": sha(definition)},
    }
    output = tmp_path / "canary-run" / "canary-results.json"
    return definition, config, output


def test_cli_translates_admission_failure(monkeypatch, tmp_path: Path, capsys):
    definition, _config, _output = inputs(tmp_path)
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({
        "admission_provider_id": "provider",
        "admission_allowlist_sha256": "a" * 64,
        "cpl_remote_closure_sha256": "b" * 64,
    }) + "\n")
    admission = tmp_path / "admission.json"
    admission.write_text("{}\n")

    class RefusingPool:
        def __init__(self, *args, **kwargs):
            raise canaries.AdmissionError("no eligible device")

    monkeypatch.setattr(canaries, "CplRemoteResourcePool", RefusingPool)
    with pytest.raises(SystemExit) as failure:
        canaries.main([
            "--runtime-config", str(config),
            "--runtime-config-sha256", sha(config),
            "--definition", str(definition),
            "--definition-sha256", sha(definition),
            "--admission", str(admission),
            "--admission-sha256", sha(admission),
            "--run-root", str(tmp_path / "run"),
        ])

    assert failure.value.code == 2
    assert "no eligible device" in capsys.readouterr().err


def test_runner_emits_gate_accepted_artifacts_and_exact_resume(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )

    result = runner.run(output)

    assert result["schema"] == "profiling-skill/audited-repair-canary-results/v1"
    assert len(result["results"]) == 4
    assert production.validate_canary_gate(config, output, sha(output))["status"] == "passed"
    assert sum(action == "observe" for _, action in FakeLauncher.calls) == 1
    assert not Path(config["run_root"]).exists()
    retained = json.loads((tmp_path / "canary-run/state/run.json").read_text())
    assert retained["status"] == "complete" and retained["production_campaign_launched"] is False


def test_dual_product_runner_emits_a3_cells_and_binds_exact_runtime_config(
    tmp_path: Path,
):
    definition, config, output = inputs(tmp_path)
    a3_task = {"path": "/inputs/a3-matmul.md", "sha256": "2" * 64}
    config["product_tasks"] = {
        "a3": {"matmul": a3_task},
        "a5": {"matmul": {"path": "/inputs/a5-matmul.md", "sha256": "3" * 64}},
    }
    config["products"] = {
        "a3": {"runtime": "py311-torch"}, "a5": {"runtime": "cann91"},
    }
    exact_runtime_sha256 = production.document_sha256(config)
    seen = []

    class DualLauncher(FakeLauncher):
        def launch(self, cell: dict, slot: dict) -> dict:
            seen.append((dict(cell), dict(slot), self.config))
            return super().launch(cell, slot)

        def observe(self, cell: dict, slot: dict, handle: str) -> dict:
            seen.append((dict(cell), dict(slot), self.config))
            return super().observe(cell, slot, handle)

    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: DualLauncher(cfg, mode),
    )

    result = runner.run(output)

    assert result["runtime_config_sha256"] == exact_runtime_sha256
    assert seen
    assert all(cell["product"] == "a3" and cell["runtime"] == "py311-torch"
               and cell["task_sha256"] == a3_task["sha256"]
               for cell, _slot, _config in seen)
    assert all(slot["target"] == "bz-a3-1" for _cell, slot, _config in seen)
    assert all(runtime["product_tasks"]["a3"]["matmul"] == a3_task
               for _cell, _slot, runtime in seen)
    assert production.validate_canary_gate(
        config, output, sha(output), exact_runtime_sha256,
    )["status"] == "passed"


def test_runner_accepts_v2_exact_profile_treatment_subset(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    document = json.loads(definition.read_text())
    selected = ["cannbot-new-profiler", "guarded-new-profiler"]
    document["schema"] = "profiling-skill/audited-repair-canaries/v2"
    document["treatments"] = selected
    document["canaries"] = [item for item in document["canaries"]
                            if item["treatment"] in selected]
    document["gate"]["minimum_repaired_canaries"] = 1
    definition.write_text(json.dumps(document, sort_keys=True) + "\n")
    config["canary_definition"] = {"path": str(definition), "sha256": sha(definition)}
    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )

    result = runner.run(output)
    calls = list(FakeLauncher.calls)
    repeated = runner.run(output)

    assert [item["treatment"] for item in result["results"]] == selected
    assert repeated == result
    assert FakeLauncher.calls == calls
    assert production.validate_canary_gate(
        config, output, sha(output), selected_treatments=tuple(selected),
    )["status"] == "passed"


def test_runner_preserves_reordered_complete_v1_definition(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    document = json.loads(definition.read_text())
    document["canaries"].reverse()
    definition.write_text(json.dumps(document, sort_keys=True) + "\n")
    config["canary_definition"] = {"path": str(definition), "sha256": sha(definition)}
    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )

    result = runner.run(output)

    assert [item["treatment"] for item in result["results"]] == [
        "guarded-old-profiler", "guarded-new-profiler",
        "cannbot-new-profiler", "cannbot-all",
    ]
    assert production.validate_canary_gate(config, output, sha(output))["status"] == "passed"


def test_completed_results_are_idempotently_verified_without_launch(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )
    first = runner.run(output)
    calls = list(FakeLauncher.calls)

    second = runner.run(output)

    assert first == second
    assert FakeLauncher.calls == calls


def test_cached_results_reject_run_state_runtime_binding_drift(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    run_root = tmp_path / "canary-run"
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), run_root, FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )
    runner.run(output)
    state_path = run_root / "state/run.json"
    state = json.loads(state_path.read_text())
    state["runtime_config_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

    with pytest.raises(canaries.CanaryError, match="run state is invalid"):
        runner.run(output)


@pytest.mark.parametrize("binding", [
    "prompt", "task", "model", "image", "starter", "baseline",
])
def test_cached_results_reject_complete_runtime_config_drift(
        tmp_path: Path, binding: str):
    definition, config, output = inputs(tmp_path)
    run_root = tmp_path / "canary-run"
    def runner(value):
        return canaries.CanaryRunner(
            value, definition, sha(definition), run_root, FakePool(),
            launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
        )

    runner(config).run(output)
    drifted = json.loads(json.dumps(config))
    if binding == "prompt":
        drifted["prompt"]["sha256"] = "2" * 64
    elif binding == "task":
        drifted["tasks"]["matmul"]["sha256"] = "2" * 64
    elif binding == "model":
        drifted["model"] = "gpt-other"
    elif binding == "image":
        drifted["runtime_image_digest"] = "sha256:" + "2" * 64
    elif binding == "starter":
        drifted["starter_sources"]["matmul"]["sha256"] = "2" * 64
    else:
        drifted["baseline_sources"]["matmul"]["sha256"] = "2" * 64

    with pytest.raises(canaries.production.ProductionError,
                       match="pinned production inputs"):
        runner(drifted).run(output)


def test_initial_slot_filters_busy_and_unhealthy_but_retained_slot_is_exact(tmp_path: Path):
    definition, config, _output = inputs(tmp_path)

    class MixedPool:
        def admit(self):
            return [
                {"target": "bz-a3-1", "device": 0, "healthy": False, "idle": True},
                {"target": "bz-a3-1", "device": 1, "healthy": True, "idle": False},
                {"target": "bz-a3-2", "device": 3, "healthy": True, "idle": True},
            ]

    run_root = tmp_path / "canary-run"
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), run_root, MixedPool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )
    cell = runner._cell(runner.definition["canaries"][0])
    assert runner._slot(cell) == {
        "target": "bz-a3-2", "device": 3, "healthy": True, "idle": True,
    }

    retained = run_root / "cells" / cell["cell_id"] / "state/placement.json"
    retained.parent.mkdir(parents=True)
    exact = {"target": "bz-a3-1", "device": 7, "healthy": True, "idle": True}
    retained.write_text(json.dumps(exact) + "\n")
    assert runner._slot(cell) == exact


def test_initial_slot_fails_when_no_healthy_idle_device(tmp_path: Path):
    definition, config, _output = inputs(tmp_path)
    pool = type("UnavailablePool", (), {"admit": lambda self: [
        {"target": "bz-a3-1", "device": 0, "healthy": False, "idle": True},
        {"target": "bz-a3-2", "device": 0, "healthy": True, "idle": False},
    ]})()
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", pool,
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )
    with pytest.raises(canaries.CanaryError, match="no healthy idle"):
        runner._slot(runner._cell(runner.definition["canaries"][0]))


def test_unclassified_infrastructure_failure_is_not_a_resume(tmp_path: Path):
    definition, config, output = inputs(tmp_path)

    class Broken(FakeLauncher):
        def launch(self, cell, slot):
            del cell, slot
            raise canaries.CanaryError("remote capacity unavailable")

    runner = canaries.CanaryRunner(
        config, definition, sha(definition), tmp_path / "canary-run", FakePool(),
        launcher_factory=lambda cfg, mode: Broken(cfg, mode),
    )
    with pytest.raises(canaries.CanaryError, match="capacity"):
        runner.run(output)
    assert not output.exists()


def test_candidate_failure_injection_strips_timing_and_runs_once(tmp_path: Path):
    marker = tmp_path / "injected.json"

    class Controller:
        def __call__(self, number, candidate, manifest):
            return {
                "status": "ok", "experiment": number, "handle": "remote:job:1",
                "candidate_sha256": candidate, "manifest_sha256": manifest,
                "device": "bz-a3-1/device-6", "samples_us": [1.0, 1.1, 1.2],
                "median_us": 1.1, "baseline_median_us": 2.0, "baseline": {},
                "calibration": {}, "normalized_samples_us": [1.0],
                "normalized_median_us": 1.0, "speedup_vs_baseline": 2.0,
                "policy": {"sample_count": 3, "variability_ratio": .1,
                           "accepted_timing": "primary", "primary": {},
                           "confirmation": {}, "confirmation_count": 1,
                           "post_control": "stable"},
            }

    wrapped = canaries.InjectedController(Controller(), "repair", marker)
    failed = wrapped(1, "a" * 64, "b" * 64)
    replayed = wrapped(1, "a" * 64, "b" * 64)
    later_round = wrapped(2, "a" * 64, "b" * 64)
    successful = wrapped(2, "c" * 64, "d" * 64)

    assert failed["status"] == "candidate_error"
    assert failed["policy"]["post_control"] == "not_run"
    assert replayed["status"] == "candidate_error"
    assert later_round["status"] == "ok"
    assert not ({"samples_us", "median_us", "baseline"} & failed.keys())
    assert successful["status"] == "ok"


def valid_receipt(candidate: str, manifest: str, number: int) -> dict:
    handle, device = f"remote:bz-a3-1:job:{number}", "bz-a3-1/device-6"
    return {
        "status": "ok", "experiment": number, "handle": handle,
        "candidate_sha256": candidate, "manifest_sha256": manifest,
        "device": device, "samples_us": [1.0, 1.1, 1.2], "median_us": 1.1,
        "policy": {
            "schema": audited_contract.CONTROLLER_POLICY_SCHEMA,
            "selected_device": device,
            "admission_controls": [{"device": device, "status": "pass",
                                    "healthy": True, "idle": True, "warmed": True}],
            "submission_candidate_sha256": candidate,
            "submitted_handles": [handle], "observed_handles": [handle],
            "infra_retries": 0, "retry_budget": 3, "quarantined_devices": [],
            "quarantine_controls": {}, "confirmation_count": 0,
            "post_control": "stable", "variability_threshold": .25,
            "sample_count": 3, "variability_ratio": .2 / 1.1,
        },
    }


def event_stream(thread: str, document: dict, command: bool = True) -> str:
    events = [{"type": "thread.started", "thread_id": thread}]
    if command:
        events.append({"type": "item.completed", "item": {
            "type": "command_execution", "command": "python smoke.py",
            "exit_code": 0, "aggregated_output": "ok\n",
        }})
    events.append({"type": "item.completed", "item": {
        "type": "agent_message", "text": json.dumps(document),
    }})
    return "\n".join(json.dumps(item) for item in events) + "\n"


def test_injected_receipt_passes_contract_and_drives_same_round_repair(tmp_path: Path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=repo, check=True)
    (repo / "candidate.py").write_text("VALUE = 0\n")
    (repo / "candidate.manifest.json").write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": "kernel",
    }) + "\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    prompt, task = tmp_path / "prompt.md", tmp_path / "task.md"
    prompt.write_text("Prepare and report one experiment.\n")
    task.write_text("Increment VALUE.\n")

    class Controller:
        last = None

        def __call__(self, number, candidate, manifest):
            self.last = valid_receipt(candidate, manifest, number)
            return self.last

    base = Controller()
    controller = canaries.InjectedController(base, "repair", tmp_path / "inject.json")
    preparations = 0

    def invoke(number, session, instruction):
        nonlocal preparations
        if instruction is None or instruction.startswith("Repair candidate attempt"):
            preparations += 1
            (repo / "candidate.py").write_text(f"VALUE = {preparations}\n")
            return event_stream("thread-one", {"prepared": True})
        candidate = hashlib.sha256((repo / "candidate.py").read_bytes()).hexdigest()
        manifest = hashlib.sha256((repo / "candidate.manifest.json").read_bytes()).hexdigest()
        report = {
            "hypothesis": "increment compiles", "expected_result": "valid kernel",
            "change": "increment VALUE", "evidence": "controller receipt",
            "observed_result": "profile completed", "decision": "retain",
            "postmortem": "bounded injected error was repaired", "next_experiment": "done",
            "candidate_sha256": candidate, "manifest_sha256": manifest,
            "controller_handle": base.last["handle"],
            "controller_receipt_sha256": audited_contract.sha256_json(base.last),
            "sources": [], "no_sources_reason": "self-contained canary",
        }
        return event_stream("thread-one", report, command=False)

    result = audited_lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
        max_candidate_repairs=2,
    ).run("canary", "repair")

    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    attempts = evidence["candidate_attempts"]
    assert result.status == "complete" and preparations == 2
    assert [item["status"] for item in attempts] == ["candidate_error", "ok"]
    failed = json.loads((repo / "experiments/01/attempts/01/controller.json").read_text())
    audited_contract.validate_controller_receipt(
        failed, failed["candidate_sha256"], failed["manifest_sha256"]
    )


@pytest.mark.parametrize("boundary", ["after-marker", "after-candidate-change"])
def test_forced_repair_crash_boundaries_resume_without_duplicate_dispatch(
        tmp_path: Path, boundary: str):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=repo, check=True)
    (repo / "candidate.py").write_text("VALUE = 0\n")
    (repo / "candidate.manifest.json").write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": "kernel",
    }) + "\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    prompt, task = tmp_path / "prompt.md", tmp_path / "task.md"
    prompt.write_text("Prepare and report one experiment.\n")
    task.write_text("Increment VALUE.\n")

    class CachedController:
        def __init__(self):
            self.cache, self.dispatches, self.last = {}, 0, None

        def __call__(self, number, candidate, manifest):
            key = (number, candidate, manifest)
            if key not in self.cache:
                self.dispatches += 1
                value = valid_receipt(candidate, manifest, self.dispatches)
                value["handle"] = f"remote:bz-a3-1:job:dispatch-{self.dispatches}"
                value["policy"]["submitted_handles"] = [value["handle"]]
                value["policy"]["observed_handles"] = [value["handle"]]
                self.cache[key] = value
            self.last = self.cache[key]
            return self.last

    base = CachedController()
    marker = tmp_path / "repair-injection.json"

    class CrashAfterMarker(canaries.InjectedController):
        crash = True

        def __call__(self, *args):
            result = super().__call__(*args)
            if self.crash and result.get("status") == "candidate_error":
                self.crash = False
                raise KeyboardInterrupt("after marker")
            return result

    controller = (CrashAfterMarker(base, "repair", marker)
                  if boundary == "after-marker"
                  else canaries.InjectedController(base, "repair", marker))
    repair_crash = boundary == "after-candidate-change"
    repair_value = 1
    repair_invocations = 0

    def invoke(number, session, instruction):
        nonlocal repair_crash, repair_value, repair_invocations
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return event_stream("thread-crash", {"prepared": True})
        if instruction.startswith("Repair candidate attempt") \
                or instruction.startswith("Resume blocked experiment"):
            repair_invocations += 1
            repair_value += 1
            (repo / "candidate.py").write_text(f"VALUE = {repair_value}\n")
            if repair_crash:
                repair_crash = False
                raise KeyboardInterrupt("after candidate change")
            return event_stream("thread-crash", {"repaired": True})
        candidate = hashlib.sha256((repo / "candidate.py").read_bytes()).hexdigest()
        manifest = hashlib.sha256((repo / "candidate.manifest.json").read_bytes()).hexdigest()
        report = {
            "hypothesis": "increment compiles", "expected_result": "valid kernel",
            "change": "increment VALUE", "evidence": "controller receipt",
            "observed_result": "profile completed", "decision": "retain",
            "postmortem": "bounded injected error was repaired", "next_experiment": "done",
            "candidate_sha256": candidate, "manifest_sha256": manifest,
            "controller_handle": base.last["handle"],
            "controller_receipt_sha256": audited_contract.sha256_json(base.last),
            "sources": [], "no_sources_reason": "self-contained canary",
        }
        return event_stream("thread-crash", report, command=False)

    runner = audited_lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
        max_candidate_repairs=2,
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run("canary", "repair")
    blocked = json.loads((repo / ".experiment/blocked.json").read_text())
    assert blocked["stage"] == ("controller" if boundary == "after-marker" else "prepare")

    resumed_controller = canaries.InjectedController(base, "repair", marker)
    result = audited_lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, resumed_controller, round_count=1,
        max_candidate_repairs=2,
    ).run("canary", "repair", resume=True)

    attempts = json.loads((repo / "experiments/01/evidence.json").read_text())[
        "candidate_attempts"
    ]
    handles = [
        json.loads((repo / f"experiments/01/attempts/{number:02d}/controller.json").read_text())[
            "handle"
        ] for number in (1, 2)
    ]
    assert result.status == "complete"
    assert [item["status"] for item in attempts] == ["candidate_error", "ok"]
    assert handles == ["remote:bz-a3-1:job:dispatch-1",
                       "remote:bz-a3-1:job:dispatch-2"]
    assert base.dispatches == 2
    if boundary == "after-candidate-change":
        assert repair_invocations == 1


def test_fresh_process_recovers_persisted_resume_checkpoint(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    run_root = tmp_path / "canary-run"
    cell = "matmul-cannbot-new-profiler-resume"
    repo = run_root / "cells" / cell / "repo"
    blocked = repo / ".experiment" / "blocked.json"
    blocked.parent.mkdir(parents=True)
    blocked.write_text(json.dumps({
        "schema": "profiling-skill/audited-blocked/v2", "round_count": 4,
        "experiment": 2, "stage": "controller", "session_id": "thread-resume",
        "candidate_sha256": "e" * 64, "manifest_sha256": "f" * 64,
        "receipt": {"handle": "remote:bz-a3-1:job:resume"},
    }, sort_keys=True) + "\n")
    marker = repo.parent / "state" / "canary-observer-interrupt.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({
        "schema": canaries.OBSERVER_SCHEMA,
        "job_request_sha256": "8" * 64,
        "controller_request_sha256": "9" * 64,
        "handle": "remote:bz-a3-1:job:resume",
        "candidate_sha256": "e" * 64, "manifest_sha256": "f" * 64,
        "target": "bz-a3-1", "device": 6, "job_identity": {"action": "check"},
    }) + "\n")
    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), run_root, FakePool(),
        launcher_factory=lambda cfg, mode: FakeLauncher(cfg, mode),
    )

    result = runner.run(output)

    resume = next(item for item in result["results"] if item["id"] == cell)
    assert resume["resume_receipt"] is not None
    actions = [action for canary, action in FakeLauncher.calls if canary == cell]
    assert actions == ["observe"]


def test_later_infrastructure_resume_preserves_original_resume_proof(tmp_path: Path):
    definition, config, output = inputs(tmp_path)
    run_root, cell = tmp_path / "canary-run", "matmul-cannbot-new-profiler-resume"
    repo = run_root / "cells" / cell / "repo"
    blocked = repo / ".experiment" / "blocked.json"
    blocked.parent.mkdir(parents=True)
    blocked.write_text(json.dumps({
        "schema": "profiling-skill/audited-blocked/v2", "round_count": 4,
        "experiment": 3, "stage": "controller", "session_id": "thread-resume",
        "candidate_sha256": "1" * 64, "manifest_sha256": "2" * 64,
        "receipt": {"handle": "remote:bz-a3-1:job:later"},
    }, sort_keys=True) + "\n")
    artifact_root = run_root / "artifacts" / cell
    artifact_root.mkdir(parents=True)
    original_checkpoint = artifact_root / "original-checkpoint.json"
    original_checkpoint.write_text(json.dumps({
        "schema": "profiling-skill/audited-blocked/v2", "round_count": 4,
        "experiment": 2, "stage": "controller", "session_id": "thread-resume",
        "candidate_sha256": "e" * 64, "manifest_sha256": "f" * 64,
        "receipt": {"handle": "remote:bz-a3-1:job:resume"},
    }, sort_keys=True) + "\n")
    original_marker = artifact_root / "original-observer-marker.json"
    original_marker.write_text(json.dumps({
        "schema": canaries.OBSERVER_SCHEMA,
        "job_request_sha256": "8" * 64,
        "controller_request_sha256": "9" * 64,
        "handle": "remote:bz-a3-1:job:resume", "candidate_sha256": "e" * 64,
        "manifest_sha256": "f" * 64, "target": "bz-a3-1", "device": 6,
        "job_identity": {"action": "check"},
    }, sort_keys=True) + "\n")
    intent = artifact_root / "resume-intent.json"
    intent.write_text(json.dumps({
        "schema": "profiling-skill/audited-repair-canary-resume-intent/v1",
        "canary_id": cell,
        "checkpoint": {"path": str(original_checkpoint),
                       "sha256": sha(original_checkpoint)},
        "marker": {"path": str(original_marker), "sha256": sha(original_marker)},
    }, sort_keys=True) + "\n")

    class LaterLauncher(FakeLauncher):
        def launch(self, target, slot):
            if target["cell_id"] != cell:
                return super().launch(target, slot)
            self.calls.append((target["cell_id"], "launch"))
            (self._repo(target) / ".experiment" / "blocked.json").unlink()
            return self._receipt(target)

    FakeLauncher.calls = []
    runner = canaries.CanaryRunner(
        config, definition, sha(definition), run_root, FakePool(),
        launcher_factory=lambda cfg, mode: LaterLauncher(cfg, mode),
    )
    runner.run(output)

    actions = [action for canary, action in FakeLauncher.calls if canary == cell]
    assert actions == ["launch"]
    resume = json.loads((artifact_root / "resume.json").read_text())
    assert resume["checkpoint_sha256"] == sha(original_checkpoint)
