from __future__ import annotations

import fcntl
import importlib.util
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audited_campaign", ROOT / "scripts" / "audited_campaign.py"
)
audited_campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = audited_campaign
SPEC.loader.exec_module(audited_campaign)


def test_run_campaign_holds_exclusive_ledger_lock(tmp_path: Path, monkeypatch):
    ledger = tmp_path / "ledger.json"

    def assert_locked(*args, **kwargs):
        del args, kwargs
        lock_path = ledger.with_name(f".{ledger.name}.lock")
        with lock_path.open("a+b") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"status": "locked"}

    monkeypatch.setattr(audited_campaign, "_run_campaign_locked", assert_locked)

    assert audited_campaign.run_campaign({}, ledger, None, None) == {"status": "locked"}


def inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("invariant prompt\n")
    tasks = {}
    for name in ("matmul", "gdn", "bsa"):
        path = tmp_path / f"{name}.md"
        path.write_text(f"optimize {name}\n")
        tasks[name] = path
    provenance = {
        "source_revision": "a" * 40,
        "controller_sha256": "b" * 64,
        "runtime_image_digest": "sha256:" + "c" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "baselines": {name: "d" * 64 for name in tasks},
        "starters": {
            name: {
                "candidate": {"path": f"/frozen/{name}/candidate.py", "sha256": "2" * 64},
                "manifest": {
                    "path": f"/frozen/{name}/candidate.manifest.json",
                    "sha256": "3" * 64,
                },
            }
            for name in tasks
        },
        "skills": {
            "cannbot": "e" * 64,
            "ascend-profiling": "f" * 64,
            "triton-guarded-kernel": "1" * 64,
        },
    }
    return prompt, tasks, provenance


def manifest(tmp_path: Path, seed: str = "campaign-1") -> dict:
    prompt, tasks, provenance = inputs(tmp_path)
    return audited_campaign.build_manifest(
        run_id="run-2026-10-06",
        prompt=prompt,
        task_files=tasks,
        provenance=provenance,
        ordering_seed=seed,
        rounds=4,
        request_budget=24,
    )


def test_manifest_is_exact_three_by_three_with_four_rounds(tmp_path: Path):
    document = manifest(tmp_path)
    assert document["schema_version"] == 2
    cells = document["cells"]
    assert len(cells) == 9
    assert {(cell["task"], cell["treatment"]) for cell in cells} == {
        (task, treatment)
        for task in audited_campaign.TASKS
        for treatment in audited_campaign.TREATMENTS
    }
    assert {cell["round_count"] for cell in cells} == {4}
    assert {cell["request_budget"] for cell in cells} == {24}
    assert len({cell["branch"] for cell in cells}) == 9
    assert all("device" not in cell and "target" not in cell for cell in cells)


def test_new_manifest_defaults_to_repair_aware_operation_budget(tmp_path: Path):
    prompt, tasks, provenance = inputs(tmp_path)
    document = audited_campaign.build_manifest(
        "repair-aware", prompt, tasks, provenance, "seed",
    )
    assert document["request_budget"] == 48
    assert {cell["request_budget"] for cell in document["cells"]} == {48}
    audited_campaign.verify_manifest(document)


@pytest.mark.parametrize("budget", [0, 23, 25, 47, 49])
def test_manifest_rejects_nonproduction_operation_budgets(tmp_path: Path, budget: int):
    prompt, tasks, provenance = inputs(tmp_path)
    with pytest.raises(audited_campaign.CampaignError, match="24 or 48"):
        audited_campaign.build_manifest(
            "bad-budget", prompt, tasks, provenance, "seed", request_budget=budget,
        )


def test_order_is_deterministic_fair_and_recorded(tmp_path: Path):
    first = manifest(tmp_path / "a")
    second = manifest(tmp_path / "b")
    assert first["order"] == second["order"]
    assert first["ordering"]["algorithm"] == "balanced-latin-v1"
    for start in range(0, 9, 3):
        block = [
            next(cell for cell in first["cells"] if cell["cell_id"] == cell_id)
            for cell_id in first["order"][start : start + 3]
        ]
        assert {cell["task"] for cell in block} == set(audited_campaign.TASKS)
        assert {cell["treatment"] for cell in block} == set(
            audited_campaign.TREATMENTS
        )


def test_treatment_visibility_is_exact_and_task_prompt_is_treatment_independent(
    tmp_path: Path,
):
    document = manifest(tmp_path)
    by_task = {}
    for cell in document["cells"]:
        by_task.setdefault(cell["task"], set()).add(cell["task_sha256"])
        assert cell["skills"] == list(
            audited_campaign.TREATMENT_SKILLS[cell["treatment"]]
        )
    assert all(len(digests) == 1 for digests in by_task.values())
    assert not any("device" in json.dumps(cell["prompt_contract"]).lower()
                   for cell in document["cells"])


class StaticPool:
    def __init__(self, slots):
        self.slots = slots
        self.calls = 0

    def admit(self):
        self.calls += 1
        return self.slots


class RecordingLauncher:
    def __init__(self, fail_once: str | None = None):
        self.calls = []
        self.fail_once = fail_once

    def launch(self, cell, slot):
        self.calls.append((cell["cell_id"], slot.copy()))
        if cell["cell_id"] == self.fail_once:
            self.fail_once = None
            raise audited_campaign.InfrastructureFailure("temporary transport loss")
        return {
            "status": "complete",
            "durable_handle": f"{slot['target']}:job-{cell['cell_id']}",
            "rounds_completed": 4,
            "baseline_median_us": 12.0,
            "rounds": [
                {
                    "round": round_number,
                    "status": "ok",
                    "handle": f"{slot['target']}:round-{round_number}",
                    "median_us": 10.0 - round_number,
                    "normalized_median_us": 9.0 - round_number,
                    "samples_us": [10.0 - round_number] * 3,
                    "case_results": [{"case": 7, "median_us": 10.0 - round_number}],
                    "controls": {"before_us": 10.0, "after_us": 10.1},
                    "policy": {"post_control": "pass"},
                }
                for round_number in range(1, 5)
            ],
            "commits": [f"commit-{number}" for number in range(1, 5)],
        }


class HandleRecoveryLauncher(RecordingLauncher):
    def __init__(self, failing_cell):
        super().__init__()
        self.failing_cell = failing_cell
        self.observe_calls = []

    def launch(self, cell, slot):
        if cell["cell_id"] == self.failing_cell:
            self.calls.append((cell["cell_id"], slot.copy()))
            self.failing_cell = None
            raise audited_campaign.InfrastructureFailure(
                "observer disconnected", f"{slot['target']}:retained-123"
            )
        return super().launch(cell, slot)

    def observe(self, cell, placement, durable_handle):
        self.observe_calls.append((cell["cell_id"], placement, durable_handle))
        return {
            "status": "complete", "durable_handle": durable_handle,
            "rounds_completed": 4,
            "rounds": [{"round": number, "status": "ok",
                        "handle": f"handle-{number}",
                        "median_us": 20.0 - number}
                       for number in range(1, 5)],
            "commits": [f"commit-{number}" for number in range(1, 5)],
        }


def test_dynamic_admission_uses_all_unique_healthy_idle_devices(tmp_path: Path):
    document = manifest(tmp_path)
    pool = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
        {"target": "bz-a3-1", "device": 1, "healthy": False, "idle": True},
        {"target": "bz-a3-2", "device": 2, "healthy": True, "idle": True},
        {"target": "bz-a3-2", "device": 3, "healthy": True, "idle": False},
        {"target": "bz-a3-2", "device": 2, "healthy": True, "idle": True},
    ])
    launcher = RecordingLauncher()
    ledger = audited_campaign.run_campaign(
        document, tmp_path / "ledger.json", pool, launcher
    )
    assert ledger["status"] == "complete"
    assert len(launcher.calls) == 9
    assert {call[1]["target"] for call in launcher.calls} == {
        "bz-a3-1", "bz-a3-2"
    }
    assert all(entry["status"] == "complete" for entry in ledger["cells"].values())
    assert all(entry["attempts"][0]["durable_handle"] for entry in ledger["cells"].values())


def test_scheduler_refills_each_free_slot_without_waiting_for_batch(tmp_path: Path):
    document = manifest(tmp_path)
    first_two = document["order"][:2]
    started = {cell_id: threading.Event() for cell_id in first_two}
    releases = {cell_id: threading.Event() for cell_id in first_two}
    third_started = threading.Event()

    class BlockingLauncher(RecordingLauncher):
        def launch(self, cell, slot):
            cell_id = cell["cell_id"]
            if cell_id in started:
                started[cell_id].set()
                assert releases[cell_id].wait(5)
            else:
                third_started.set()
            return super().launch(cell, slot)

    pool = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
        {"target": "bz-a3-2", "device": 1, "healthy": True, "idle": True},
    ])
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            audited_campaign.run_campaign, document, tmp_path / "ledger.json",
            pool, BlockingLauncher(),
        )
        assert all(event.wait(5) for event in started.values())
        releases[first_two[0]].set()
        assert third_started.wait(5)
        assert not releases[first_two[1]].is_set()
        releases[first_two[1]].set()
        assert running.result(timeout=5)["status"] == "complete"
    assert pool.calls >= 3


def test_resume_preserves_completed_and_retries_only_infrastructure_failure(
    tmp_path: Path,
):
    document = manifest(tmp_path)
    slots = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
        {"target": "bz-a3-2", "device": 1, "healthy": True, "idle": True},
    ])
    failed_cell = document["order"][1]
    first = RecordingLauncher(fail_once=failed_cell)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(document, ledger_path, slots, first)
    checkpoint = json.loads(ledger_path.read_text())
    completed = {
        cell_id for cell_id, state in json.loads(ledger_path.read_text())["cells"].items()
        if state["status"] == "complete"
    }
    assert len(completed) == 8
    assert checkpoint["cells"][failed_cell]["status"] == "infrastructure_pending"
    resumed = RecordingLauncher()
    ledger = audited_campaign.run_campaign(
        document, ledger_path, slots, resumed, resume=True
    )
    assert ledger["status"] == "complete"
    assert completed.isdisjoint({cell_id for cell_id, _ in resumed.calls})
    assert failed_cell in {cell_id for cell_id, _ in resumed.calls}
    assert len(ledger["cells"][failed_cell]["attempts"]) == 2


def test_resume_observes_existing_durable_handle_without_redispatch(tmp_path: Path):
    document = manifest(tmp_path)
    pool = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
    ])
    target = document["order"][0]
    launcher = HandleRecoveryLauncher(target)
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(document, ledger_path, pool, launcher)
    launches_before = [cell for cell, _ in launcher.calls].count(target)
    ledger = audited_campaign.run_campaign(
        document, ledger_path, pool, launcher, resume=True
    )
    assert ledger["status"] == "complete"
    assert [cell for cell, _ in launcher.calls].count(target) == launches_before
    assert launcher.observe_calls == [
        (target, {"target": "bz-a3-1", "device": 0},
         "bz-a3-1:retained-123")
    ]
    report = audited_campaign.build_report(document, ledger)
    assert report["discarded_infrastructure_attempts"][0]["durable_handle"] == (
        "bz-a3-1:retained-123"
    )


def test_report_contains_evolution_best_round_and_failures(tmp_path: Path):
    document = manifest(tmp_path)
    pool = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
    ])
    ledger = audited_campaign.run_campaign(
        document, tmp_path / "ledger.json", pool, RecordingLauncher()
    )
    report = audited_campaign.build_report(document, ledger)
    assert report["schema_version"] == 2
    assert report["summary"] == {"complete": 9, "candidate_failed": 0,
                                  "infrastructure_pending": 0}
    assert len(report["cells"]) == 9
    assert all(row["best_round"] == 4 and row["best_median_us"] == 6.0
               for row in report["cells"])
    assert all(row["speedup_vs_baseline"] == 12.0 / 5.0 for row in report["cells"])
    assert all(len(row["raw_evolution"]) == 4 for row in report["cells"])
    assert all(len(row["per_case_evidence"]) == 4 for row in report["cells"])
    assert all(len(row["controls"]) == 4 for row in report["cells"])
    assert all(row["best_normalized_median_us"] == 5.0 for row in report["cells"])
    assert report["discarded_infrastructure_attempts"] == []


def test_report_separates_raw_attempt_and_repaired_round_success(tmp_path: Path):
    document = manifest(tmp_path)
    ledger = audited_campaign._new_ledger(document)
    for cell_id in document["order"]:
        ledger["cells"][cell_id] = {
            "status": "complete", "attempts": [{"status": "complete", "receipt": {
                "status": "complete", "durable_handle": f"local:{cell_id}",
                "rounds_completed": 4, "commits": ["1", "2", "3", "4"],
                "rounds": [
                    {"round": 1, "status": "ok", "handle": "h1", "median_us": 9.0,
                     "attempt_statuses": ["compile_error", "ok"]},
                    {"round": 2, "status": "ok", "handle": "h2", "median_us": 8.0,
                     "attempt_statuses": ["ok"]},
                    {"round": 3, "status": "ok", "handle": "h3", "median_us": 7.0,
                     "attempt_statuses": ["runtime_error", "correctness_error", "ok"]},
                    {"round": 4, "status": "ok", "handle": "h4", "median_us": 6.0,
                     "attempt_statuses": ["ok"]},
                ],
            }}],
        }
    report = audited_campaign.build_report(document, ledger)
    row = report["cells"][0]
    assert row["attempt_summary"] == {
        "attempts": 7, "successful_attempts": 4, "raw_attempt_success_rate": 4 / 7,
        "successful_rounds": 4, "repair_attempted_rounds": 2,
        "repaired_rounds": 2, "repaired_round_success_rate": 1.0,
        "repair_count": 3,
        "failure_transitions": [
            {"round": 1, "from": "compile_error", "to": "ok"},
            {"round": 3, "from": "runtime_error", "to": "correctness_error"},
            {"round": 3, "from": "correctness_error", "to": "ok"},
        ],
        "final_timing": {"round": 4, "median_us": 6.0,
                         "normalized_median_us": None},
    }
    assert report["attempt_summary"]["attempts"] == 63
    assert report["attempt_summary"]["repaired_rounds"] == 18


def test_report_retains_discarded_infrastructure_attempts(tmp_path: Path):
    document = manifest(tmp_path)
    pool = StaticPool([
        {"target": "bz-a3-1", "device": 0, "healthy": True, "idle": True},
    ])
    failed = document["order"][0]
    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(
            document, ledger_path, pool, RecordingLauncher(fail_once=failed)
        )
    audited_campaign.run_campaign(
        document, ledger_path, pool, RecordingLauncher(), resume=True
    )
    report = audited_campaign.build_report(
        document, json.loads(ledger_path.read_text())
    )
    assert report["discarded_infrastructure_attempts"] == [{
        "cell_id": failed, "attempt": 1, "target": "bz-a3-1", "device": 0,
        "durable_handle": None, "error": "temporary transport loss",
    }]


@pytest.mark.parametrize("malformation", ["partial", "no-candidate-error"])
def test_malformed_candidate_failure_remains_infrastructure_pending(
    tmp_path: Path, malformation: str,
):
    document = manifest(tmp_path)

    class MalformedLauncher:
        def launch(self, cell, slot):
            count = 2 if malformation == "partial" else 4
            return {
                "status": "candidate_failed", "durable_handle": "local:failed",
                "rounds_completed": count,
                "rounds": [{"round": number, "status": "ok",
                            "handle": f"local:round-{number}"}
                           for number in range(1, count + 1)],
                "commits": [f"commit-{number}" for number in range(1, count + 1)],
            }

    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(
            document, ledger_path,
            StaticPool([{"target": "bz-a3-1", "device": 0,
                         "healthy": True, "idle": True}]),
            MalformedLauncher(),
        )
    ledger = json.loads(ledger_path.read_text())
    assert ledger["cells"][document["order"][0]]["status"] == "infrastructure_pending"


def test_complete_without_four_commits_remains_infrastructure_pending(tmp_path: Path):
    document = manifest(tmp_path)

    class MissingCommitLauncher(RecordingLauncher):
        def launch(self, cell, slot):
            receipt = super().launch(cell, slot)
            receipt["commits"].pop()
            return receipt

    ledger_path = tmp_path / "ledger.json"
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(
            document, ledger_path,
            StaticPool([{"target": "bz-a3-1", "device": 0,
                         "healthy": True, "idle": True}]),
            MissingCommitLauncher(),
        )
    ledger = json.loads(ledger_path.read_text())
    assert ledger["cells"][document["order"][0]]["status"] == "infrastructure_pending"


def test_malformed_terminal_receipt_relaunches_instead_of_observing(tmp_path: Path):
    document = manifest(tmp_path)
    target = document["order"][0]

    class RepairingLauncher(RecordingLauncher):
        def __init__(self):
            super().__init__()
            self.malformed = True
            self.observe_calls = []

        def launch(self, cell, slot):
            receipt = super().launch(cell, slot)
            if cell["cell_id"] == target and self.malformed:
                self.malformed = False
                receipt["commits"].pop()
            return receipt

        def observe(self, cell, placement, durable_handle):
            self.observe_calls.append((cell["cell_id"], durable_handle))
            raise AssertionError("malformed terminal evidence must be reconstructed")

    launcher = RepairingLauncher()
    ledger_path = tmp_path / "ledger.json"
    pool = StaticPool([{"target": "bz-a3-1", "device": 0,
                        "healthy": True, "idle": True}])
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(document, ledger_path, pool, launcher)
    failed_attempt = json.loads(ledger_path.read_text())["cells"][target]["attempts"][0]
    assert "durable_handle" not in failed_attempt

    ledger = audited_campaign.run_campaign(
        document, ledger_path, pool, launcher, resume=True
    )
    assert ledger["status"] == "complete"
    assert [cell_id for cell_id, _ in launcher.calls].count(target) == 2
    assert launcher.observe_calls == []


def test_malformed_observation_drops_retained_handle_before_relaunch(tmp_path: Path):
    document = manifest(tmp_path)
    target = document["order"][0]

    class MalformedObservationLauncher(HandleRecoveryLauncher):
        def observe(self, cell, placement, durable_handle):
            receipt = super().observe(cell, placement, durable_handle)
            receipt["commits"].pop()
            return receipt

    launcher = MalformedObservationLauncher(target)
    ledger_path = tmp_path / "ledger.json"
    pool = StaticPool([{"target": "bz-a3-1", "device": 0,
                        "healthy": True, "idle": True}])
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(document, ledger_path, pool, launcher)
    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(
            document, ledger_path, pool, launcher, resume=True
        )
    latest = json.loads(ledger_path.read_text())["cells"][target]["attempts"][-1]
    assert "durable_handle" not in latest

    ledger = audited_campaign.run_campaign(
        document, ledger_path, pool, launcher, resume=True
    )
    assert ledger["status"] == "complete"
    assert [cell_id for cell_id, _ in launcher.calls].count(target) == 2


def test_resume_does_not_skip_running_dispatch_without_retained_handle(tmp_path: Path):
    document = manifest(tmp_path)
    target = document["order"][0]
    ledger_path = tmp_path / "ledger.json"
    ledger = audited_campaign._new_ledger(document)
    ledger["cells"][target] = {
        "status": "running",
        "attempts": [{"target": "bz-a3-1", "device": 0, "status": "running"}],
    }
    ledger_path.write_text(json.dumps(ledger))
    launcher = RecordingLauncher()

    with pytest.raises(audited_campaign.CampaignPaused):
        audited_campaign.run_campaign(
            document, ledger_path,
            StaticPool([{"target": "bz-a3-1", "device": 0,
                         "healthy": True, "idle": True}]),
            launcher, resume=True,
        )

    checkpoint = json.loads(ledger_path.read_text())
    assert checkpoint["status"] == "infrastructure_pending"
    assert checkpoint["cells"][target]["status"] == "infrastructure_pending"
    assert checkpoint["cells"][target]["attempts"][0]["retryable"] is False
    assert target not in {cell_id for cell_id, _ in launcher.calls}
    assert sum(state["status"] == "complete"
               for state in checkpoint["cells"].values()) == 8


def test_report_aggregates_all_candidate_error_rounds(tmp_path: Path):
    document = manifest(tmp_path)

    class CandidateFailureLauncher:
        def launch(self, cell, slot):
            rounds = [
                {"round": 1, "status": "candidate_error", "handle": "h1",
                 "failure_type": "compile_error", "reason": "bad tl.load"},
                {"round": 2, "status": "ok", "handle": "h2"},
                {"round": 3, "status": "candidate_error", "handle": "h3",
                 "failure_type": "correctness_error", "reason": "mismatch"},
                {"round": 4, "status": "ok", "handle": "h4"},
            ]
            return {
                "status": "candidate_failed", "durable_handle": "h4",
                "rounds_completed": 4, "rounds": rounds,
                "commits": ["c1", "c2", "c3", "c4"],
            }

    ledger = audited_campaign.run_campaign(
        document, tmp_path / "ledger.json",
        StaticPool([{"target": "bz-a3-1", "device": 0,
                     "healthy": True, "idle": True}]),
        CandidateFailureLauncher(),
    )
    report = audited_campaign.build_report(document, ledger)
    assert report["summary"]["candidate_failed"] == 9
    assert report["cells"][0]["candidate_errors"] == [
        {"round": 1, "failure_type": "compile_error", "reason": "bad tl.load"},
        {"round": 3, "failure_type": "correctness_error", "reason": "mismatch"},
    ]


def test_report_rejects_ledger_from_another_run_before_consuming_cells(tmp_path: Path):
    first = manifest(tmp_path / "first")
    second = manifest(tmp_path / "second", seed="campaign-2")
    second["run_id"] = "different-run"
    second["manifest_sha256"] = audited_campaign._document_digest(second)
    ledger = audited_campaign._new_ledger(first)

    with pytest.raises(audited_campaign.CampaignError, match="ledger run_id"):
        audited_campaign.build_report(second, ledger)


def test_report_prefers_controller_normalized_receipt_for_comparison(tmp_path: Path):
    document = manifest(tmp_path)
    baseline = {
        "schema": "profiling-skill/baseline-timing/v1",
        "benchmark": "matmul", "case_medians_us": [{"case": 7, "median_us": 16.0}],
        "control_median_us": 10.0, "sha256": "a" * 64,
    }

    class NormalizedLauncher(RecordingLauncher):
        def launch(self, cell, slot):
            rounds = []
            for number, raw, normalized in [
                (1, 5.0, 10.0), (2, 6.0, 8.0),
                (3, 7.0, 9.0), (4, 8.0, 11.0),
            ]:
                rounds.append({
                    "round": number, "median_us": raw,
                    "status": "ok", "handle": f"round-{number}",
                    "samples_us": [raw] * 3,
                    "normalized_samples_us": [normalized] * 3,
                    "normalized_median_us": normalized,
                    "baseline_median_us": 16.0, "baseline": baseline,
                    "speedup_vs_baseline": 16.0 / normalized,
                    "calibration": {
                        "before": {"median_us": 10.0},
                        "after": {"median_us": 10.0},
                        "normalization_factor": normalized / raw,
                    },
                    "policy": {"post_control": "pass"},
                    "case_results": [{"case": 7, "median_us": raw}],
                })
            return {
                "status": "complete", "durable_handle": f"{slot['target']}:job",
                "rounds_completed": 4, "rounds": rounds,
                "commits": ["c1", "c2", "c3", "c4"],
            }

    ledger = audited_campaign.run_campaign(
        document, tmp_path / "ledger.json",
        StaticPool([{"target": "bz-a3-1", "device": 0,
                     "healthy": True, "idle": True}]),
        NormalizedLauncher(),
    )
    row = audited_campaign.build_report(document, ledger)["cells"][0]
    assert row["best_round"] == 2
    assert row["best_median_us"] == 5.0
    assert row["best_normalized_median_us"] == 8.0
    assert row["comparison_basis"] == "calibration_normalized_median_us"
    assert row["comparison_median_us"] == 8.0
    assert row["speedup_vs_baseline"] == 2.0
    assert row["baseline"] == baseline
    expected_calibration = {
        "before": {"median_us": 10.0}, "after": {"median_us": 10.0},
        "normalization_factor": 8.0 / 6.0,
    }
    assert row["normalized_evolution"][1] == {
        "round": 2, "normalized_samples_us": [8.0, 8.0, 8.0],
        "normalized_median_us": 8.0, "speedup_vs_baseline": 2.0,
        "baseline_median_us": 16.0,
        "calibration": expected_calibration,
    }
    assert row["controls"][1]["calibration"] == expected_calibration


def test_manifest_rejects_unpinned_provenance_and_wrong_dimensions(tmp_path: Path):
    prompt, tasks, provenance = inputs(tmp_path)
    provenance["controller_sha256"] = "mutable"
    with pytest.raises(audited_campaign.CampaignError, match="controller_sha256"):
        audited_campaign.build_manifest(
            "run", prompt, tasks, provenance, "seed", rounds=4, request_budget=24
        )
    _, _, provenance = inputs(tmp_path / "other")
    with pytest.raises(audited_campaign.CampaignError, match="exactly four rounds"):
        audited_campaign.build_manifest(
            "run", prompt, tasks, provenance, "seed", rounds=3, request_budget=24
        )


@pytest.mark.parametrize("mutation", ["missing-task", "relative-path", "bad-hash"])
def test_manifest_rejects_unpinned_task_starters(tmp_path: Path, mutation: str):
    prompt, tasks, provenance = inputs(tmp_path)
    if mutation == "missing-task":
        del provenance["starters"]["bsa"]
    elif mutation == "relative-path":
        provenance["starters"]["gdn"]["candidate"]["path"] = "candidate.py"
    else:
        provenance["starters"]["matmul"]["manifest"]["sha256"] = "mutable"

    with pytest.raises(audited_campaign.CampaignError, match="starter"):
        audited_campaign.build_manifest(
            "run", prompt, tasks, provenance, "seed", rounds=4, request_budget=24
        )


def test_legacy_v1_manifest_without_starters_still_verifies_and_reports(tmp_path: Path):
    document = manifest(tmp_path)
    document["schema_version"] = 1
    del document["provenance"]["starters"]
    document["manifest_sha256"] = audited_campaign._document_digest(document)
    audited_campaign.verify_manifest(document)
    ledger = audited_campaign._new_ledger(document)
    report = audited_campaign.build_report(document, ledger)
    assert report["manifest_sha256"] == document["manifest_sha256"]


def test_v2_manifest_requires_starter_provenance(tmp_path: Path):
    document = manifest(tmp_path)
    del document["provenance"]["starters"]
    document["manifest_sha256"] = audited_campaign._document_digest(document)
    with pytest.raises(audited_campaign.CampaignError, match="starter"):
        audited_campaign.verify_manifest(document)


def test_cli_fake_controller_end_to_end(tmp_path: Path):
    prompt, tasks, provenance = inputs(tmp_path / "inputs")
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.json"
    report_path = tmp_path / "report.json"
    script = ROOT / "scripts" / "audited_campaign.py"
    subprocess.run([
        sys.executable, str(script), "generate", "--run-id", "cli-e2e",
        "--prompt", str(prompt), "--matmul-task", str(tasks["matmul"]),
        "--gdn-task", str(tasks["gdn"]), "--bsa-task", str(tasks["bsa"]),
        "--provenance", str(provenance_path), "--ordering-seed", "fixed",
        "--output", str(manifest_path),
    ], check=True)
    subprocess.run([
        sys.executable, str(script), "simulate", "--manifest", str(manifest_path),
        "--ledger", str(ledger_path), "--slots", "3",
    ], check=True)
    subprocess.run([
        sys.executable, str(script), "report", "--manifest", str(manifest_path),
        "--ledger", str(ledger_path), "--output", str(report_path),
    ], check=True)
    assert json.loads(ledger_path.read_text())["status"] == "complete"
    assert json.loads(report_path.read_text())["summary"]["complete"] == 9
