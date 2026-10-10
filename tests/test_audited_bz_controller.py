import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load("audited_bz_controller")
contract = load("audited_contract")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "candidate.py").write_text("VALUE = 1\n")
    (repo / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"kernel"}\n'
    )
    return repo, digest(repo / "candidate.py"), digest(repo / "candidate.manifest.json")


class FakeBackend:
    def __init__(self, overrides=None):
        self.requests = []
        self.overrides = list(overrides or [])
        self.serial = 0

    def __call__(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        self.serial += 1
        if self.overrides:
            override = self.overrides.pop(0)
            if override is not None:
                return override
        handle = f"bz-a3-1:job-{self.serial}"
        if request["action"] == "calibrate":
            return {"status": "ok", "handle": handle, "latency_us": 10.0,
                    "samples_us": [9.0, 10.0, 11.0], "median_us": 10.0,
                    "artifacts": {"remote_profile_evidence": "/remote/calibration.json"}}
        if request["action"] == "check":
            return {"status": "ok", "handle": handle, "passed": True}
        rows = [
            {"case": case, "samples_us": [10.0 + index, 20.0 + index, 40.0 + index],
             "median_us": 20.0 + index}
            for index, case in enumerate(request["cases"])
        ]
        return {"status": "ok", "handle": handle, "cases": rows,
                "kernel_name": "kernel",
                "fusion_gate": {
                    "entrypoint": "kernel", "kernel_name": "kernel",
                    "cases": list(range(10)), "logical_launches_per_case": 1,
                },
                "fusion_handle": f"{handle}:full-domain-gate",
                "artifacts": {"remote_profile_evidence": "/remote/evidence.json"}}


def baseline(values=(30.0, 33.0, 36.0), control=20.0):
    document = {
        "schema": "profiling-skill/baseline-timing/v1", "benchmark": "matmul",
        "case_medians_us": [
            {"case": case, "median_us": value}
            for case, value in zip([7, 8, 9], values)
        ],
        "control_median_us": control,
    }
    document["sha256"] = hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return document


def make_controller(repo: Path, backend, **options):
    settings = {"variability_threshold": 2.0, "request_budget": 24,
                "baseline": baseline(), **options}
    return controller.AuditedBzController(
        repo=repo,
        state_dir=repo.parent / "state",
        benchmark="matmul",
        backend=backend,
        devices=[{"id": "bz-a3-1/device-0", "device": 0}],
        development_cases=[7, 8, 9],
        all_cases=list(range(10)),
        round_count=4,
        **settings,
    )


def test_success_runs_controls_full_final_check_and_three_profile_repetitions(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    receipt = make_controller(repo, backend).run(4, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert [request["action"] for request in backend.requests] == [
        "calibrate", "check", "profile", "calibrate",
    ]
    assert backend.requests[1]["scope"] == "full"
    assert backend.requests[1]["cases"] == list(range(10))
    assert backend.requests[2]["repeats"] == 3
    assert receipt["case_results"][0]["case"] == 7
    assert receipt["declared_kernel_name"] == "kernel"
    assert receipt["resolved_kernel_name"] == "kernel"
    assert receipt["policy"]["primary"]["declared_kernel_name"] == "kernel"
    assert receipt["policy"]["primary"]["resolved_kernel_name"] == "kernel"
    assert receipt["fusion_handle"] == "bz-a3-1:job-3:full-domain-gate"
    assert receipt["fusion_gate"]["cases"] == list(range(10))
    assert receipt["policy"]["primary"]["fusion_handle"] == receipt["fusion_handle"]
    profile_operation = next(
        row for row in receipt["policy"]["operation_history"]
        if row["action"] == "profile"
    )
    assert profile_operation["fusion_handle"] == receipt["fusion_handle"]
    assert profile_operation["fusion_gate"] == receipt["fusion_gate"]
    expected = [
        math.exp(sum(math.log(value) for value in sample) / 3)
        for sample in zip([10.0, 20.0, 40.0], [11.0, 21.0, 41.0], [12.0, 22.0, 42.0])
    ]
    assert receipt["samples_us"] == expected
    assert receipt["policy"]["operations_consumed"] == 4
    assert receipt["compact_artifacts"] == ["/remote/evidence.json"]
    expected_baseline = math.exp(sum(math.log(value) for value in [30.0, 33.0, 36.0]) / 3)
    assert receipt["baseline_median_us"] == expected_baseline
    assert receipt["baseline"]["sha256"] == baseline()["sha256"]
    assert receipt["calibration"]["before"]["median_us"] == 10.0
    assert receipt["calibration"]["after"]["median_us"] == 10.0
    assert receipt["normalized_median_us"] == receipt["median_us"] * 2
    assert receipt["speedup_vs_baseline"] == (
        receipt["baseline_median_us"] / receipt["normalized_median_us"]
    )

    missing_history = json.loads(json.dumps(receipt))
    operation = next(
        row for row in missing_history["policy"]["operation_history"]
        if row["action"] == "profile"
    )
    operation.pop("fusion_gate")
    operation.pop("fusion_handle")
    with pytest.raises(contract.AuditError, match="operation history"):
        contract.validate_controller_receipt(
            missing_history, candidate_hash, manifest_hash,
        )

    unhashable_cases = json.loads(json.dumps(receipt))
    unhashable_cases["fusion_gate"]["cases"] = [[0]]
    with pytest.raises(contract.AuditError, match="fusion authorization"):
        contract.validate_controller_receipt(
            unhashable_cases, candidate_hash, manifest_hash,
        )


def test_compile_error_is_counted_without_profile_or_post_control(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend([
        None,
        {"status": "compile_error", "failure_type": "compile_error",
         "diagnostics": "bad tl.load", "handle": "bz-a3-1:compile"},
    ])
    receipt = make_controller(repo, backend).run(1, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "candidate_error"
    assert receipt["failure_type"] == "compile_error"
    assert receipt["policy"]["post_control"] == "not_run"
    assert receipt["policy"]["operations_consumed"] == 2
    assert [request["action"] for request in backend.requests] == ["calibrate", "check"]
    assert not ({"samples_us", "median_us", "baseline_median_us", "baseline",
                 "calibration", "normalized_median_us", "speedup_vs_baseline"} & receipt.keys())


def test_rejects_baseline_hash_or_case_value_mismatch(tmp_path: Path):
    repo, _candidate_hash, _manifest_hash = repository(tmp_path)
    wrong_hash = baseline()
    wrong_hash["sha256"] = "0" * 64
    with pytest.raises(controller.ControllerError, match="baseline sha256"):
        make_controller(repo, FakeBackend(), baseline=wrong_hash)
    wrong_cases = baseline()
    wrong_cases["case_medians_us"][0]["case"] = 6
    wrong_cases["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in wrong_cases.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    with pytest.raises(controller.ControllerError, match="baseline cases"):
        make_controller(repo, FakeBackend(), baseline=wrong_cases)


def test_receipt_validation_binds_baseline_and_normalized_values(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    receipt = make_controller(repo, FakeBackend()).run(
        1, candidate_hash, manifest_hash)
    tampered_hash = json.loads(json.dumps(receipt))
    tampered_hash["baseline"]["sha256"] = "0" * 64
    with pytest.raises(contract.AuditError, match="baseline timing hash"):
        contract.validate_controller_receipt(tampered_hash, candidate_hash, manifest_hash)
    tampered_value = json.loads(json.dumps(receipt))
    tampered_value["normalized_median_us"] *= 2
    with pytest.raises(contract.AuditError, match="normalized timing"):
        contract.validate_controller_receipt(tampered_value, candidate_hash, manifest_hash)


def test_infrastructure_handle_is_checkpointed_and_observed_without_budget_charge(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend([
        None,
        {"status": "infrastructure_error", "failure_type": "observer_error",
         "diagnostics": "observer disconnected", "handle": "bz-a3-1:retained"},
        {"status": "ok", "handle": "bz-a3-1:retained", "passed": True},
    ])
    adapter = make_controller(repo, backend)
    interrupted = adapter.run(1, candidate_hash, manifest_hash)

    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["terminal"] is False
    assert interrupted["handle"] == "bz-a3-1:retained"
    resumed = adapter.run(
        1, candidate_hash, manifest_hash, observe_handle="bz-a3-1:retained"
    )
    contract.validate_controller_receipt(resumed, candidate_hash, manifest_hash)
    assert backend.requests[1] == backend.requests[2]
    assert resumed["policy"]["operations_consumed"] == 4
    assert resumed["policy"]["infrastructure_attempts"] == 1
    assert resumed["policy"]["infra_retries"] == 1
    assert resumed["policy"]["submitted_handles"] == [
        "bz-a3-1:job-1", "bz-a3-1:retained", "bz-a3-1:job-4", "bz-a3-1:job-5",
    ]
    assert resumed["policy"]["observed_handles"] == resumed["policy"]["submitted_handles"]
    assert [item["mode"] for item in resumed["policy"]["operation_history"]] == [
        "submit", "submit", "observe", "submit", "submit",
    ]
    tampered = json.loads(json.dumps(resumed))
    tampered["policy"]["infra_retries"] = 0
    with pytest.raises(contract.AuditError, match="operation history"):
        contract.validate_controller_receipt(tampered, candidate_hash, manifest_hash)


def test_observe_transition_authenticates_later_pending_handle(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    check_handle, profile_handle = "bz-a3-1:check", "bz-a3-1:profile"
    rows = [{"case": case, "samples_us": [10.0, 20.0, 40.0],
             "median_us": 20.0} for case in [7, 8, 9]]
    backend = FakeBackend([
        None,
        {"status": "infrastructure_error", "failure_type": "observer_error",
         "diagnostics": "check disconnected", "handle": check_handle},
        {"status": "ok", "handle": check_handle, "passed": True},
        {"status": "infrastructure_error", "failure_type": "observer_error",
         "diagnostics": "profile disconnected", "handle": profile_handle},
        {"status": "ok", "handle": profile_handle, "cases": rows,
         "kernel_name": "kernel"},
    ])
    adapter = make_controller(repo, backend)
    first = adapter.run(1, candidate_hash, manifest_hash)
    assert first["handle"] == check_handle

    transition = adapter.run(
        1, candidate_hash, manifest_hash, observe_handle=check_handle,
    )
    assert transition["handle"] == profile_handle
    contract.validate_observe_transition(
        transition, check_handle, candidate_hash, manifest_hash, 1,
    )

    terminal = adapter.run(
        1, candidate_hash, manifest_hash, observe_handle=profile_handle,
    )
    contract.validate_controller_receipt(terminal, candidate_hash, manifest_hash)
    contract.validate_observe_transaction(terminal, profile_handle)
    assert [item["mode"] for item in terminal["policy"]["operation_history"]] == [
        "submit", "submit", "observe", "submit", "observe", "submit",
    ]


def test_subprocess_backend_marks_only_explicit_observation(monkeypatch, tmp_path: Path):
    calls = []

    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"status":"ok"}', "")

    monkeypatch.setattr(controller.subprocess, "run", invoke)
    backend = controller.SubprocessBackend(["backend"], tmp_path, 20)
    request = {"action": "check"}

    monkeypatch.setenv("PROFILING_SKILL_CONTROLLER_MODE", "observe")
    backend(request)
    backend.observe(request)

    assert "PROFILING_SKILL_CONTROLLER_MODE" not in calls[0][1]["env"]
    assert calls[1][1]["env"]["PROFILING_SKILL_CONTROLLER_MODE"] == "observe"
    assert calls[0][1]["env"]["PROFILING_SKILL_CONTROLLER_REQUEST_SHA256"] == \
        calls[1][1]["env"]["PROFILING_SKILL_CONTROLLER_REQUEST_SHA256"]


def test_infrastructure_retry_budget_stops_before_another_dispatch(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)

    class AlwaysInfrastructure:
        def __init__(self):
            self.requests = []

        def __call__(self, request):
            self.requests.append(json.loads(json.dumps(request)))
            return {"status": "infrastructure_error", "handle": "bz-a3-1:retained",
                    "failure_type": "observer_error", "diagnostics": "still running"}

    backend = AlwaysInfrastructure()
    adapter = make_controller(repo, backend, infrastructure_retry_budget=2)
    receipt = adapter.run(1, candidate_hash, manifest_hash)
    for _ in range(3):
        receipt = adapter.run(1, candidate_hash, manifest_hash,
                              observe_handle="bz-a3-1:retained")

    assert receipt["status"] == "infrastructure_error"
    assert "retry budget exhausted" in receipt["reason"]
    assert len(backend.requests) == 3


def test_noisy_profile_gets_exactly_one_confirmation_then_measurement_pending(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    receipt = make_controller(
        repo, backend, variability_threshold=0.01
    ).run(1, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "measurement_pending"
    assert receipt["policy"]["confirmation_count"] == 1
    assert [request["action"] for request in backend.requests] == [
        "calibrate", "check", "profile", "profile", "calibrate",
    ]


def test_stable_confirmation_replaces_noisy_primary_as_accepted_evidence(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    cases = [7, 8, 9]
    noisy = {
        "status": "ok", "handle": "bz-a3-1:primary", "kernel_name": "kernel_mix_aiv",
        "declared_kernel_name": "kernel_mix_aiv", "resolved_kernel_name": "kernel",
        "cases": [{"case": case, "samples_us": [1.0, 10.0, 100.0],
                   "median_us": 10.0} for case in cases],
        "artifacts": {"remote_profile_evidence": "/remote/primary.json"},
    }
    stable = {
        "status": "ok", "handle": "bz-a3-1:confirmation", "kernel_name": "kernel_mix_aiv",
        "declared_kernel_name": "kernel_mix_aiv", "resolved_kernel_name": "kernel",
        "cases": [{"case": case, "samples_us": [5.0, 5.0, 5.0],
                   "median_us": 5.0} for case in cases],
        "artifacts": {"remote_profile_evidence": "/remote/confirmation.json"},
    }
    backend = FakeBackend([None, None, noisy, stable, None])

    receipt = make_controller(repo, backend, variability_threshold=0.1).run(
        1, candidate_hash, manifest_hash)

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "ok"
    assert receipt["handle"] == "bz-a3-1:confirmation"
    assert receipt["samples_us"] == pytest.approx([5.0, 5.0, 5.0])
    assert receipt["median_us"] == pytest.approx(5.0)
    assert receipt["case_results"] == stable["cases"]
    assert receipt["compact_artifacts"] == ["/remote/confirmation.json"]
    assert receipt["kernel_name"] == "kernel_mix_aiv"
    assert receipt["declared_kernel_name"] == "kernel_mix_aiv"
    assert receipt["resolved_kernel_name"] == "kernel"
    assert receipt["policy"]["accepted_timing"] == "confirmation"
    assert receipt["policy"]["primary"]["handle"] == "bz-a3-1:primary"
    assert receipt["policy"]["confirmation"]["handle"] == "bz-a3-1:confirmation"
    assert backend.requests[2]["attempt_id"] == "experiment-1-measurement-0-primary"
    assert backend.requests[3]["attempt_id"] == "experiment-1-measurement-0-confirmation"


def test_confirmation_kernel_identity_drift_is_infrastructure_error(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    cases = [7, 8, 9]
    noisy = {"status": "ok", "handle": "bz-a3-1:primary", "kernel_name": "kernel",
             "cases": [{"case": case, "samples_us": [1.0, 10.0, 100.0],
                        "median_us": 10.0} for case in cases]}
    drifted = {"status": "ok", "handle": "bz-a3-1:confirmation",
               "kernel_name": "different_kernel",
               "cases": [{"case": case, "samples_us": [5.0, 5.0, 5.0],
                          "median_us": 5.0} for case in cases]}
    backend = FakeBackend([None, None, noisy, drifted])

    receipt = make_controller(repo, backend, variability_threshold=0.1).run(
        1, candidate_hash, manifest_hash)

    assert receipt["status"] == "infrastructure_error"
    assert "confirmation identity" in receipt["reason"]


def test_confirmation_resolved_selector_drift_is_infrastructure_error(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    cases = [7, 8, 9]
    rows = [{"case": case, "samples_us": [1.0, 10.0, 100.0],
             "median_us": 10.0} for case in cases]
    noisy = {"status": "ok", "handle": "bz-a3-1:primary",
             "kernel_name": "kernel_mix_aiv", "declared_kernel_name": "kernel_mix_aiv",
             "resolved_kernel_name": "kernel", "cases": rows}
    drifted = {"status": "ok", "handle": "bz-a3-1:confirmation",
               "kernel_name": "kernel_mix_aiv", "declared_kernel_name": "kernel_mix_aiv",
               "resolved_kernel_name": "kernel_mix_aiv",
               "cases": [{"case": case, "samples_us": [5.0, 5.0, 5.0],
                          "median_us": 5.0} for case in cases]}
    backend = FakeBackend([None, None, noisy, drifted])

    receipt = make_controller(repo, backend, variability_threshold=0.1).run(
        1, candidate_hash, manifest_hash)

    assert receipt["status"] == "infrastructure_error"
    assert "confirmation identity" in receipt["reason"]


def test_receipt_contract_rejects_published_resolved_selector_drift(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    receipt = make_controller(repo, FakeBackend()).run(1, candidate_hash, manifest_hash)

    tampered = json.loads(json.dumps(receipt))
    tampered["resolved_kernel_name"] = "different"
    with pytest.raises(contract.AuditError, match="published timing"):
        contract.validate_controller_receipt(tampered, candidate_hash, manifest_hash)

    tampered = json.loads(json.dumps(receipt))
    tampered["policy"]["primary"]["resolved_kernel_name"] = "different"
    with pytest.raises(contract.AuditError, match="selector identity"):
        contract.validate_controller_receipt(tampered, candidate_hash, manifest_hash)


def test_receipt_contract_accepts_legacy_exact_selector_timing(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    receipt = make_controller(repo, FakeBackend()).run(1, candidate_hash, manifest_hash)
    legacy = json.loads(json.dumps(receipt))
    legacy.pop("declared_kernel_name")
    legacy.pop("resolved_kernel_name")
    legacy["policy"]["primary"].pop("declared_kernel_name")
    legacy["policy"]["primary"].pop("resolved_kernel_name")

    contract.validate_controller_receipt(legacy, candidate_hash, manifest_hash)


def test_measurement_pending_can_remeasure_without_rechecking_candidate(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    adapter = make_controller(repo, backend, variability_threshold=0.01)
    pending = adapter.run(1, candidate_hash, manifest_hash)
    backend.overrides.extend([
        {"status": "ok", "handle": "bz-a3-1:profile-stable", "kernel_name": "kernel", "cases": [
            {"case": case, "samples_us": [10.0, 10.0, 10.0], "median_us": 10.0}
            for case in [7, 8, 9]
        ]},
        {"status": "ok", "handle": "bz-a3-1:post-stable", "latency_us": 10.0,
         "samples_us": [9.0, 10.0, 11.0], "median_us": 10.0},
    ])

    receipt = adapter.run(
        1, candidate_hash, manifest_hash, remeasure_handle=pending["handle"]
    )

    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
    assert receipt["status"] == "ok"
    assert [request["action"] for request in backend.requests[-2:]] == [
        "profile", "calibrate",
    ]
    assert receipt["policy"]["operations_consumed"] == 7


def test_remeasure_transition_can_end_in_new_measurement_pending_handles(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    adapter = make_controller(repo, backend, variability_threshold=0.01)
    pending = adapter.run(1, candidate_hash, manifest_hash)
    backend.overrides.extend([
        {"status": "ok", "handle": "bz-a3-1:profile-fresh",
         "kernel_name": "kernel", "cases": [
             {"case": case, "samples_us": [10.0, 10.0, 10.0], "median_us": 10.0}
             for case in [7, 8, 9]
         ]},
        {"status": "ok", "handle": "bz-a3-1:post-drift", "latency_us": 20.0,
         "samples_us": [19.0, 20.0, 21.0], "median_us": 20.0},
    ])

    remeasured = adapter.run(
        1, candidate_hash, manifest_hash, remeasure_handle=pending["handle"],
    )

    assert remeasured["status"] == "measurement_pending"
    assert remeasured["handle"] == "bz-a3-1:profile-fresh"
    assert remeasured["calibration"]["after"]["handle"] == "bz-a3-1:post-drift"
    contract.validate_remeasure_transition(
        remeasured, pending, candidate_hash, manifest_hash, 1,
    )


def test_measurement_remeasure_generation_forces_fresh_cached_captures(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)

    class CachingBackend:
        def __init__(self):
            self.cache = {}
            self.requests = []

        def __call__(self, request):
            self.requests.append(json.loads(json.dumps(request)))
            key = json.dumps(request, sort_keys=True, separators=(",", ":"))
            if key in self.cache:
                return json.loads(json.dumps(self.cache[key]))
            handle = f"bz-a3-1:cache-{len(self.cache)}"
            if request["action"] == "calibrate":
                result = {"status": "ok", "handle": handle, "latency_us": 10.0,
                          "samples_us": [9.0, 10.0, 11.0], "median_us": 10.0}
            elif request["action"] == "check":
                result = {"status": "ok", "handle": handle, "passed": True}
            else:
                noisy = "measurement-0" in request["attempt_id"]
                samples = [1.0, 10.0, 100.0] if noisy else [5.0, 5.0, 5.0]
                result = {"status": "ok", "handle": handle, "kernel_name": "kernel",
                          "cases": [{"case": case, "samples_us": samples,
                                     "median_us": statistics.median(samples)}
                                    for case in request["cases"]]}
            self.cache[key] = result
            return json.loads(json.dumps(result))

    backend = CachingBackend()
    adapter = make_controller(repo, backend, variability_threshold=0.1)
    pending = adapter.run(1, candidate_hash, manifest_hash)
    receipt = adapter.run(1, candidate_hash, manifest_hash,
                          remeasure_handle=pending["handle"])

    assert pending["status"] == "measurement_pending"
    assert receipt["status"] == "ok"
    assert receipt["handle"] != pending["handle"]
    assert receipt["calibration"]["after"]["handle"] not in {
        pending["handle"], receipt["handle"],
    }
    assert receipt["policy"]["measurement_generation"] == 1
    contract.validate_remeasure_transition(
        receipt, pending, candidate_hash, manifest_hash, 1,
    )
    transition = receipt["remeasure_transition"]
    assert transition["pending_handle"] == pending["handle"]
    assert transition["profile_handle"] == receipt["handle"]
    assert transition["post_control_handle"] == receipt["calibration"]["after"]["handle"]
    assert receipt["samples_us"] == pytest.approx([5.0, 5.0, 5.0])
    measurement_requests = [request for request in backend.requests
                            if request["action"] in {"profile", "calibrate"}
                            and "measurement-" in request.get("attempt_id", "")]
    assert [request["attempt_id"] for request in measurement_requests[-2:]] == [
        "experiment-1-measurement-1-primary",
        "experiment-1-measurement-1-after",
    ]

    for mutation in ("pending", "profile", "control", "generation", "history"):
        tampered = json.loads(json.dumps(receipt))
        if mutation == "pending":
            tampered["remeasure_transition"]["pending_handle"] = "bz-a3-1:unrelated"
        elif mutation == "profile":
            tampered["remeasure_transition"]["profile_handle"] = pending["handle"]
        elif mutation == "control":
            tampered["remeasure_transition"]["post_control_handle"] = pending["handle"]
        elif mutation == "generation":
            tampered["remeasure_transition"]["to_measurement_generation"] = 2
        else:
            tampered["policy"]["operation_history"] = \
                tampered["policy"]["operation_history"][1:]
        with pytest.raises(contract.AuditError, match="remeasure transition"):
            contract.validate_remeasure_transition(
                tampered, pending, candidate_hash, manifest_hash, 1,
            )
    unrelated_pending = json.loads(json.dumps(pending))
    unrelated_pending["candidate_sha256"] = "0" * 64
    with pytest.raises(contract.AuditError, match="remeasure transition"):
        contract.validate_remeasure_transition(
            receipt, unrelated_pending, candidate_hash, manifest_hash, 1,
        )


def test_backend_operation_budget_is_shared_across_branch_rounds(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = FakeBackend()
    adapter = make_controller(repo, backend, request_budget=7)
    first = adapter.run(1, candidate_hash, manifest_hash)
    second = adapter.run(2, candidate_hash, manifest_hash)

    assert first["status"] == "ok"
    assert second["status"] == "infrastructure_error"
    assert second["reason"] == "7-operation experiment budget exhausted"
    assert len(backend.requests) == 7


def test_cli_matches_command_controller_arguments(tmp_path: Path):
    repo, candidate_hash, manifest_hash = repository(tmp_path)
    backend = tmp_path / "backend.py"
    backend.write_text(
        """import json, sys
request=json.load(sys.stdin)
action=request['action']
handle='bz-a3-1:'+action
if action == 'calibrate': result={'status':'ok','handle':handle,'latency_us':10.0,'samples_us':[9.0,10.0,11.0],'median_us':10.0}
elif action == 'check': result={'status':'ok','handle':handle,'passed':True}
else:
 rows=[{'case':c,'samples_us':[10.0,10.0,10.0],'median_us':10.0} for c in request['cases']]
 result={'status':'ok','handle':handle,'kernel_name':'kernel','cases':rows}
print(json.dumps(result))
"""
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema": "profiling-skill/audited-bz-controller-config/v1",
        "benchmark": "matmul", "round_count": 4, "request_budget": 24,
        "profile_repeats": 3, "variability_threshold": 0.25,
        "control_drift_threshold": 0.2,
        "baseline": baseline(),
        "devices": [{"id": "bz-a3-1/device-0", "device": 0}],
        "development_cases": [7, 8, 9], "all_cases": list(range(10)),
        "backend_command": [sys.executable, str(backend)],
    }))
    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/audited_bz_controller.py"),
        "--config", str(config), "--state-dir", str(tmp_path / "state"),
        "--experiment", "1", "--candidate-sha256", candidate_hash,
        "--manifest-sha256", manifest_hash,
    ], cwd=repo, text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr
    receipt = json.loads(run.stdout)
    contract.validate_controller_receipt(receipt, candidate_hash, manifest_hash)
