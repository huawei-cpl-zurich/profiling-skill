import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load(ROOT / "scripts/a3_terminal_gate.py", "a3_terminal_gate")
builder = load(ROOT / "scripts/build_a3_gate_bundle.py", "build_a3_gate_bundle")
protocol = load(ROOT / "scripts/simple_experiment_runner.py", "simple_experiment_runner")


def manifest(task):
    spec = gate.SPECS[task]
    return {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": spec["kernel_name"], "entrypoint": spec["entrypoint"],
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }


def fake_modules(task, *, check_status="ok", profile_status="success",
                 fusion_error: str | None = None, include_pipe_activity=True,
                 resolved_kernel_name: str | None = None,
                 pipe_kernel_name: str | None = None,
                 pipe_selector_miss: bool = False,
                 empty_pipe_activity: bool = False):
    spec = gate.SPECS[task]
    resolved_kernel_name = resolved_kernel_name or spec["kernel_name"]
    pipe_kernel_name = pipe_kernel_name or resolved_kernel_name
    fusion_rows = [
        {"schema": "profiling-skill/fusion-evidence/v1", "case": case,
         "output_launch_id": "launch-0", "operators": [{
             "name": spec["kernel_name"], "origin": "triton",
             "entrypoint": spec["entrypoint"], "launch_id": "launch-0",
             "component": "aic",
         }]}
        for case in spec["cases"]
    ]
    runner = SimpleNamespace(execute=lambda job: {
        "status": check_status, "passed": check_status == "ok",
        "diagnostics": "compile failed" if check_status != "ok" else "",
        "fusion_evidence": fusion_rows,
    })

    class FusionError(ValueError):
        pass

    def validate_fusion(value, evidence, expected_cases):
        if fusion_error is not None:
            raise FusionError(fusion_error)
        return {
            "entrypoint": value["entrypoint"], "kernel_name": value["kernel_name"],
            "cases": list(expected_cases), "logical_launches_per_case": 1,
        }

    fusion = SimpleNamespace(
        FusionContractError=FusionError,
        validate_manifest=lambda value: value,
        validate_fusion_evidence=validate_fusion,
    )

    def batch_execute(job, **kwargs):
        output, response = kwargs["output"], kwargs["response_path"]
        output.mkdir(parents=True, exist_ok=True)
        ok = profile_status == "success" and not pipe_selector_miss
        response.write_text(json.dumps({
            "status": "ok" if ok else "infrastructure_error",
            "diagnostics": (
                "no PipeUtilization export for resolved kernel"
                if pipe_selector_miss else "msprof unavailable" if not ok else ""
            ),
        }))
        pipe_activity = {
            "schema_version": 1, "product": "a3",
            "evidence_kind": "per-block-pipe-activity",
            "provenance": {"kernel_selector": pipe_kernel_name},
            "observations": [{
                "observation_id": f"{spec['kernel_name']}:block-0:cube0",
                "activity": {} if empty_pipe_activity else {"aic_mte2": 0.5},
                "saturation": {"state": "unknown", "reason": "no capacity denominator"},
            }],
        }
        (output / "evidence.json").write_text(json.dumps({
            "status": "failure" if pipe_selector_miss else profile_status,
            "profiler": "msprof-op",
            "timing_scope": "device-task", "kernel_name": spec["kernel_name"],
            "declared_kernel_name": spec["kernel_name"],
            "resolved_kernel_name": resolved_kernel_name,
            "cases": [{"case": case, "median_us": 2.0, "samples_us": [2.0] * 3}
                      for case in spec["ranked"]],
            "repeats": 3, "geomean_us": 2.0, "captures": [
                {"case": case, "iteration": iteration, "duration_us": 2.0,
                 "kernel_name": resolved_kernel_name,
                 "declared_kernel_name": spec["kernel_name"],
                 "resolved_kernel_name": resolved_kernel_name,
                 "evidence_sha256": "1" * 64, "msprof_log_sha256": "2" * 64}
                for case in spec["ranked"] for iteration in range(3)
            ],
            **({"pipe_activity": pipe_activity} if include_pipe_activity else {}),
        }))
        return 0 if ok else 1

    return runner, fusion, SimpleNamespace(execute=batch_execute)


def prepare(tmp_path, task):
    root = tmp_path / "root"
    for path in (root / "scripts", root / f"benchmarks/{task}"):
        path.mkdir(parents=True, exist_ok=True)
    for name in ("a3_benchmark_runner.py", "fully_fused_contract.py",
                 "batch_profile_a3.py", "profile_a3.py"):
        (root / "scripts" / name).write_text("fixture\n")
    for name in ("baseline.py", "cases.jsonl"):
        (root / f"benchmarks/{task}/{name}").write_text("fixture\n")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate\n")
    declaration = tmp_path / "candidate-manifest.json"
    declaration.write_text(json.dumps(manifest(task)))
    return root, candidate, declaration


def test_gate_requires_host_selected_logical_device_zero(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ASCEND_DEVICE_ID", "0")
    assert "host placement" in gate.validate_environment()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ASCEND_DEVICE_ID", "1")
    assert "logical device 0" in gate.validate_environment()


def test_infrastructure_envelope_retains_request_bound_autotune_checkpoint(
    tmp_path, monkeypatch, capsys
):
    checkpoint = {
        "schema": gate.AUTOTUNE_CHECKPOINT_SCHEMA, "request_sha256": "1" * 64,
        "optimization_round": 1, "screens": {"incumbent": {"outcome": "success"}},
        "profiles": {},
    }

    def fail_with_checkpoint(*args):
        output = args[4]
        gate.write_json(output / "autotune-checkpoint.json", checkpoint)
        return "infrastructure_failure", "device unavailable", {}, {}

    monkeypatch.setattr(gate, "run_autotune_gate", fail_with_checkpoint)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ASCEND_DEVICE_ID", "0")
    monkeypatch.setattr(sys, "argv", [
        "a3_terminal_gate.py", "--root", str(tmp_path),
        "--candidate", str(tmp_path / "candidate.py"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--benchmark", "matmul", "--output", str(tmp_path / "out"),
        "--autotune-request", str(tmp_path / "request.json"),
    ])
    assert gate.main() == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["outcome"] == "infrastructure_failure"
    assert envelope["autotune_checkpoint"] == checkpoint


def test_unexpected_autotune_exception_still_retains_checkpoint(
    tmp_path, monkeypatch, capsys
):
    checkpoint = {
        "schema": gate.AUTOTUNE_CHECKPOINT_SCHEMA, "request_sha256": "1" * 64,
        "optimization_round": 1, "screens": {"incumbent": {"outcome": "success"}},
        "profiles": {}, "profile_failures": {},
    }

    def crash_after_checkpoint(*args):
        gate.write_json(args[4] / "autotune-checkpoint.json", checkpoint)
        raise RuntimeError("late host failure")

    monkeypatch.setattr(gate, "run_autotune_gate", crash_after_checkpoint)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ASCEND_DEVICE_ID", "0")
    monkeypatch.setattr(sys, "argv", [
        "a3_terminal_gate.py", "--root", str(tmp_path),
        "--candidate", str(tmp_path / "candidate.py"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--benchmark", "matmul", "--output", str(tmp_path / "out"),
        "--autotune-request", str(tmp_path / "request.json"),
    ])

    assert gate.main() == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["outcome"] == "infrastructure_failure"
    assert envelope["diagnostics"] == (
        "host gate failed unexpectedly: RuntimeError: late host failure"
    )
    assert envelope["autotune_checkpoint"] == checkpoint


def test_gate_runs_full_correctness_before_ranked_profile(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "gdn")
    modules = fake_modules("gdn")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "gdn", tmp_path / "out", 4,
    )
    assert outcome == "success" and diagnostics == ""
    assert summary["correctness_cases"] == 50
    assert summary["schema"] == "profiling-skill/a3-round-profile-summary/v1"
    assert [row["case"] for row in summary["timing"]["cases"]] == [40, 49, 47, 46, 45]
    assert summary["timing"]["repeats"] == 3
    assert summary["timing"]["profiler"] == "msprof-op"
    assert summary["timing"]["kernel_name"] == gate.SPECS["gdn"]["kernel_name"]
    assert len(summary["timing"]["captures"]) == len(gate.SPECS["gdn"]["ranked"]) * 3
    assert summary["pipe_activity"]["observations"][0]["saturation"]["state"] == "unknown"
    assert summary["timeline"]["available"] is False
    assert summary["timeline"]["kind"] == "a3-temporal-pipe-phases"
    assert summary["fusion"]["logical_launches_per_case"] == 1


def test_gate_reports_suffix_fallback_as_exact_resolved_timing_identity(
    tmp_path, monkeypatch
):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    declared = gate.SPECS["matmul"]["kernel_name"]
    resolved = "streaming_matmul_add_kernel"
    modules = fake_modules("matmul", resolved_kernel_name=resolved)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 2,
    )
    assert outcome == "success" and diagnostics == ""
    assert summary["timing"]["kernel_name"] == resolved
    assert summary["timing"]["declared_kernel_name"] == declared
    assert summary["timing"]["resolved_kernel_name"] == resolved
    assert summary["pipe_activity"]["provenance"]["kernel_selector"] == resolved


def test_gate_rejects_timing_and_pipe_selector_mismatch(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = fake_modules(
        "matmul", resolved_kernel_name="streaming_matmul_add_kernel",
        pipe_kernel_name=gate.SPECS["matmul"]["kernel_name"],
    )
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 2,
    )
    assert outcome == "infrastructure_failure" and summary == {}
    assert diagnostics == (
        "managed A3 timing and pipe activity use different exact kernel selectors"
    )


def test_compile_failure_stops_before_profiling(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "bsa")
    modules = fake_modules("bsa", check_status="compile_error")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "bsa", tmp_path / "out", 1,
    )
    assert outcome == "candidate_failure"
    assert diagnostics == "compile failed" and summary == {}


def test_fusion_failure_preserves_actionable_entrypoint_diagnostics(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    detail = (
        "Triton launch entrypoint mismatch: expected='streaming_matmul_add_kernel'; "
        "observed=['matmul_kernel']; "
        "kernel_name selector='streaming_matmul_add_kernel_mix_aic'"
    )
    modules = fake_modules("matmul", fusion_error=detail)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "candidate_failure" and summary == {}
    assert diagnostics == f"fully-fused audit failed: {detail}"


def test_profile_tool_failure_is_infrastructure(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = fake_modules("matmul", profile_status="failure")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure"
    assert "msprof unavailable" in diagnostics and summary == {}


def test_pipe_selector_miss_after_timing_success_is_infrastructure(
    tmp_path, monkeypatch
):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = fake_modules("matmul", pipe_selector_miss=True)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure" and summary == {}
    assert diagnostics == "no PipeUtilization export for resolved kernel"


def test_missing_pipe_activity_fails_instead_of_claiming_profile_guidance(
    tmp_path, monkeypatch
):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = fake_modules("matmul", include_pipe_activity=False)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure" and summary == {}
    assert diagnostics == "managed A3 profile is missing pipe activity evidence"


def test_claimed_a3_saturation_is_rejected(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = list(fake_modules("matmul"))
    original = modules[2].execute

    def saturated(job, **kwargs):
        rc = original(job, **kwargs)
        path = kwargs["output"] / "evidence.json"
        evidence = json.loads(path.read_text())
        evidence["pipe_activity"]["observations"][0]["saturation"]["state"] = "saturated"
        path.write_text(json.dumps(evidence))
        return rc

    modules[2] = SimpleNamespace(execute=saturated)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure" and summary == {}
    assert "keep saturation unknown" in diagnostics


def test_empty_modeled_pipe_activity_is_rejected(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "matmul")
    modules = fake_modules("matmul", empty_pipe_activity=True)
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure" and summary == {}
    assert diagnostics == (
        "every managed A3 pipe observation requires a finite modeled activity fraction"
    )

@pytest.mark.parametrize("rejected_full_gate_values,rejected_profile_values", [
    ((), ()), ((4,), ()), ((4, 8, 20), ()), ((), (4,)), ((), (4, 8, 20)),
])
def test_autotune_screens_broad_set_and_profiles_only_three_finalists(
    tmp_path, monkeypatch, rejected_full_gate_values, rejected_profile_values
):
    root, _candidate, declaration = prepare(tmp_path, "matmul")
    spec = gate.SPECS["matmul"]
    calls = {"checks": [], "profiles": []}

    def runner_execute(job):
        source = Path(job["candidate"]).read_text()
        value = int(source.split("TUNE = ", 1)[1].splitlines()[0])
        calls["checks"].append((value, job["scope"], tuple(job["cases"])))
        if value == 99:
            return {"status": "compile_error", "passed": False,
                    "diagnostics": "resource overflow"}
        if value in rejected_full_gate_values and job["scope"] == "full":
            return {"status": "correctness_error", "passed": False,
                    "diagnostics": "full shape mismatch"}
        rows = [{"case": case, "host_elapsed_us": float(value), "passed": True}
                for case in job["cases"]]
        fusion_rows = [
            {"case": case, "operators": [], "output_launch_id": "launch-0"}
            for case in job["cases"]
        ]
        return {"status": "ok", "passed": True, "diagnostics": "",
                "case_evidence": rows, "fusion_evidence": fusion_rows}

    runner_module = SimpleNamespace(execute=runner_execute)

    class FusionError(ValueError):
        pass

    fusion_module = SimpleNamespace(
        FusionContractError=FusionError,
        validate_manifest=lambda value: value,
        validate_fusion_evidence=lambda value, evidence, expected_cases: {
            "entrypoint": value["entrypoint"], "kernel_name": value["kernel_name"],
            "cases": list(expected_cases), "logical_launches_per_case": 1,
        },
    )

    def batch_execute(job, **kwargs):
        value = int(Path(job["candidate"]).read_text().split("TUNE = ", 1)[1].splitlines()[0])
        pipe = bool(job["profiling"]["pipe_activity"])
        calls["profiles"].append((value, pipe, job["repeats"]))
        output, response = kwargs["output"], kwargs["response_path"]
        output.mkdir(parents=True, exist_ok=True)
        profile_rejected = not pipe and value in rejected_profile_values
        response.write_text(json.dumps({
            "status": "compile_error" if profile_rejected else "ok",
            "diagnostics": "repeated profile launch failed" if profile_rejected else "",
        }))
        evidence = {
            "status": "failure" if profile_rejected else "success",
            "failure": "repeated profile launch failed" if profile_rejected else "",
            "profiler": "msprof-op", "timing_scope": "device-task",
            "kernel_name": spec["kernel_name"],
            "declared_kernel_name": spec["kernel_name"],
            "resolved_kernel_name": spec["kernel_name"], "cases": [
                {"case": case, "median_us": float(value), "samples_us": [float(value)] * 3}
                for case in spec["ranked"]
            ], "repeats": 3, "geomean_us": float(value), "captures": [
                {"case": case, "iteration": iteration,
                 "duration_us": float(value), "kernel_name": spec["kernel_name"],
                 "declared_kernel_name": spec["kernel_name"],
                 "resolved_kernel_name": spec["kernel_name"],
                 "evidence_sha256": "3" * 64, "msprof_log_sha256": "4" * 64}
                for case in spec["ranked"] for iteration in range(3)
            ],
        }
        if pipe:
            evidence["pipe_activity"] = {
                "product": "a3", "evidence_kind": "per-block-pipe-activity",
                    "provenance": {"kernel_selector": spec["kernel_name"]},
                "observations": [{"activity": {"aic_mte2": 0.5},
                                  "saturation": {"state": "unknown"}}],
            }
        (output / "evidence.json").write_text(json.dumps(evidence))
        return 1 if profile_rejected else 0

    batch_module = SimpleNamespace(execute=batch_execute)

    def fake_load(path, _name):
        return {
            "a3_benchmark_runner.py": runner_module,
            "fully_fused_contract.py": fusion_module,
            "batch_profile_a3.py": batch_module,
        }[path.name]

    monkeypatch.setattr(gate, "load", fake_load)
    sources = {
        "incumbent": "TUNE = 20\n",
        "fast": "TUNE = 4\n",
        "medium": "TUNE = 8\n",
        "slow": "TUNE = 40\n",
        "broken": "TUNE = 99\n",
    }
    request = {
        "schema": gate.AUTOTUNE_REQUEST_SCHEMA, "incumbent_id": "incumbent",
        "optimization_round": 1, "benchmark": "matmul",
        "required_correctness_cases": len(spec["cases"]),
        "expected_entrypoint": spec["entrypoint"],
        "expected_declared_kernel_name": spec["kernel_name"],
        "expected_profile_cases": spec["ranked"],
        "options": [
            {"id": identifier, "source": source,
             "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
            for identifier, source in sources.items()
        ],
    }
    request_path = tmp_path / "autotune.json"
    request_path.write_text(json.dumps(request))

    outcome, diagnostics, profile, tuning = gate.run_autotune_gate(
        root, request_path, declaration, "matmul", tmp_path / "out", 1,
    )

    expected_outcome = (
        "candidate_failure"
        if len(rejected_full_gate_values) == 3 or len(rejected_profile_values) == 3
        else "success"
    )
    assert outcome == expected_outcome
    assert diagnostics == (
        (
            "all selected finalists failed official profiling"
            if len(rejected_profile_values) == 3 else
            "all officially profiled finalists failed the complete winner gate"
        ) if expected_outcome == "candidate_failure" else ""
    )
    assert len(calls["checks"]) == 9 + (
        0 if len(rejected_profile_values) == 3
        else 1 + min(len(rejected_full_gate_values), 2)
    )
    assert calls["checks"][:9] == [
        (value, scope, tuple(spec["ranked"]))
        for value in (20, 4, 8, 40)
        for scope in ("autotune-warmup", "autotune-ranked")
    ] + [(99, "autotune-warmup", tuple(spec["ranked"]))]
    assert tuning["finalists"] == ["fast", "medium", "incumbent"]
    assert tuning["optimization_round"] == 1
    expected_winner = (
        None if expected_outcome == "candidate_failure"
        else "medium" if rejected_full_gate_values or rejected_profile_values else "fast"
    )
    if expected_winner is not None:
        assert tuning["winner_id"] == expected_winner
        assert tuning["winner_source_sha256"] == hashlib.sha256(
            sources[expected_winner].encode()
        ).hexdigest()
        assert tuning["winner_full_gate"] == {
            "winner_id": expected_winner,
            "source_sha256": tuning["winner_source_sha256"],
            "candidate_artifact_sha256": tuning["winner_source_sha256"],
            "profile_summary_sha256": gate.canonical_json_sha256(profile),
        }
    successful_profiles = [identifier for identifier, value in (
        ("fast", 4), ("medium", 8), ("incumbent", 20)
    ) if value not in rejected_profile_values]
    assert tuning["profiled_finalists"] == successful_profiles
    assert set(tuning["finalist_profiles"]) == set(successful_profiles)
    if "fast" in successful_profiles:
        assert tuning["finalist_profiles"]["fast"] == {
            "option_id": "fast",
            "source_sha256": tuning["options"][1]["source_sha256"],
            "candidate_artifact_sha256": tuning["options"][1]["source_sha256"],
            "timing": tuning["finalist_profiles"]["fast"]["timing"],
        }
        assert tuning["finalist_profiles"]["fast"]["timing"]["metric"] == "BasicInfo"
    assert [row["option_id"] for row in tuning["profile_failures"]] == [
        identifier for identifier, value in (
            ("fast", 4), ("medium", 8), ("incumbent", 20)
        ) if value in rejected_profile_values
    ]
    if rejected_profile_values:
        assert tuning["profile_failures"][0]["diagnostics"] == "repeated profile launch failed"
    expected_failures = [
        identifier for identifier, value in (("fast", 4), ("medium", 8), ("incumbent", 20))
        if value in rejected_full_gate_values
    ]
    assert [row["option_id"] for row in tuning["winner_gate_failures"]] == expected_failures
    if expected_failures:
        assert tuning["winner_gate_failures"][0]["diagnostics"] == "full shape mismatch"
    if expected_winner is not None:
        selected, accepted = protocol.ExperimentCell._validate_autotune_success(
            request, {"profile_summary": profile, "autotune_summary": tuning},
        )
        assert selected == sources[expected_winner] and accepted == tuning
        assert profile["pipe_activity"]["observations"][0]["saturation"]["state"] == "unknown"
    else:
        assert profile == {}
    assert calls["profiles"] == (
        [(4, False, 3), (8, False, 3), (20, False, 3)]
        + ([] if expected_winner is None else [
            ({"fast": 4, "medium": 8, "incumbent": 20}[expected_winner], True, 3),
        ])
    )
    assert next(row for row in tuning["options"] if row["id"] == "broken")["outcome"] == "candidate_failure"
    checkpoint = json.loads((tmp_path / "out/autotune-checkpoint.json").read_text())
    assert set(checkpoint["screens"]) == set(sources)

    prior_checks, prior_profiles = len(calls["checks"]), len(calls["profiles"])
    repeated = gate.run_autotune_gate(
        root, request_path, declaration, "matmul", tmp_path / "out", 1,
    )
    assert repeated[0] == expected_outcome
    assert len(calls["checks"]) == prior_checks
    assert len(calls["profiles"]) == prior_profiles


def test_autotune_request_is_bound_to_optimization_round(tmp_path):
    source = "TUNE = 1\n"
    request = {
        "schema": gate.AUTOTUNE_REQUEST_SCHEMA, "incumbent_id": "incumbent",
        "optimization_round": 2, "benchmark": "matmul",
        "required_correctness_cases": 10,
        "expected_entrypoint": gate.SPECS["matmul"]["entrypoint"],
        "expected_declared_kernel_name": gate.SPECS["matmul"]["kernel_name"],
        "expected_profile_cases": gate.SPECS["matmul"]["ranked"],
        "options": [{
            "id": "incumbent", "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }],
    }
    path = tmp_path / "autotune.json"
    path.write_text(json.dumps(request))

    with pytest.raises(ValueError, match="invalid autotune request"):
        gate.load_autotune_request(path, 1)


def test_autotune_request_pins_cannot_rename_the_benchmark_contract(tmp_path):
    root, _candidate, declaration = prepare(tmp_path, "matmul")
    source = "TUNE = 1\n"
    request = {
        "schema": gate.AUTOTUNE_REQUEST_SCHEMA, "incumbent_id": "incumbent",
        "optimization_round": 1, "benchmark": "matmul",
        "required_correctness_cases": 10,
        "expected_entrypoint": "renamed_kernel",
        "expected_declared_kernel_name": gate.SPECS["matmul"]["kernel_name"],
        "expected_profile_cases": gate.SPECS["matmul"]["ranked"],
        "options": [{"id": "incumbent", "source": source,
                     "source_sha256": hashlib.sha256(source.encode()).hexdigest()}],
    }
    path = tmp_path / "autotune.json"
    path.write_text(json.dumps(request))
    outcome, diagnostics, profile, tuning = gate.run_autotune_gate(
        root, path, declaration, "matmul", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure" and profile == {} and tuning == {}
    assert diagnostics == "host autotune request does not match the benchmark contract"


def test_autotune_screen_retries_only_infrastructure_failures(tmp_path):
    calls = 0

    class FusionError(ValueError):
        pass

    def execute(job):
        nonlocal calls
        calls += 1
        if calls < 3:
            return {"status": "infrastructure_error", "passed": False,
                    "diagnostics": "device unavailable"}
        return {
            "status": "ok", "passed": True,
            "case_evidence": [{"host_elapsed_us": 5.0}],
            "fusion_evidence": [{}],
        }

    fusion = SimpleNamespace(
        FusionContractError=FusionError,
        validate_fusion_evidence=lambda *args, **kwargs: {},
    )
    outcome, diagnostics, row = gate.screen_candidate(
        SimpleNamespace(execute=execute), fusion, {}, {}, {"ranked": [7]},
        tmp_path / "candidate.py",
    )
    assert outcome == "success" and diagnostics == ""
    assert calls == 4 and len(row["commands"]) == 4
    assert "scope=autotune-warmup" in row["commands"][-2]
    assert "scope=autotune-ranked" in row["commands"][-1]


def test_autotune_remote_retry_reuses_completed_screens_and_profiles(
    tmp_path, monkeypatch
):
    root, _candidate, declaration = prepare(tmp_path, "matmul")
    spec = gate.SPECS["matmul"]
    fusion = SimpleNamespace(
        FusionContractError=ValueError,
        validate_manifest=lambda value: value,
    )
    monkeypatch.setattr(
        gate, "load", lambda path, _name: (
            fusion if path.name == "fully_fused_contract.py" else SimpleNamespace()
        ),
    )
    calls = {"screens": 0, "profiles": 0, "full": 0}

    def screen(*args):
        calls["screens"] += 1
        value = int(args[-1].read_text().split("=", 1)[1])
        return "success", "", {"cheap_device_timing_us": float(value), "commands": []}

    def profile(root, common, spec, candidate, output, round_number, *, pipe_activity):
        calls["profiles"] += 1
        value = float(candidate.read_text().split("=", 1)[1])
        return "success", "", {"metric": "BasicInfo", "profiler": "msprof-op",
            "timing_scope": "device-task", "kernel_name": spec["kernel_name"],
            "declared_kernel_name": spec["kernel_name"],
            "resolved_kernel_name": spec["kernel_name"], "cases": [],
            "repeats": 3, "geomean_us": value, "captures": []}

    def full(*args):
        calls["full"] += 1
        if calls["full"] <= 3:
            return "infrastructure_failure", "device unavailable", {}
        return "success", "", {"schema": "profiling-skill/a3-round-profile-summary/v1"}

    monkeypatch.setattr(gate, "screen_candidate", screen)
    monkeypatch.setattr(gate, "profile_candidate", profile)
    monkeypatch.setattr(gate, "run_gate", full)
    sources = {"incumbent": "TUNE=3", "fast": "TUNE=1", "medium": "TUNE=2"}
    request = {
        "schema": gate.AUTOTUNE_REQUEST_SCHEMA, "incumbent_id": "incumbent",
        "optimization_round": 1, "benchmark": "matmul",
        "required_correctness_cases": 10, "expected_profile_cases": spec["ranked"],
        "expected_entrypoint": spec["entrypoint"],
        "expected_declared_kernel_name": spec["kernel_name"],
        "options": [{"id": identifier, "source": source,
                     "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
                    for identifier, source in sources.items()],
    }
    request_path = tmp_path / "autotune.json"
    request_path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")))
    output = tmp_path / "out"

    assert gate.run_autotune_gate(
        root, request_path, declaration, "matmul", output, 1,
    )[0] == "infrastructure_failure"
    assert gate.run_autotune_gate(
        root, request_path, declaration, "matmul", output, 1,
    )[0] == "success"
    assert calls == {"screens": 3, "profiles": 3, "full": 4}


def test_wrong_host_task_manifest_is_infrastructure_failure(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "gdn")
    declaration.write_text(json.dumps(manifest("bsa")))
    modules = fake_modules("gdn")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "gdn", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure"
    assert "benchmark contract" in diagnostics and summary == {}


def test_invalid_host_manifest_is_infrastructure_failure(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "gdn")
    declaration.write_text("not json")
    modules = fake_modules("gdn")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "gdn", tmp_path / "out", 1,
    )
    assert outcome == "infrastructure_failure"
    assert "invalid host candidate contract" in diagnostics and summary == {}


def test_bundle_is_deterministic_and_self_describing(tmp_path):
    first, second = tmp_path / "one.tar", tmp_path / "two.tar"
    one = builder.build(ROOT, first)
    two = builder.build(ROOT, second)
    assert first.read_bytes() == second.read_bytes()
    assert one == two and one["archive_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    with tarfile.open(first) as archive:
        names = set(archive.getnames())
        bundled_manifest = json.load(archive.extractfile("bundle-manifest.json"))
    assert names == set(builder.FILES) | {"bundle-manifest.json"}
    assert bundled_manifest["schema"] == "profiling-skill/a3-terminal-gate-bundle/v1"


def test_thin_gate_classifies_bundle_failure_as_infrastructure(tmp_path):
    broken = tmp_path / "broken.tar"
    broken.write_text("not a tar archive")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate\n")
    declaration = tmp_path / "manifest.json"
    declaration.write_text(json.dumps(manifest("matmul")))
    environment = {
        **os.environ, "TERMINAL_GATE_BUNDLE_PATH": str(broken),
        "CANDIDATE_PATH": str(candidate), "CANDIDATE_MANIFEST_PATH": str(declaration),
        "EXPERIMENT_ROUND": "1",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/a3-gates/matmul.sh")], cwd=tmp_path,
        env=environment, text=True, capture_output=True, check=False,
    )
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode == 1
    assert envelope["schema"] == gate.RESULT_SCHEMA
    assert envelope["outcome"] == "infrastructure_failure"


@pytest.mark.parametrize("task", ["gdn", "bsa", "matmul"])
def test_thin_gate_classifies_missing_entrypoint_as_infrastructure(tmp_path, task):
    bundle = tmp_path / "missing-entrypoint.tar"
    with tarfile.open(bundle, "w") as archive:
        info = tarfile.TarInfo("placeholder")
        info.size = 0
        archive.addfile(info, io.BytesIO())
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate\n")
    declaration = tmp_path / "manifest.json"
    declaration.write_text(json.dumps(manifest(task)))
    environment = {
        **os.environ, "TERMINAL_GATE_BUNDLE_PATH": str(bundle),
        "CANDIDATE_PATH": str(candidate), "CANDIDATE_MANIFEST_PATH": str(declaration),
        "EXPERIMENT_ROUND": "1",
    }
    result = subprocess.run(
        ["bash", str(ROOT / f"scripts/a3-gates/{task}.sh")], cwd=tmp_path,
        env=environment, text=True, capture_output=True, check=False,
    )
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode == 1
    assert envelope == {
        "schema": gate.RESULT_SCHEMA,
        "outcome": "infrastructure_failure",
        "diagnostics": "host gate entrypoint produced no result envelope",
    }


def test_thin_gate_preserves_valid_candidate_failure_envelope(tmp_path):
    bundle = tmp_path / "candidate-failure.tar"
    source = (
        "import json,sys\n"
        "print(json.dumps({'schema':'simple-experiment-result/v1',"
        "'outcome':'candidate_failure','diagnostics':'compile failed'}))\n"
        "raise SystemExit(1)\n"
    ).encode()
    with tarfile.open(bundle, "w") as archive:
        info = tarfile.TarInfo("scripts/a3_terminal_gate.py")
        info.size = len(source)
        archive.addfile(info, io.BytesIO(source))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("candidate\n")
    declaration = tmp_path / "manifest.json"
    declaration.write_text(json.dumps(manifest("matmul")))
    environment = {
        **os.environ, "TERMINAL_GATE_BUNDLE_PATH": str(bundle),
        "CANDIDATE_PATH": str(candidate), "CANDIDATE_MANIFEST_PATH": str(declaration),
        "EXPERIMENT_ROUND": "1",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/a3-gates/matmul.sh")], cwd=tmp_path,
        env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout.strip().splitlines()[-1])["outcome"] == "candidate_failure"
