import hashlib
import importlib.util
import io
import json
import os
import subprocess
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
                 pipe_kernel_name: str | None = None):
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
        output.mkdir(parents=True)
        ok = profile_status == "success"
        response.write_text(json.dumps({
            "status": "ok" if ok else "infrastructure_error",
            "diagnostics": "msprof unavailable" if not ok else "",
        }))
        pipe_activity = {
            "schema_version": 1, "product": "a3",
            "evidence_kind": "per-block-pipe-activity",
            "provenance": {"kernel_selector": pipe_kernel_name},
            "observations": [{
                "observation_id": f"{spec['kernel_name']}:block-0:cube0",
                "activity": {"aic_mte2": 0.5},
                "saturation": {"state": "unknown", "reason": "no capacity denominator"},
            }],
        }
        (output / "evidence.json").write_text(json.dumps({
            "status": profile_status, "profiler": "msprof-op",
            "timing_scope": "device-task", "kernel_name": resolved_kernel_name,
            "declared_kernel_name": spec["kernel_name"],
            "resolved_kernel_name": resolved_kernel_name,
            "cases": [{"case": case, "median_us": 2.0, "samples_us": [2.0] * 3}
                      for case in spec["ranked"]],
            "repeats": 3, "geomean_us": 2.0, "captures": [],
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
