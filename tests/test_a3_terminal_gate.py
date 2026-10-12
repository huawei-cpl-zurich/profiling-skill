import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace


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


def fake_modules(task, *, check_status="ok", profile_status="success"):
    spec = gate.SPECS[task]
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

    fusion = SimpleNamespace(
        FusionContractError=FusionError,
        validate_manifest=lambda value: value,
        validate_fusion_evidence=lambda value, evidence, expected_cases: {
            "entrypoint": value["entrypoint"], "kernel_name": value["kernel_name"],
            "cases": list(expected_cases), "logical_launches_per_case": 1,
        },
    )

    def batch_execute(job, **kwargs):
        output, response = kwargs["output"], kwargs["response_path"]
        output.mkdir(parents=True)
        ok = profile_status == "success"
        response.write_text(json.dumps({
            "status": "ok" if ok else "infrastructure_error",
            "diagnostics": "msprof unavailable" if not ok else "",
        }))
        (output / "evidence.json").write_text(json.dumps({
            "status": profile_status, "profiler": "msprof-op",
            "timing_scope": "device-task", "kernel_name": spec["kernel_name"],
            "cases": [{"case": case, "median_us": 2.0, "samples_us": [2.0] * 3}
                      for case in spec["ranked"]],
            "repeats": 3, "geomean_us": 2.0, "captures": [],
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
    assert [row["case"] for row in summary["cases"]] == [40, 49, 47, 46, 45]
    assert summary["repeats"] == 3 and summary["profiler"] == "msprof-op"
    assert summary["fusion"]["logical_launches_per_case"] == 1


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


def test_wrong_task_manifest_fails_before_execution(tmp_path, monkeypatch):
    root, candidate, declaration = prepare(tmp_path, "gdn")
    declaration.write_text(json.dumps(manifest("bsa")))
    modules = fake_modules("gdn")
    loaded = iter(modules)
    monkeypatch.setattr(gate, "load", lambda *_: next(loaded))
    outcome, diagnostics, summary = gate.run_gate(
        root, candidate, declaration, "gdn", tmp_path / "out", 1,
    )
    assert outcome == "candidate_failure"
    assert "benchmark contract" in diagnostics and summary == {}


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
