from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "scripts" / "benchmark_backend.py"
GENERATOR = ROOT / "scripts" / "generate_benchmark_config.py"


def load_backend():
    spec = importlib.util.spec_from_file_location("benchmark_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def fake_client(tmp_path: Path) -> Path:
    path = tmp_path / "client.py"
    path.write_text("""#!/usr/bin/env python3
import json, os, sys
job = json.load(sys.stdin)
mode = os.environ.get("FAKE_MODE", "ok")
print("client diagnostic", file=sys.stderr)
if mode == "echo":
    print(json.dumps({"status": "ok", "diagnostics": "", "job": job,
                      "latency_us": 12.5, "passed": True,
                      "device": job["device"], "handle": "gz-a3:job-1"}))
elif mode in {"compile_error", "runtime_error", "correctness_error", "infrastructure_error"}:
    print(json.dumps({"status": mode, "diagnostics": mode + " details",
                      "handle": "gz-a3:job-2", "device": job["device"]}))
else:
    print("not-json")
""")
    path.chmod(0o755)
    return path


def request(benchmark="gdn", action="profile", **extra):
    default = {"protocol_version": 1, "action": action, "benchmark": benchmark,
               "device": 0 if benchmark == "gdn" else 1}
    if action in {"profile", "measure"}:
        default.update(case=40 if benchmark == "gdn" else 47, iteration=0)
    else:
        module = load_backend()
        default.update(cases=module.ALL_CASES, scope="full", round=3)
    default.update(extra)
    return default


def run_backend(tmp_path: Path, payload: dict, benchmark="gdn", mode="echo"):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# candidate\n")
    client = fake_client(tmp_path)
    return subprocess.run(
        [sys.executable, str(BACKEND), "--benchmark", benchmark, "--candidate", str(candidate),
         "--job-client", str(client)], input=json.dumps(payload), text=True,
        capture_output=True, env={**__import__("os").environ, "FAKE_MODE": mode}, check=False)


def test_pinned_assets_are_exact_and_have_fifty_cases():
    module = load_backend()
    expected = {
        "gdn": ("498a0b0c255c883e4307801de09db7479d06415daf33084e919cd3ea253c0b4f",
                "e720bf1d97942e9b2a1e0020745cdbb862e6bb9a5ee88b49bd5effcd3003cd67"),
        "bsa": ("37db28c5be1b5af617dcc556cdf26d33667f90db6c56f177099a61727ca65c6e",
                "63f2c51b7d5552313979f8620688910b1d73c1c492140ad0473fcc6f26e11806"),
    }
    import hashlib
    for name, hashes in expected.items():
        spec = module.BENCHMARKS[name]
        files = [ROOT / spec["asset"], ROOT / spec["cases"]]
        assert tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in files) == hashes
        assert len([line for line in files[1].read_text(encoding="utf-8-sig").splitlines() if line]) == 50


def test_profile_job_has_exact_binding_and_msprof_selector(tmp_path: Path):
    run = run_backend(tmp_path, request(), mode="echo")
    assert run.returncode == 0
    result = json.loads(run.stdout)
    job = result["job"]
    assert job["profile"] == "gz-a3"
    assert job["runtime"] == "py311-torch"
    assert (job["device"], job["logical_device"]) == (0, 0)
    assert job["case"] == 40
    assert job["reference_revision"] == "a42c54b916189500e2f7cb47640980f230f2eb65"
    assert job["profiling"] == {
        "driver": str((ROOT / "scripts/profile_a3.py").resolve()),
        "tool": "msprof op", "captures": 1, "aic_metrics": "BasicInfo", "warm_up": 3,
        "launch_count": 1, "replay_mode": "kernel", "kernel_selector_source": "candidate-manifest",
    }
    assert result["handle"] == "gz-a3:job-1"
    assert "client diagnostic" in result["diagnostics"]


@pytest.mark.parametrize("status", ["compile_error", "runtime_error", "correctness_error"])
def test_candidate_failures_and_diagnostics_are_preserved(tmp_path: Path, status: str):
    result = json.loads(run_backend(tmp_path, request(), mode=status).stdout)
    assert result["status"] == status
    assert result["handle"] == "gz-a3:job-2"
    assert status + " details" in result["diagnostics"]
    assert "client diagnostic" in result["diagnostics"]


def test_environment_failure_is_separate(tmp_path: Path):
    result = json.loads(run_backend(tmp_path, request(), mode="infrastructure_error").stdout)
    assert result["status"] == "infrastructure_error"
    assert result["handle"] == "gz-a3:job-2"


@pytest.mark.parametrize("change,message", [
    ({"device": 1}, "device binding"),
    ({"case": 0}, "invalid profile case"),
    ({"benchmark": "bsa"}, "does not match"),
])
def test_adapter_rejects_controller_drift(tmp_path: Path, change: dict, message: str):
    payload = request()
    payload.update(change)
    result = json.loads(run_backend(tmp_path, payload).stdout)
    assert result["status"] == "infrastructure_error"
    assert message in result["diagnostics"]


def test_full_check_requires_ordered_all_fifty_and_forwards_it(tmp_path: Path):
    run = run_backend(tmp_path, request(action="check"), mode="echo")
    job = json.loads(run.stdout)["job"]
    assert job["cases"] == list(range(50))
    assert (job["scope"], job["round"]) == ("full", 3)


def test_missing_candidate_is_compile_failure(tmp_path: Path):
    client = fake_client(tmp_path)
    run = subprocess.run(
        [sys.executable, str(BACKEND), "--benchmark", "gdn", "--candidate", str(tmp_path / "missing.py"),
         "--job-client", str(client)], input=json.dumps(request()), text=True, capture_output=True, check=False)
    result = json.loads(run.stdout)
    assert result["status"] == "compile_error"
    assert "does not exist" in result["diagnostics"]


def test_generator_emits_exact_controller_cells(tmp_path: Path):
    output = tmp_path / "cells.json"
    subprocess.run([sys.executable, str(GENERATOR), "--job-client", "/bin/true", "--output", str(output)], check=True)
    cells = json.loads(output.read_text())["cells"]
    assert cells["gdn"]["development_cases"] == [40, 49, 47, 46, 45]
    assert cells["bsa"]["development_cases"] == [47, 46, 49, 44, 43]
    assert (cells["gdn"]["device"], cells["bsa"]["device"]) == (0, 1)
    assert cells["gdn"]["all_cases"] == list(range(50))
