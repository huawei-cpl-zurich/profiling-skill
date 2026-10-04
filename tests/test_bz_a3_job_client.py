from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bz_a3_job_client.py"


def load():
    spec = importlib.util.spec_from_file_location("bz_a3_job_client", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTransport:
    def __init__(self, module, status="ok", fail=None):
        self.module, self.status, self.fail = module, status, fail
        self.uploads, self.executions, self.observations = [], [], []

    def upload(self, profile, source, destination, timeout):
        self.uploads.append((profile, source, destination, timeout))
        if self.fail == "upload":
            raise self.module.JobError("staging_error", "cpl-remote upload failed")

    def execute(self, profile, device, operation, script, timeout):
        self.executions.append((profile, device, operation, script, timeout))
        if self.fail == "observer":
            raise self.module.JobError("observer_error", "observer interrupted",
                                       f"{profile}:retained")
        return self.completed(profile)

    def observe(self, profile, handle, timeout):
        self.observations.append((profile, handle, timeout))
        return self.completed(profile)[0]

    def completed(self, profile):
        payload = {
            "status": self.status,
            "diagnostics": "NameError: tl" if self.status == "compile_error" else "",
            "benchmark": "gdn", "action": "profile", "device": 0,
            "cases": [40, 49], "repeats": 3, "round": 1,
            "kernel_name": "gdn_kernel", "passed": self.status == "ok",
            "profile_cases": [
                {"case": 40, "samples_us": [8.0, 9.0, 10.0], "median_us": 9.0},
                {"case": 49, "samples_us": [10.0, 11.0, 12.0], "median_us": 11.0},
            ],
            "profile": {
                "schema_version": 1, "status": "success", "profiler": "msprof-op",
                "kernel_name": "gdn_kernel", "repeats": 3,
                "geomean_us": 9.949874371,
            },
        }
        if self.status != "ok":
            payload.pop("profile", None)
            payload.pop("profile_cases", None)
        line = "BZ_PRODUCTION_RESULT=" + json.dumps(payload) + "\n"
        return self.module.CommandResult(0, line, ""), f"{profile}:job-1"


def files(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name in ("candidate.py", "baseline.py", "cases.jsonl"):
        path = tmp_path / name
        path.write_text("{}\n" if name.endswith("jsonl") else "# source\n")
        assets[name] = str(path)
    (tmp_path / "baseline.json").write_text('{"metadata": true}\n')
    return assets


def profile_job(tmp_path: Path):
    assets = files(tmp_path)
    return {
        "protocol_version": 1, "profile": "gz-a3", "runtime": "py311-torch",
        "benchmark": "gdn", "action": "profile", "device": 0,
        "logical_device": 0, "candidate": assets["candidate.py"],
        "baseline": assets["baseline.py"], "case_spec": assets["cases.jsonl"],
        "cases": [40, 49], "repeats": 3, "round": 1,
        "tolerances": {"rtol": 0.02, "atol": 0.02},
        "profiling": {"kernel_name": "gdn_kernel", "tool": "msprof op"},
    }


def client(tmp_path: Path, transport, placements=None):
    module = transport.module if transport is not None else load()
    placements = placements or {"0": {"profile": "bz-a3-2", "device": 3}}
    return module, module.BzA3JobClient(
        transport or FakeTransport(module), tmp_path / "state", placements,
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        profiler=ROOT / "scripts/profile_a3.py",
        batch_profiler=ROOT / "scripts/batch_profile_a3.py",
    )


def test_profile_stages_supplement_and_returns_compact_remote_evidence(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    _module, subject = client(tmp_path, transport)

    result = subject.run(profile_job(tmp_path))

    assert result["status"] == "ok"
    assert result["device"] == 0
    assert result["placement"] == {"profile": "bz-a3-2", "device": 3}
    assert result["handle"] == "bz-a3-2:job-1"
    assert result["profile"]["profiler"] == "msprof-op"
    assert len(result["artifacts"]["placements_sha256"]) == 64
    assert result["artifacts"]["remote_profile_evidence"].endswith("/profile/evidence.json")
    assert "vendor_report" not in result["artifacts"]
    assert len(transport.uploads) == 1
    with tarfile.open(transport.uploads[0][1], "r") as archive:
        assert set(archive.getnames()) == {
            "candidate.py", "baseline.py", "baseline.json", "cases.jsonl",
            "runner.py", "profile_a3.py", "batch_profile_a3.py", "job.json",
        }
    profile, device, _operation, script, _timeout = transport.executions[0]
    assert (profile, device) == ("bz-a3-2", 3)
    assert "batch_profile_a3.py" in script
    assert "--kernel-name gdn_kernel" in script


def test_completed_receipt_is_idempotent_and_receipt_drift_is_rejected(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    _module, subject = client(tmp_path, transport)
    job = profile_job(tmp_path)
    first = subject.run(job)
    second = subject.run(job)
    assert second == first
    assert len(transport.executions) == 1

    receipt = next((tmp_path / "state").glob("*/completed.json"))
    record = json.loads(receipt.read_text())
    record["request_sha256"] = "0" * 64
    receipt.write_text(json.dumps(record))
    drift = subject.run(job)
    assert drift["status"] == "infrastructure_error"
    assert drift["failure_type"] == "request_error"
    assert "request" in drift["diagnostics"]
    assert len(transport.executions) == 1


def test_retained_dispatch_is_observed_without_duplicate_upload_or_execute(tmp_path: Path):
    module = load()
    interrupted = FakeTransport(module, fail="observer")
    _module, first_client = client(tmp_path, interrupted)
    first = first_client.run(profile_job(tmp_path))
    assert first["status"] == "infrastructure_error"
    assert first["failure_type"] == "observer_error"
    assert first["handle"] == "bz-a3-2:retained"

    resumed = FakeTransport(module)
    _module, second_client = client(tmp_path, resumed)
    result = second_client.run(profile_job(tmp_path))
    assert result["status"] == "ok"
    assert resumed.uploads == [] and resumed.executions == []
    assert resumed.observations[0][:2] == ("bz-a3-2", "bz-a3-2:retained")


def test_counted_candidate_failures_are_not_infrastructure(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, status="compile_error")
    _module, subject = client(tmp_path, transport)
    result = subject.run(profile_job(tmp_path))
    assert result["status"] == "compile_error"
    assert result["failure_type"] == "compile_error"
    assert result["placement"]["device"] == 3


def test_placement_and_profile_contract_are_validated_before_transport(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    bad_placements = [
        {},
        {"0": {"profile": "gz-a3", "device": 0}},
        {"0": {"profile": "bz-a3-1", "device": -1}},
        {"0": {"profile": "bz-a3-1", "device": True}},
    ]
    for index, placements in enumerate(bad_placements):
        try:
            module.BzA3JobClient(
                transport, tmp_path / f"state-{index}", placements,
                runner=ROOT / "scripts/a3_benchmark_runner.py",
                profiler=ROOT / "scripts/profile_a3.py",
                batch_profiler=ROOT / "scripts/batch_profile_a3.py",
            )
        except module.JobError as exc:
            assert exc.failure_type == "request_error"
        else:
            raise AssertionError("invalid placements must be rejected")
    assert transport.uploads == [] and transport.executions == []


def test_check_and_measure_use_runner_and_preserve_logical_identity(tmp_path: Path):
    module = load()

    class ActionTransport(FakeTransport):
        def completed(self, profile):
            job = self.current
            payload = {"status": "ok", "diagnostics": "", "passed": True,
                       "benchmark": "gdn", "action": job["action"], "device": 2}
            if job["action"] == "check":
                payload.update(cases=[40], scope="full")
            else:
                payload.update(case=40, phase="sample", latency_us=8.5)
            return self.module.CommandResult(
                0, "BZ_PRODUCTION_RESULT=" + json.dumps(payload) + "\n", ""), f"{profile}:x"

    transport = ActionTransport(module)
    _module, subject = client(tmp_path, transport,
                              {"2": {"profile": "bz-a3-1", "device": 1}})
    base = profile_job(tmp_path)
    for action in ("check", "measure"):
        job = {**base, "action": action, "device": 2, "logical_device": 2}
        job.pop("profiling")
        job.pop("repeats")
        job.pop("round")
        if action == "check":
            job.update(cases=[40], scope="full")
        else:
            job.pop("cases")
            job.update(case=40, phase="sample")
        transport.current = job
        result = subject.run(job)
        assert result["status"] == "ok" and result["device"] == 2
    assert all("runner.py" in call[3] for call in transport.executions)


def test_upload_and_profile_tool_failures_are_infrastructure(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, fail="upload")
    _module, subject = client(tmp_path, transport)
    failed = subject.run(profile_job(tmp_path))
    assert failed["status"] == "infrastructure_error"
    assert failed["failure_type"] == "staging_error"

    transport = FakeTransport(module, status="infrastructure_error")
    _module, subject = client(tmp_path / "other", transport)
    failed = subject.run(profile_job(tmp_path / "other"))
    assert failed["status"] == "infrastructure_error"
    assert failed["failure_type"] == "profile_tool_error"
