from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bz_a3_job_client.py"
BACKEND = ROOT / "scripts/benchmark_backend.py"


def load():
    spec = importlib.util.spec_from_file_location("bz_a3_job_client", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_backend():
    spec = importlib.util.spec_from_file_location("benchmark_backend_for_bz", BACKEND)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
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
        rows = [
            {"case": 40, "samples_us": [8.0, 9.0, 10.0], "median_us": 9.0},
            {"case": 49, "samples_us": [10.0, 11.0, 12.0], "median_us": 11.0},
        ]
        captures = [
            {"case": row["case"], "iteration": iteration,
             "duration_us": sample, "kernel_name": "gdn_kernel",
             "evidence_sha256": "a" * 64, "msprof_log_sha256": "b" * 64}
            for row in rows for iteration, sample in enumerate(row["samples_us"])
        ]
        payload = {
            "status": self.status,
            "diagnostics": "NameError: tl" if self.status == "compile_error" else "",
            "benchmark": "gdn", "action": "profile", "device": 0,
            "cases": [40, 49], "repeats": 3, "round": 1,
            "kernel_name": "gdn_kernel", "passed": self.status == "ok",
            "profile_cases": rows,
            "profile": {
                "schema_version": 1, "status": "success", "profiler": "msprof-op",
                "kernel_name": "gdn_kernel", "repeats": 3,
                "cases": rows, "captures": captures,
                "geomean_us": math.sqrt(99.0),
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
        remote_root="/srv/profiling-skill-production",
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
                remote_root="/srv/profiling-skill-production",
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
                       "benchmark": "gdn", "action": job["action"], "device": 0}
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
        job = {**base, "action": action, "device": 2, "logical_device": 0}
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


def test_configured_remote_root_owns_staging_runs_and_evidence(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    remote_root = "/srv/campaigns/profiling-skill"
    subject = module.BzA3JobClient(
        transport, tmp_path / "state", {"0": {"profile": "bz-a3-2", "device": 3}},
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        profiler=ROOT / "scripts/profile_a3.py",
        batch_profiler=ROOT / "scripts/batch_profile_a3.py",
        remote_root=remote_root,
    )

    result = subject.run(profile_job(tmp_path))

    assert result["status"] == "ok"
    assert transport.uploads[0][2].startswith(remote_root + "/staging/")
    assert result["artifacts"]["remote_run_root"].startswith(remote_root + "/runs/")
    assert result["artifacts"]["remote_profile_evidence"].startswith(
        remote_root + "/runs/")
    assert remote_root in transport.executions[0][3]


@pytest.mark.parametrize("remote_root", [
    "relative/path", "/srv/../escape", "/srv/root;touch-pwned", "/srv/root\nnext",
])
def test_unsafe_remote_root_is_rejected_before_transport(
        tmp_path: Path, remote_root: str):
    module = load()
    transport = FakeTransport(module)
    with pytest.raises(module.JobError, match="remote root"):
        module.BzA3JobClient(
            transport, tmp_path / "state",
            {"0": {"profile": "bz-a3-2", "device": 3}},
            runner=ROOT / "scripts/a3_benchmark_runner.py",
            profiler=ROOT / "scripts/profile_a3.py",
            batch_profiler=ROOT / "scripts/batch_profile_a3.py",
            remote_root=remote_root,
        )
    assert transport.uploads == [] and transport.executions == []


def production_client(module, transport, state: Path, remote_root: str):
    return module.BzA3JobClient(
        transport, state, {"0": {"profile": "bz-a3-2", "device": 3}},
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        profiler=ROOT / "scripts/profile_a3.py",
        batch_profiler=ROOT / "scripts/batch_profile_a3.py",
        remote_root=remote_root,
    )


def test_remote_root_drift_never_observes_foreign_retained_dispatch(tmp_path: Path):
    module = load()
    state = tmp_path / "state"
    root_a = "/srv/campaigns/root-a"
    root_b = "/srv/campaigns/root-b"
    interrupted = FakeTransport(module, fail="observer")
    first = production_client(module, interrupted, state, root_a).run(
        profile_job(tmp_path))
    assert first["handle"] == "bz-a3-2:retained"

    replacement = FakeTransport(module)
    second = production_client(module, replacement, state, root_b).run(
        profile_job(tmp_path))

    assert second["status"] == "ok"
    assert replacement.observations == []
    assert len(replacement.uploads) == len(replacement.executions) == 1
    assert replacement.uploads[0][2].startswith(root_b + "/staging/")
    assert len(list(state.glob("*/dispatch.json"))) == 1
    assert len(list(state.glob("*/completed.json"))) == 1


def test_remote_root_drift_never_reuses_foreign_completed_receipt(tmp_path: Path):
    module = load()
    state = tmp_path / "state"
    root_a = "/srv/campaigns/root-a"
    root_b = "/srv/campaigns/root-b"
    first_transport = FakeTransport(module)
    first = production_client(module, first_transport, state, root_a).run(
        profile_job(tmp_path))

    second_transport = FakeTransport(module)
    second = production_client(module, second_transport, state, root_b).run(
        profile_job(tmp_path))

    assert first["artifacts"]["request_digest"] != second["artifacts"]["request_digest"]
    assert first["artifacts"]["remote_run_root"].startswith(root_a + "/runs/")
    assert second["artifacts"]["remote_run_root"].startswith(root_b + "/runs/")
    assert len(second_transport.uploads) == len(second_transport.executions) == 1
    assert second_transport.observations == []
    assert len(list(state.glob("*/completed.json"))) == 2


def test_actual_backend_nonzero_campaign_device_round_trips_for_all_actions(
        monkeypatch, tmp_path: Path):
    module = load()
    backend = load_backend()

    class BackendTransport(FakeTransport):
        def completed(self, profile):
            job = self.controller_job
            payload = {"status": "ok", "diagnostics": "", "passed": True,
                       "benchmark": job["benchmark"], "action": job["action"],
                       "device": 0}
            if job["action"] == "check":
                payload.update(cases=job["cases"], scope=job["scope"])
            elif job["action"] == "measure":
                payload.update(case=job["case"], phase=job["phase"], latency_us=8.5)
            else:
                cases, repeats = job["cases"], job["repeats"]
                rows = [{"case": case, "samples_us": [8.0] * repeats,
                         "median_us": 8.0} for case in cases]
                captures = [{"case": case, "iteration": iteration,
                             "duration_us": 8.0,
                             "kernel_name": job["profiling"]["kernel_name"]}
                            for case in cases for iteration in range(repeats)]
                payload.update(
                    cases=cases, repeats=repeats, round=job["round"],
                    kernel_name=job["profiling"]["kernel_name"],
                    profile_cases=rows,
                    profile={"status": "success", "profiler": "msprof-op",
                             "kernel_name": job["profiling"]["kernel_name"],
                             "repeats": repeats, "cases": rows,
                             "captures": captures, "geomean_us": 8.0},
                )
            return self.module.CommandResult(
                0, "BZ_PRODUCTION_RESULT=" + json.dumps(payload) + "\n", ""), f"{profile}:x"

    transport = BackendTransport(module)
    subject = module.BzA3JobClient(
        transport, tmp_path / "state", {"2": {"profile": "bz-a3-1", "device": 5}},
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        profiler=ROOT / "scripts/profile_a3.py",
        batch_profiler=ROOT / "scripts/batch_profile_a3.py",
        remote_root="/srv/profiling-skill-production",
    )

    def route(_command, *, input, **_kwargs):
        transport.controller_job = json.loads(input)
        result = subject.run(transport.controller_job)
        return subprocess.CompletedProcess([], 0 if result["status"] == "ok" else 2,
                                           json.dumps(result), "")

    monkeypatch.setattr(backend.subprocess, "run", route)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# candidate\n")
    requests = [
        {"action": "check", "device": 2, "cases": [40], "scope": "full"},
        {"action": "measure", "device": 2, "case": 40, "phase": "sample"},
        {"action": "profile", "device": 2, "cases": [40], "repeats": 3,
         "round": 1},
    ]
    for request in requests:
        job = backend.make_job(
            request, "gdn", candidate, ROOT,
            "gdn_kernel" if request["action"] == "profile" else None,
        )
        result = backend.invoke(["bz-client"], job, 60)
        assert result["status"] == "ok"
        assert result["device"] == 2
        assert result["placement"] == {"profile": "bz-a3-1", "device": 5}
        assert transport.controller_job["logical_device"] == 0


@pytest.mark.parametrize("corruption", [
    "missing", "kernel", "repeats", "cases", "sample_count", "nonfinite",
    "captures", "geomean", "non_object_row",
])
def test_profile_rejects_malformed_compact_evidence_before_receipting(
        tmp_path: Path, corruption: str):
    module = load()

    class MalformedTransport(FakeTransport):
        def completed(self, profile):
            result, handle = super().completed(profile)
            payload = json.loads(result.stdout.removeprefix("BZ_PRODUCTION_RESULT="))
            evidence = payload["profile"]
            if corruption == "missing":
                payload.pop("profile")
            elif corruption == "kernel":
                evidence["kernel_name"] = "other_kernel"
            elif corruption == "repeats":
                evidence["repeats"] = 2
            elif corruption == "cases":
                evidence["cases"] = list(reversed(evidence["cases"]))
            elif corruption == "sample_count":
                evidence["cases"][0]["samples_us"].pop()
            elif corruption == "nonfinite":
                evidence["cases"][0]["samples_us"][0] = float("nan")
            elif corruption == "captures":
                evidence["captures"].pop()
            elif corruption == "geomean":
                evidence["geomean_us"] = float("inf")
            else:
                payload["profile_cases"].append("not-a-case-row")
                evidence["cases"].append("not-a-case-row")
            return self.module.CommandResult(
                0, "BZ_PRODUCTION_RESULT=" + json.dumps(payload) + "\n", ""), handle

    transport = MalformedTransport(module)
    _module, subject = client(tmp_path, transport)
    result = subject.run(profile_job(tmp_path))
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "profile_tool_error"
    assert not list((tmp_path / "state").glob("*/completed.json"))
