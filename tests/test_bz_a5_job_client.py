from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bz_a5_job_client.py"
BACKEND = ROOT / "scripts/benchmark_backend.py"


def load():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("bz_a5_job_client", MODULE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def load_backend():
    spec = importlib.util.spec_from_file_location("a5_benchmark_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class FakeTransport:
    expected_sha256 = "f" * 64

    def __init__(self, module, status="ok", fail_observe=False):
        self.module, self.status, self.fail_observe = module, status, fail_observe
        self.uploads, self.executions, self.observations = [], [], []

    def upload(self, target, source, destination, timeout):
        self.uploads.append((target, source, destination, timeout))

    def dispatch(self, target, device, runtime, operation, script, timeout):
        self.executions.append((target, device, runtime, operation, script, timeout))
        return "remote:bz-a5:job:retained" if self.fail_observe else "remote:bz-a5:job:one"

    def observe(self, target, handle, timeout):
        self.observations.append((target, handle, timeout))
        if self.fail_observe:
            raise self.module.JobError("observer_error", "interrupted", handle)
        rows = [{"case": 3, "samples_us": [5.0, 6.0, 7.0], "median_us": 6.0}]
        captures = [
            {"case": 3, "iteration": index, "duration_us": value,
             "kernel_name": "fused_kernel", "evidence_sha256": "a" * 64,
             "msprof_log_sha256": "b" * 64}
            for index, value in enumerate(rows[0]["samples_us"])
        ]
        result = {
            "status": self.status, "diagnostics": "",
            "benchmark": "matmul", "action": "profile", "device": 0,
            "cases": [3], "repeats": 3, "round": 1,
            "kernel_name": "fused_kernel", "passed": self.status == "ok",
        }
        if self.status == "ok":
            result.update(profile_cases=rows, geomean_us=6.0, profile={
                "schema_version": 1, "status": "success", "profiler": "msprof-op",
                "target_family": "Ascend-A5",
                "kernel_name": "fused_kernel", "repeats": 3, "cases": rows,
                "captures": captures, "geomean_us": math.prod([6.0]),
            })
        return self.module.CommandResult(
            0, "BZ_PRODUCTION_RESULT=" + json.dumps(result) + "\n", "")


def job(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name in ("candidate.py", "baseline.py", "cases.jsonl"):
        path = tmp_path / name
        path.write_text("{}\n" if name.endswith("jsonl") else "# source\n")
        assets[name] = str(path)
    return {
        "protocol_version": 1, "product": "a5", "runtime": "cann91",
        "benchmark": "matmul", "action": "profile", "device": 0,
        "logical_device": 0, "candidate": assets["candidate.py"],
        "baseline": assets["baseline.py"], "case_spec": assets["cases.jsonl"],
        "cases": [3], "repeats": 3, "round": 1,
        "tolerances": {"rtol": 0.02, "atol": 0.02},
        "profiling": {"kernel_name": "fused_kernel", "tool": "msprof op"},
    }


def client(tmp_path: Path, transport):
    module = transport.module
    return module.BzA5JobClient(
        transport, tmp_path / "state", {"0": {"target": "bz-a5", "device": 4}},
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        profiler=ROOT / "scripts/profile_a3.py",
        batch_profiler=ROOT / "scripts/batch_profile_a3.py",
        remote_root="/home/mariodrumond/.profiling-skill/production",
    )


def test_a5_dispatch_stages_content_addressed_payload_and_returns_provenance(
        tmp_path: Path):
    module = load()
    transport = FakeTransport(module)

    result = client(tmp_path, transport).run(job(tmp_path))

    assert result["status"] == "ok"
    assert result["product"] == "a5"
    assert result["profile"]["target_family"] == "Ascend-A5"
    assert result["placement"] == {"target": "bz-a5", "device": 4}
    assert result["artifacts"]["execution_provenance"]["runtime"] == "cann91"
    assert result["artifacts"]["execution_provenance"]["product"] == "a5"
    assert result["artifacts"]["execution_provenance"]["staging"] == (
        "content-addressed-upload")
    assert result["artifacts"]["remote_profile_evidence"].endswith("/profile/evidence.json")
    assert len(transport.uploads) == 1
    upload_target, archive, remote_archive, _upload_timeout = transport.uploads[0]
    archive_sha = module._sha(archive)
    assert upload_target == "bz-a5"
    assert remote_archive.endswith(f"/payload-{archive_sha}.tar")
    target, device, runtime, _operation, script, _timeout = transport.executions[0]
    assert (target, device, runtime) == ("bz-a5", 4, "cann91")
    assert remote_archive in script
    assert archive_sha in script
    assert "base64" not in script
    assert len(script.encode()) < 8_192
    assert "ASCEND_RT_VISIBLE_DEVICES=4" not in script
    assert "vendor_report" not in json.dumps(result)


def test_large_a5_payload_is_uploaded_and_does_not_expand_dispatch_script(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    request = job(tmp_path)
    Path(request["candidate"]).write_bytes(b"# candidate\n" + b"x" * 2_000_000)

    result = client(tmp_path, transport).run(request)

    assert result["status"] == "ok"
    assert transport.uploads[0][1].stat().st_size > 2_000_000
    script = transport.executions[0][4]
    assert len(script.encode()) < 8_192
    assert "x" * 1_000 not in script


def test_remote_script_rejects_corrupt_staged_archive_before_extraction(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    subject = client(tmp_path, transport)
    archive = tmp_path / "expected.tar"
    archive.write_bytes(b"expected immutable payload")
    remote_archive = tmp_path / f"payload-{module._sha(archive)}.tar"
    remote_archive.write_bytes(b"corrupt payload")

    script = subject._remote_script(
        str(remote_archive), module._sha(archive), str(tmp_path / "run"), 60,
        job(tmp_path / "assets"),
    )
    result = subprocess.run(["bash", "-c", script], text=True,
                            capture_output=True, check=False)

    assert result.returncode == 91
    assert "payload-digest-mismatch" in result.stderr
    assert not (tmp_path / "run" / "work").exists()


def test_completed_a5_request_reuses_receipt_without_reupload_or_redispatch(
        tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    subject = client(tmp_path, transport)
    request = job(tmp_path)

    first = subject.run(request)
    second = subject.run(request)

    assert second == first
    assert len(transport.uploads) == 1
    assert len(transport.executions) == 1
    assert len(transport.observations) == 1


def test_a5_resume_observes_exact_handle_without_redispatch(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, fail_observe=True)
    subject = client(tmp_path, transport)
    request = job(tmp_path)

    first = subject.run(request)
    assert first["status"] == "infrastructure_error"
    assert first["handle"] == "remote:bz-a5:job:retained"
    transport.fail_observe = False
    second = subject.run(request)

    assert second["status"] == "ok"
    assert len(transport.uploads) == 1
    assert len(transport.executions) == 1
    assert transport.observations[-1][1] == "remote:bz-a5:job:retained"


def test_a5_rejects_wrong_product_runtime_and_target_before_dispatch(tmp_path: Path):
    module = load()
    for mutation in ({"product": "a3"}, {"runtime": "py311-torch"}):
        transport = FakeTransport(module)
        request = {**job(tmp_path / mutation[next(iter(mutation))]), **mutation}
        result = client(tmp_path / ("state-" + next(iter(mutation))), transport).run(request)
        assert result["status"] == "infrastructure_error"
        assert result["failure_type"] == "request_error"
        assert transport.executions == []

    transport = FakeTransport(module)
    try:
        module.BzA5JobClient(
            transport, tmp_path / "bad", {"0": {"target": "bz-a3-1", "device": 0}},
            runner=ROOT / "scripts/a3_benchmark_runner.py",
            profiler=ROOT / "scripts/profile_a3.py",
            batch_profiler=ROOT / "scripts/batch_profile_a3.py", remote_root="/remote",
        )
    except module.JobError as error:
        assert error.failure_type == "request_error"
    else:
        raise AssertionError("A5 client accepted an A3 placement")


def test_global_transport_uses_named_runtime_and_approved_upload(tmp_path: Path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []

    def invoke(argv, _timeout):
        calls.append(argv)
        state = "completed" if "upload" in argv else "running"
        payload = {"target": "bz-a5", "state": state}
        if state == "running":
            payload["handle"] = "remote:bz-a5:job:one"
        return module.CommandResult(0, json.dumps(payload) + "\n", "")

    transport = module.A5GlobalCplRemoteTransport(
        module._sha(executable), "/remote", invoke, executable=executable)
    source = tmp_path / "payload.tar"
    source.write_bytes(b"payload")
    transport.upload("bz-a5", source, "/remote/payload.tar", 60)
    handle = transport.dispatch("bz-a5", 7, "cann91", "operation", "echo ok", 60)

    assert handle == "remote:bz-a5:job:one"
    upload = calls[0]
    assert upload[1:4] == ["--json", "upload", "--timeout"]
    assert upload[-3:] == ["bz-a5", str(source), "/remote/payload.tar"]
    argv = calls[1]
    assert argv[1:4] == ["--json", "run", "bz-a5"]
    assert argv[argv.index("--runtime") + 1] == "cann91"
    assert "--file" in argv and "upload" not in argv and "download" not in argv
    script = Path(argv[argv.index("--file") + 1]).read_text() if Path(
        argv[argv.index("--file") + 1]).exists() else None
    assert script is None  # the exact temporary script is removed after dispatch


def test_a5_dispatch_timeout_preserves_emitted_handle(monkeypatch, tmp_path: Path):
    module = load()
    base = sys.modules["bz_a3_job_client"]
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    handle = "remote:bz-a5:job:20261010T170000Z-retained"

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            [str(executable)], 60, output=(handle + "\n").encode())

    monkeypatch.setattr(base.subprocess, "run", timeout)
    transport = module.A5GlobalCplRemoteTransport(
        module._sha(executable), "/remote", executable=executable)
    try:
        transport.dispatch("bz-a5", 2, "cann91", "operation", "true", 60)
    except module.JobError as error:
        assert error.failure_type == "observer_error"
        assert error.handle == handle
        assert error.dispatch_uncertain is False
    else:
        raise AssertionError("timeout after dispatch did not preserve the A5 handle")


def test_backend_builds_product_specific_a5_job(tmp_path: Path):
    backend = load_backend()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# candidate\n")
    request = {"action": "check", "device": 0, "cases": list(range(10)),
               "scope": "full"}

    value = backend.make_job(
        request, "matmul", candidate, ROOT, product="a5", runtime="cann91")

    assert value["product"] == "a5"
    assert value["runtime"] == "cann91"
    assert value["logical_device"] == 0


def test_backend_cli_defaults_a5_to_cann91(monkeypatch):
    backend = load_backend()
    monkeypatch.setattr(sys, "argv", [
        "benchmark_backend.py", "--product", "a5", "--benchmark", "matmul",
        "--job-client-json", '["client"]',
    ])
    assert backend.parse_args().runtime == "cann91"


@pytest.mark.parametrize("product,runtime", [
    ("a3", "cann91"), ("a5", "py311-torch"),
])
def test_backend_cli_rejects_crossed_product_runtime(monkeypatch, product, runtime):
    backend = load_backend()
    monkeypatch.setattr(sys, "argv", [
        "benchmark_backend.py", "--product", product, "--runtime", runtime,
        "--benchmark", "matmul", "--job-client-json", '["client"]',
    ])
    with pytest.raises(SystemExit) as failure:
        backend.parse_args()
    assert failure.value.code == 2
