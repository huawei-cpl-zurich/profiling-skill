from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bz_a3_diagnostic_client.py"


def load():
    spec = importlib.util.spec_from_file_location("bz_a3_diagnostic_client", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTransport:
    def __init__(self, module, mode="ok"):
        self.module, self.mode = module, mode
        self.uploads = []
        self.executions = []
        self.observations = []

    def upload(self, profile, source, destination, timeout):
        self.uploads.append((profile, source, destination, timeout))
        if self.mode == "staging":
            raise self.module.DiagnosticError("staging_error", "rsync unavailable")

    def execute(self, profile, device, operation, script, timeout):
        self.executions.append((profile, device, operation, script, timeout))
        if self.mode == "transport":
            raise self.module.DiagnosticError("transport_error", "vpn unavailable")
        if self.mode == "observer":
            raise self.module.DiagnosticError("observer_error", "retained job is not terminal", f"{profile}:kept")
        if self.mode == "candidate_timeout":
            return self.module.CommandResult(124, "timed out", ""), f"{profile}:timeout"
        if self.mode == "device":
            return self.module.CommandResult(1, "", "NPU device unavailable"), f"{profile}:device"
        if self.mode == "digest":
            return self.module.CommandResult(91, "", "common-digest-mismatch"), f"{profile}:digest"
        status = self.mode if self.mode in {"compile_error", "runtime_error", "correctness_error"} else "ok"
        diagnostics = "Traceback\n  File candidate.py, line 17\nNameError: tl" if status == "compile_error" else ""
        payload = {"status": status, "diagnostics": diagnostics, "passed": status == "ok",
                   "case_evidence": [{"case": 0, "passed": status == "ok", "host_elapsed_us": 99.0}],
                   "host_elapsed_us": 99.0}
        stdout = "BZ_DIAGNOSTIC_RESULT=" + json.dumps(payload) + "\n"
        return self.module.CommandResult(0, stdout, ""), f"{profile}:job-1"

    def observe(self, profile, handle, timeout):
        self.observations.append((profile, handle, timeout))
        payload = {"status": "ok", "passed": True, "diagnostics": "",
                   "case_evidence": [{"case": 0, "passed": True}]}
        return self.module.CommandResult(
            0, "BZ_DIAGNOSTIC_RESULT=" + json.dumps(payload) + "\n", "")


def request(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("candidate", "candidate_manifest", "baseline", "case_spec", "runner"):
        suffix = ".json" if "manifest" in name else ".py"
        path = tmp_path / f"{name}{suffix}"
        path.write_text("{}\n" if "manifest" in name else "# source\n")
        files[name] = str(path)
    return {"campaign": "quick", "wave": 1, "cell": "arm-a", "profile": "bz-a3-1",
            "device": 3, "timeout": 30, "cases": list(range(7)), **files}


def run(tmp_path: Path, mode="ok"):
    module = load()
    transport = FakeTransport(module, mode)
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(request(tmp_path))
    return module, transport, result


def test_success_stages_content_addressed_assets_and_uses_logical_zero(tmp_path: Path):
    _module, transport, result = run(tmp_path)
    assert result["status"] == "ok"
    assert result["failure_type"] == "success"
    assert result["handle"] == "bz-a3-1:job-1"
    assert len(transport.uploads) == 2
    common, candidate = transport.uploads
    assert f"diagnostic-common-{result['artifacts']['common_sha256']}.tar" in common[2]
    assert "diagnostic-candidate-quick-1-arm-a-" in candidate[2]
    assert result["artifacts"]["candidate_sha256"] in candidate[2]
    profile, device, _operation, script, timeout = transport.executions[0]
    assert (profile, device, timeout) == ("bz-a3-1", 3, 30)
    job_line = next(line for line in script.splitlines() if "job.json" in line and "printf" in line)
    assert '"device":0' in job_line and '"logical_device":0' in job_line
    assert "host_elapsed_us" not in result
    assert "host_elapsed_us" not in result["case_evidence"][0]


def test_verified_common_archive_is_reused(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    assert client.run(value)["status"] == "ok"
    assert client.run(value)["status"] == "ok"
    common_uploads = [upload for upload in transport.uploads if "diagnostic-common-" in upload[2]]
    candidate_uploads = [upload for upload in transport.uploads if "diagnostic-candidate-" in upload[2]]
    assert len(common_uploads) == 1
    assert len(candidate_uploads) == 2


def test_compiler_traceback_is_complete_and_counted(tmp_path: Path):
    _module, _transport, result = run(tmp_path, "compile_error")
    assert result["status"] == result["failure_type"] == "compile_error"
    assert "candidate.py, line 17" in result["diagnostics"]
    assert result["handle"] == "bz-a3-1:job-1"


def test_runtime_and_correctness_are_candidate_failures(tmp_path: Path):
    for mode in ("runtime_error", "correctness_error"):
        _module, _transport, result = run(tmp_path / mode, mode)
        assert result["status"] == result["failure_type"] == mode


def test_missing_submission_is_counted_without_remote_dispatch(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    value = request(tmp_path)
    Path(value["candidate_manifest"]).unlink()
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
    assert result["status"] == "submission_error"
    assert result["failure_type"] == "missing_submission"
    assert not transport.uploads and not transport.executions


def test_candidate_timeout_is_not_infrastructure(tmp_path: Path):
    _module, _transport, result = run(tmp_path, "candidate_timeout")
    assert result["status"] == "candidate_timeout"
    assert result["failure_type"] == "candidate_timeout"
    assert result["handle"] == "bz-a3-1:timeout"


def test_infrastructure_failures_are_disjoint(tmp_path: Path):
    expected = {"staging": "staging_error", "transport": "transport_error",
                "observer": "observer_error", "device": "device_error",
                "digest": "digest_mismatch"}
    for mode, failure in expected.items():
        _module, _transport, result = run(tmp_path / mode, mode)
        assert result["status"] == "infrastructure_error"
        assert result["failure_type"] == failure


def test_adapter_transport_observes_same_handle_after_interruption():
    module = load()
    calls = []

    def invoke(argv, _timeout):
        calls.append(argv)
        if "observe" in argv:
            return module.CommandResult(0, "CATLASS_VALIDATION_STATE=completed\n", "")
        return module.CommandResult(75, "CATLASS_VALIDATION_STATE=observation-unavailable\n",
                                    "next bz-a3-2:retained-7")

    transport = module.AdapterTransport(["cpl-remote"], ["adapter"], invoke)
    result, handle = transport.execute("bz-a3-2", 6, "diagnostic-x", "true", 30)
    assert result.returncode == 0
    assert handle == "bz-a3-2:retained-7"
    assert calls[1] == ["adapter", "--profile", "bz-a3-2", "observe", "--handle", handle]
    assert sum("run" in call for call in calls) == 1


def test_client_observes_retained_handle_without_upload_or_execution(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    handle = "bz-a3-1:retained-9"

    result = client.observe(value, handle)

    assert result["status"] == "ok" and result["failure_type"] == "success"
    assert result["handle"] == handle
    assert transport.observations == [("bz-a3-1", handle, 30)]
    assert not transport.uploads and not transport.executions


def test_dispatch_observer_timeout_retains_handle_for_observe_only_resume(tmp_path: Path):
    module = load()
    calls = []
    observe_attempts = 0
    handle = "bz-a3-1:retained-timeout"

    def invoke(argv, timeout):
        nonlocal observe_attempts
        calls.append(argv)
        if "observe" in argv:
            observe_attempts += 1
            if observe_attempts == 1:
                raise subprocess.TimeoutExpired(argv, timeout)
            payload = {"status": "ok", "passed": True, "diagnostics": "",
                       "case_evidence": [{"case": 0, "passed": True}]}
            return module.CommandResult(
                0, "BZ_DIAGNOSTIC_RESULT=" + json.dumps(payload) + "\n", "")
        if "run" in argv:
            return module.CommandResult(
                75, "CATLASS_VALIDATION_STATE=observation-unavailable\n" + handle + "\n", "")
        return module.CommandResult(0, "", "")

    transport = module.AdapterTransport(["remote"], ["adapter"], invoke)
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)

    interrupted = client.run(value)
    resumed = client.observe(value, interrupted["handle"])

    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["failure_type"] == "transport_error"
    assert interrupted["handle"] == handle
    assert resumed["status"] == "ok" and resumed["handle"] == handle
    assert sum("run" in argv for argv in calls) == 1
    assert sum("observe" in argv for argv in calls) == 2


def test_dispatch_observer_error_reattaches_extracted_handle():
    module = load()
    handle = "bz-a3-2:retained-observer"

    def invoke(argv, _timeout):
        if "observe" in argv:
            raise module.DiagnosticError("observer_error", "listener unavailable")
        return module.CommandResult(
            75, "CATLASS_VALIDATION_STATE=observation-unavailable\n" + handle + "\n", "")

    transport = module.AdapterTransport(["remote"], ["adapter"], invoke)
    try:
        transport.execute("bz-a3-2", 0, "diagnostic", "true", 30)
    except module.DiagnosticError as error:
        assert error.failure_type == "observer_error"
        assert error.handle == handle
    else:
        raise AssertionError("observer failure was not propagated")


def test_remote_script_verifies_both_digests_and_bounds_execution():
    module = load()
    script = module._remote_script("/common.tar", "a" * 64, "/candidate.tar", "b" * 64,
                                   "/run", 42, {"device": 0})
    assert script.count("sha256sum") == 2
    assert "timeout --signal=TERM --kill-after=10 42" in script
    assert "BZ_DIAGNOSTIC_RESULT=" in script
