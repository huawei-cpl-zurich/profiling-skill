from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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
        self.stale_returned = False
        self.observer_returned = False

    def upload(self, profile, source, destination, timeout):
        self.uploads.append((profile, source, destination, timeout))
        if self.mode == "staging" or (self.mode == "stale_common_upload_failure"
                                      and "diagnostic-common-" in destination):
            raise self.module.DiagnosticError("staging_error", "rsync unavailable")

    def execute(self, profile, device, operation, script, timeout):
        self.executions.append((profile, device, operation, script, timeout))
        if self.mode == "transport":
            raise self.module.DiagnosticError("transport_error", "vpn unavailable")
        if self.mode == "observer":
            raise self.module.DiagnosticError("observer_error", "retained job is not terminal", f"{profile}:kept")
        if self.mode == "observer_then_stale" and not self.observer_returned:
            self.observer_returned = True
            raise self.module.DiagnosticError("observer_error", "retained job is not terminal",
                                              f"{profile}:kept")
        if self.mode == "candidate_timeout":
            return self.module.CommandResult(124, "timed out", ""), f"{profile}:timeout"
        if self.mode == "candidate_kill_timeout":
            return self.module.CommandResult(137, "killed after timeout", ""), f"{profile}:killed"
        if self.mode == "device":
            return self.module.CommandResult(1, "", "NPU device unavailable"), f"{profile}:device"
        if self.mode == "digest":
            return self.module.CommandResult(91, "", "common-digest-mismatch"), f"{profile}:digest"
        if self.mode in {"stale_common", "stale_common_upload_failure"} and not self.stale_returned:
            self.stale_returned = True
            return self.module.CommandResult(91, "", "common-digest-mismatch"), f"{profile}:stale"
        return self.completed(profile)

    def observe(self, profile, handle, timeout):
        self.observations.append((profile, handle, timeout))
        if self.mode == "observer_then_stale" and not self.stale_returned:
            self.stale_returned = True
            return self.module.CommandResult(91, "", "common-digest-mismatch")
        return self.completed(profile)[0]

    def completed(self, profile):
        status = self.mode if self.mode in {
            "compile_error", "runtime_error", "correctness_error", "infrastructure_error",
        } else "ok"
        diagnostics = "Traceback\n  File candidate.py, line 17\nNameError: tl" if status == "compile_error" else ""
        payload = {"status": status, "diagnostics": diagnostics, "passed": status == "ok",
                   "case_evidence": [{"case": 0, "passed": status == "ok", "host_elapsed_us": 99.0}],
                   "host_elapsed_us": 99.0}
        if status == "infrastructure_error":
            payload.update(failure_type="runner_setup_error", diagnostics="torch import failed")
        stdout = "BZ_DIAGNOSTIC_RESULT=" + json.dumps(payload) + "\n"
        return self.module.CommandResult(0, stdout, ""), f"{profile}:job-1"


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
    assert len(candidate_uploads) == 1
    assert len(transport.executions) == 1
    completed = tmp_path / "state" / "quick" / "1" / "arm-a" / "completed.json"
    assert completed.is_file()
    assert not (completed.parent / "dispatch.json").exists()


def test_retained_handle_observes_without_upload_or_execute(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    value = request(tmp_path)
    value.update(retained_handle="bz-a3-1:reconciled", observe_timeout=17)

    result = module.BzA3DiagnosticClient(
        transport, tmp_path / "state").run(value)

    assert result["status"] == "ok"
    assert result["handle"] == "bz-a3-1:reconciled"
    assert transport.uploads == []
    assert transport.executions == []
    assert transport.observations == [
        ("bz-a3-1", "bz-a3-1:reconciled", pytest.approx(17, abs=1)),
    ]


@pytest.mark.parametrize("receipt", ["dispatch", "completed"])
def test_retained_handle_rejects_conflicting_durable_receipt(
    tmp_path: Path, receipt: str,
):
    module = load()
    transport = FakeTransport(module, "observer" if receipt == "dispatch" else "ok")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    first = client.run(value)
    assert first["handle"] == (
        "bz-a3-1:kept" if receipt == "dispatch" else "bz-a3-1:job-1")
    executions = len(transport.executions)

    result = client.run({**value, "retained_handle": "bz-a3-1:other",
                         "observe_timeout": 17})

    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "request_error"
    assert len(transport.executions) == executions
    assert transport.observations == []


def test_retained_digest_mismatch_fails_without_redispatch(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "observer_then_stale")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    first = client.run(value)
    assert first["handle"] == "bz-a3-1:kept"

    result = client.run({**value, "retained_handle": first["handle"],
                         "observe_timeout": 17})

    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "digest_mismatch"
    assert result["handle"] == first["handle"]
    assert len(transport.executions) == 1


def test_resume_observes_exact_selected_fallback_dispatch_receipt(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "observer")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    primary = request(tmp_path)
    primary["cell"] = "arm-attempt-1"
    fallback = {**primary, "cell": "arm-attempt-2", "device": 2}
    first = client.run(fallback)
    assert first["handle"] == "bz-a3-1:kept"
    transport.mode = "ok"

    result = client.resume([fallback], first["handle"], 17)

    assert result["status"] == "ok"
    assert result["cell"] == "arm-attempt-2"
    assert len(transport.executions) == 1
    assert transport.observations[-1][1] == first["handle"]


@pytest.mark.parametrize("owner", ["attempt", "wave", "campaign"])
def test_resume_rejects_handle_owned_by_different_request(
    tmp_path: Path, owner: str,
):
    module = load()
    transport = FakeTransport(module, "observer")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    primary = request(tmp_path)
    primary["cell"] = "arm-attempt-1"
    fallback = {**primary, "cell": "arm-attempt-2", "device": 2}
    if owner == "wave":
        fallback["wave"] = 2
    elif owner == "campaign":
        fallback["campaign"] = "other-campaign"
    first = client.run(fallback)
    assert first["handle"] == "bz-a3-1:kept"
    executions = len(transport.executions)

    result = client.resume([primary], first["handle"], 17)

    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "request_error"
    assert len(transport.executions) == executions
    assert transport.observations == []


def test_completed_result_is_durable_before_dispatch_cleanup(tmp_path: Path, monkeypatch):
    module = load()
    transport = FakeTransport(module)
    state = tmp_path / "state"
    completed = state / "quick" / "1" / "arm-a" / "completed.json"
    original_unlink = Path.unlink

    def ordered_unlink(path, *args, **kwargs):
        if path.name == "dispatch.json":
            assert completed.is_file()
            record = json.loads(completed.read_text())
            assert record["result"]["status"] == "ok"
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", ordered_unlink)
    result = module.BzA3DiagnosticClient(transport, state).run(request(tmp_path))
    assert result["status"] == "ok"


def test_stale_common_receipt_reuploads_once_and_self_heals(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    assert client.run(value)["status"] == "ok"
    value["cell"] = "arm-b"
    transport.mode = "stale_common"
    assert client.run(value)["status"] == "ok"
    common_uploads = [upload for upload in transport.uploads if "diagnostic-common-" in upload[2]]
    assert len(common_uploads) == 2
    assert len(transport.executions) == 3


def test_stale_common_repair_failure_does_not_preserve_terminal_handle(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    state = tmp_path / "state"
    client = module.BzA3DiagnosticClient(transport, state)
    value = request(tmp_path)
    assert client.run(value)["status"] == "ok"
    value["cell"] = "arm-b"
    transport.mode = "stale_common_upload_failure"
    failed = client.run(value)
    assert failed["status"] == "infrastructure_error"
    assert failed["handle"] is None
    assert not (state / "quick" / "1" / "arm-b" / "dispatch.json").exists()
    transport.mode = "ok"
    assert client.run(value)["status"] == "ok"
    assert not transport.observations


def test_repeated_common_digest_mismatch_never_poisons_dispatch_receipt(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "digest")
    state = tmp_path / "state"
    client = module.BzA3DiagnosticClient(transport, state)
    value = request(tmp_path)
    assert client.run(value)["failure_type"] == "digest_mismatch"
    assert client.run(value)["failure_type"] == "digest_mismatch"
    assert len(transport.executions) == 2
    assert not transport.observations
    assert not (state / "quick" / "1" / "arm-a" / "dispatch.json").exists()


def test_resumed_stale_common_result_repairs_instead_of_reobserving(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    assert client.run(value)["status"] == "ok"
    value["cell"] = "arm-b"
    transport.mode = "observer_then_stale"
    interrupted = client.run(value)
    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["handle"] == "bz-a3-1:kept"
    resumed = client.run(value)
    assert resumed["status"] == "ok"
    assert transport.observations[-1][1] == "bz-a3-1:kept"
    assert len([upload for upload in transport.uploads
                if "diagnostic-common-" in upload[2]]) == 2


def test_first_dispatch_observed_digest_mismatch_is_repaired(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "observer_then_stale")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    interrupted = client.run(value)
    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["handle"] == "bz-a3-1:kept"
    resumed = client.run({**value, "observe_timeout": 30})
    assert resumed["status"] == "ok"
    assert transport.observations[-1][2] == 30
    assert len([upload for upload in transport.uploads
                if "diagnostic-common-" in upload[2]]) == 2


def test_observe_timeout_cannot_dispatch_without_receipt(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    value = {**request(tmp_path), "observe_timeout": 7}
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "request_error"
    assert not transport.uploads and not transport.executions


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


def test_candidate_kill_after_timeout_is_not_infrastructure(tmp_path: Path):
    _module, _transport, result = run(tmp_path, "candidate_kill_timeout")
    assert result["status"] == "candidate_timeout"
    assert result["failure_type"] == "candidate_timeout"
    assert result["handle"] == "bz-a3-1:killed"


def test_remote_timeout_returns_counted_result_within_outer_grace(tmp_path: Path,
                                                                   monkeypatch):
    module = load()
    calls = []
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def invoke(argv, timeout):
        calls.append((argv, timeout))
        if "upload" in argv:
            return module.CommandResult(0)
        # Simulate setup, the complete workload cap, ten-second kill-after,
        # and result propagation without sleeping.
        clock[0] += 2 + 5 + 10 + 2
        return module.CommandResult(124, "bz-a3-1:timed-out\n", "")

    transport = module.AdapterTransport(["remote"], ["adapter"], invoke)
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(request(tmp_path))
    assert result["status"] == result["failure_type"] == "candidate_timeout"
    adapter_argv, outer_timeout = calls[-1]
    assert outer_timeout == 30
    assert adapter_argv[adapter_argv.index("--timeout") + 1] == "30"
    assert "timeout --signal=TERM --kill-after=10 5" in adapter_argv[-1]
    assert clock[0] < outer_timeout


def test_timeout_without_response_grace_is_rejected_before_staging(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "candidate_timeout")
    value = {**request(tmp_path), "timeout": 25}
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "request_error"
    assert not transport.uploads and not transport.executions


def test_adapter_over_response_grace_remains_infrastructure(tmp_path: Path):
    module = load()
    calls = []

    def invoke(argv, timeout):
        calls.append((argv, timeout))
        if "upload" in argv:
            return module.CommandResult(0)
        raise module.DiagnosticError("transport_error", "outer response deadline expired")

    transport = module.AdapterTransport(["remote"], ["adapter"], invoke)
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(request(tmp_path))
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "transport_error"
    assert calls[-1][1] == 30
    assert "timeout --signal=TERM --kill-after=10 5" in calls[-1][0][-1]


def test_observe_failure_preserves_handle_from_initial_dispatch():
    module = load()

    def invoke(argv, _timeout):
        if "observe" in argv:
            raise module.DiagnosticError("transport_error", "observer timed out")
        return module.CommandResult(75, "CATLASS_VALIDATION_STATE=running\n",
                                    "bz-a3-1:retained")

    try:
        module.AdapterTransport(["remote"], ["adapter"], invoke).execute(
            "bz-a3-1", 2, "diagnostic", "true", 30)
    except module.DiagnosticError as error:
        assert error.failure_type == "observer_error"
        assert error.handle == "bz-a3-1:retained"
    else:
        raise AssertionError("observer failure should preserve retained handle")


def test_structured_remote_infrastructure_result_is_preserved(tmp_path: Path):
    _module, _transport, result = run(tmp_path, "infrastructure_error")
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "runner_setup_error"
    assert result["diagnostics"] == "torch import failed"
    assert result["handle"] == "bz-a3-1:job-1"


def test_non_object_request_returns_structured_request_error(tmp_path: Path):
    module = load()
    transport = FakeTransport(module)
    for value in (None, [], "request"):
        result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
        assert result == {"status": "infrastructure_error", "failure_type": "request_error",
                          "diagnostics": "request must be a JSON object", "handle": None}
    assert not transport.uploads and not transport.executions


def test_non_string_asset_paths_return_structured_request_error(tmp_path: Path):
    module = load()
    for invalid in (None, 7, 1.5, [], {}):
        transport = FakeTransport(module)
        value = request(tmp_path / str(type(invalid).__name__))
        value["candidate"] = invalid
        result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
        assert result["status"] == "infrastructure_error"
        assert result["failure_type"] == "request_error"
        assert result["diagnostics"] == "asset paths must be JSON strings"
        assert not transport.uploads and not transport.executions


def test_second_invocation_observes_durable_handle_without_redispatch(tmp_path: Path):
    module = load()
    transport = FakeTransport(module, "observer")
    client = module.BzA3DiagnosticClient(transport, tmp_path / "state")
    value = request(tmp_path)
    interrupted = client.run(value)
    assert interrupted["status"] == "infrastructure_error"
    assert interrupted["handle"] == "bz-a3-1:kept"
    assert len(transport.uploads) == 2 and len(transport.executions) == 1

    transport.mode = "ok"
    resumed = client.run(value)
    assert resumed["status"] == "ok"
    assert transport.observations == [("bz-a3-1", "bz-a3-1:kept", 30)]
    assert len(transport.uploads) == 2 and len(transport.executions) == 1


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


def test_client_uses_one_decreasing_deadline_across_all_phases(tmp_path: Path, monkeypatch):
    module = load()
    ticks = iter((100.0, 101.0, 105.0, 108.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    transport = FakeTransport(module)
    result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(request(tmp_path))
    assert result["status"] == "infrastructure_error"
    assert result["failure_type"] == "transport_error"
    assert [entry[3] for entry in transport.uploads] == [29, 25]
    assert not transport.executions


def test_execute_observation_shares_one_deadline(monkeypatch):
    module = load()
    clock = [0.0]
    calls = []

    def invoke(argv, timeout):
        calls.append((argv, timeout))
        if "observe" not in argv:
            clock[0] = 20.0
            return module.CommandResult(75, "CATLASS_VALIDATION_STATE=running\n",
                                        "bz-a3-1:kept")
        return module.CommandResult(0, "CATLASS_VALIDATION_STATE=completed\n", "")

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    module.AdapterTransport(["remote"], ["adapter"], invoke).execute(
        "bz-a3-1", 2, "diagnostic", "true", 30)
    assert [timeout for _argv, timeout in calls] == [30, 10]


def test_host_timeout_preserves_handle_from_partial_adapter_output(monkeypatch):
    module = load()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["adapter"], 30,
                                        output=b"submitted bz-a3-2:retained-timeout\n")

    monkeypatch.setattr(module.subprocess, "run", timeout)
    try:
        module._run(["adapter"], 30)
    except module.DiagnosticError as error:
        assert error.failure_type == "observer_error"
        assert error.handle == "bz-a3-2:retained-timeout"
    else:
        raise AssertionError("timeout should raise DiagnosticError")


def test_identifier_dot_segments_are_rejected_without_dispatch(tmp_path: Path):
    module = load()
    for name in ("campaign", "wave", "cell"):
        for invalid in (".", ".."):
            transport = FakeTransport(module)
            value = request(tmp_path / name / invalid.replace(".", "dot"))
            value[name] = invalid
            result = module.BzA3DiagnosticClient(transport, tmp_path / "state").run(value)
            assert result["status"] == "infrastructure_error"
            assert result["failure_type"] == "request_error"
            assert not transport.uploads and not transport.executions


def test_remote_script_verifies_both_digests_and_bounds_execution():
    module = load()
    script = module._remote_script("/common.tar", "a" * 64, "/candidate.tar", "b" * 64,
                                   "/run", 42, {"device": 0})
    assert script.count("sha256sum") == 2
    assert "timeout --signal=TERM --kill-after=10 42" in script
    assert 'test "$rc" -ne 124 && test "$rc" -ne 137 || exit 124' in script
    assert "BZ_DIAGNOSTIC_RESULT=" in script
