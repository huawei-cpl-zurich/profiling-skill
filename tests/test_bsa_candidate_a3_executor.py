from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bsa_candidate_a3_executor.py"


def load():
    spec = importlib.util.spec_from_file_location("bsa_candidate_a3_executor", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def request(action="discover"):
    value = {
        "schema": "profiling-skill/bsa-candidate-executor-request/v1",
        "action": action,
        "candidate": {
            "execution_id": "candidate-0123456789abcdef0123",
            "benchmark": "bsa",
            "source_sha256": hashlib.sha256(b"# candidate\n").hexdigest(),
            "source": "# candidate\n",
            "parameters": {"BLOCK_N": 128},
            "environment": {"product": "a3", "runtime": "py311-torch"},
            "declared_kernel_name": "block_sparse_attention_kernel",
        },
    }
    if action == "discover":
        value.update(correctness_cases=[47, 46], metric="BasicInfo")
    else:
        value.update(case=47, sample_index=0,
                     kernel_name="block_sparse_attention_kernel",
                     component={"selector": "block_sparse_attention_kernel", "kind": "aiv"},
                     metric="PipeUtilization")
    return value


class FakeTransport:
    def __init__(self, module, remote_result, *, interrupted=False):
        self.module = module
        self.remote_result = remote_result
        self.interrupted = interrupted
        self.probes = []
        self.uploads = []
        self.runs = []
        self.observations = []

    def capabilities(self, target):
        self.probes.append(("capabilities", target))

    def preflight(self, target):
        self.probes.append(("preflight", target))

    def upload(self, target, source, destination):
        self.uploads.append((target, source, destination))

    def run(self, target, runtime, script, cwd):
        self.runs.append((target, runtime, script, cwd))
        self.last_command = ["cpl-remote", "run", target, "--runtime", runtime,
                             "--file", str(script), "--cwd", cwd]
        return f"remote:{target}:job:retained-{len(self.runs)}"

    def observe_result(self, target, handle):
        self.observations.append((target, handle))
        if self.interrupted:
            raise self.module.ExecutorError("infrastructure_error", "observer unavailable",
                                            handle=handle, terminal=False)
        return self.remote_result


def config(module, tmp_path, transport):
    return module.ExecutorConfig(
        target="bz-a3-1", remote_root="/srv/profiling", state_dir=tmp_path / "state",
        baseline=ROOT / "benchmarks/bsa/baseline.py",
        cases=ROOT / "benchmarks/bsa/cases.jsonl",
        runner=ROOT / "scripts/a3_benchmark_runner.py",
    ), transport


def success(action="discover"):
    base = {
        "schema": "profiling-skill/bsa-candidate-remote-result/v1",
        "status": "ok", "physical_device": 3,
        "correctness": [{"case": 47, "passed": True}],
    }
    if action == "discover":
        base.update(components=[
                        {"selector": "block_sparse_attention_kernel_mix_aic", "kind": "aic"},
                        {"selector": "block_sparse_attention_kernel_mix_aiv", "kind": "aiv"},
                    ],
                    basic_info={"metric": "BasicInfo", "rows": [
                        {"Op Name": "block_sparse_attention_kernel_mix_aic",
                         "Task Duration(us)": "10.0"}]})
    else:
        base.update(duration_us=10.0, pipe_utilization={
            "metric": "PipeUtilization", "selector": "block_sparse_attention_kernel",
            "rows": [{"aiv_vec_ratio": "0.75"}],
        })
    return base


def test_invalid_request_is_rejected_before_remote_calls(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    bad = request()
    bad["candidate"]["source_sha256"] = "0" * 64

    result = module.Executor(cfg, transport).execute(bad)

    assert result["status"] == "infrastructure_error"
    assert "source SHA-256" in result["diagnostics"]
    assert not transport.probes


def test_staged_remote_driver_is_valid_python():
    module = load()

    compile(module.remote_driver_source(), "remote_driver.py", "exec")


def test_discover_stages_self_contained_bundle_and_returns_compact_evidence(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request())

    assert result["status"] == "ok"
    assert [item["kind"] for item in result["components"]] == ["aic", "aiv"]
    assert result["basic_info"]["rows"][0]["Task Duration(us)"] == "10.0"
    assert result["provenance"]["physical_device"] == 3
    assert result["provenance"]["runtime"] == "py311-torch"
    assert result["provenance"]["handle"] == "remote:bz-a3-1:job:retained-1"
    assert result["provenance"]["product"] == "a3"
    assert result["terminal"] is True
    assert result["correctness"] == [{"case": 47, "passed": True}]
    assert len(result["evidence_sha256"]) == 64
    assert transport.probes == [("capabilities", "bz-a3-1"),
                                ("preflight", "bz-a3-1")]
    assert len(transport.uploads) == len(transport.runs) == 1
    target, archive, remote = transport.uploads[0]
    assert target == "bz-a3-1"
    assert remote.startswith("/srv/profiling/payload-")
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {
            "candidate.py", "baseline.py", "baseline.json", "cases.jsonl", "runner.py",
            "request.json", "remote_driver.py",
        }
        staged = json.load(bundle.extractfile("request.json"))
        assert staged["action"] == "discover"
        assert staged["correctness_cases"] == [47, 46]
    _target, runtime, script_path, cwd = transport.runs[0]
    script = Path(script_path).read_text()
    assert runtime == "py311-torch" and cwd == "/srv/profiling"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in script
    assert 'CPL_DEVICE_LEASE_ROOT="$HOME/.cpl-device-leases"' in script
    assert "BSA_MIN_FREE_DEVICE_BYTES=34359738368" in script
    assert 'flock 9' in script and 'compact-result.log' in script
    assert "remote_driver.py" in script
    with tarfile.open(archive) as bundle:
        driver = bundle.extractfile("remote_driver.py").read().decode()
    assert '"device-lease-%d.lock"%physical' in driver


def test_device_probe_rejects_low_free_memory_with_compact_diagnostics(
    tmp_path, monkeypatch,
):
    module = load()
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(module.remote_driver_source(), "remote_driver.py", "exec"), namespace)
    monkeypatch.setenv("CPL_DEVICE_LEASE_ROOT", str(tmp_path / "leases"))
    monkeypatch.setenv("BSA_MIN_FREE_DEVICE_BYTES", str(10_000))
    monkeypatch.setattr(namespace["time"], "sleep", lambda _seconds: None)
    monkeypatch.setattr(namespace["subprocess"], "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0,
                                        stdout='{"free":5410,"total":10000}\n',
                                        stderr=""))

    with pytest.raises(RuntimeError, match=r"free=5410 required=10000") as error:
        namespace["pick_device"]()
    assert str(error.value).count("free=5410 required=10000") == 8


def test_capture_binds_exact_selector_and_returns_pipe_rows(tmp_path):
    module = load()
    transport = FakeTransport(module, success("capture"))
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request("capture"))

    assert result["status"] == "ok"
    assert result["duration_us"] == 10.0
    assert result["pipe_utilization"]["selector"] == "block_sparse_attention_kernel"
    assert result["evidence_sha256"] == module.digest_json(success("capture"))


def test_capture_samples_use_independent_captures_in_one_retained_batch(tmp_path):
    module = load()
    remote = success("capture")
    remote.pop("duration_us")
    remote.pop("pipe_utilization")
    remote["captures"] = [
        {"sample_index": 0, "duration_us": 10.0, "pipe_utilization": {
            "metric": "PipeUtilization", "selector": "block_sparse_attention_kernel",
            "rows": [{"aiv_vec_ratio": "0.75"}],
        }},
        {"sample_index": 1, "duration_us": 11.0, "pipe_utilization": {
            "metric": "PipeUtilization", "selector": "block_sparse_attention_kernel",
            "rows": [{"aiv_vec_ratio": "0.76"}],
        }},
    ]
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request("capture"))
    second_request = request("capture")
    second_request["sample_index"] = 1
    second = executor.execute(second_request)

    assert first["duration_us"] == 10.0
    assert second["duration_us"] == 11.0
    assert second["provenance"]["handle"] == first["provenance"]["handle"]
    assert first["evidence_sha256"] != second["evidence_sha256"]
    assert len(transport.runs) == 1


@pytest.mark.parametrize("remote_status", [
    "compile_error", "runtime_error", "correctness_error",
])
def test_candidate_failures_are_preserved(remote_status, tmp_path):
    module = load()
    remote = {"schema": "profiling-skill/bsa-candidate-remote-result/v1",
              "status": remote_status, "diagnostics": "candidate broke",
              "physical_device": 2}
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request())

    assert result["status"] == remote_status
    assert result["diagnostics"] == "candidate broke"


def test_prelease_memory_failure_preserves_compact_remote_diagnostics(tmp_path):
    module = load()
    remote = {"schema": module.REMOTE_SCHEMA, "status": "infrastructure_error",
              "diagnostics": "no device passed: 3:free=5410 required=10000"}
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request())

    assert result["status"] == "infrastructure_error"
    assert result["diagnostics"] == remote["diagnostics"]
    assert result["provenance"]["physical_device"] == -1


def test_observer_interruption_preserves_handle_and_resume_does_not_redispatch(tmp_path):
    module = load()
    transport = FakeTransport(module, success(), interrupted=True)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())
    assert first["status"] == "infrastructure_error"
    assert first["provenance"]["handle"] == "remote:bz-a3-1:job:retained-1"
    assert first["terminal"] is False
    assert len(transport.runs) == 1

    transport.interrupted = False
    retry = request()
    retry["retry"] = {"mode": "observe", "handle": first["provenance"]["handle"]}
    second = executor.execute(retry)
    assert second["status"] == "ok"
    assert len(transport.runs) == 1
    assert transport.observations == [
        ("bz-a3-1", "remote:bz-a3-1:job:retained-1"),
        ("bz-a3-1", "remote:bz-a3-1:job:retained-1"),
    ]


def test_terminal_infrastructure_requires_numbered_new_attempt(tmp_path):
    module = load()
    remote = {"schema": module.REMOTE_SCHEMA, "status": "infrastructure_error",
              "diagnostics": "device reset", "physical_device": 1}
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())
    assert first["terminal"] is True
    handle = first["provenance"]["handle"]
    transport.remote_result = success()
    retry = request()
    retry["retry"] = {"mode": "new_attempt", "prior_handle": handle}
    second = executor.execute(retry)

    assert second["status"] == "ok"
    assert second["provenance"]["handle"] != handle
    assert len(transport.runs) == 2
    receipts = sorted((tmp_path / "state").rglob("dispatch.json"))
    assert [path.parent.name for path in receipts] == ["attempt-0001", "attempt-0002"]


def test_incomplete_attempt_is_redispatched_idempotently_in_same_attempt(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())
    assert first["status"] == "ok"
    receipt = next((tmp_path / "state").rglob("attempt-0001/dispatch.json"))
    receipt.unlink()

    second = executor.execute(request())

    assert second["status"] == "ok"
    assert len(transport.runs) == 2
    assert len(list((tmp_path / "state").rglob("attempt-*"))) == 1
    first_script = Path(transport.runs[0][2]).read_text()
    second_script = Path(transport.runs[1][2]).read_text()
    assert first_script == second_script
    assert "execution.lock" in second_script


def test_upload_failure_does_not_poison_attempt_identity(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)
    upload = transport.upload
    failures = 0

    def fail_once(*args):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise module.ExecutorError("infrastructure_error", "upload unavailable")
        upload(*args)

    transport.upload = fail_once
    assert executor.execute(request())["status"] == "infrastructure_error"

    result = executor.execute(request())

    assert result["status"] == "ok"
    assert len(transport.runs) == 1
    assert len(list((tmp_path / "state").rglob("attempt-0001/dispatch.json"))) == 1


def test_target_is_part_of_identity_and_persisted_handle_target_is_checked(tmp_path):
    module = load()
    first_transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, first_transport)
    assert module.Executor(cfg, first_transport).execute(request())["status"] == "ok"

    second_transport = FakeTransport(module, success())
    second_cfg = module.ExecutorConfig(**{**cfg.__dict__, "target": "bz-a3-2"})
    second = module.Executor(second_cfg, second_transport).execute(request())
    assert second["status"] == "ok"
    assert second["provenance"]["target"] == "bz-a3-2"
    assert len(second_transport.runs) == 1

    receipt = next(path for path in (tmp_path / "state").rglob(
        "attempt-0001/dispatch.json") if json.loads(path.read_text())["target"] == "bz-a3-1")
    document = json.loads(receipt.read_text())
    document["handle"] = "remote:bz-a3-2:job:wrong-target"
    receipt.write_text(json.dumps(document))
    observed = module.Executor(cfg, first_transport).execute(request())
    assert observed["status"] == "infrastructure_error"
    assert "target" in observed["diagnostics"]


def test_remote_result_rejects_wrong_selector_and_malformed_pipe_evidence(tmp_path):
    module = load()
    remote = success("capture")
    remote["pipe_utilization"]["selector"] = "other_kernel"
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request("capture"))

    assert result["status"] == "infrastructure_error"
    assert "selector" in result["diagnostics"]


def test_cli_without_local_target_configuration_writes_failure_response(
    tmp_path, monkeypatch,
):
    module = load()
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    request_path.write_text(json.dumps(request()))
    monkeypatch.delenv("BSA_A3_TARGET", raising=False)
    monkeypatch.delenv("BSA_A3_REMOTE_ROOT", raising=False)

    status = module.main(["--request", str(request_path), "--response", str(response_path)])

    assert status == 2
    response = json.loads(response_path.read_text())
    assert response["status"] == "infrastructure_error"
    assert "local configuration" in response["diagnostics"]
