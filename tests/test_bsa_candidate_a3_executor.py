from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
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
        return "remote:bz-a3-1:job:retained-1"

    def observe_result(self, target, handle):
        self.observations.append((target, handle))
        if self.interrupted:
            raise self.module.ExecutorError("infrastructure_error", "observer unavailable",
                                            handle=handle)
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
        base.update(kernel_names=["block_sparse_attention_kernel"],
                    basic_info={"metric": "BasicInfo", "rows": [
                        {"Op Name": "block_sparse_attention_kernel",
                         "Task Duration(us)": "10.0"}]})
    else:
        base.update(duration_us=10.0, duration_samples_us=[10.0, 10.2, 9.9], pipe_utilization={
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


def test_discover_stages_self_contained_bundle_and_returns_compact_evidence(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request())

    assert result["status"] == "ok"
    assert result["kernel_names"] == ["block_sparse_attention_kernel"]
    assert result["basic_info"]["rows"][0]["Task Duration(us)"] == "10.0"
    assert result["provenance"]["physical_device"] == 3
    assert result["provenance"]["runtime"] == "py311-torch"
    assert result["provenance"]["handle"] == "remote:bz-a3-1:job:retained-1"
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
    assert "remote_driver.py" in script


def test_capture_binds_exact_selector_and_returns_pipe_rows(tmp_path):
    module = load()
    transport = FakeTransport(module, success("capture"))
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request("capture"))

    assert result["status"] == "ok"
    assert result["duration_us"] == 10.0
    assert result["pipe_utilization"]["selector"] == "block_sparse_attention_kernel"
    assert result["evidence_sha256"] == module.digest_json(success("capture"))


def test_capture_caches_distinct_msprof_launch_samples_without_redispatch(tmp_path):
    module = load()
    transport = FakeTransport(module, success("capture"))
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request("capture"))
    second_request = request("capture")
    second_request["sample_index"] = 1
    second = executor.execute(second_request)

    assert first["duration_us"] == 10.0
    assert second["duration_us"] == 10.2
    assert second["provenance"]["handle"] == first["provenance"]["handle"]
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


def test_observer_interruption_preserves_handle_and_resume_does_not_redispatch(tmp_path):
    module = load()
    transport = FakeTransport(module, success(), interrupted=True)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())
    assert first["status"] == "infrastructure_error"
    assert first["provenance"]["handle"] == "remote:bz-a3-1:job:retained-1"
    assert len(transport.runs) == 1

    transport.interrupted = False
    second = executor.execute(request())
    assert second["status"] == "ok"
    assert len(transport.runs) == 1
    assert transport.observations == [
        ("bz-a3-1", "remote:bz-a3-1:job:retained-1"),
        ("bz-a3-1", "remote:bz-a3-1:job:retained-1"),
    ]


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
