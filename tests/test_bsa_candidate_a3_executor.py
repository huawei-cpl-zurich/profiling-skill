from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/bsa_candidate_a3_executor.py"
QUALIFICATION_CANDIDATE = ROOT / "benchmarks/bsa/qualification_candidate.py"


def load():
    spec = importlib.util.spec_from_file_location("bsa_candidate_a3_executor", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_qualification_candidate():
    spec = importlib.util.spec_from_file_location(
        "bsa_qualification_candidate", QUALIFICATION_CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def request(action="discover", *, target="bz-a3-1"):
    activation = "named-runtime"
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
        "execution_environment": {
            "target": target,
            "runtime": "py311-torch",
            "runtime_activation": activation,
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
    def __init__(self, module, remote_result, *, interrupted=False,
                 transfer_interrupted=False, upload_state="observation-unavailable"):
        self.module = module
        self.remote_result = remote_result
        self.interrupted = interrupted
        self.transfer_interrupted = transfer_interrupted
        self.upload_state = upload_state
        self.probes = []
        self.uploads = []
        self.transfer_observations = []
        self.runs = []
        self.observations = []

    def capabilities(self, target):
        self.probes.append(("capabilities", target))

    def preflight(self, target):
        self.probes.append(("preflight", target))

    def upload(self, target, source, destination):
        self.uploads.append((target, source, destination))
        return {
            "handle": f"remote:{target}:transfer:retained-{len(self.uploads)}",
            "state": self.upload_state,
        }

    def observe_transfer(self, target, handle):
        self.transfer_observations.append((target, handle))
        if self.transfer_interrupted:
            raise self.module.ExecutorError(
                "infrastructure_error", "transfer observer unavailable",
                handle=handle, terminal=False,
            )

    def run(self, target, runtime, script, cwd):
        self.runs.append((target, runtime, script, cwd))
        self.last_command = ["cpl-remote", "run", target, "--runtime", runtime]
        self.last_command.extend(["--file", str(script), "--cwd", cwd])
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


def runtime_fingerprint(tag="a"):
    return {
        "schema": "profiling-skill/a3-python-runtime-fingerprint/v1",
        "executable": {
            "basename": "python",
            "realpath_sha256": tag * 64,
            "content_sha256": tag * 64,
        },
        "python": {"implementation": "CPython", "version": "3.11.10"},
        "packages": {
            "torch": "2.10.0",
            "torch_npu": "2.10.0",
            "triton": "3.2.0",
            "triton_distribution": {"name": "triton-ascend", "version": "3.2.0"},
        },
    }


def qualification_receipt(module, cfg, *, campaign_revision="a" * 40,
                          fingerprint=None):
    identity = module.qualification_identity(cfg, "py311-torch", campaign_revision)
    fingerprint = fingerprint or runtime_fingerprint()
    return {
        "schema": module.QUALIFICATION_SCHEMA,
        "passed": True,
        **identity,
        "runtime_fingerprint": fingerprint,
        "runtime_fingerprint_sha256": module.digest_json(fingerprint),
        "cases": module.QUALIFICATION_CASES,
        "correctness": [
            {"case": case, "passed": True, "max_abs_error": 0.0}
            for case in module.QUALIFICATION_CASES
        ],
        "measurements": {str(case): {
            "samples_us": [10.0, 10.1, 10.2],
            "stable_window_us": [10.0, 10.1, 10.2],
            "relative_range": 0.02,
        } for case in module.QUALIFICATION_CASES},
        "compile_failure": {"status": "compile_error", "diagnostics": "expected"},
        "durable_handles": [],
    }


def success(action="discover"):
    base = {
        "schema": "profiling-skill/bsa-candidate-remote-result/v2",
        "status": "ok", "physical_device": 3,
        "runtime_fingerprint": runtime_fingerprint(),
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


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("execution_environment"),
    lambda value: value["execution_environment"].update(target="gz-a3"),
    lambda value: value["execution_environment"].update(runtime="other-runtime"),
    lambda value: value["execution_environment"].update(
        runtime_activation="target-default"),
    lambda value: value["execution_environment"].update(extra="not-exact"),
])
def test_execution_environment_mismatch_is_rejected_before_remote_calls(
    tmp_path, mutation,
):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    invalid = request()
    mutation(invalid)

    result = module.Executor(cfg, transport).execute(invalid)

    assert result["status"] == "infrastructure_error"
    assert "execution environment" in result["diagnostics"]
    assert not transport.probes


def test_staged_remote_driver_is_valid_python():
    module = load()

    compile(module.remote_driver_source(), "remote_driver.py", "exec")


@pytest.mark.parametrize("rows,pipe,expected", [
    ([{"Op Name":"kernel", "Task Duration(us)":"NA"}], [], "missing_pipe_rows"),
    ([{"Op Name":"kernel", "Task Duration(us)":"NA"}], [{}], "invalid_duration"),
    ([{"Op Name":"kernel", "Task Duration(us)":"1"}]*2, [{}], "duration_row_count"),
    ([{"Op Name":"kernel", "Task Duration(us)":"1"}], [{}], None),
])
def test_capture_diagnostics_distinguishes_missing_and_invalid_evidence(rows, pipe, expected):
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(load().remote_driver_source(), "remote_driver.py", "exec"), namespace)
    result = namespace["capture_diagnostics"](rows, pipe, "kernel")
    assert result is None if expected is None else result.startswith(expected + ":")


@pytest.mark.parametrize("heads", [3, 24, 48, 96])
def test_qualification_candidate_blockmask_access_plan_is_bounded(heads):
    candidate = load_qualification_candidate()
    blocks = heads // 3

    plan = candidate.blockmask_access_plan(heads, blocks)

    active = [(head, rank) for head, rank, should_load in plan if should_load]
    assert active == [
        (head, head - (heads - 2 * blocks))
        for head in range(heads - 2 * blocks, heads - blocks)
    ]
    assert all(0 <= rank < blocks for _head, rank, _should_load in plan)

    source = QUALIFICATION_CANDIDATE.read_text()
    assert "mask=valid & sparse_head" in source
    assert "safe_block_rank" in source


def test_remote_classifier_uses_bounded_out_of_memory_markers():
    module = load()
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(module.remote_driver_source(), "remote_driver.py", "exec"), namespace)

    assert namespace["is_device_infra"]("torch.OutOfMemoryError: NPU out of memory")
    assert namespace["is_device_infra"]("runtime failed: OOM while allocating")
    assert not namespace["is_device_infra"]("candidate boom in user code")
    assert not namespace["is_device_infra"]("room size mismatch")


def test_pipe_compaction_rejects_duplicate_block_identity_across_csvs(tmp_path):
    module = load()
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(module.remote_driver_source(), "remote_driver.py", "exec"), namespace)
    first, second = tmp_path / "PipeUtilization-0.csv", tmp_path / "PipeUtilization-1.csv"
    header = "block_id,sub_block_id,aiv_vec_ratio\n"
    first.write_text(header + "0,vector0,0.5\n")
    second.write_text(header + "0,vector0,0.7\n")

    rows = namespace["read_rows"]([first, second])

    with pytest.raises(ValueError, match="duplicate PipeUtilization identity"):
        namespace["compact_pipe"](rows)


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


def test_gz_uses_named_runtime_and_requires_matching_qualification(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3",
        "remote_root": "profiling-skill-bsa-qualification-pr100",
        "campaign_revision": "a" * 40,
        "qualification_receipt": tmp_path / "qualification.json",
    })
    cfg.qualification_receipt.write_text(json.dumps(qualification_receipt(module, cfg)))

    result = module.Executor(cfg, transport).execute(request(target="gz-a3"))

    assert result["status"] == "ok"
    assert result["provenance"]["runtime"] == "py311-torch"
    assert result["provenance"]["runtime_activation"] == "named-runtime"
    assert transport.runs[0][0:2] == ("gz-a3", "py311-torch")
    assert transport.uploads[0][2].startswith("profiling-skill-bsa-qualification-pr100/")
    assert transport.runs[0][3] == "profiling-skill-bsa-qualification-pr100"
    script = Path(transport.runs[0][2]).read_text()
    assert "archive=payload-" in script
    assert "run_root=runs/" in script
    assert "unset PYTHONHOME\nexport PYTHONPATH=" in script
    assert result["provenance"]["command"][3:5] == ["--runtime", "py311-torch"]
    dispatch = json.loads(next((tmp_path / "state").rglob("dispatch.json")).read_text())
    assert dispatch["runtime_activation"] == "named-runtime"
    assert dispatch["campaign_revision"] == "a" * 40


def test_gz_rendered_script_keeps_result_paths_valid_after_entering_workdir(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3", "remote_root": "profiling-gz",
        "campaign_revision": "a" * 40,
        "qualification_receipt": tmp_path / "qualification.json",
    })
    cfg.qualification_receipt.write_text(json.dumps(qualification_receipt(module, cfg)))

    assert module.Executor(cfg, transport).execute(request(target="gz-a3"))["status"] == "ok"

    uploaded = Path(transport.uploads[0][1])
    remote = tmp_path / "remote"
    remote.mkdir()
    shutil.copy2(uploaded, remote / Path(transport.uploads[0][2]).name)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "test -z \"${PYTHONHOME+x}\" || exit 41\n"
        "test \"${PYTHONPATH+x}\" = x || exit 42\n"
        "test -z \"$PYTHONPATH\" || exit 43\n"
        "printf '%s\\n' 'BSA_CANDIDATE_RESULT={\"status\":\"ok\"}'\n"
    )
    fake_python.chmod(0o755)
    rendered = Path(transport.runs[0][2])

    completed = subprocess.run(
        ["bash", str(rendered)], cwd=remote, text=True, capture_output=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
             "PYTHONHOME": "/hostile/python/home", "PYTHONPATH": "/hostile/python/path"},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == 'BSA_CANDIDATE_RESULT={"status":"ok"}\n'
    result_logs = list((remote / "runs").rglob("compact-result.log"))
    assert len(result_logs) == 1
    assert result_logs[0].read_text() == completed.stdout


def test_gz_rejects_runtime_fingerprint_drift_since_qualification(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    receipt = tmp_path / "qualification.json"
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3", "remote_root": "profiling-gz",
        "campaign_revision": "a" * 40, "qualification_receipt": receipt,
    })
    receipt.write_text(json.dumps(qualification_receipt(
        module, cfg, fingerprint=runtime_fingerprint("b"))))

    result = module.Executor(cfg, transport).execute(request(target="gz-a3"))

    assert result["status"] == "infrastructure_error"
    assert "runtime fingerprint" in result["diagnostics"]
    assert result["terminal"] is True


def test_gz_rejects_missing_or_drifted_qualification_before_remote_calls(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    receipt = tmp_path / "qualification.json"
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3", "remote_root": "profiling-gz",
        "campaign_revision": "a" * 40,
        "qualification_receipt": receipt,
    })

    missing = module.Executor(cfg, transport).execute(request(target="gz-a3"))
    assert missing["status"] == "infrastructure_error"
    assert "qualification" in missing["diagnostics"]
    assert not transport.probes

    stale = qualification_receipt(module, cfg)
    stale["executor_sha256"] = "0" * 64
    receipt.write_text(json.dumps(stale))
    drifted = module.Executor(cfg, transport).execute(request(target="gz-a3"))
    assert drifted["status"] == "infrastructure_error"
    assert "qualification" in drifted["diagnostics"]
    assert not transport.probes


def test_cpl_remote_gz_run_uses_named_runtime(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "target": "gz-a3", "state": "running",
            "handle": "remote:gz-a3:job:abc",
        }) + "\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    script = tmp_path / "run.sh"
    script.write_text("true\n")

    handle = module.CplRemoteTransport(executable).run(
        "gz-a3", "py311-torch", script, "/remote/work")

    assert handle == "remote:gz-a3:job:abc"
    assert calls[0][2:8] == [
        "run", "gz-a3", "--runtime", "py311-torch", "--file", str(script),
    ]
    assert calls[0][-4:] == ["--cwd", "/remote/work", "--timeout", "900"]


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("field", ["target", "handle"])
def test_observe_result_rejects_mismatched_log_receipt_identity(
    monkeypatch, tmp_path, stream, field,
):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    handle = "remote:gz-a3:job:retained-1"
    compact = {
        "schema": module.REMOTE_SCHEMA,
        "status": "infrastructure_error",
        "diagnostics": "expected test result",
    }

    def fake_run(command, **_kwargs):
        action = command[2]
        if action == "observe":
            receipt = {"target": "gz-a3", "state": "completed", "handle": handle}
        else:
            selected = command[command.index("--stream") + 1]
            content = ("BSA_CANDIDATE_RESULT=" + json.dumps(compact) + "\n"
                       if selected == "stdout" else "")
            receipt = {
                "target": "gz-a3", "state": "completed", "handle": handle,
                "content": content,
            }
            if selected == stream:
                receipt[field] = (
                    "bz-a3-1" if field == "target"
                    else "remote:gz-a3:job:different"
                )
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt) + "\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(module.ExecutorError, match="log receipt identity mismatch"):
        module.CplRemoteTransport(executable).observe_result("gz-a3", handle)


def test_observe_result_rejects_wrong_target_observation_receipt(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    handle = "remote:gz-a3:job:retained-1"
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps({
                            "target": "bz-a3-1", "state": "completed", "handle": handle,
                        }) + "\n", stderr=""))

    with pytest.raises(module.ExecutorError, match="observation identity mismatch"):
        module.CplRemoteTransport(executable).observe_result("gz-a3", handle)


def test_observe_result_rejects_wrong_target_handle_before_transport(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **_kwargs: calls.append(args))

    with pytest.raises(module.ExecutorError, match="handle target mismatch"):
        module.CplRemoteTransport(executable).observe_result(
            "gz-a3", "remote:bz-a3-1:job:retained-1")

    assert calls == []


def test_cpl_remote_upload_returns_then_observes_retained_transfer(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []
    replies = [
        {"target": "gz-a3", "state": "observation-unavailable",
         "handle": "remote:gz-a3:transfer:upload-1"},
        {"target": "gz-a3", "state": "completed",
         "handle": "remote:gz-a3:transfer:upload-1"},
    ]

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(replies.pop(0)) + "\n",
                               stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    source = tmp_path / "payload.tar"
    source.write_bytes(b"payload")

    transport = module.CplRemoteTransport(executable)
    upload = transport.upload("gz-a3", source, "qualification/payload.tar")
    transport.observe_transfer("gz-a3", upload["handle"])

    assert calls[0][-2:] == ["--timeout", "0"]
    assert calls[1][2:] == ["observe", "remote:gz-a3:transfer:upload-1",
                            "--wait", "--timeout", "900"]


def test_cpl_remote_bz_upload_remains_synchronous(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"target": "bz-a3-1", "state": "completed"}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    source = tmp_path / "payload.tar"
    source.write_bytes(b"payload")

    handle = module.CplRemoteTransport(executable).upload(
        "bz-a3-1", source, "/srv/profiling/payload.tar")

    assert handle is None
    assert "--timeout" not in calls[0]


def test_cpl_remote_upload_failure_returns_terminal_transfer_receipt(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=1, stdout=json.dumps({
                            "target": "gz-a3", "state": "failed",
                            "handle": "remote:gz-a3:transfer:upload-failed",
                        }) + "\n", stderr=""))
    source = tmp_path / "payload.tar"
    source.write_bytes(b"payload")

    upload = module.CplRemoteTransport(executable).upload(
        "gz-a3", source, "qualification-payload.tar")

    assert upload == {
        "handle": "remote:gz-a3:transfer:upload-failed", "state": "failed",
    }


def test_terminal_transfer_failure_is_persisted_and_new_attempt_can_upload(tmp_path):
    module = load()
    transport = FakeTransport(module, success(), upload_state="failed")
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())

    assert first["status"] == "infrastructure_error"
    assert first["terminal"] is True
    failed_handle = first["provenance"]["handle"]
    assert failed_handle == "remote:bz-a3-1:transfer:retained-1"
    upload_receipt = next((tmp_path / "state").rglob("upload.json"))
    assert json.loads(upload_receipt.read_text())["handle"] == failed_handle

    transport.upload_state = "observation-unavailable"
    retry = request()
    retry["retry"] = {"mode": "new_attempt", "prior_handle": failed_handle}
    second = executor.execute(retry)

    assert second["status"] == "ok"
    assert len(transport.uploads) == 2
    assert len(transport.runs) == 1
    assert len(list((tmp_path / "state").rglob("attempt-*"))) == 2


def test_transfer_observer_error_preserves_known_handle(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="controller disconnected"),
    )

    with pytest.raises(module.ExecutorError) as error:
        module.CplRemoteTransport(executable).observe_transfer(
            "gz-a3", "remote:gz-a3:transfer:upload-1")

    assert error.value.handle == "remote:gz-a3:transfer:upload-1"
    assert error.value.terminal is False


def test_qualification_runs_five_cases_stability_and_compile_diagnostic(tmp_path):
    module = load()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# known good\n")
    calls = []

    class FakeExecutor:
        config = SimpleNamespace(target="gz-a3")

        def execute(self, value):
            calls.append(value)
            if "intentional_compile_failure" in value["candidate"]["source"]:
                return {"status": "compile_error", "diagnostics": "CompilationError: static assertion",
                        "terminal": True, "runtime_fingerprint": runtime_fingerprint(),
                        "provenance": {"handle": "remote:gz-a3:job:bad"}}
            if value["action"] == "discover":
                return {"status": "ok", "components": [{
                    "selector": "block_sparse_attention_kernel_mix_aiv", "kind": "aiv"}],
                    "runtime_fingerprint": runtime_fingerprint(),
                    "correctness": [{"case": case, "passed": True}
                                    for case in module.QUALIFICATION_CASES],
                    "provenance": {"handle": "remote:gz-a3:job:discover"}}
            return {"status": "ok", "duration_us": [10.0, 10.2, 10.1][value["sample_index"]],
                    "runtime_fingerprint": runtime_fingerprint(),
                    "pipe_utilization": {"selector": value["kernel_name"], "rows": [{}]},
                    "provenance": {"handle": f"remote:gz-a3:job:{value['case']}"}}

    cfg = SimpleNamespace(
        target="gz-a3", baseline=ROOT / "benchmarks/bsa/baseline.py",
        cases=ROOT / "benchmarks/bsa/cases.jsonl",
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        qualification_candidate=candidate,
    )
    receipt = module.qualify(
        FakeExecutor(), cfg, candidate, "block_sparse_attention_kernel", "a" * 40)

    assert receipt["passed"] is True
    assert receipt["runtime_fingerprint"] == runtime_fingerprint()
    assert receipt["runtime_fingerprint_sha256"] == module.digest_json(runtime_fingerprint())
    assert list(receipt["measurements"]) == [str(case) for case in module.QUALIFICATION_CASES]
    assert all(len(item["samples_us"]) == 3 for item in receipt["measurements"].values())
    assert receipt["compile_failure"]["status"] == "compile_error"
    assert receipt["correctness"] == [
        {"case": case, "passed": True}
        for case in module.QUALIFICATION_CASES
    ]
    assert "static assertion" in receipt["compile_failure"]["diagnostics"]
    assert len(calls) == 1 + len(module.QUALIFICATION_CASES) * 3 + 1
    assert all(call["execution_environment"] == {
        "target": "gz-a3",
        "runtime": "py311-torch",
        "runtime_activation": "named-runtime",
    } for call in calls)


def test_qualification_rejects_runtime_change_between_remote_stages(tmp_path):
    module = load()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# known good\n")

    class FakeExecutor:
        config = SimpleNamespace(target="gz-a3")

        def execute(self, value):
            if value["action"] == "discover":
                return {
                    "status": "ok",
                    "components": [{"selector": "kernel", "kind": "aiv"}],
                    "correctness": [
                        {"case": case, "passed": True}
                        for case in module.QUALIFICATION_CASES
                    ],
                    "runtime_fingerprint": runtime_fingerprint("a"),
                    "provenance": {"handle": "remote:gz-a3:job:discover"},
                }
            return {
                "status": "ok", "duration_us": 10.0,
                "pipe_utilization": {"selector": "kernel", "rows": [{}]},
                "runtime_fingerprint": runtime_fingerprint("b"),
                "provenance": {"handle": "remote:gz-a3:job:capture"},
            }

    cfg = SimpleNamespace(
        target="gz-a3", baseline=ROOT / "benchmarks/bsa/baseline.py",
        cases=ROOT / "benchmarks/bsa/cases.jsonl",
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        qualification_candidate=candidate,
    )

    receipt = module.qualify(
        FakeExecutor(), cfg, candidate, "kernel", "a" * 40)

    assert receipt["passed"] is False
    assert "resolved runtime changed" in receipt["failure"]
    assert receipt["runtime_fingerprint"] == runtime_fingerprint("a")
    assert receipt["observed_runtime_fingerprint"] == runtime_fingerprint("b")


@pytest.mark.parametrize("mutation", [
    lambda value: value["executable"].update(raw_path="/must/not/leak"),
    lambda value: value["executable"].update(content_sha256="not-a-digest"),
    lambda value: value["packages"]["triton_distribution"].update(name="unknown"),
])
def test_runtime_fingerprint_schema_rejects_unbounded_or_invalid_identity(mutation):
    module = load()
    value = runtime_fingerprint()
    mutation(value)

    with pytest.raises(module.ExecutorError, match="runtime|executable|Triton"):
        module.validate_runtime_fingerprint(value)


@pytest.mark.parametrize("stage", ["discover", "capture", "compile"])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("handle_kind", ["job", "transfer"])
def test_qualification_retries_exact_infrastructure_handle_for_every_stage(
    tmp_path, stage, terminal, handle_kind,
):
    module = load()
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# known good\n")
    calls = []
    failed = False
    failed_handle = f"remote:gz-a3:{handle_kind}:{stage}-failure"

    class FakeExecutor:
        config = SimpleNamespace(target="gz-a3")

        def execute(self, value):
            nonlocal failed
            calls.append(value)
            is_compile = "intentional_compile_failure" in value["candidate"]["source"]
            matches = (
                (stage == "discover" and value["action"] == "discover" and not is_compile)
                or (stage == "capture" and value["action"] == "capture")
                or (stage == "compile" and is_compile)
            )
            if matches and not failed:
                failed = True
                return {
                    "status": "infrastructure_error",
                    "diagnostics": "remote observer failed",
                    "terminal": terminal,
                    "provenance": {"handle": failed_handle},
                }
            if is_compile:
                return {
                    "status": "compile_error",
                    "diagnostics": "CompilationError: static assertion",
                    "terminal": True,
                    "runtime_fingerprint": runtime_fingerprint(),
                    "provenance": {"handle": "remote:gz-a3:job:compile-success"},
                }
            if value["action"] == "discover":
                return {
                    "status": "ok",
                    "components": [{
                        "selector": "block_sparse_attention_kernel_mix_aiv",
                        "kind": "aiv",
                    }],
                    "correctness": [
                        {"case": case, "passed": True}
                        for case in module.QUALIFICATION_CASES
                    ],
                    "runtime_fingerprint": runtime_fingerprint(),
                    "provenance": {"handle": "remote:gz-a3:job:discover-success"},
                }
            return {
                "status": "ok",
                "duration_us": [10.0, 10.2, 10.1][value["sample_index"]],
                "pipe_utilization": {"selector": value["kernel_name"], "rows": [{}]},
                "runtime_fingerprint": runtime_fingerprint(),
                "provenance": {"handle": "remote:gz-a3:job:capture-success"},
            }

    cfg = SimpleNamespace(
        target="gz-a3", baseline=ROOT / "benchmarks/bsa/baseline.py",
        cases=ROOT / "benchmarks/bsa/cases.jsonl",
        runner=ROOT / "scripts/a3_benchmark_runner.py",
        qualification_candidate=candidate,
    )
    receipt = module.qualify(
        FakeExecutor(), cfg, candidate, "block_sparse_attention_kernel", "a" * 40)

    assert receipt["passed"] is True
    failed_index = next(
        index for index, call in enumerate(calls)
        if call.get("retry", {}).get("handle") == failed_handle
        or call.get("retry", {}).get("prior_handle") == failed_handle
    )
    expected = ({"mode": "new_attempt", "prior_handle": failed_handle}
                if terminal else {"mode": "observe", "handle": failed_handle})
    assert calls[failed_index]["retry"] == expected
    retried = {key: value for key, value in calls[failed_index].items() if key != "retry"}
    assert retried == calls[failed_index - 1]
    assert failed_handle in receipt["durable_handles"]


def test_remote_driver_classifies_correctness_timeout_as_candidate_runtime_error(
    tmp_path, monkeypatch,
):
    module = load()
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(module.remote_driver_source(), "remote_driver.py", "exec"), namespace)

    def timeout(*_args, **_kwargs):
        raise namespace["subprocess"].TimeoutExpired("runner.py", 300)

    monkeypatch.setattr(namespace["subprocess"], "run", timeout)
    result = namespace["run_workload"](
        request(), {}, tmp_path / "correctness.json")

    assert result["status"] == "runtime_error"
    assert result["case_evidence"] == []
    assert "timed out after 300" in result["diagnostics"]
    assert len(result["diagnostics"]) < 300


def test_qualification_infrastructure_retry_budget_is_bounded():
    module = load()
    calls, handles, retry_evidence = [], [], []

    class AlwaysUnavailable:
        def execute(self, value):
            calls.append(value)
            index = len(calls)
            return {
                "status": "infrastructure_error",
                "diagnostics": "still unavailable",
                "terminal": True,
                "provenance": {"handle": f"remote:gz-a3:job:attempt-{index}"},
            }

    original = request(target="gz-a3")
    result = module._qualification_execute(
        AlwaysUnavailable(), original, "gz-a3", handles, retry_evidence,
        "discovery",
    )

    assert result["status"] == "infrastructure_error"
    assert len(calls) == module.QUALIFICATION_INFRA_RETRIES + 1
    assert len(retry_evidence) == module.QUALIFICATION_INFRA_RETRIES
    assert calls[0] == original
    assert calls[-1]["retry"] == {
        "mode": "new_attempt",
        "prior_handle": f"remote:gz-a3:job:attempt-{len(calls) - 1}",
    }


@pytest.mark.parametrize("state", ["cancelled", "observation-unavailable", "error"])
def test_cpl_remote_preflight_rejects_nonsuccess_state_before_upload(
    monkeypatch, tmp_path, state,
):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps({
                            "target": "gz-a3", "state": state,
                        }) + "\n", stderr=""))

    with pytest.raises(module.ExecutorError, match="preflight"):
        module.CplRemoteTransport(executable).preflight("gz-a3")


def test_cpl_remote_preflight_rejects_nonzero_client_exit(monkeypatch, tmp_path):
    module = load()
    executable = tmp_path / "cpl-remote"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=7, stdout=json.dumps({
                            "target": "gz-a3", "state": "completed",
                        }) + "\n", stderr="preflight failed"))

    with pytest.raises(module.ExecutorError, match="exited 7"):
        module.CplRemoteTransport(executable).preflight("gz-a3")


def test_failed_qualification_receipt_is_not_accepted(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    receipt = tmp_path / "qualification.json"
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3", "remote_root": "profiling-gz",
        "campaign_revision": "a" * 40,
        "qualification_receipt": receipt,
    })
    document = qualification_receipt(module, cfg)
    document["passed"] = False
    receipt.write_text(json.dumps(document))

    result = module.Executor(cfg, transport).execute(request(target="gz-a3"))

    assert result["status"] == "infrastructure_error"
    assert "did not pass" in result["diagnostics"]
    assert not transport.probes


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("correctness"),
    lambda value: value.__setitem__("correctness", value["correctness"][:-1]),
    lambda value: value["correctness"][2].update(passed=False),
    lambda value: value["correctness"].reverse(),
])
def test_gz_rejects_missing_partial_false_or_reordered_correctness_receipt(
    tmp_path, mutation,
):
    module = load()
    transport = FakeTransport(module, success())
    bz, _ = config(module, tmp_path, transport)
    receipt = tmp_path / "qualification.json"
    cfg = module.ExecutorConfig(**{
        **bz.__dict__, "target": "gz-a3", "remote_root": "profiling-gz",
        "campaign_revision": "a" * 40, "qualification_receipt": receipt,
    })
    document = qualification_receipt(module, cfg)
    mutation(document)
    receipt.write_text(json.dumps(document))

    result = module.Executor(cfg, transport).execute(request(target="gz-a3"))

    assert result["status"] == "infrastructure_error"
    assert "correctness" in result["diagnostics"]
    assert not transport.probes


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
    offline = FakeTransport(module, None, interrupted=True)
    second = module.Executor(cfg, offline).execute(second_request)

    assert first["duration_us"] == 10.0
    assert second["duration_us"] == 11.0
    assert second["provenance"]["handle"] == first["provenance"]["handle"]
    assert first["evidence_sha256"] != second["evidence_sha256"]
    assert len(transport.runs) == 1
    assert offline.observations == []
    assert offline.runs == []


def test_zero_exit_invalid_capture_retains_vendor_transcript(tmp_path, monkeypatch, capsys):
    module = load()
    namespace = {"__name__": "staged_remote_driver"}
    exec(compile(module.remote_driver_source(), "remote_driver.py", "exec"), namespace)
    monkeypatch.chdir(tmp_path)
    req = request("capture")
    (tmp_path / "request.json").write_text(json.dumps(req))
    namespace["pick_device"] = lambda: (0, {}, None)
    namespace["run_workload"] = lambda *_: {"status":"ok", "case_evidence":[]}
    namespace["valid_fusion"] = lambda *_: True
    namespace["runtime_fingerprint"] = runtime_fingerprint
    stdout = namespace["SUCCESS"] + "\nvendor: no PMU records\n"
    stderr = "vendor counter diagnostic\n"

    def fake_profile(command, **_kwargs):
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
        report.mkdir()
        (report / "OpBasicInfo.csv").write_text(
            "Op Name,Task Duration(us)\nblock_sparse_attention_kernel,NA\n")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(namespace["subprocess"], "run", fake_profile)
    namespace["main"]()
    payload = json.loads(capsys.readouterr().out.split("BSA_CANDIDATE_RESULT=", 1)[1])
    assert payload["status"] == "infrastructure_error"
    assert payload["diagnostics"].startswith("missing_pipe_rows:")
    assert len(payload["retained_logs"]) == 2
    for log in payload["retained_logs"]:
        data = Path(log["path"]).read_bytes()
        assert data.decode() == (stdout if log["stream"] == "stdout" else stderr)
        assert hashlib.sha256(data).hexdigest() == log["sha256"]
        assert len(data) == log["bytes"]
    transport = FakeTransport(module, payload)
    cfg, _ = config(module, tmp_path, transport)
    response = module.Executor(cfg, transport).execute(request("capture"))
    assert response["retained_logs"] == payload["retained_logs"]


@pytest.mark.parametrize("remote_status", [
    "compile_error", "runtime_error", "correctness_error",
])
def test_candidate_failures_are_preserved(remote_status, tmp_path):
    module = load()
    remote = {"schema": module.REMOTE_SCHEMA,
              "status": remote_status, "diagnostics": "candidate broke",
              "physical_device": 2,
              "runtime_fingerprint": runtime_fingerprint()}
    transport = FakeTransport(module, remote)
    cfg, _ = config(module, tmp_path, transport)

    result = module.Executor(cfg, transport).execute(request())

    assert result["status"] == remote_status
    assert result["diagnostics"] == "candidate broke"
    assert result["runtime_fingerprint"] == runtime_fingerprint()
    assert result["provenance"]["runtime_fingerprint"] == runtime_fingerprint()


def test_prelease_memory_failure_preserves_compact_remote_diagnostics(tmp_path):
    module = load()
    remote = {"schema": module.REMOTE_SCHEMA, "status": "infrastructure_error",
              "diagnostics": "no device passed: 3:free=5410 required=10000",
              "runtime_fingerprint": runtime_fingerprint()}
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


def test_terminal_observe_error_is_persisted_and_new_attempt_can_dispatch(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)
    observe = transport.observe_result

    def terminal_log_failure(target, handle):
        transport.observations.append((target, handle))
        raise module.ExecutorError(
            "infrastructure_error", "terminal job logs unavailable",
            handle=handle, terminal=True,
        )

    transport.observe_result = terminal_log_failure
    first = executor.execute(request())

    assert first["status"] == "infrastructure_error"
    assert first["terminal"] is True
    handle = first["provenance"]["handle"]
    terminal_path = next((tmp_path / "state").rglob("attempt-0001/terminal.json"))
    assert json.loads(terminal_path.read_text()) == {
        "handle": handle,
        "status": "infrastructure_error",
        "terminal": True,
        "evidence_sha256": hashlib.sha256(
            json.dumps({
                "diagnostics": "terminal job logs unavailable",
                "handle": handle,
                "status": "infrastructure_error",
            }, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }

    transport.observe_result = observe
    retry = request()
    retry["retry"] = {"mode": "new_attempt", "prior_handle": handle}
    second = executor.execute(retry)

    assert second["status"] == "ok"
    assert len(transport.runs) == 2
    assert [path.name for path in sorted((tmp_path / "state").rglob("attempt-*"))] == [
        "attempt-0001", "attempt-0002",
    ]


def test_nonterminal_observe_error_does_not_create_terminal_receipt(tmp_path):
    module = load()
    transport = FakeTransport(module, success(), interrupted=True)
    cfg, _ = config(module, tmp_path, transport)

    first = module.Executor(cfg, transport).execute(request())

    assert first["terminal"] is False
    assert not list((tmp_path / "state").rglob("terminal.json"))


def test_observation_identity_mismatch_cannot_authorize_replacement(tmp_path):
    module = load()
    transport = FakeTransport(module, success())
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    def mismatched_observation(target, handle):
        transport.observations.append((target, handle))
        client = module.CplRemoteTransport.__new__(module.CplRemoteTransport)
        client._call = lambda *args, **kwargs: {
            "target": target,
            "handle": f"remote:{target}:job:different",
            "state": "completed",
        }
        return client.observe_result(target, handle)

    transport.observe_result = mismatched_observation
    first = executor.execute(request())

    assert first["status"] == "infrastructure_error"
    assert first["terminal"] is False
    handle = first["provenance"]["handle"]
    assert not list((tmp_path / "state").rglob("terminal.json"))

    retry = request()
    retry["retry"] = {"mode": "new_attempt", "prior_handle": handle}
    replacement = executor.execute(retry)

    assert replacement["status"] == "infrastructure_error"
    assert replacement["terminal"] is False
    assert "cannot replace a nonterminal retained handle" in replacement["diagnostics"]
    assert len(transport.runs) == 1
    assert not list((tmp_path / "state").rglob("attempt-0002"))


def test_transfer_interruption_resumes_exact_handle_then_dispatches_without_reupload(
    tmp_path,
):
    module = load()
    transport = FakeTransport(
        module, success(), transfer_interrupted=True)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)

    first = executor.execute(request())

    assert first["status"] == "infrastructure_error"
    assert first["terminal"] is False
    transfer = first["provenance"]["handle"]
    assert transfer == "remote:bz-a3-1:transfer:retained-1"
    assert len(transport.uploads) == 1
    assert not transport.runs
    upload_receipt = next((tmp_path / "state").rglob("upload.json"))
    assert json.loads(upload_receipt.read_text())["handle"] == transfer

    transport.transfer_interrupted = False
    retry = request()
    retry["retry"] = {"mode": "observe", "handle": transfer}
    second = executor.execute(retry)

    assert second["status"] == "ok"
    assert len(transport.uploads) == 1
    assert len(transport.runs) == 1
    assert transport.transfer_observations == [
        ("bz-a3-1", transfer), ("bz-a3-1", transfer),
    ]


def test_nonterminal_transfer_cannot_be_replaced_by_new_attempt(tmp_path):
    module = load()
    transport = FakeTransport(
        module, success(), transfer_interrupted=True)
    cfg, _ = config(module, tmp_path, transport)
    executor = module.Executor(cfg, transport)
    first = executor.execute(request())
    transfer = first["provenance"]["handle"]

    retry = request()
    retry["retry"] = {"mode": "new_attempt", "prior_handle": transfer}
    second = executor.execute(retry)

    assert second["status"] == "infrastructure_error"
    assert "nonterminal transfer" in second["diagnostics"]
    assert len(transport.uploads) == 1
    assert not transport.runs


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
        return upload(*args)

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
    second = module.Executor(second_cfg, second_transport).execute(
        request(target="bz-a3-2"))
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
