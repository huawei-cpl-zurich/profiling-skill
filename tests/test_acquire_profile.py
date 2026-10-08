from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "acquire_profile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("acquire_profile", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_cpl_remote_resolution_prefers_path_client(tmp_path, monkeypatch):
    module = load_module()
    path_client = tmp_path / "path" / "cpl-remote"
    path_client.parent.mkdir()
    path_client.write_text("#!/bin/sh\n")
    path_client.chmod(0o755)
    fallback = tmp_path / "home" / ".agents/skills/remote-access/scripts/cpl-remote"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("#!/bin/sh\n")
    fallback.chmod(0o755)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(path_client))

    assert module.resolve_cpl_remote(None, home=tmp_path / "home") == path_client


def test_cpl_remote_resolution_uses_user_wide_fallback(tmp_path, monkeypatch):
    module = load_module()
    fallback = tmp_path / "home" / ".agents/skills/remote-access/scripts/cpl-remote"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("#!/bin/sh\n")
    fallback.chmod(0o755)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    assert module.resolve_cpl_remote(None, home=tmp_path / "home") == fallback


def test_cpl_remote_resolution_fails_when_no_approved_client(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    with pytest.raises(module.AcquisitionError, match="approved cpl-remote"):
        module.resolve_cpl_remote(None, home=tmp_path / "empty-home")


def identity(workload: Path, arguments: list[str]) -> dict:
    raw = workload.read_bytes()
    item = {
        "path": workload.name,
        "mode": stat.S_IMODE(workload.stat().st_mode),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    manifest = {"schema_version": 1, "entrypoint": workload.name, "files": [item]}
    return {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "entrypoint": workload.name,
        "arguments": arguments,
        "file_count": 1,
    }


def evidence_bytes(
    product: str,
    target: str,
    phase: str,
    workload_identity: dict,
    *,
    selector: str | None = None,
) -> bytes:
    value = {
        "schema_version": 2,
        "schema": "cpl.profile-acquisition.v2",
        "status": "success",
        "product": product,
        "target": target,
        "runtime": "py311-torch" if product == "a3" else "cann91",
        "dispatch_key": "test-key",
        "phase": phase,
        "device": {"physical_id": 2, "logical_id": 0},
        "provenance": {"product": product, "target": target},
        "workload": workload_identity,
        "normal_run": {
            "exit_code": 0,
            "stdout": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
            "stderr": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        },
        "capture": {
            "metric": "BasicInfo" if phase == "basic" else "PipeUtilization",
            "observed_kernel_names": ["setup", "kernel.alpha/v1", "kernel beta"],
            "inventory_scope": {"launch_bound": 5000, "complete": False},
            "selected_kernel": selector,
            "rows": [{"Op Name": selector}] if selector else [],
            "sources": [],
        },
        "saturation": {
            "state": "unknown",
            "reason": "activity-only evidence has no reviewed capacity denominator",
        },
    }
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fake_cpl_remote(
    tmp_path: Path, evidence: bytes, scenario: str = "ok", receipt_format: str = "json"
) -> Path:
    path = tmp_path / "cpl-remote"
    path.write_text(
        f"""#!/usr/bin/env python3
import base64, hashlib, json, os, pathlib, sys
args = sys.argv[1:]
log = pathlib.Path(os.environ["CPL_TEST_LOG"])
with log.open("a") as f: f.write(json.dumps(args) + "\\n")
joined = " ".join(args)
target = "bz-a5" if "bz-a5" in joined else ("bz-a3-2" if "bz-a3-2" in joined else "bz-a3-1")
handle = "remote:" + target + ":job:job-7"
action = next(x for x in args if x in ("run", "observe", "result", "logs"))
scenario = {scenario!r}
receipt_format = {receipt_format!r}
def emit(value):
    if receipt_format == "json":
        print(json.dumps(value))
        return
    order = ("target", "backend", "operation", "state", "handle", "exit", "next", "stream", "content")
    for key in order:
        if value.get(key) is not None:
            print("REMOTE_" + key.upper() + "=" + str(value[key]))
if action == "run":
    emit({{"target": target, "backend": "broker", "operation": "run", "state": "running", "handle": handle}})
elif action == "observe":
    marker = log.with_suffix(".observe")
    if scenario == "interrupt-always" or (scenario == "interrupt-once" and not marker.exists()):
        marker.write_text("1")
        print("temporary observer loss", file=sys.stderr)
        raise SystemExit(9)
    if scenario == "nonterminal-once" and not marker.exists():
        marker.write_text("1")
        emit({{"target": target, "backend": "broker", "operation": "observe", "state": "running", "handle": handle}})
        raise SystemExit(0)
    state = "failed" if scenario.startswith("terminal-") else "completed"
    emit({{"target": target, "backend": "broker", "operation": "observe", "state": state, "handle": handle, "exit": 17 if state == "failed" else 0}})
    if state == "failed": raise SystemExit(1)
elif action == "result":
    marker = log.with_suffix(".result")
    if scenario == "retrieval-interrupt-once" and not marker.exists():
        marker.write_text("1")
        raise SystemExit(9)
    state = "failed" if scenario.startswith("terminal-") else "completed"
    emit({{"target": target, "backend": "broker", "operation": "result", "state": state, "handle": handle, "exit": 17 if state == "failed" else 0}})
    if state == "failed": raise SystemExit(1)
else:
    stream = args[args.index("--stream") + 1]
    marker = log.with_suffix(".logs-" + stream)
    if scenario == "retrieval-interrupt-once" and not marker.exists():
        marker.write_text("1")
        raise SystemExit(9)
    failures = {{
      "terminal-failure": ("workload_failure", "workload", "compile failed in user workload"),
      "terminal-profiler": ("profiler_failure", "profiler", "profiler rejected metric"),
      "terminal-device": ("device_unavailable", "device-probe", "no device passed probe"),
      "terminal-evidence": ("evidence_failure", "evidence", "missing exact row"),
    }}
    content = ""
    if stream == "stderr" and scenario in failures:
        classification, phase, message = failures[scenario]
        content = "ACQUIRE_FAILURE_JSON=" + json.dumps({{"classification": classification, "phase": phase, "message": message}})
    if stream == "stdout" and not scenario.startswith("terminal-"):
        raw = {evidence!r}
        digest = hashlib.sha256(raw).hexdigest()
        content = "ACQUIRE_EVIDENCE_SHA256=" + digest + "\\n"
        content += "REMOTE_CONTENT_SHA256=" + digest + "\\n"
        content += "ACQUIRE_EVIDENCE_B64=" + base64.b64encode(raw).decode() + "\\n"
    emit({{"target": target, "backend": "broker", "operation": "logs", "state": "completed", "handle": handle, "stream": stream, "content": content}})
"""
    )
    path.chmod(0o755)
    return path


def run_cli(
    tmp_path: Path,
    phase: str,
    *,
    product="a3",
    target="bz-a3-1",
    selector: str | None = None,
    basic: Path | None = None,
    scenario="ok",
    resume: str | None = None,
    attempts: int = 2,
    cpl_override: bool = True,
    receipt_format: str = "json",
    remote_mode: str | None = None,
):
    workload = tmp_path / "user workload.py"
    if not workload.exists():
        workload.write_text("#!/usr/bin/env python3\nprint('arbitrary workload')\n")
    args = ["--mode", "fast", "value with spaces"]
    raw = evidence_bytes(
        product, target, phase, identity(workload, args), selector=selector
    )
    cpl = fake_cpl_remote(tmp_path, raw, scenario, receipt_format)
    output = tmp_path / f"{phase}.json"
    log = tmp_path / "calls.jsonl"
    command = [
        "python3",
        str(SCRIPT),
        phase,
        "--product",
        product,
        "--target",
        target,
        "--dispatch-key",
        "test-key",
        "--workload",
        str(workload),
        "--evidence",
        str(output),
        "--transport-attempts",
        str(attempts),
    ]
    if cpl_override:
        command += ["--cpl-remote", str(cpl)]
    for argument in args:
        command += ["--workload-arg", argument]
    if basic:
        command += ["--basic-evidence", str(basic)]
    if selector:
        command += ["--kernel-name", selector]
    if resume:
        command += ["--resume-handle", resume]
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CPL_TEST_LOG": str(log),
    }
    environment.pop("CPL_REMOTE_MODE", None)
    if remote_mode is not None:
        environment["CPL_REMOTE_MODE"] = remote_mode
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, output, raw, calls, workload


def test_cli_uses_broker_client_on_path_without_hidden_override(tmp_path):
    result, output, expected, calls, _ = run_cli(tmp_path, "basic", cpl_override=False)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected
    assert len([call for call in calls if "run" in call]) == 1


def test_text_receipts_use_one_action_first_dispatch_and_decode_evidence(tmp_path):
    result, output, expected, calls, _ = run_cli(
        tmp_path,
        "basic",
        receipt_format="text",
        cpl_override=False,
        remote_mode="retained-broker",
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected
    assert [call[0] for call in calls] == [
        "run",
        "observe",
        "result",
        "logs",
    ]
    assert all("--json" not in call for call in calls)
    assert len([call for call in calls if call[0] == "run"]) == 1
    run = calls[0]
    assert run.count("--dispatch-key") == 1
    assert run[run.index("--dispatch-key") + 1] == "test-key"


def test_global_transport_omits_broker_dispatch_key(tmp_path):
    result, _, _, calls, _ = run_cli(tmp_path, "basic")
    assert result.returncode == 0, result.stderr
    run = next(call for call in calls if call[0] == "run")
    assert "--dispatch-key" not in run


def test_payload_is_staged_beside_evidence_and_cleaned_after_dispatch(tmp_path):
    result, output, _, calls, _ = run_cli(
        tmp_path, "basic", remote_mode="retained-broker"
    )
    assert result.returncode == 0, result.stderr
    run = next(call for call in calls if call[0] == "run")
    payload = Path(run[run.index("--file") + 1])
    assert payload.parent == output.parent
    assert payload.name.startswith(".acquire-payload-")
    assert not payload.exists()


def test_text_receipt_parser_preserves_metadata_and_multiline_content():
    module = load_module()
    result = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "REMOTE_TARGET=bz-a3-1\n"
            "REMOTE_OPERATION=logs\n"
            "REMOTE_STATE=completed\n"
            "REMOTE_HANDLE=remote:bz-a3-1:job:7\n"
            "REMOTE_EXIT=0\n"
            "REMOTE_STREAM=stdout\n"
            'REMOTE_CONTENT=first\n{"message":"inside"}\n'
            "REMOTE_NOT_METADATA=inside\n"
        ),
        stderr="",
    )
    receipt = module._receipt(result, "logs")
    assert receipt == {
        "target": "bz-a3-1",
        "operation": "logs",
        "state": "completed",
        "handle": "remote:bz-a3-1:job:7",
        "exit": 0,
        "stream": "stdout",
        "content": 'first\n{"message":"inside"}\nREMOTE_NOT_METADATA=inside\n',
    }


def test_text_receipt_accepts_repeated_identical_metadata():
    module = load_module()
    result = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "REMOTE_TARGET=bz-a3-1\n"
            "REMOTE_HANDLE=remote:bz-a3-1:job:7\n"
            "REMOTE_HANDLE=remote:bz-a3-1:job:7\n"
            "REMOTE_STATE=completed\n"
            "REMOTE_STATE=completed\n"
        ),
        stderr="",
    )
    assert module._receipt(result, "observe")["handle"].endswith(":7")


def test_text_receipt_rejects_conflicting_duplicate_metadata():
    module = load_module()
    result = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "REMOTE_HANDLE=remote:bz-a3-1:job:7\n"
            "REMOTE_HANDLE=remote:bz-a3-1:job:8\n"
            "REMOTE_STATE=completed\n"
        ),
        stderr="",
    )
    with pytest.raises(module.AcquisitionError, match="unambiguous receipt"):
        module._receipt(result, "observe")


@pytest.mark.parametrize(
    ("product", "target", "runtime"),
    [
        ("a3", "bz-a3-1", "py311-torch"),
        ("a3", "bz-a3-2", "py311-torch"),
        ("a5", "bz-a5", "cann91"),
    ],
)
def test_basic_uses_product_runtime_and_writes_exact_remote_bytes(
    tmp_path, product, target, runtime
):
    result, output, expected, calls, _ = run_cli(
        tmp_path, "basic", product=product, target=target
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected
    run = next(call for call in calls if "run" in call)
    assert run[run.index("--runtime") + 1] == runtime
    assert run[run.index("run") + 1] == target
    assert hashlib.sha256(expected).hexdigest() in result.stdout


def test_payload_owns_flags_but_not_workload_or_selector_semantics(tmp_path):
    module = load_module()
    workload = tmp_path / "anything.sh"
    workload.write_text("#!/bin/sh\nprintf arbitrary\n")
    payload = module.render_remote_payload(
        "basic", "a3", "bz-a3-1", "key", workload, ["a b", "--x=7"], None
    )
    assert "--aic-metrics=BasicInfo" in payload
    assert "--metrics=" not in payload
    assert "--device=" not in payload
    assert '"--launch-count="+str(LAUNCH_COUNT)' in payload
    assert '"--kill=off"' in payload
    assert "--replay-mode" not in payload
    assert "ASCEND_RT_VISIBLE_DEVICES" in payload
    assert "range(8)" in payload and "timeout=20" in payload
    assert "matmul" not in payload.lower()
    assert "FILES=json.loads(base64.b64decode(" in payload
    assert payload.count('print("ACQUIRE_EVIDENCE_B64="') == 1
    assert payload.count('print("ACQUIRE_EVIDENCE_SHA256="') == 1
    assert payload.count('print("REMOTE_CONTENT_SHA256="') == 1
    assert "tempfile.mkdtemp" in payload
    assert "dispatch key already retained" not in payload
    assert "retained report path leaked into compact evidence" in payload


def test_pipe_requires_agent_selected_exact_exported_name_and_same_workload(tmp_path):
    workload = tmp_path / "user workload.py"
    workload.write_text("#!/usr/bin/env python3\nprint('arbitrary workload')\n")
    args = ["--mode", "fast", "value with spaces"]
    basic = tmp_path / "prior.json"
    basic.write_bytes(
        evidence_bytes("a3", "bz-a3-1", "basic", identity(workload, args))
    )
    result, output, expected, calls, _ = run_cli(
        tmp_path, "pipe", selector="kernel.alpha/v1", basic=basic
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected
    assert len([call for call in calls if "run" in call]) == 1


@pytest.mark.parametrize("selector", [None, "", "not-exported", "kernel*"])
def test_pipe_surfaces_zero_or_nonexact_selection_before_dispatch(tmp_path, selector):
    workload = tmp_path / "user workload.py"
    workload.write_text("#!/usr/bin/env python3\nprint('arbitrary workload')\n")
    args = ["--mode", "fast", "value with spaces"]
    basic = tmp_path / "prior.json"
    basic.write_bytes(
        evidence_bytes("a3", "bz-a3-1", "basic", identity(workload, args))
    )
    result, output, _, calls, _ = run_cli(
        tmp_path, "pipe", selector=selector, basic=basic
    )
    assert result.returncode != 0
    assert "exact exported kernel" in result.stderr or "--kernel-name" in result.stderr
    assert not calls
    assert not output.exists()


def test_pipe_rejects_changed_workload_or_arguments(tmp_path):
    workload = tmp_path / "user workload.py"
    workload.write_text("#!/usr/bin/env python3\nprint('first')\n")
    basic = tmp_path / "prior.json"
    basic.write_bytes(
        evidence_bytes(
            "a3",
            "bz-a3-1",
            "basic",
            identity(workload, ["different"]),
        )
    )
    result, _, _, calls, _ = run_cli(
        tmp_path, "pipe", selector="kernel beta", basic=basic
    )
    assert result.returncode == 2
    assert "workload identity" in result.stderr
    assert not calls


def test_transport_interruption_reobserves_same_handle_without_redispatch(tmp_path):
    result, _, _, calls, _ = run_cli(tmp_path, "basic", scenario="interrupt-once")
    assert result.returncode == 0, result.stderr
    observes = [call for call in calls if "observe" in call]
    assert len(observes) == 2
    assert (
        observes[0][observes[0].index("observe") + 1]
        == observes[1][observes[1].index("observe") + 1]
    )
    assert len([call for call in calls if "run" in call]) == 1


def test_nonterminal_observation_reobserves_same_handle(tmp_path):
    result, _, _, calls, _ = run_cli(tmp_path, "basic", scenario="nonterminal-once")
    assert result.returncode == 0, result.stderr
    observes = [call for call in calls if "observe" in call]
    assert len(observes) == 2
    assert len([call for call in calls if "run" in call]) == 1


def test_process_restart_resumes_persisted_handle_without_redispatch(tmp_path):
    first, output, _, calls, _ = run_cli(
        tmp_path, "basic", scenario="interrupt-always", attempts=1
    )
    assert first.returncode == 1
    receipt = Path(str(output) + ".dispatch.json")
    saved = json.loads(receipt.read_text())
    assert saved["handle"] == "remote:bz-a3-1:job:job-7"
    assert '"status": "dispatched"' in first.stdout
    second, output, expected, calls, _ = run_cli(
        tmp_path,
        "basic",
        resume=saved["handle"],
        scenario="ok",
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == expected
    assert len([call for call in calls if "run" in call]) == 1


def test_text_receipt_restart_resumes_same_handle_without_redispatch(tmp_path):
    first, output, _, _, _ = run_cli(
        tmp_path,
        "basic",
        scenario="interrupt-always",
        attempts=1,
        receipt_format="text",
        remote_mode="retained-broker",
    )
    assert first.returncode == 1
    saved = json.loads(Path(str(output) + ".dispatch.json").read_text())

    second, output, expected, calls, _ = run_cli(
        tmp_path,
        "basic",
        resume=saved["handle"],
        scenario="ok",
        receipt_format="text",
        remote_mode="retained-broker",
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == expected
    assert len([call for call in calls if call[0] == "run"]) == 1
    run = next(call for call in calls if call[0] == "run")
    assert run[run.index("--dispatch-key") + 1] == "test-key"


def test_resume_rejects_corrupted_receipt_metadata_hash(tmp_path):
    first, output, _, _, _ = run_cli(
        tmp_path, "basic", scenario="interrupt-always", attempts=1
    )
    assert first.returncode == 1
    receipt = Path(str(output) + ".dispatch.json")
    saved = json.loads(receipt.read_text())
    saved["request_sha256"] = "0" * 64
    receipt.write_text(json.dumps(saved))

    resumed, _, _, calls, _ = run_cli(
        tmp_path, "basic", resume=saved["handle"], scenario="ok"
    )
    assert resumed.returncode == 2
    assert "metadata hash mismatch" in resumed.stderr
    assert len([call for call in calls if "run" in call]) == 1


def test_result_and_logs_retry_same_handle(tmp_path):
    result, output, expected, calls, _ = run_cli(
        tmp_path, "basic", scenario="retrieval-interrupt-once", attempts=2
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == expected
    assert len([call for call in calls if "run" in call]) == 1
    assert len([call for call in calls if "result" in call]) == 2
    assert len([call for call in calls if "logs" in call]) == 2


def test_terminal_failure_includes_handle_phase_and_log_excerpt(tmp_path):
    result, output, _, calls, _ = run_cli(
        tmp_path, "basic", scenario="terminal-failure"
    )
    assert result.returncode == 1
    assert "remote:bz-a3-1:job:job-7" in result.stderr
    assert "classification=workload_failure phase=workload" in result.stderr
    assert "compile failed in user workload" in result.stderr
    assert not output.exists()
    assert len([call for call in calls if "run" in call]) == 1


def test_text_terminal_failure_preserves_remote_classification(tmp_path):
    result, output, _, calls, _ = run_cli(
        tmp_path,
        "basic",
        scenario="terminal-profiler",
        receipt_format="text",
    )
    assert result.returncode == 1
    assert "classification=profiler_failure phase=profiler" in result.stderr
    assert "profiler rejected metric" in result.stderr
    assert not output.exists()
    assert len([call for call in calls if call[0] == "run"]) == 1


@pytest.mark.parametrize(
    ("scenario", "classification", "phase"),
    [
        ("terminal-profiler", "profiler_failure", "profiler"),
        ("terminal-device", "device_unavailable", "device-probe"),
        ("terminal-evidence", "evidence_failure", "evidence"),
    ],
)
def test_terminal_failure_classification_survives_cli(
    tmp_path, scenario, classification, phase
):
    result, _, _, _, _ = run_cli(tmp_path, "basic", scenario=scenario)
    assert result.returncode == 1
    assert f"classification={classification} phase={phase}" in result.stderr
    assert "handle=remote:bz-a3-1:job:job-7" in result.stderr


def test_activity_evidence_cannot_claim_saturation(tmp_path):
    module = load_module()
    workload = tmp_path / "w"
    workload.write_bytes(b"x")
    raw = evidence_bytes(
        "a5", "bz-a5", "pipe", identity(workload, []), selector="kernel beta"
    )
    value = json.loads(raw)
    value["saturation"]["state"] = "saturated"
    with pytest.raises(module.AcquisitionError, match="reviewed capacity denominator"):
        module.validate_evidence(
            value, "pipe", "a5", "bz-a5", "test-key", identity(workload, [])
        )


@pytest.mark.parametrize("mode", ["basic", "pipe"])
def test_rendered_payload_parses_deployed_csv_shapes(tmp_path, mode):
    module = load_module()
    workload = tmp_path / "workload.sh"
    workload.write_text("#!/bin/sh\nexit 0\n")
    workload.chmod(0o755)
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        "float32 = object()\n"
        "class NPU:\n"
        " def set_device(self, value): pass\n"
        " def synchronize(self): pass\n"
        "npu = NPU()\n"
        "class V:\n"
        " def __add__(self, value): return self\n"
        " def cpu(self): return self\n"
        " def __getitem__(self, key): return 4.0\n"
        "def arange(*args, **kwargs): return V()\n"
    )
    (modules / "torch_npu.py").write_text("")
    msprof = tmp_path / "msprof"
    msprof.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "output = pathlib.Path(next(x.split('=',1)[1] for x in sys.argv if x.startswith('--output='))) / 'OPPROF_live'\n"
        "output.mkdir(parents=True)\n"
        "(output/'OpBasicInfo.csv').write_text('Op Name,Task Duration(us)\\nexact.kernel/7,3.5\\n')\n"
        "if '--aic-metrics=PipeUtilization' in sys.argv:\n"
        " (output/'PipeUtilization.csv').write_text('block_id,sub_block_id,aic_cube_ratio\\n0,cube0,0.75\\n')\n"
        "print('Profiling running finished. All task success. output='+str(output))\n"
    )
    msprof.chmod(0o755)
    payload = module.render_remote_payload(
        mode,
        "a3",
        "bz-a3-1",
        f"payload-{mode}",
        workload,
        [],
        "exact.kernel/7" if mode == "pipe" else None,
    )
    payload_path = tmp_path / "payload.py"
    payload_path.write_text(payload)
    results = [
        subprocess.run(
            ["python3", str(payload_path)],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
                "PYTHONPATH": str(modules),
            },
        )
        for _ in range(2)
    ]
    assert all(result.returncode == 0 for result in results), [
        result.stderr for result in results
    ]
    raw_values = [
        base64.b64decode(
            next(
                line.split("=", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("ACQUIRE_EVIDENCE_B64=")
            )
        )
        for result in results
    ]
    assert raw_values[0] == raw_values[1]
    raw = raw_values[0]
    retained = list((tmp_path / ".cpl-profile-evidence").iterdir())
    assert len(retained) == 2
    assert all(path.is_dir() for path in retained)
    assert all(str(path).encode() not in raw for path in retained)
    digest = hashlib.sha256(raw).hexdigest()
    markers = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in results[0].stdout.splitlines()
        if line.startswith(("ACQUIRE_EVIDENCE_SHA256=", "REMOTE_CONTENT_SHA256="))
    }
    assert markers == {
        "ACQUIRE_EVIDENCE_SHA256": digest,
        "REMOTE_CONTENT_SHA256": digest,
    }
    evidence = json.loads(raw)
    assert evidence["schema"] == "cpl.profile-acquisition.v2"
    assert evidence["provenance"] == {"product": "a3", "target": "bz-a3-1"}
    assert evidence["normal_run"]["exit_code"] == 0
    assert set(evidence["normal_run"]) == {"exit_code", "stdout", "stderr"}
    assert evidence["capture"]["observed_kernel_names"] == ["exact.kernel/7"]
    assert len(evidence["capture"]["rows"]) == (1 if mode == "pipe" else 0)
    if mode == "pipe":
        assert evidence["capture"]["sources"][0]["role"] == "PipeUtilization"
        assert evidence["capture"]["sources"][1]["role"] == "selector_binding"


def test_bundle_manifest_binds_sibling_imports_and_data(tmp_path):
    module = load_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "main.py").write_text("import helper\nprint(open('data.txt').read())\n")
    (bundle / "helper.py").write_text("VALUE = 7\n")
    (bundle / "data.txt").write_text("payload\n")
    spec = module.build_bundle(bundle=bundle, entrypoint="main.py", arguments=["--x"])
    assert spec.identity["file_count"] == 3
    original = spec.identity["manifest_sha256"]
    (bundle / "helper.py").write_text("VALUE = 8\n")
    changed = module.build_bundle(
        bundle=bundle, entrypoint="main.py", arguments=["--x"]
    )
    assert changed.identity["manifest_sha256"] != original


def test_pipe_rejects_changed_sibling_file_before_dispatch(tmp_path):
    module = load_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "main.py").write_text("import helper\n")
    helper = bundle / "helper.py"
    helper.write_text("VALUE=1\n")
    spec = module.build_bundle(bundle=bundle, entrypoint="main.py", arguments=[])
    basic = tmp_path / "basic.json"
    basic.write_bytes(evidence_bytes("a3", "bz-a3-1", "basic", spec.identity))
    helper.write_text("VALUE=2\n")
    cpl = fake_cpl_remote(tmp_path, b"unused")
    log = tmp_path / "calls.jsonl"
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "pipe",
            "--product",
            "a3",
            "--target",
            "bz-a3-1",
            "--dispatch-key",
            "changed",
            "--bundle",
            str(bundle),
            "--entrypoint",
            "main.py",
            "--basic-evidence",
            str(basic),
            "--kernel-name",
            "kernel beta",
            "--evidence",
            str(tmp_path / "pipe.json"),
            "--cpl-remote",
            str(cpl),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "CPL_TEST_LOG": str(log)},
    )
    assert result.returncode == 2
    assert "workload identity" in result.stderr
    assert not log.exists()


def test_failure_marker_preserves_phase_and_classification():
    module = load_module()
    stderr = 'noise\nACQUIRE_FAILURE_JSON={"classification":"workload_failure","message":"compile broke","phase":"workload"}\n'
    error = module.remote_failure(stderr, "remote:bz-a3-1:job:7")
    assert error.phase == "workload"
    assert error.classification == "workload_failure"
    assert "compile broke" in str(error)


def test_launch_count_accepts_deployed_bound_and_rejects_outside_range():
    module = load_module()
    assert module.launch_count("5000") == 5000
    with pytest.raises(Exception, match="1..5000"):
        module.launch_count("5001")


def test_bundle_payload_runs_sibling_import_and_discovers_kernel_after_twenty(tmp_path):
    module = load_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "helper.py").write_text("VALUE='sibling-ok'\n")
    (bundle / "data.txt").write_text("data-ok\n")
    (bundle / "main.py").write_text(
        "import helper\n"
        "assert helper.VALUE == 'sibling-ok'\n"
        "assert open('data.txt').read().strip() == 'data-ok'\n"
    )
    spec = module.build_bundle(bundle=bundle, entrypoint="main.py", arguments=[])
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        "float32=object()\n"
        "class NPU:\n def set_device(self,x): pass\n def synchronize(self): pass\n"
        "npu=NPU()\n"
        "class V:\n def __add__(self,x): return self\n def cpu(self): return self\n def __getitem__(self,x): return 4.0\n"
        "def arange(*a,**k): return V()\n"
    )
    (modules / "torch_npu.py").write_text("")
    msprof = tmp_path / "msprof"
    msprof.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib,sys\n"
        "limit=int(next(x.split('=',1)[1] for x in sys.argv if x.startswith('--launch-count=')))\n"
        "name='late.kernel' if limit>20 else 'early.kernel'\n"
        "out=pathlib.Path(next(x.split('=',1)[1] for x in sys.argv if x.startswith('--output=')))/'OPPROF'\n"
        "out.mkdir(parents=True)\n"
        "(out/'OpBasicInfo.csv').write_text('Op Name,Task Duration(us)\\n'+name+',1\\n')\n"
        "print('Profiling running finished. All task success.')\n"
    )
    msprof.chmod(0o755)
    payload = module.render_remote_payload(
        "basic", "a3", "bz-a3-1", "late-target", spec, [], None, 5000
    )
    script = tmp_path / "payload.py"
    script.write_text(payload)
    result = subprocess.run(
        ["python3", str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PYTHONPATH": str(modules),
        },
    )
    assert result.returncode == 0, result.stderr
    encoded = next(
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("ACQUIRE_EVIDENCE_B64=")
    )
    evidence = json.loads(base64.b64decode(encoded))
    assert evidence["schema"] == "cpl.profile-acquisition.v2"
    assert evidence["provenance"] == {"product": "a3", "target": "bz-a3-1"}
    assert evidence["capture"]["observed_kernel_names"] == ["late.kernel"]
    assert evidence["capture"]["inventory_scope"] == {
        "launch_bound": 5000,
        "complete": False,
        "reason": "msprof exports observed names but no total application launch count",
    }
