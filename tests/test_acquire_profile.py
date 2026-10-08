from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
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


def identity(workload: Path, arguments: list[str]) -> dict:
    return {
        "sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
        "arguments": arguments,
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
        "schema_version": 1,
        "status": "success",
        "product": product,
        "target": target,
        "runtime": "py311-torch" if product == "a3" else "cann91",
        "dispatch_key": "test-key",
        "phase": phase,
        "device": {"physical_id": 2, "logical_id": 0},
        "workload": workload_identity,
        "capture": {
            "metric": "BasicInfo" if phase == "basic" else "PipeUtilization",
            "exported_kernel_names": ["setup", "kernel.alpha/v1", "kernel beta"],
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


def fake_cpl_remote(tmp_path: Path, evidence: bytes, scenario: str = "ok") -> Path:
    path = tmp_path / "cpl-remote"
    path.write_text(
        f'''#!/usr/bin/env python3
import base64, hashlib, json, os, pathlib, sys
args = sys.argv[1:]
log = pathlib.Path(os.environ["CPL_TEST_LOG"])
with log.open("a") as f: f.write(json.dumps(args) + "\\n")
joined = " ".join(args)
target = "bz-a5" if "bz-a5" in joined else ("bz-a3-2" if "bz-a3-2" in joined else "bz-a3-1")
handle = "remote:" + target + ":job:job-7"
action = next(x for x in args if x in ("run", "observe", "result", "logs"))
scenario = {scenario!r}
if action == "run":
    print(json.dumps({{"target": target, "state": "running", "handle": handle}}))
elif action == "observe":
    marker = log.with_suffix(".observe")
    if scenario == "interrupt-once" and not marker.exists():
        marker.write_text("1")
        print("temporary observer loss", file=sys.stderr)
        raise SystemExit(9)
    state = "failed" if scenario == "terminal-failure" else "completed"
    print(json.dumps({{"target": target, "state": state, "handle": handle, "exit": 17 if state == "failed" else 0}}))
    if state == "failed": raise SystemExit(1)
elif action == "result":
    state = "failed" if scenario == "terminal-failure" else "completed"
    print(json.dumps({{"target": target, "state": state, "handle": handle, "exit": 17 if state == "failed" else 0}}))
    if state == "failed": raise SystemExit(1)
else:
    stream = args[args.index("--stream") + 1]
    content = "compile failed in user workload" if stream == "stderr" else ""
    if stream == "stdout" and scenario != "terminal-failure":
        raw = {evidence!r}
        content = "ACQUIRE_EVIDENCE_SHA256=" + hashlib.sha256(raw).hexdigest() + "\\n"
        content += "ACQUIRE_EVIDENCE_B64=" + base64.b64encode(raw).decode() + "\\n"
    print(json.dumps({{"target": target, "state": "completed", "handle": handle, "content": content}}))
'''
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
):
    workload = tmp_path / "user workload.py"
    if not workload.exists():
        workload.write_text("#!/usr/bin/env python3\nprint('arbitrary workload')\n")
    args = ["--mode", "fast", "value with spaces"]
    raw = evidence_bytes(product, target, phase, identity(workload, args), selector=selector)
    cpl = fake_cpl_remote(tmp_path, raw, scenario)
    output = tmp_path / f"{phase}.json"
    log = tmp_path / "calls.jsonl"
    command = [
        "python3", str(SCRIPT), phase, "--product", product, "--target", target,
        "--dispatch-key", "test-key", "--workload", str(workload),
        "--evidence", str(output), "--cpl-remote", str(cpl),
        "--observe-attempts", "2",
    ]
    for argument in args:
        command += ["--workload-arg", argument]
    if basic:
        command += ["--basic-evidence", str(basic)]
    if selector:
        command += ["--kernel-name", selector]
    result = subprocess.run(
        command, text=True, capture_output=True,
        env={**os.environ, "CPL_TEST_LOG": str(log)},
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, output, raw, calls, workload


@pytest.mark.parametrize(
    ("product", "target", "runtime"),
    [("a3", "bz-a3-1", "py311-torch"), ("a3", "bz-a3-2", "py311-torch"), ("a5", "bz-a5", "cann91")],
)
def test_basic_uses_product_runtime_and_writes_exact_remote_bytes(tmp_path, product, target, runtime):
    result, output, expected, calls, _ = run_cli(tmp_path, "basic", product=product, target=target)
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
    assert '"--launch-count=20"' in payload
    assert '"--kill=off"' in payload
    assert "--replay-mode" not in payload
    assert "ASCEND_RT_VISIBLE_DEVICES" in payload
    assert "range(8)" in payload and "timeout=20" in payload
    assert "matmul" not in payload.lower()
    assert base64.b64encode(workload.read_bytes()).decode() in payload
    assert payload.count('print("ACQUIRE_EVIDENCE_B64="') == 1
    assert payload.count('print("ACQUIRE_EVIDENCE_SHA256="') == 1


def test_pipe_requires_agent_selected_exact_exported_name_and_same_workload(tmp_path):
    workload = tmp_path / "user workload.py"
    workload.write_text("#!/usr/bin/env python3\nprint('arbitrary workload')\n")
    args = ["--mode", "fast", "value with spaces"]
    basic = tmp_path / "prior.json"
    basic.write_bytes(evidence_bytes("a3", "bz-a3-1", "basic", identity(workload, args)))
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
    basic.write_bytes(evidence_bytes("a3", "bz-a3-1", "basic", identity(workload, args)))
    result, output, _, calls, _ = run_cli(tmp_path, "pipe", selector=selector, basic=basic)
    assert result.returncode != 0
    assert "exact exported kernel" in result.stderr or "--kernel-name" in result.stderr
    assert not calls
    assert not output.exists()


def test_pipe_rejects_changed_workload_or_arguments(tmp_path):
    workload = tmp_path / "user workload.py"
    workload.write_text("#!/usr/bin/env python3\nprint('first')\n")
    basic = tmp_path / "prior.json"
    basic.write_bytes(evidence_bytes(
        "a3", "bz-a3-1", "basic",
        {"sha256": hashlib.sha256(workload.read_bytes()).hexdigest(), "arguments": ["different"]},
    ))
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
    assert observes[0][observes[0].index("observe") + 1] == observes[1][observes[1].index("observe") + 1]
    assert len([call for call in calls if "run" in call]) == 1


def test_terminal_failure_includes_handle_phase_and_log_excerpt(tmp_path):
    result, output, _, calls, _ = run_cli(tmp_path, "basic", scenario="terminal-failure")
    assert result.returncode == 1
    assert "remote:bz-a3-1:job:job-7" in result.stderr
    assert "phase=remote-job" in result.stderr
    assert "compile failed in user workload" in result.stderr
    assert not output.exists()
    assert len([call for call in calls if "run" in call]) == 1


def test_activity_evidence_cannot_claim_saturation(tmp_path):
    module = load_module()
    workload = tmp_path / "w"
    workload.write_bytes(b"x")
    raw = evidence_bytes("a5", "bz-a5", "pipe", identity(workload, []), selector="kernel beta")
    value = json.loads(raw)
    value["saturation"]["state"] = "saturated"
    with pytest.raises(module.AcquisitionError, match="reviewed capacity denominator"):
        module.validate_evidence(value, "pipe", "a5", "bz-a5", "test-key", identity(workload, []))


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
        "print('Profiling running finished. All task success.')\n"
    )
    msprof.chmod(0o755)
    payload = module.render_remote_payload(
        mode, "a3", "bz-a3-1", f"payload-{mode}", workload, [],
        "exact.kernel/7" if mode == "pipe" else None,
    )
    payload_path = tmp_path / "payload.py"
    payload_path.write_text(payload)
    result = subprocess.run(
        ["python3", str(payload_path)], cwd=tmp_path, text=True, capture_output=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "PYTHONPATH": str(modules)},
    )
    assert result.returncode == 0, result.stderr
    encoded = next(line.split("=", 1)[1] for line in result.stdout.splitlines()
                   if line.startswith("ACQUIRE_EVIDENCE_B64="))
    evidence = json.loads(base64.b64decode(encoded))
    assert evidence["capture"]["exported_kernel_names"] == ["exact.kernel/7"]
    assert len(evidence["capture"]["rows"]) == (1 if mode == "pipe" else 0)
    if mode == "pipe":
        assert evidence["capture"]["sources"][0]["role"] == "PipeUtilization"
        assert evidence["capture"]["sources"][1]["role"] == "selector_binding"
