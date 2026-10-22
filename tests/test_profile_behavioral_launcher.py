from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import socketserver
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_behavioral_launcher", ROOT / "scripts/profile_behavioral_launcher.py"
)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

GATE_TEST_SPEC = importlib.util.spec_from_file_location(
    "profile_gate_tests", ROOT / "tests/test_profile_behavioral_gate.py")
gate_tests = importlib.util.module_from_spec(GATE_TEST_SPEC)
assert GATE_TEST_SPEC.loader
sys.modules[GATE_TEST_SPEC.name] = gate_tests
GATE_TEST_SPEC.loader.exec_module(gate_tests)


def executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    value = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        value.update(item.relative_to(path).as_posix().encode() + b"\0")
        value.update(hashlib.sha256(item.read_bytes()).digest())
    return value.hexdigest()


def fixture(tmp_path: Path, kind: str = "acquisition"):
    skill = tmp_path / "candidate-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Use the supplied profiling evidence.\n")
    remote = executable(tmp_path / "remote" / "cpl-remote")
    runtime = tmp_path / "node"
    executable(runtime / "bin" / "node")
    codex = executable(runtime / "bin" / "codex")
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("secret")
    bwrap = executable(tmp_path / "bwrap")
    cases = []
    for index, product in enumerate(("a3", "a3", "a5", "a5"), 1):
        evidence = tmp_path / f"case-{index}.json"
        evidence.write_text(json.dumps({"case": index}) + "\n")
        cases.append({"case_id": f"case-{index}", "product": product,
                      "evidence": str(evidence), "evidence_sha256": digest(evidence)})
    model = {"name": "gpt-test", "reasoning_effort": "low"}
    model_hash = hashlib.sha256(json.dumps(
        model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request = {
        "schema": launcher.REQUEST_SCHEMA,
        "kind": kind,
        "session_id": "session-1",
        "arm": "candidate" if kind == "interpretation" else None,
        "manifest_sha256": "1" * 64,
        "prompt": str(prompt), "prompt_sha256": digest(prompt),
        "skill": str(skill), "skill_sha256": digest(skill),
        "launcher": {"identity": "profiling-behavioral-launcher/v1",
                     "sha256": launcher.launcher_digest()},
        "model": {"identity": "gpt", "config_sha256": model_hash, **model},
        "reviewer": {"identity": "reviewer/v1", "config_sha256": "5" * 64},
        "allowed_targets": {"a3": ["bz-a3-1", "bz-a3-2"], "a5": ["bz-a5"]},
        "units": ([{"product": "a3", "target": "bz-a3-1"},
                   {"product": "a5", "target": "bz-a5"}]
                  if kind == "acquisition" else cases),
    }
    instance = launcher.BehavioralLauncher(
        remote_command=[str(remote)], codex=str(codex), bwrap=str(bwrap), auth_home=auth
    )
    return instance, request


def test_bubblewrap_exposes_only_selected_skill_and_remote_broker(tmp_path: Path):
    instance, request = fixture(tmp_path)
    request["session_id"] = "candidate-treatment-session"
    checked = instance.validate_request(request)
    root = tmp_path / "run"
    root.mkdir()
    paths = instance.prepare(root, checked)
    command = instance.base_command(paths, checked)
    joined = "\0".join(map(str, command))

    assert "--unshare-all" in command and "--share-net" in command
    assert f"{paths.skill}\0/workspace/.agents/skills/ascend-profiling" in joined
    assert "/tools/cpl-remote" in command
    assert f"{ROOT / 'scripts/production_launcher.py'}\0/experiment/production_launcher.py" in joined
    assert f"{instance.auth_home}/auth.json\0/codex-home/auth.json" in joined
    assert str(Path.home() / ".agents" / "skills") not in joined
    assert str(Path.home() / ".codex" / "skills") not in joined
    public = json.loads((paths.workspace / "request.json").read_text())
    assert set(public) == {"schema", "turns"}
    assert "candidate" not in json.dumps(public).lower()
    assert "session" not in json.dumps(public).lower()


def test_base_command_exposes_retained_broker_mode_to_agent_process(tmp_path: Path):
    instance, request = fixture(tmp_path)
    fake_bwrap = executable(
        tmp_path / "functional-bwrap",
        """#!/usr/bin/env python3
import os
import subprocess
import sys

args = sys.argv[1:]
environment = dict(os.environ)
index = 0
no_value = {"--die-with-parent", "--new-session", "--unshare-all", "--share-net"}
one_value = {"--tmpfs", "--proc", "--dev", "--dir", "--chdir"}
two_values = {"--ro-bind", "--bind"}
while index < len(args):
    option = args[index]
    if option in no_value:
        index += 1
    elif option == "--clearenv":
        environment = {}
        index += 1
    elif option in one_value:
        index += 2
    elif option in two_values:
        index += 3
    elif option == "--setenv":
        environment[args[index + 1]] = args[index + 2]
        index += 3
    else:
        break
raise SystemExit(subprocess.run(args[index:], env=environment, check=False).returncode)
""",
    )
    instance.bwrap = str(fake_bwrap)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))

    run = subprocess.run(
        [*instance.base_command(paths, request), "/bin/sh", "-ceu",
         'test "$CPL_REMOTE_MODE" = retained-broker'],
        text=True, capture_output=True, check=False,
    )

    assert run.returncode == 0, run.stderr


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="Bubblewrap is unavailable")
def test_real_bubblewrap_executes_mounted_remote_wrapper_and_dependency(tmp_path: Path):
    instance, request = fixture(tmp_path)
    instance.bwrap = shutil.which("bwrap")
    executable(Path(instance.remote_command[0]), "#!/bin/sh\necho REMOTE_STATE=completed\n")
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    broker = launcher.RemoteBroker(instance.remote_command, paths.journal, paths.workspace,
                                   {"bz-a3-1", "bz-a3-2", "bz-a5"})
    server = socketserver.UnixStreamServer(
        str(paths.socket_dir / "broker.sock"), launcher._BrokerHandler)
    server.broker = broker
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = subprocess.run(
            [*instance.base_command(paths, request),
             "/tools/cpl-remote", "preflight", "bz-a3-1"],
            text=True, capture_output=True, check=False, timeout=30)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert run.returncode == 0, run.stderr
    assert "REMOTE_STATE=completed" in run.stdout


def test_prepare_snapshots_pinned_prompt_skill_and_case_bytes(tmp_path: Path):
    instance, request = fixture(tmp_path, "interpretation")
    original_prompt = Path(request["prompt"]).read_text()
    original_skill = (Path(request["skill"]) / "SKILL.md").read_text()
    original_case = Path(request["units"][0]["evidence"]).read_text()
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))

    Path(request["prompt"]).write_text("mutated prompt\n")
    (Path(request["skill"]) / "SKILL.md").write_text("mutated skill\n")
    Path(request["units"][0]["evidence"]).write_text("mutated case\n")

    assert paths.prompt.read_text() == original_prompt
    assert (paths.skill / "SKILL.md").read_text() == original_skill
    assert (paths.workspace / "case-1.json").read_text() == original_case


def test_request_requires_pinned_skill_prompt_and_exact_unit_shape(tmp_path: Path):
    instance, request = fixture(tmp_path)
    request["skill_sha256"] = "0" * 64
    with pytest.raises(launcher.LaunchError, match="skill changed"):
        instance.validate_request(request)


def test_request_rejects_non_string_session_and_case_ids(tmp_path: Path):
    instance, request = fixture(tmp_path)
    request["session_id"] = 7
    with pytest.raises(launcher.LaunchError, match="behavioral session"):
        instance.validate_request(request)

    instance, request = fixture(tmp_path / "interpretation", "interpretation")
    request["units"][0]["case_id"] = 7
    with pytest.raises(launcher.LaunchError, match="unique case IDs"):
        instance.validate_request(request)


@pytest.mark.parametrize("targets,error", [
    ({"a3": ["bz-a3-1", "bz-a3-1"], "a5": ["bz-a5"]}, "unique"),
    ({"a3": ["shared"], "a5": ["shared"]}, "product-scoped"),
    ({"a3": [7], "a5": ["bz-a5"]}, "valid target IDs"),
    ({"a3": ["bad target"], "a5": ["bz-a5"]}, "valid target IDs"),
])
def test_request_requires_unique_product_scoped_valid_target_ids(
        tmp_path: Path, targets: dict, error: str):
    instance, request = fixture(tmp_path)
    request["allowed_targets"] = targets

    with pytest.raises(launcher.LaunchError, match=error):
        instance.validate_request(request)


def test_acquisition_requires_distinct_assigned_targets(tmp_path: Path):
    instance, request = fixture(tmp_path)
    request["allowed_targets"]["a5"].append("bz-a3-1")
    request["units"][1]["target"] = "bz-a3-1"

    with pytest.raises(launcher.LaunchError, match="distinct assigned targets"):
        instance.validate_request(request)

    _, request = fixture(tmp_path / "other")
    request["units"].append({"product": "a3", "target": "bz-a3-1"})
    with pytest.raises(launcher.LaunchError, match="paired A3 and A5"):
        instance.validate_request(request)


@pytest.mark.parametrize("identity", ["launcher", "model"])
def test_request_rejects_actual_launcher_or_model_identity_drift(tmp_path: Path, identity: str):
    instance, request = fixture(tmp_path)
    key = "sha256" if identity == "launcher" else "config_sha256"
    request[identity][key] = "0" * 64
    with pytest.raises(launcher.LaunchError, match=identity):
        instance.validate_request(request)


def test_acquisition_turns_resume_one_session_for_a5(tmp_path: Path):
    instance, request = fixture(tmp_path)
    turns = instance.turn_plan(instance.validate_request(request))
    assert [turn["payload"]["product"] for turn in turns] == ["a3", "a5"]
    assert all(set(turn["payload"]) == {"product", "target", "prompt_sha256"}
               for turn in turns)
    assert [turn["payload"]["target"] for turn in turns] == ["bz-a3-1", "bz-a5"]
    assert turns[0]["resume"] is False and turns[1]["resume"] is True


def test_interpretation_uses_one_persistent_session_for_four_blinded_cases(tmp_path: Path):
    instance, request = fixture(tmp_path, "interpretation")
    turns = instance.turn_plan(instance.validate_request(request))
    assert len(turns) == 4 and [turn["resume"] for turn in turns] == [False, True, True, True]
    assert all("arm" not in turn["payload"] and "skill" not in turn["payload"] for turn in turns)
    assert {turn["payload"]["case_id"] for turn in turns} == {f"case-{i}" for i in range(1, 5)}


@pytest.mark.parametrize("mutation,error", [
    (lambda units: units[1].update(case_id=units[0]["case_id"]), "unique case IDs"),
    (lambda units: units[1].update(product="a5"), "two A3 and two A5"),
])
def test_interpretation_request_validates_case_identity_and_product_distribution(
        tmp_path: Path, mutation, error: str):
    instance, request = fixture(tmp_path, "interpretation")
    mutation(request["units"])

    with pytest.raises(launcher.LaunchError, match=error):
        instance.validate_request(request)


def test_broker_target_scope_can_be_narrowed_for_each_active_turn(tmp_path: Path):
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path,
                                   {"bz-a3-1", "bz-a5"})
    broker.restrict_targets({"bz-a3-1"})

    with pytest.raises(launcher.LaunchError, match="unapproved target"):
        broker.execute(["preflight", "bz-a5"])


def test_remote_broker_checkpoints_before_dispatch_and_reobserves_same_handle(
        tmp_path: Path, monkeypatch):
    (tmp_path / "job.sh").write_text("echo profile\n")
    calls = []
    outputs = [
        launcher.CommandResult(0, "REMOTE_HANDLE=remote:bz-a3-1:job:abc\nREMOTE_STATE=running\n", ""),
        launcher.CommandResult(0, "REMOTE_HANDLE=remote:bz-a3-1:job:abc\nREMOTE_STATE=completed\n", ""),
    ]

    def run(command, **_kwargs):
        calls.append(command)
        result = outputs.pop(0)
        return type("Result", (), {"returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path)
    first = broker.execute(["run", "bz-a3-1", "--dispatch-key", "capture-a3-1",
                            "--file", "/workspace/job.sh",
                            "--cwd", "/home/agent/run"])
    second = broker.execute(["run", "bz-a3-1", "--dispatch-key", "capture-a3-1",
                             "--file", "/workspace/job.sh",
                             "--cwd", "/home/agent/run"])
    journal = json.loads((tmp_path / "journal.json").read_text())

    assert first["handle"] == second["handle"] == "remote:bz-a3-1:job:abc"
    assert calls[0][:3] == ["cpl-remote", "run", "bz-a3-1"]
    assert calls[1] == ["cpl-remote", "observe", "remote:bz-a3-1:job:abc", "--wait"]
    assert journal["dispatches"][0]["state"] == "completed"
    assert journal["dispatches"][0]["request_sha256"]


def test_remote_broker_never_redispatches_handleless_uncertain_request(
        tmp_path: Path, monkeypatch):
    (tmp_path / "job.sh").write_text("echo profile\n")
    calls = []

    def timeout(command, **_kwargs):
        calls.append(command)
        raise launcher.subprocess.TimeoutExpired(command, 5, output="dispatch accepted\n",
                                                 stderr="observer connection lost\n")

    monkeypatch.setattr(launcher.subprocess, "run", timeout)
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path)
    arguments = ["run", "bz-a3-1", "--dispatch-key", "uncertain-a3",
                 "--file", "/workspace/job.sh"]
    first = broker.execute(arguments)
    with pytest.raises(launcher.LaunchError, match="uncertain dispatch.*never resubmit"):
        broker.execute(arguments)
    journal = json.loads((tmp_path / "journal.json").read_text())

    assert len(calls) == 1 and first["returncode"] == 124
    assert journal["dispatches"][0]["state"] == "uncertain"
    assert "observer connection lost" in journal["dispatches"][0]["result"]["stderr"]


@pytest.mark.parametrize("operation", ["observe", "result"])
def test_explicit_observe_or_result_updates_originating_dispatch_with_final_result(
        tmp_path: Path, monkeypatch, operation: str):
    (tmp_path / "job.sh").write_text("echo profile\n")
    outputs = [
        launcher.CommandResult(0, "REMOTE_HANDLE=remote:bz-a3-1:job:abc\nREMOTE_STATE=running\n", ""),
        launcher.CommandResult(0, "REMOTE_HANDLE=remote:bz-a3-1:job:abc\nREMOTE_STATE=completed\n", "final"),
    ]

    def run(_command, **_kwargs):
        result = outputs.pop(0)
        return type("Result", (), {"returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    journal_path = tmp_path / "journal.json"
    broker = launcher.RemoteBroker(["cpl-remote"], journal_path, tmp_path)
    broker.execute(["run", "bz-a3-1", "--dispatch-key", "observe-a3",
                    "--file", "/workspace/job.sh"])
    arguments = [operation, "remote:bz-a3-1:job:abc"]
    if operation == "observe":
        arguments.append("--wait")
    broker.execute(arguments)
    dispatch = json.loads(journal_path.read_text())["dispatches"][0]

    assert dispatch["state"] == "completed"
    assert dispatch["result"]["returncode"] == 0
    assert dispatch["result"]["stderr"] == "final"


def test_dispatch_key_binds_argv_and_payload_but_new_key_allows_corrected_run(
        tmp_path: Path, monkeypatch):
    payload = tmp_path / "job.sh"
    payload.write_text("echo first\n")
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        handle = f"remote:bz-a3-1:job:job-{len(calls)}"
        return type("Result", (), {"returncode": 0,
                                    "stdout": f"REMOTE_HANDLE={handle}\nREMOTE_STATE=completed\n",
                                    "stderr": ""})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    journal_path = tmp_path / "journal.json"
    broker = launcher.RemoteBroker(["cpl-remote"], journal_path, tmp_path)
    first = ["run", "bz-a3-1", "--dispatch-key", "attempt-1",
             "--file", "/workspace/job.sh"]
    broker.execute(first)
    broker.execute(first)
    payload.write_text("echo corrected\n")
    with pytest.raises(launcher.LaunchError, match="dispatch key.*changed"):
        broker.execute(first)
    broker.execute(["run", "bz-a3-1", "--dispatch-key", "attempt-2",
                    "--file", "/workspace/job.sh"])
    journal = json.loads(journal_path.read_text())

    assert len(calls) == 2
    assert all("--dispatch-key" not in command for command in calls)
    assert Path(calls[0][calls[0].index("--file") + 1]).read_text() == "echo first\n"
    assert Path(calls[1][calls[1].index("--file") + 1]).read_text() == "echo corrected\n"
    assert [row["dispatch_key"] for row in journal["dispatches"]] == ["attempt-1", "attempt-2"]
    assert journal["dispatches"][0]["file_sha256"] != journal["dispatches"][1]["file_sha256"]
    assert all("--dispatch-key" not in row["arguments"] for row in journal["dispatches"])


def test_remote_broker_rejects_raw_or_unapproved_commands(tmp_path: Path):
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path,
                                   allowed_targets={"bz-a3-1"})
    with pytest.raises(launcher.LaunchError, match="unsupported remote operation"):
        broker.execute(["ssh", "host"])
    with pytest.raises(launcher.LaunchError, match="unapproved target"):
        broker.execute(["preflight", "bz-a3-2"])
    with pytest.raises(launcher.LaunchError, match="unapproved target"):
        broker.execute(["observe", "remote:bz-a3-2:job:other", "--wait"])
    with pytest.raises(launcher.LaunchError, match="not dispatched"):
        broker.execute(["observe", "remote:bz-a3-1:job:other", "--wait"])
    with pytest.raises(launcher.LaunchError, match="workspace path"):
        broker.execute(["run", "bz-a3-1", "--dispatch-key", "bad-path",
                        "--file", "/etc/passwd"])
    with pytest.raises(launcher.LaunchError, match="file-backed"):
        broker.execute(["run", "bz-a3-1", "--", "echo", "unsafe"])
    with pytest.raises(launcher.LaunchError, match="dispatch key"):
        broker.execute(["run", "bz-a3-1", "--file", "/workspace/job.sh"])
    with pytest.raises(launcher.LaunchError, match="credentials"):
        broker.execute(["preflight", "bz-a3-1", "--token=do-not-log"])
    with pytest.raises(launcher.LaunchError, match="unsupported remote operation"):
        broker.execute(["download", "bz-a3-1", "/remote/evidence.json", "/tmp/output.json"])
    with pytest.raises(launcher.LaunchError, match="unsupported remote operation"):
        broker.execute(["upload", "bz-a3-1", "/workspace/input", "/remote/output"])


@pytest.mark.parametrize("document,expected", [
    ({"returncode": 1, "state": "failed", "stdout": "", "stderr": "compile error"},
     ("counted_failure", "compile", "compile")),
    ({"returncode": 1, "state": "observation-unavailable", "stdout": "", "stderr": "vpn"},
     ("discarded_infrastructure", "observer", "observer")),
    ({"returncode": 124, "state": "", "stdout": "", "stderr": "agent deadline"},
     ("counted_failure", "launcher", "launcher")),
    ({"returncode": 2, "state": "failed", "failure_type": "target_unavailable",
      "stdout": "", "stderr": "preflight failed"},
     ("discarded_infrastructure", "target_unavailable", "transport")),
    ({"returncode": 1, "state": "failed", "failure_type": "runtime",
      "stdout": "", "stderr": "remote job failed"},
     ("counted_failure", "runtime", "runtime")),
])
def test_failure_classification_is_host_owned_and_fail_closed(document, expected):
    assert launcher.classify_failure(document) == expected


@pytest.mark.parametrize("remote_output,expected", [
    (
        "PROFILE_COMMAND=msprof op --application=python workload.py "
        "--output=/tmp/basic --metrics=BasicInfo\n"
        "[ERROR] unexpected argument --metrics=BasicInfo\n"
        "no exported kernel row in BasicInfo\n",
        ("counted_failure", "profiler_command", "profiler"),
    ),
    (
        "Traceback (most recent call last):\n"
        "RuntimeError: expected one exported matmul selector, got []\n",
        ("counted_failure", "evidence", "evidence"),
    ),
    (
        'EXPERIMENT_FAILURE={"message": "BasicInfo profiler failure rc=0: '
        "[ERROR] unexpected argument --device=0\\n"
        '[INFO] Use msprof op --help", "type": "RuntimeError"}\n',
        ("counted_failure", "profiler_command", "profiler"),
    ),
])
def test_remote_profiler_evidence_outranks_incidental_agent_compilation_prose(
        remote_output: str, expected: tuple[str, str, str]):
    document = {
        "returncode": 1,
        "state": "failed",
        "trusted_stdout": remote_output,
        "trusted_stderr": "",
        "stdout": f"remote result:\n{remote_output}",
        "stderr": (
            "codex stderr:\nThe profiling guide says compile and compilation errors remain actionable.\n"
            "launcher diagnostic:\ncounted remote failure cannot be replaced"
        ),
    }

    assert launcher.classify_failure(document) == expected


def test_agent_profiler_prose_cannot_relabel_trusted_remote_compile_failure():
    document = {
        "returncode": 1,
        "state": "failed",
        "trusted_stdout": "Triton compile error at candidate.py:17\n",
        "trusted_stderr": "",
        "stdout": "remote result:\nTriton compile error at candidate.py:17\n",
        "stderr": "codex stderr:\nI consulted the msprof profiler guide.\n",
    }

    assert launcher.classify_failure(document) == (
        "counted_failure", "compile", "compile")


@pytest.mark.parametrize("remote_output", [
    "Triton compile error in profiler_wrapper.py: invalid layout\n",
    "Triton compile error: selector expression has invalid type\n",
])
def test_incidental_profiler_or_selector_words_do_not_relabel_remote_compile(
        remote_output: str):
    document = {
        "returncode": 1,
        "state": "failed",
        "trusted_stdout": remote_output,
        "trusted_stderr": "",
        "stdout": remote_output,
        "stderr": "",
    }

    assert launcher.classify_failure(document) == (
        "counted_failure", "compile", "compile")


def test_counted_remote_failure_cannot_be_hidden_by_later_success():
    failed = {"target": "bz-a3-1", "handle": "remote:bz-a3-1:job:bad",
              "state": "failed", "result": {"returncode": 1, "state": "failed",
              "failure_type": None, "stdout": "", "stderr": ""}}
    succeeded = {"target": "bz-a3-1", "handle": "remote:bz-a3-1:job:good",
                 "state": "completed", "result": {"returncode": 0,
                 "state": "completed", "failure_type": None, "stdout": "", "stderr": ""}}
    journal = {"dispatches": [failed, succeeded], "calls": [{
        "handle": failed["handle"], "target": "bz-a3-1",
        "result": {"returncode": 0, "state": "completed", "failure_type": None,
                   "stdout": 'REMOTE_CONTENT={"status":"failure","phase":"missing_row"}',
                   "stderr": ""}}]}

    assert launcher.counted_dispatch_failure(journal, (0, 0), "bz-a3-1") == failed


def test_evidenced_infrastructure_dispatch_may_be_replaced():
    failed = {"target": "bz-a5", "handle": "remote:bz-a5:job:busy",
              "state": "failed", "result": {"returncode": 1, "state": "device-busy",
              "failure_type": "device_busy", "stdout": "", "stderr": ""}}
    journal = {"dispatches": [failed], "calls": []}

    assert launcher.counted_dispatch_failure(journal, (0, 0), "bz-a5") is None


def test_mixed_infrastructure_then_counted_failure_uses_selected_dispatch_calls_only():
    busy_handle = "remote:bz-a3-1:job:busy"
    failed_handle = "remote:bz-a3-1:job:runtime"
    busy = {"target": "bz-a3-1", "handle": busy_handle, "state": "failed",
            "result": {"returncode": 1, "state": "device-busy",
                       "failure_type": "device_busy", "stdout": "", "stderr": ""}}
    failed = {"target": "bz-a3-1", "handle": failed_handle, "state": "failed",
              "result": {"returncode": 1, "state": "failed", "failure_type": None,
                         "stdout": "", "stderr": "remote job failed"}}
    journal = {"dispatches": [busy, failed], "calls": [
        {"operation": "logs", "target": "bz-a3-1", "handle": busy_handle,
         "result": {"returncode": 0, "state": "completed",
                    "failure_type": "device_busy", "stdout": "device busy", "stderr": ""}},
        {"operation": "logs", "target": "bz-a3-1", "handle": failed_handle,
         "result": {"returncode": 0, "state": "completed", "failure_type": "compile",
                    "stdout": "triton compile error", "stderr": ""}},
    ]}

    assert launcher.counted_dispatch_failure(journal, (0, 0), "bz-a3-1") == failed
    evidence = launcher.dispatch_failure_evidence(journal, failed, 0)
    assert launcher.classify_failure(evidence) == ("counted_failure", "compile", "compile")
    assert "device busy" not in evidence["stdout"]


@pytest.mark.parametrize("marker_classification,expected", [
    ("device_unavailable", ("discarded_infrastructure", "device_busy", "transport")),
    ("host_environment", ("discarded_infrastructure", "target_unavailable", "transport")),
    ("workload_failure", ("counted_failure", "runtime", "runtime")),
    ("profiler_failure", ("counted_failure", "profiler_command", "profiler")),
    ("evidence_failure", ("counted_failure", "evidence", "evidence")),
    ("bundle_failure", ("counted_failure", "launcher", "launcher")),
])
def test_selected_remote_acquisition_failure_marker_is_classified(
        marker_classification: str, expected: tuple[str, str, str]):
    handle = "remote:bz-a3-1:job:structured-failure"
    marker = "ACQUIRE_FAILURE_JSON=" + json.dumps({
        "classification": marker_classification, "phase": "retention",
        "detail": "bounded remote failure",
    }, separators=(",", ":"))
    dispatch = {
        "target": "bz-a3-1", "handle": handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
    }
    journal = {"dispatches": [dispatch], "calls": [{
        "operation": "logs", "target": "bz-a3-1", "handle": handle,
        "result": {"returncode": 0, "state": "completed",
                   "stdout": marker + "\n", "stderr": ""},
    }]}

    evidence = launcher.dispatch_failure_evidence(journal, dispatch, 0)

    assert launcher.classify_failure(evidence) == expected


@pytest.mark.parametrize("log_text", [
    'ACQUIRE_FAILURE_JSON={"classification":"host_environment"}\n',
    'ACQUIRE_FAILURE_JSON={"classification":"unknown","phase":"retention"}\n',
    'ACQUIRE_FAILURE_JSON={not-json}\n',
    ('ACQUIRE_FAILURE_JSON={"classification":"host_environment","phase":"retention"}\n'
     'ACQUIRE_FAILURE_JSON={"classification":"host_environment","phase":"retention"}\n'),
])
def test_malformed_unknown_or_duplicate_acquisition_failure_marker_fails_closed(
        log_text: str):
    handle = "remote:bz-a3-1:job:invalid-marker"
    dispatch = {
        "target": "bz-a3-1", "handle": handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
    }
    journal = {"dispatches": [dispatch], "calls": [{
        "operation": "logs", "target": "bz-a3-1", "handle": handle,
        "result": {"returncode": 0, "state": "completed",
                   "stdout": log_text, "stderr": ""},
    }]}

    evidence = launcher.dispatch_failure_evidence(journal, dispatch, 0)

    assert launcher.classify_failure(evidence) == (
        "counted_failure", "launcher", "launcher")


def test_agent_prose_cannot_spoof_acquisition_infrastructure_marker():
    marker = ('ACQUIRE_FAILURE_JSON={"classification":"host_environment",'
              '"phase":"retention"}')
    document = {
        "returncode": 1, "state": "failed",
        "trusted_stdout": "", "trusted_stderr": "",
        "stdout": "", "stderr": f"agent explanation:\n{marker}\n",
    }

    assert launcher.classify_failure(document) == (
        "counted_failure", "launcher", "launcher")


@pytest.mark.parametrize("marker_location", ["dispatch_stdout", "unrelated_log"])
def test_acquisition_failure_marker_outside_selected_remote_stderr_or_logs_is_ignored(
        marker_location: str):
    handle = "remote:bz-a3-1:job:selected"
    marker = ('ACQUIRE_FAILURE_JSON={"classification":"host_environment",'
              '"phase":"retention"}\n')
    dispatch = {
        "target": "bz-a3-1", "handle": handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed",
                   "stdout": marker if marker_location == "dispatch_stdout" else "",
                   "stderr": ""},
    }
    calls = ([{
        "operation": "logs", "target": "bz-a3-1",
        "handle": "remote:bz-a3-1:job:unrelated",
        "result": {"returncode": 0, "state": "completed", "stdout": marker, "stderr": ""},
    }] if marker_location == "unrelated_log" else [])

    evidence = launcher.dispatch_failure_evidence(
        {"dispatches": [dispatch], "calls": calls}, dispatch, 0)

    assert launcher.classify_failure(evidence) == (
        "counted_failure", "launcher", "launcher")


def test_failure_result_uses_selected_retained_acquisition_marker(tmp_path: Path):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    handle = "remote:bz-a3-1:job:20261008T210414Z-9865c9544328"
    marker = ("ACQUIRE_FAILURE_JSON="
              '{"classification": "host_environment", '
              '"message": "dispatch key already retained", "phase": "retention"}\n')
    paths.journal.write_text(json.dumps({
        "schema": "profiling-skill/remote-journal/v1",
        "dispatches": [{
            "target": "bz-a3-1", "handle": handle, "state": "failed",
            "arguments": ["run", "bz-a3-1", "--file", "/workspace/job.sh"],
            "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
        }],
        "calls": [{
            "operation": "logs", "target": "bz-a3-1", "handle": handle,
            "arguments": ["logs", "--stream", "stderr", handle],
            "result": {"returncode": 0, "state": "completed",
                       "stdout": marker, "stderr": ""},
        }],
    }))

    result = instance._failure_result(
        request, paths, store, request["units"][0], "remote acquisition failed",
        journal_start=(0, 0), failed_handle=handle,
    )

    assert result["status"] == "discarded_infrastructure"
    assert result["record"]["failure_type"] == "target_unavailable"
    assert result["record"]["stage"] == "transport"
    assert result["record"]["handle"] == handle


def test_structured_infrastructure_retry_can_succeed_in_same_turn():
    failed_handle = "remote:bz-a3-1:job:host-environment"
    failed = {
        "target": "bz-a3-1", "handle": failed_handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
    }
    succeeded = {
        "target": "bz-a3-1", "handle": "remote:bz-a3-1:job:success",
        "state": "completed",
        "result": {"returncode": 0, "state": "completed", "stdout": "", "stderr": ""},
    }
    marker = ('ACQUIRE_FAILURE_JSON={"classification":"host_environment",'
              '"phase":"retention"}\n')
    journal = {"dispatches": [failed, succeeded], "calls": [{
        "operation": "logs", "target": "bz-a3-1", "handle": failed_handle,
        "result": {"returncode": 0, "state": "completed", "stdout": marker, "stderr": ""},
    }]}

    assert launcher.counted_dispatch_failure(journal, (0, 0), "bz-a3-1") is None


def test_structured_counted_failure_cannot_be_hidden_by_retry_success():
    failed_handle = "remote:bz-a3-1:job:workload-failure"
    failed = {
        "target": "bz-a3-1", "handle": failed_handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
    }
    succeeded = {
        "target": "bz-a3-1", "handle": "remote:bz-a3-1:job:success",
        "state": "completed",
        "result": {"returncode": 0, "state": "completed", "stdout": "", "stderr": ""},
    }
    marker = ('ACQUIRE_FAILURE_JSON={"classification":"workload_failure",'
              '"phase":"execution"}\n')
    journal = {"dispatches": [failed, succeeded], "calls": [{
        "operation": "logs", "target": "bz-a3-1", "handle": failed_handle,
        "result": {"returncode": 0, "state": "completed", "stdout": marker, "stderr": ""},
    }]}

    assert launcher.counted_dispatch_failure(journal, (0, 0), "bz-a3-1") == failed


def test_structured_counted_failure_keeps_existing_diagnostic_precedence():
    handle = "remote:bz-a3-1:job:diagnostic-precedence"
    dispatch = {
        "target": "bz-a3-1", "handle": handle, "state": "failed",
        "result": {"returncode": 1, "state": "failed", "stdout": "", "stderr": ""},
    }
    marker = ('ACQUIRE_FAILURE_JSON={"classification":"workload_failure",'
              '"phase":"profiling"}\nmsprof op --help\n')
    journal = {"dispatches": [dispatch], "calls": [{
        "operation": "logs", "target": "bz-a3-1", "handle": handle,
        "result": {"returncode": 0, "state": "completed", "stdout": marker, "stderr": ""},
    }]}

    evidence = launcher.dispatch_failure_evidence(journal, dispatch, 0)

    assert launcher.classify_failure(evidence) == (
        "counted_failure", "profiler_command", "profiler")


@pytest.mark.parametrize("operation,stdout,returncode,expected", [
    ("preflight", "REMOTE_STATE=failed\nREMOTE_FAILURE_TYPE=device_busy\n", 1,
     "device_busy"),
])
def test_remote_broker_preserves_or_normalizes_evidenced_infrastructure_failure(
        tmp_path: Path, monkeypatch, operation: str, stdout: str, returncode: int,
        expected: str):
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": returncode, "stdout": stdout,
                            "stderr": "trusted remote preflight failed"})())
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path,
                                   {"bz-a3-1"})

    result = broker.execute([operation, "bz-a3-1"])

    assert result["failure_type"] == expected
    assert launcher.classify_failure(result)[0] == "discarded_infrastructure"


@pytest.mark.parametrize("stdout,stderr", [
    ("REMOTE_STATE=failed\n", "preflight rejected malformed profile"),
    ("not remote protocol output\n", "preflight command failed"),
])
def test_preflight_errors_without_explicit_availability_evidence_are_counted(
        tmp_path: Path, monkeypatch, stdout: str, stderr: str):
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 2, "stdout": stdout, "stderr": stderr})())
    broker = launcher.RemoteBroker(["cpl-remote"], tmp_path / "journal.json", tmp_path,
                                   {"bz-a3-1"})

    result = broker.execute(["preflight", "bz-a3-1"])

    assert result["failure_type"] is None
    assert launcher.classify_failure(result)[0] == "counted_failure"


def test_trusted_cpl_remote_transport_diagnostic_is_normalized():
    raw = launcher.CommandResult(
        2, "", "error: remote transport unavailable; retrying attempt 3/3")
    assert launcher._remote_failure_type("run", {}, raw) == "transport"


def test_artifacts_are_bounded_and_receipt_matches_gate_schema(tmp_path: Path):
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    command = store.text("failures/s-command.txt", "cpl-remote run ...\n")
    log = store.text("failures/s-log.txt", "compile error\n")
    diagnostic = store.text("failures/s-diagnostic.txt", "compile\n")
    record = launcher.failure_record(
        store=store, session_id="s", kind="acquisition", arm=None,
        classification="counted_failure", failure_type="compile", stage="compile",
        product="a3", target="bz-a3-1", handle=None,
        command=command, log=log, diagnostic=diagnostic,
        manifest_sha256="1" * 64,
        launcher_identity={"identity": "behavioral-launcher/v1", "sha256": "2" * 64},
        model_identity={"identity": "gpt", "config_sha256": "3" * 64},
        skill_sha256="4" * 64,
    )
    receipt = json.loads((store.root / record["receipt"]["path"]).read_text())
    assert receipt["schema"] == "profiling-skill/launcher-receipt/v1"
    assert receipt["artifacts"] == {"command": command["sha256"], "log": log["sha256"],
                                     "diagnostic": diagnostic["sha256"]}
    with pytest.raises(launcher.LaunchError, match="artifact exceeds"):
        store.text("too-long.txt", "x" * (launcher.MAX_ARTIFACT + 1))


def test_review_queue_binds_reasoning_and_requires_pinned_external_reviewer(tmp_path: Path):
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    reasoning = store.text("reasoning/s-a3.txt", "MTE2 is active; capacity is unknown.\n")
    queue = launcher.review_queue(
        [{"session_id": "s", "unit": "a3", "reasoning": reasoning}],
        {"identity": "reviewer/v1", "config_sha256": "5" * 64},
    )
    assert queue["status"] == "review_pending" and queue["items"][0]["reasoning"] == reasoning
    assert "passed" not in queue["items"][0]


def test_acquisition_draft_binds_actual_thread_handle_target_and_remote_command(tmp_path: Path):
    instance, request = fixture(tmp_path)
    checked = instance.validate_request(request)
    root = tmp_path / "run"
    root.mkdir()
    paths = instance.prepare(root, checked)
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    journal_calls = []
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:job-{index}"
        provenance = {"product": unit["product"], "target": unit["target"]}
        evidence = paths.workspace / f"profile-{index}.json"
        evidence.write_text(json.dumps({"schema": "compact/v1", "provenance": provenance}))
        evidence_sha256 = digest(evidence)
        command = store.text(f"agent-command-{index}.txt", "codex exec\n")
        log = store.text(f"agent-log-{index}.txt", "agent output\n")
        turns.append({"payload": {"product": unit["product"], "target": unit["target"],
                                  "prompt_sha256": request["prompt_sha256"]},
                      "answer": {"product": unit["product"], "target": unit["target"],
                                 "handle": handle, "evidence": evidence.name,
                                 "reasoning": "capacity evidence is explicit"},
                      "command": command, "log": log})
        result_stdout = ("REMOTE_STATE=completed\n" if index == 1 else
                         f"REMOTE_CONTENT_SHA256={evidence_sha256}\n")
        dispatches.append({"request_sha256": str(index) * 64, "target": unit["target"],
                           "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
                           "state": "completed", "handle": handle,
                           "result": {"returncode": 0, "state": "completed",
                                      "stdout": result_stdout,
                                      "stderr": ""}})
        if index == 1:
            journal_calls.extend([
                {"operation": "logs", "handle": "remote:bz-a3-1:job:unrelated",
                 "result": {"stdout": f"REMOTE_CONTENT_SHA256={'0' * 64}\n"}},
                {"operation": "logs", "handle": handle,
                 "result": {"stdout": f"REMOTE_CONTENT_SHA256={evidence_sha256}\n"}},
            ])
    paths.journal.write_text(json.dumps({"schema": "profiling-skill/remote-journal/v1",
                                         "dispatches": dispatches, "calls": journal_calls}))

    draft, _ = instance._success_draft(request, turns, paths, store, "actual-thread")

    assert draft["session_id"] == "actual-thread"
    assert all(outcome["session_id"] == "actual-thread" for outcome in draft["outcomes"])
    command_text = (store.root / draft["outcomes"][0]["command"]["path"]).read_text()
    assert command_text.startswith("cpl-remote run bz-a3-1 --file")


def test_acquisition_retains_exact_noncanonical_remote_evidence_bytes(tmp_path: Path):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    expected = {}
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:exact-{index}"
        provenance = {"product": unit["product"], "target": unit["target"]}
        evidence = paths.workspace / f"exact-{index}.json"
        evidence.write_text(
            '{\n  "provenance": ' + json.dumps(provenance, separators=(", ", ": "))
            + ',\n  "schema": "compact/v1"\n}\n')
        evidence_sha256 = digest(evidence)
        expected[unit["product"]] = (evidence.read_bytes(), evidence_sha256)
        turns.append({"payload": {"product": unit["product"], "target": unit["target"],
                                  "prompt_sha256": request["prompt_sha256"]},
                      "answer": {"product": unit["product"], "target": unit["target"],
                                 "handle": handle, "evidence": evidence.name,
                                 "reasoning": "exact bytes retained"},
                      "command": store.text(f"command-exact-{index}.txt", "codex\n"),
                      "log": store.text(f"log-exact-{index}.txt", "agent\n")})
        dispatches.append({"request_sha256": str(index) * 64, "target": unit["target"],
                           "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
                           "state": "completed", "handle": handle,
                           "result": {"returncode": 0, "state": "completed",
                                      "stdout": f"REMOTE_CONTENT_SHA256={evidence_sha256}\n",
                                      "stderr": ""}})
    paths.journal.write_text(json.dumps({"schema": "profiling-skill/remote-journal/v1",
                                         "dispatches": dispatches, "calls": []}))

    draft, _ = instance._success_draft(request, turns, paths, store, "actual-thread")

    for outcome in draft["outcomes"]:
        content, emitted_hash = expected[outcome["product"]]
        retained = store.root / outcome["evidence"]["path"]
        assert retained.read_bytes() == content
        assert outcome["evidence"]["sha256"] == emitted_hash
        assert outcome["evidence"]["remote_sha256"] == emitted_hash
        assert outcome["evidence"]["provenance"] == {
            "product": outcome["product"], "target": outcome["target"]}


@pytest.mark.parametrize("invalid_provenance", [
    {"product": "a3", "target": "bz-a3-1", "handle": "remote:bz-a3-1:job:job-1"},
    {"product": "a5", "target": "bz-a3-1"},
    {"product": "a3", "target": "bz-a3-2"},
])
def test_acquisition_draft_requires_exact_producer_provenance(
        tmp_path: Path, invalid_provenance: dict):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:job-{index}"
        provenance = (invalid_provenance if index == 1 else
                      {"product": unit["product"], "target": unit["target"]})
        evidence = paths.workspace / f"profile-{index}.json"
        evidence.write_text(json.dumps({"schema": "compact/v1", "provenance": provenance}))
        evidence_sha256 = digest(evidence)
        turns.append({"payload": {"product": unit["product"], "target": unit["target"],
                                  "prompt_sha256": request["prompt_sha256"]},
                      "answer": {"product": unit["product"], "target": unit["target"],
                                 "handle": handle, "evidence": evidence.name,
                                 "reasoning": "producer provenance is bounded"},
                      "command": store.text(f"command-{index}.txt", "codex\n"),
                      "log": store.text(f"log-{index}.txt", "agent\n")})
        dispatches.append({"request_sha256": str(index) * 64, "target": unit["target"],
                           "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
                           "state": "completed", "handle": handle,
                           "result": {"returncode": 0, "state": "completed",
                                      "stdout": f"REMOTE_CONTENT_SHA256={evidence_sha256}\n",
                                      "stderr": ""}})
    paths.journal.write_text(json.dumps({"schema": "profiling-skill/remote-journal/v1",
                                         "dispatches": dispatches, "calls": []}))

    with pytest.raises(launcher.LaunchError, match="exact provenance"):
        instance._success_draft(request, turns, paths, store, "actual-thread")


@pytest.mark.parametrize("state,returncode", [("running", 0), ("failed", 1),
                                                ("completed", 1)])
def test_acquisition_draft_rejects_nonterminal_or_failed_dispatch(
        tmp_path: Path, state: str, returncode: int):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:job-{index}"
        provenance = {"product": unit["product"], "target": unit["target"]}
        evidence = paths.workspace / f"profile-{index}.json"
        evidence.write_text(json.dumps({"schema": "compact/v1", "provenance": provenance}))
        evidence_sha256 = digest(evidence)
        turns.append({"payload": {"product": unit["product"], "target": unit["target"],
                                  "prompt_sha256": request["prompt_sha256"]},
                      "answer": {"product": unit["product"], "target": unit["target"],
                                 "handle": handle, "evidence": evidence.name,
                                 "reasoning": "bounded by explicit evidence"},
                      "command": store.text(f"command-{index}.txt", "codex\n"),
                      "log": store.text(f"log-{index}.txt", "agent\n")})
        dispatches.append({"request_sha256": str(index) * 64, "target": unit["target"],
                           "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
                           "state": state if index == 1 else "completed", "handle": handle,
                           "result": {"returncode": returncode if index == 1 else 0,
                                      "state": state if index == 1 else "completed",
                                      "stdout": f"REMOTE_CONTENT_SHA256={evidence_sha256}\n",
                                      "stderr": ""}})
    paths.journal.write_text(json.dumps({"schema": "profiling-skill/remote-journal/v1",
                                         "dispatches": dispatches, "calls": []}))

    with pytest.raises(launcher.LaunchError, match="terminal successful dispatch"):
        instance._success_draft(request, turns, paths, store, "actual-thread")


def test_acquisition_draft_rejects_agent_evidence_not_bound_to_remote_bytes(tmp_path: Path):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    journal_calls = []
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:job-{index}"
        provenance = {"product": unit["product"], "target": unit["target"]}
        evidence = paths.workspace / f"profile-{index}.json"
        evidence.write_text(json.dumps({"schema": "compact/v1", "provenance": provenance}))
        turns.append({"payload": {"product": unit["product"], "target": unit["target"],
                                  "prompt_sha256": request["prompt_sha256"]},
                      "answer": {"product": unit["product"], "target": unit["target"],
                                 "handle": handle, "evidence": evidence.name,
                                 "reasoning": "remote evidence is retained"},
                      "command": store.text(f"command-{index}.txt", "codex\n"),
                      "log": store.text(f"log-{index}.txt", "agent\n")})
        emitted_hash = (None if index == 1 else digest(evidence))
        dispatches.append({"request_sha256": str(index) * 64, "target": unit["target"],
                           "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
                           "state": "completed", "handle": handle,
                           "result": {"returncode": 0, "state": "completed",
                                      "stdout": (f"REMOTE_CONTENT_SHA256={emitted_hash}\n"
                                                 if emitted_hash else "REMOTE_STATE=completed\n"),
                                      "stderr": ""}})
        if index == 1:
            journal_calls.append({
                "operation": "logs", "handle": "remote:bz-a3-1:job:unrelated",
                "result": {"stdout": f"REMOTE_CONTENT_SHA256={digest(evidence)}\n"},
            })
    paths.journal.write_text(json.dumps({"schema": "profiling-skill/remote-journal/v1",
                                         "dispatches": dispatches, "calls": journal_calls}))

    with pytest.raises(launcher.LaunchError, match="remote evidence bytes"):
        instance._success_draft(request, turns, paths, store, "actual-thread")


def test_failed_job_retains_complete_journal_and_compiler_diagnostics_from_logs(
        tmp_path: Path, monkeypatch):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    (paths.workspace / "job.sh").write_text("python compile_kernel.py\n")
    outputs = [
        launcher.CommandResult(
            0, "REMOTE_HANDLE=remote:bz-a3-1:job:compile-1\nREMOTE_STATE=running\n", ""),
        launcher.CommandResult(
            1, "REMOTE_HANDLE=remote:bz-a3-1:job:compile-1\nREMOTE_STATE=failed\n"
               "REMOTE_EXIT=1\n", ""),
        launcher.CommandResult(
            0, "REMOTE_HANDLE=remote:bz-a3-1:job:compile-1\nREMOTE_STATE=completed\n"
               "REMOTE_CONTENT=triton compile error: invalid layout\n", ""),
    ]

    def run(_command, **_kwargs):
        result = outputs.pop(0)
        return type("Result", (), {"returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    broker = launcher.RemoteBroker(["cpl-remote"], paths.journal, paths.workspace,
                                   {"bz-a3-1", "bz-a3-2", "bz-a5"})
    broker.execute(["run", "bz-a3-1", "--dispatch-key", "compile-attempt-1",
                    "--file", "/workspace/job.sh"])
    broker.execute(["result", "remote:bz-a3-1:job:compile-1"])
    broker.execute(["logs", "remote:bz-a3-1:job:compile-1", "--tail", "200"])

    store = launcher.ArtifactStore(tmp_path / "artifacts")
    failed = instance._failure_result(
        request, paths, store, request["units"][0], "agent stopped after failed result",
        codex_command="codex exec",
        codex_stderr="I consulted the msprof guide; remote check failed",
        codex_returncode=1)

    assert failed["record"]["failure_type"] == "compile"
    log = (store.root / failed["record"]["log"]["path"]).read_text()
    assert "triton compile error: invalid layout" in log
    journal = json.loads((store.root / failed["remote_journal"]["path"]).read_text())
    assert journal["dispatches"][0]["handle"] == "remote:bz-a3-1:job:compile-1"
    assert [call["operation"] for call in journal["calls"]] == ["run", "result", "logs"]


def test_oversized_success_journal_is_compact_and_keeps_remote_digest_binding(tmp_path: Path):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    turns = []
    dispatches = []
    original_payload = "profile sample\n" * 10_000
    for index, unit in enumerate(request["units"], 1):
        handle = f"remote:{unit['target']}:job:large-{index}"
        evidence = paths.workspace / f"large-{index}.json"
        evidence.write_text(json.dumps({
            "schema": "compact/v1",
            "provenance": {"product": unit["product"], "target": unit["target"]},
        }))
        evidence_sha256 = digest(evidence)
        turns.append({
            "payload": {"product": unit["product"], "target": unit["target"],
                        "prompt_sha256": request["prompt_sha256"]},
            "answer": {"product": unit["product"], "target": unit["target"],
                       "handle": handle, "evidence": evidence.name,
                       "reasoning": "large remote output remains digest-bound"},
            "command": store.text(f"large-command-{index}.txt", "codex\n"),
            "log": store.text(f"large-log-{index}.txt", "agent\n"),
        })
        stdout = (f"REMOTE_CONTENT_SHA256={evidence_sha256}\n" + original_payload
                  if index == 1 else json.dumps({
                      "REMOTE_CONTENT_SHA256": evidence_sha256,
                      "content": original_payload,
                  }))
        dispatches.append({
            "request_sha256": str(index) * 64,
            "target": unit["target"],
            "dispatch_key": f"large-{index}",
            "file_sha256": str(index + 2) * 64,
            "arguments": ["run", unit["target"], "--file", "/workspace/job.sh"],
            "state": "completed", "handle": handle,
            "result": {"returncode": 0, "state": "completed", "handle": handle,
                       "stdout": stdout,
                       "stderr": ""},
        })
    live_journal = {"schema": "profiling-skill/remote-journal/v1",
                    "dispatches": dispatches, "calls": []}
    paths.journal.write_text(json.dumps(live_journal))

    draft, _ = instance._success_draft(request, turns, paths, store, "actual-thread")
    retained_ref = launcher.retain_remote_journal(
        store, "remote/large-success.json", json.loads(paths.journal.read_text()))
    retained_path = store.root / retained_ref["path"]
    retained = json.loads(retained_path.read_text())

    assert len(retained_path.read_bytes()) < launcher.MAX_ARTIFACT
    assert paths.journal.stat().st_size > launcher.MAX_ARTIFACT
    source_bytes = json.dumps(
        live_journal, sort_keys=True, separators=(",", ":")).encode()
    assert retained["source_canonical_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert retained["dispatches"][0]["handle"] == dispatches[0]["handle"]
    assert retained["dispatches"][0]["arguments"] == dispatches[0]["arguments"]
    assert retained["dispatches"][0]["request_sha256"] == dispatches[0]["request_sha256"]
    assert retained["dispatches"][0]["file_sha256"] == dispatches[0]["file_sha256"]
    assert f"sha256={hashlib.sha256(dispatches[0]['result']['stdout'].encode()).hexdigest()}" \
        in retained["dispatches"][0]["result"]["stdout"]
    assert f"REMOTE_CONTENT_SHA256={draft['outcomes'][0]['evidence']['remote_sha256']}" \
        in retained["dispatches"][0]["result"]["stdout"]
    assert f"REMOTE_CONTENT_SHA256={draft['outcomes'][1]['evidence']['remote_sha256']}" \
        in retained["dispatches"][1]["result"]["stdout"]
    assert json.loads(paths.journal.read_text())["dispatches"][0]["result"]["stdout"] \
        .endswith(original_payload)


def test_oversized_failure_journal_retains_classification_and_never_recurses(
        tmp_path: Path):
    instance, request = fixture(tmp_path)
    paths = instance.prepare(tmp_path / "run", instance.validate_request(request))
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    handle = "remote:bz-a3-1:job:large-failure"
    diagnostic = "triton compile error: invalid layout\n" + ("compiler trace\n" * 10_000)
    journal = {
        "schema": "profiling-skill/remote-journal/v1",
        "dispatches": [{
            "request_sha256": "a" * 64, "target": "bz-a3-1",
            "dispatch_key": "large-failure", "file_sha256": "b" * 64,
            "arguments": ["run", "bz-a3-1", "--file", "/workspace/job.sh"],
            "state": "failed", "handle": handle,
            "result": {"returncode": 1, "state": "failed", "handle": handle,
                       "stdout": diagnostic, "stderr": ""},
        }],
        "calls": [{
            "operation": "logs", "target": "bz-a3-1", "handle": handle,
            "arguments": ["logs", handle], "request_sha256": "c" * 64,
            "result": {"returncode": 0, "state": "completed", "handle": handle,
                       "content": diagnostic, "stdout": diagnostic, "stderr": ""},
        }],
    }
    paths.journal.write_text(json.dumps(journal))

    failed = instance._failure_result(
        request, paths, store, request["units"][0], "remote compilation failed",
        journal_start=(0, 0), failed_handle=handle)
    retained_path = store.root / failed["remote_journal"]["path"]
    retained = json.loads(retained_path.read_text())
    retained_evidence = launcher.dispatch_failure_evidence(
        retained, retained["dispatches"][0], 0)

    assert failed["record"]["failure_type"] == "compile"
    assert launcher.classify_failure(retained_evidence) == (
        "counted_failure", "compile", "compile")
    assert len(retained_path.read_bytes()) < launcher.MAX_ARTIFACT
    assert retained["dispatches"][0]["handle"] == handle
    assert retained["calls"][0]["handle"] == handle
    assert "sha256=" in retained["calls"][0]["result"]["content"]


def test_pathological_journal_uses_bounded_representative_audit_summary(tmp_path: Path):
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    calls = []
    for index in range(2_000):
        handle = f"remote:bz-a3-1:job:pathological-{index}"
        calls.append({
            "operation": "logs", "target": "bz-a3-1", "handle": handle,
            "arguments": ["logs", handle], "request_sha256": f"{index:064x}",
            "result": {"returncode": 0, "state": "completed", "handle": handle,
                       "stdout": "short result", "stderr": ""},
        })
    journal = {"schema": "profiling-skill/remote-journal/v1",
               "dispatches": [], "calls": calls}

    retained_ref = launcher.retain_remote_journal(
        store, "remote/pathological.json", journal)
    retained_path = store.root / retained_ref["path"]
    retained = json.loads(retained_path.read_text())

    assert len(retained_path.read_bytes()) < launcher.MAX_ARTIFACT
    assert retained["retention"]["mode"] == "representative-summary"
    assert retained["retention"]["call_count"] == 2_000
    assert len(retained["calls"]) == 4
    assert retained["calls"][0]["handle"] == calls[0]["handle"]
    assert retained["calls"][-1]["handle"] == calls[-1]["handle"]
    assert retained["source_canonical_sha256"] == launcher._canonical_sha256(journal)
    assert len(retained["retention"]["structural_metadata_sha256"]) == 64


def test_compaction_preserves_late_decisive_compile_after_runtime_setup_noise():
    setup = "\n".join(f"runtime setup diagnostic {index}" for index in range(8))
    original = setup + "\ntriton compile error: invalid layout\n" + ("trace\n" * 10_000)

    compact = launcher._compact_remote_text(original, 512)

    assert launcher.classify_failure({"stdout": compact, "stderr": ""}) == (
        "counted_failure", "compile", "compile")
    assert "triton compile error: invalid layout" in compact


def test_compaction_keeps_authoritative_last_remote_content_hash():
    historical = [f"{index:064x}" for index in range(100)]
    authoritative = "f" * 64
    original = "\n".join(
        f"REMOTE_CONTENT_SHA256={value}" for value in [*historical, authoritative]
    ) + "\n" + ("profile output\n" * 10_000)

    compact = launcher._compact_remote_text(original, 512)

    assert launcher._parse_remote(compact)["content_sha256"] == authoritative
    assert f"REMOTE_CONTENT_SHA256={authoritative}" in compact
    assert len(compact.encode()) <= 512


@pytest.mark.parametrize("diagnostic", [
    "selector chosen: foo\nevidence mismatch: exported row belongs to another kernel",
    "expected one exported kernel row\nselector details: foo",
])
def test_compaction_preserves_multiline_evidence_classification(diagnostic: str):
    original = diagnostic + "\n" + ("profile trace\n" * 10_000)
    original_evidence = {"stdout": original, "stderr": "", "returncode": 1}

    compact = launcher._compact_remote_text(original, 512)
    compact_evidence = {"stdout": compact, "stderr": "", "returncode": 1}

    assert launcher.classify_failure(original_evidence) == (
        "counted_failure", "evidence", "evidence")
    assert launcher.classify_failure(compact_evidence) == (
        "counted_failure", "evidence", "evidence")


def test_retained_compaction_preserves_cross_field_evidence_classification(tmp_path: Path):
    stdout = "expected one exported kernel row\n" + ("stdout trace\n" * 10_000)
    stderr = "selector details: foo\n" + ("stderr trace\n" * 10_000)
    handle = "remote:bz-a3-1:job:cross-field"
    result = {"returncode": 1, "state": "failed", "handle": handle,
              "stdout": stdout, "stderr": stderr}
    journal = {
        "schema": "profiling-skill/remote-journal/v1",
        "dispatches": [{
            "request_sha256": "d" * 64, "target": "bz-a3-1",
            "dispatch_key": "cross-field", "file_sha256": "e" * 64,
            "arguments": ["run", "bz-a3-1", "--file", "/workspace/job.sh"],
            "state": "failed", "handle": handle, "result": result,
        }],
        "calls": [],
    }
    store = launcher.ArtifactStore(tmp_path / "artifacts")

    retained_ref = launcher.retain_remote_journal(store, "remote/cross-field.json", journal)
    retained_path = store.root / retained_ref["path"]
    retained = json.loads(retained_path.read_text())
    compact_result = retained["dispatches"][0]["result"]

    assert launcher.classify_failure(result) == (
        "counted_failure", "evidence", "evidence")
    assert launcher.classify_failure(compact_result) == (
        "counted_failure", "evidence", "evidence")
    assert len(retained_path.read_bytes()) < launcher.MAX_ARTIFACT
    assert journal["dispatches"][0]["result"]["stdout"] == stdout
    assert journal["dispatches"][0]["result"]["stderr"] == stderr


def test_paired_a5_process_failure_ignores_prior_a3_remote_history_and_is_gate_compatible(
        tmp_path: Path, monkeypatch):
    instance, request = fixture(tmp_path / "launch")
    calls = 0
    prompts = []

    def run(command, **_kwargs):
        nonlocal calls
        calls += 1
        prompts.append(_kwargs.get("input", ""))
        if calls == 1:
            state_mount = Path(command[command.index("/experiment-state") - 1])
            journal = state_mount.parent / "remote-journal.json"
            handle = "remote:bz-a3-1:job:a3-completed"
            journal.write_text(json.dumps({
                "schema": "profiling-skill/remote-journal/v1",
                "dispatches": [{
                    "request_sha256": "7" * 64, "target": "bz-a3-1",
                    "arguments": ["run", "bz-a3-1", "--file", "/workspace/a3.sh"],
                    "state": "completed", "handle": handle,
                    "result": {"returncode": 0, "state": "completed", "stdout": "ok",
                               "stderr": "", "failure_type": None},
                }],
                "calls": [{
                    "operation": "logs", "target": "bz-a3-1", "arguments": ["logs", handle],
                    "handle": handle, "result": {"returncode": 1, "state": "failed",
                                                   "stdout": "old triton compile error",
                                                   "stderr": "", "failure_type": "compile"},
                }],
            }))
            answer = {"product": "a3", "target": "bz-a3-1", "handle": handle,
                      "evidence": "unused-a3.json", "reasoning": "first turn completed"}
            events = [
                {"type": "thread.started", "thread_id": "paired-thread"},
                {"type": "item.completed",
                 "item": {"type": "agent_message", "text": json.dumps(answer)}},
            ]
            return type("Result", (), {"returncode": 0,
                                        "stdout": "\n".join(map(json.dumps, events)),
                                        "stderr": ""})()
        return type("Result", (), {"returncode": 2, "stdout": "",
                                    "stderr": "codex resume process failed"})()

    monkeypatch.setattr(instance, "sandbox_preflight", lambda _base: None)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    output = tmp_path / "launch-output.json"
    artifact_root = tmp_path / "launcher-artifacts"
    result = instance.run(request, output, artifact_root)

    record = result["record"]
    assert result["status"] == "counted_failure"
    assert (record["product"], record["target"], record["handle"]) == (
        "a5", "bz-a5", None)
    assert record["failure_type"] == "launcher"
    failure_log = (artifact_root / record["log"]["path"]).read_text()
    assert "old triton compile error" not in failure_log
    assert "REMOTE_CONTENT_SHA256=<hex>" in prompts[0]
    assert "provenance containing exactly the assigned `product` and `target`" in prompts[0]
    assert "do not put a durable handle" in prompts[0]
    assert '"target": "bz-a3-1"' in prompts[0]
    retained = json.loads((artifact_root / result["remote_journal"]["path"]).read_text())
    assert retained["dispatches"][0]["handle"] == "remote:bz-a3-1:job:a3-completed"

    gate_path = tmp_path / "gate"
    gate_path.mkdir()
    manifest_path, records_path, gate_root, manifest, records = gate_tests.fixture(gate_path)
    record["manifest_sha256"] = digest(manifest_path)
    record["launcher"] = deepcopy(manifest["launcher"])
    record["model"] = deepcopy(manifest["model"])
    record["skill_sha256"] = manifest["skills"]["candidate"]
    for name in ("command", "log", "diagnostic", "receipt"):
        relative = Path(record[name]["path"])
        destination = gate_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_root / relative, destination)
    records["acquisition"][-1] = record
    records_path.write_text(json.dumps(records))

    report = gate_tests.gate.evaluate(manifest_path, records_path, gate_root)
    assert report["acquisition"] == {
        "terminal_sessions": 3, "successful_pairs": 2,
        "score": {"passed": 2, "total": 3},
    }
    assert report["acceptance"]["passed"] is False


def test_finalize_review_produces_gate_compatible_manual_decisions(tmp_path: Path):
    store = launcher.ArtifactStore(tmp_path / "artifacts")
    reasoning = store.text("reasoning/s-a3.txt", "Because the capacity metric is below its ceiling.\n")
    draft = {"kind": "acquisition", "session_id": "s", "classification": "success",
             "outcomes": [{"product": "a3", "reasoning": reasoning, "manual_review": None},
                          {"product": "a5", "reasoning": reasoning, "manual_review": None}]}
    reviewer = {"identity": "reviewer/v1", "config_sha256": "5" * 64}
    reviewed = launcher.finalize_reviews(
        draft, {"a3": {"passed": True, "notes": "Correctly distinguishes activity."},
                "a5": {"passed": False, "notes": "Overclaims saturation."}},
        reviewer, store,
    )
    assert reviewed["outcomes"][0]["manual_review"]["passed"] is True
    assert reviewed["outcomes"][1]["manual_review"]["passed"] is False
    for outcome in reviewed["outcomes"]:
        assert outcome["manual_review"]["reviewer"] == reviewer
        assert (store.root / outcome["manual_review"]["notes"]["path"]).is_file()


def test_cli_finalizes_review_pending_record_atomically(tmp_path: Path):
    gate_path = tmp_path / "gate"
    gate_path.mkdir()
    manifest_path, records_path, artifacts, manifest, records = gate_tests.fixture(gate_path)
    draft = deepcopy(records["acquisition"][0])
    for outcome in draft["outcomes"]:
        outcome["manual_review"] = None
    reviewer = manifest["reviewer"]
    pending = {"status": "review_pending", "review": {"reviewer": reviewer},
               "draft_record": draft}
    launch_output = tmp_path / "launch-output.json"
    decisions = tmp_path / "decisions.json"
    output = tmp_path / "final-record.json"
    launch_output.write_text(json.dumps(pending))
    decisions.write_text(json.dumps(
        {"a3": {"passed": True, "notes": "Correct A3 interpretation."},
         "a5": {"passed": True, "notes": "Correct A5 interpretation."}}))

    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/profile_behavioral_launcher.py"), "finalize",
         "--launch-output", str(launch_output), "--decisions", str(decisions),
         "--artifact-root", str(artifacts), "--output", str(output)],
        text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr
    record = json.loads(output.read_text())
    assert record["outcomes"][0]["manual_review"]["passed"] is True
    assert record["outcomes"][0]["manual_review"]["reviewer"] == reviewer
    records["acquisition"][0] = record
    records_path.write_text(json.dumps(records))
    assert gate_tests.gate.evaluate(manifest_path, records_path, artifacts)["acceptance"]["passed"]


def test_cli_creates_reproducible_valid_operator_request(tmp_path: Path):
    destination = tmp_path / "operator-example"
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/profile_behavioral_launcher.py"),
         "request-example", "--output-dir", str(destination)],
        text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr
    request = json.loads((destination / "request.json").read_text())
    instance, _ = fixture(tmp_path / "launcher")
    assert instance.validate_request(request) == request
    assert len(request["units"]) == 4


def test_live_adapter_writes_review_pending_draft_after_four_resumed_turns(
        tmp_path: Path, monkeypatch):
    instance, request = fixture(tmp_path, "interpretation")
    commands = []
    prompts = []
    count = 0

    def run(command, **kwargs):
        nonlocal count
        commands.append(command)
        if len(command) >= 3 and command[-3] == "sh" and command[-2] == "-ceu":
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        prompts.append(kwargs.get("input", ""))
        count += 1
        case = f"case-{count}"
        answer = {"case_id": case, "conclusions": [f"fact-{count}"],
                  "saturation_claims": [], "reasoning": f"reason {count}"}
        events = []
        if count == 1:
            events.append({"type": "thread.started", "thread_id": "thread-opaque"})
        events.append({"type": "item.completed",
                       "item": {"type": "agent_message", "text": json.dumps(answer)}})
        return type("Result", (), {"returncode": 0,
                                    "stdout": "\n".join(map(json.dumps, events)),
                                    "stderr": ""})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    output = tmp_path / "retained" / "output.json"
    result = instance.run(request, output, tmp_path / "retained" / "artifacts")

    assert result["status"] == "review_pending"
    assert output.is_file() and json.loads(output.read_text()) == result
    assert result["draft_record"]["session_id"] == "thread-opaque"
    assert len(result["draft_record"]["answers"]) == 4
    codex = [command for command in commands if "/runtime/node/bin/codex" in command]
    assert len(codex) == 4
    assert "resume" not in codex[0]
    assert all("resume" in command and "thread-opaque" in command for command in codex[1:])
    assert all("candidate" not in prompt and "current" not in prompt for prompt in prompts)
    assert all("REMOTE_CONTENT_SHA256" not in prompt for prompt in prompts)


def test_interpretation_process_failure_is_end_to_end_gate_compatible(
        tmp_path: Path, monkeypatch):
    instance, request = fixture(tmp_path / "launcher", "interpretation")

    def run(command, **_kwargs):
        if len(command) >= 3 and command[-3] == "sh" and command[-2] == "-ceu":
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("Result", (), {"returncode": 9, "stdout": "",
                                    "stderr": "Codex process broke"})()

    monkeypatch.setattr(launcher.subprocess, "run", run)
    launch_artifacts = tmp_path / "launcher-artifacts"
    launched = instance.run(request, tmp_path / "launcher-output.json", launch_artifacts)
    record = launched["record"]
    assert record["product"] is record["target"] is record["handle"] is None
    assert (launch_artifacts / launched["remote_journal"]["path"]).is_file()

    gate_path = tmp_path / "gate"
    gate_path.mkdir()
    manifest_path, records_path, gate_root, manifest, records = \
        gate_tests.fixture(gate_path)
    for label in ("command", "log", "diagnostic", "receipt"):
        reference = record[label]
        destination = gate_root / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(launch_artifacts / reference["path"], destination)
    record["manifest_sha256"] = gate_tests.sha(manifest_path)
    record["launcher"] = manifest["launcher"]
    record["model"] = manifest["model"]
    record["skill_sha256"] = manifest["skills"]["candidate"]
    records["interpretation"][3] = record
    records_path.write_text(json.dumps(records))

    report = gate_tests.gate.evaluate(manifest_path, records_path, gate_root)
    assert report["interpretation"]["candidate"]["total"] == 12
    assert report["acceptance"]["passed"] is False
