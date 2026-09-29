from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "production_launcher", ROOT / "scripts" / "production_launcher.py"
)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def launcher_fixture(tmp_path: Path, **kwargs):
    runtime = tmp_path / "node"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(runtime / "bin" / "codex", "#!/bin/sh\nexit 0\n")
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "real-auth"
    auth.mkdir()
    (auth / "auth.json").write_text("secret-token")
    bwrap = executable(tmp_path / "bwrap", "#!/bin/sh\nexit 0\n")
    return launcher.ProductionLauncher(
        ["controller"], codex=str(codex), bwrap=str(bwrap), auth_home=auth, **kwargs
    )


def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "cell"
    (root / "workspace" / ".agents" / "skills" / "ascend-profiling").mkdir(parents=True)
    (root / "PROMPT.md").write_text("identical prompt\n")
    return root


def production_cell(cell_id: str = "gdn-project-only") -> dict:
    return {
        "cell_id": cell_id, "benchmark": "gdn", "device": 0,
        "rounds": 3, "request_budget": 18,
        "development_cases": [40, 49, 47, 46, 45],
        "all_cases": list(range(50)),
    }


def identified(document: dict, cell: dict, operation: str) -> dict:
    return {"operation": operation, "cell": cell["cell_id"],
            "benchmark": cell["benchmark"], "device": cell["device"], **document}


def test_host_calibration_uses_controller_and_retains_evidence(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    cell = production_cell()
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        document = identified({
            "status": "ok", "latency_us": 17.5, "handles": ["gz-a3:cal"],
            "selector": "streaming_matmul_add_kernel_mix_aic",
        }, cell, "calibrate")
        return type("Result", (), {"returncode": 0, "stdout": json.dumps(document),
                                    "stderr": ""})()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    evidence = instance.calibrate(root, cell, "before", 2, "wave-2-retry-1")
    assert observed["command"][-7:] == [
        "calibrate", "--phase", "before", "--wave", "2",
        "--attempt-id", "wave-2-retry-1",
    ]
    assert observed["cwd"] == root / "workspace"
    assert evidence["status"] == "complete"
    assert evidence["result"]["handles"] == ["gz-a3:cal"]
    assert evidence["timestamp"].endswith("+00:00")
    assert Path(evidence["evidence_path"]).is_file()


def test_bwrap_argv_mounts_only_workspace_minimal_state_and_readonly_auth(tmp_path: Path):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    argv = instance._base_command(root, root / "attempt")
    joined = "\0".join(map(str, argv))
    assert "--unshare-all" in argv and "--share-net" in argv
    assert f"{root.resolve()}/workspace\0/workspace" in joined
    assert f"{instance.auth_home}/auth.json\0/codex-home/auth.json" in joined
    assert "--ro-bind\0" + str(instance.auth_home / "auth.json") in joined
    assert str(instance.auth_home / "skills") not in joined
    # The request-gate file is mounted alone; its source repository is not mounted.
    assert f"{ROOT}\0" not in joined
    assert argv[argv.index("--setenv") + 1:argv.index("--setenv") + 3] == ["HOME", "/home/agent"]


def test_bwrap_mounts_only_systemd_resolver_target_from_host_run(tmp_path: Path):
    fake_root = tmp_path / "host"
    resolver = fake_root / "run" / "systemd" / "resolve" / "stub-resolv.conf"
    resolver.parent.mkdir(parents=True)
    resolver.write_text("nameserver 127.0.0.53\n")
    etc = fake_root / "etc"
    etc.mkdir()
    resolv_conf = etc / "resolv.conf"
    resolv_conf.symlink_to("../run/systemd/resolve/stub-resolv.conf")
    instance = launcher_fixture(tmp_path / "fixture", resolv_conf=resolv_conf)

    mounts = instance._resolver_mounts()

    assert mounts == [
        "--dir", "/run", "--dir", "/run/systemd", "--dir", "/run/systemd/resolve",
        "--ro-bind", str(resolver), "/run/systemd/resolve/stub-resolv.conf",
    ]
    assert "--ro-bind" in mounts
    assert "/run" not in mounts[mounts.index("--ro-bind") + 1:]


def test_missing_systemd_resolver_target_fails_before_agent_start(tmp_path: Path):
    fake_root = tmp_path / "host"
    etc = fake_root / "etc"
    etc.mkdir(parents=True)
    resolv_conf = etc / "resolv.conf"
    resolv_conf.symlink_to("../run/systemd/resolve/stub-resolv.conf")
    instance = launcher_fixture(tmp_path / "fixture", resolv_conf=resolv_conf)

    with pytest.raises(launcher.LaunchError, match="resolver target is unavailable"):
        instance._base_command(sandbox(tmp_path / "cell"), tmp_path / "attempt")


def test_preflight_checks_auth_local_skills_and_forbidden_host_paths(tmp_path: Path, monkeypatch):
    forbidden = tmp_path / "orchestration"
    forbidden.mkdir()
    instance = launcher_fixture(tmp_path, forbidden_paths=[forbidden, Path.home() / ".codex" / "skills"])
    root = sandbox(tmp_path)
    base = instance._base_command(root, root / "attempt")
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    state = root / "controller-state"
    state.mkdir()
    instance._preflight(base, state)
    shell = observed["argv"][-1]
    assert "test -r /codex-home/auth.json" in shell
    assert "test -d /workspace/.agents/skills" in shell
    assert str(forbidden.resolve()) in shell
    assert "test ! -e /codex-home/skills" in shell


def test_preflight_failure_is_clear(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    base = instance._base_command(root, root / "attempt")
    monkeypatch.setattr(
        launcher.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "outside path visible"),
    )
    state = root / "controller-state"
    state.mkdir()
    with pytest.raises(launcher.LaunchError, match="outer isolation preflight failed"):
        instance._preflight(base, state)


def test_budget_controller_exposes_free_contract_and_enforces_limit(tmp_path: Path):
    backend = executable(
        tmp_path / "backend",
        "#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n",
    )
    socket_path = tmp_path / "state" / "controller.sock"
    workspace = tmp_path / "cell" / "workspace"
    workspace.mkdir(parents=True)
    with launcher.BudgetController(socket_path, [str(backend)], 2, workspace,
                                   {"cell_id": "gdn-project-only", "device": 0}) as controller:
        def request(arguments):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
                return json.loads(client.makefile("rb").readline())

        help_result = request(["help"])
        assert help_result["exit_code"] == 0
        assert json.loads(help_result["stdout"]) == launcher.controller_help_payload()
        assert "--scope development" in json.loads(help_result["stdout"])["usage"]
        contract = launcher.controller_contract(2)
        assert contract["help"] == json.loads(help_result["stdout"])
        assert contract["request_budget"] == 2
        assert len(contract["help_sha256"]) == 64
        invalid = request(["check", "--config", "secret.json"])
        assert invalid["exit_code"] == 4
        assert json.loads(invalid["stdout"])["status"] == "config_error"
        assert "--config" in json.loads(invalid["stdout"])["diagnostics"]
        assert json.loads(request(["budget"])["stdout"])["remaining"] == 2
        assert request(["check", "--scope", "development", "--round", "1"])["exit_code"] == 0
        assert request(["profile", "--repeats", "3", "--round", "1"])["exit_code"] == 0
        exhausted = request(["check", "--scope", "development", "--round", "2"])
        assert exhausted["exit_code"] == 75
        assert "budget exhausted" in exhausted["stderr"]
        assert controller.used == 2
        assert controller.invalid_requests == 1
        assert controller.over_budget_requests == 1
    assert not socket_path.exists()


def test_controller_client_forwards_exit_and_streams(tmp_path: Path, capsys):
    backend = executable(tmp_path / "backend", "#!/bin/sh\necho output\necho diagnostic >&2\nexit 7\n")
    socket_path = tmp_path / "controller.sock"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with launcher.BudgetController(socket_path, [str(backend)], 1, workspace,
                                   {"cell_id": "bsa-cannbot", "device": 1}):
        assert launcher.controller_client(
            socket_path, ["check", "--scope", "development", "--round", "1"]
        ) == 7
    captured = capsys.readouterr()
    assert captured.out == "output\n"
    assert captured.err == "diagnostic\n"


def test_agent_command_matrix_is_exact_and_invalid_combinations_are_free(tmp_path: Path):
    backend = executable(tmp_path / "backend", "#!/bin/sh\nexit 0\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    socket_path = tmp_path / "controller.sock"
    valid = [
        ["check", "--scope", "development", "--round", str(round_number)]
        for round_number in (1, 2)
    ] + [["check", "--scope", "full", "--round", "3"]] + [
        ["profile", "--repeats", "3", "--round", str(round_number)]
        for round_number in (1, 2, 3)
    ]
    invalid = [
        ["check", "--scope", scope, "--round", str(round_number)]
        for scope in ("development", "full") for round_number in (1, 2, 3)
        if [scope, round_number] not in [["development", 1], ["development", 2], ["full", 3]]
    ] + [
        ["profile", "--repeats", repeats, "--round", round_number]
        for repeats, round_number in (("2", "1"), ("3", "0"), ("3", "4"))
    ]
    with launcher.BudgetController(
        socket_path, [str(backend)], len(valid), workspace,
        {"cell_id": "gdn", "device": 0},
    ) as controller:
        def request(arguments):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(json.dumps({"arguments": arguments}).encode() + b"\n")
                return json.loads(client.makefile("rb").readline())

        for arguments in invalid:
            response = request(arguments)
            assert response["exit_code"] == 4
            assert json.loads(response["stdout"])["status"] == "config_error"
        assert controller.used == 0
        for arguments in valid:
            assert request(arguments)["exit_code"] == 0
        assert controller.used == len(valid)
        assert controller.invalid_requests == len(invalid)


def test_launcher_rejects_frozen_controller_or_model_drift(tmp_path: Path):
    with pytest.raises(launcher.LaunchError, match="controller contract"):
        launcher_fixture(tmp_path / "interface", agent_interface={
            **launcher.controller_contract(18), "help_sha256": "0" * 64,
        })
    with pytest.raises(launcher.LaunchError, match="model configuration"):
        launcher_fixture(tmp_path / "model", agent_model={
            "name": "gpt-5.6-sol", "reasoning_effort": "medium",
        })


def test_three_round_persistent_session_uses_fixed_model_and_prompt(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    cell = production_cell()
    calls = []
    budget_limits = []

    class FakeBudget:
        used = 18
        rejected = 0
        invalid_requests = 0
        over_budget_requests = 0
        limit = 18
        command = ("controller",)
        def __init__(self, *args, **kwargs): budget_limits.append(args[2])
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-2:] and "sh" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-3:] == ["check", "--scope", "full"]:
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"status": "ok", "passed": True,
                                     "cases": list(range(50)), "handles": ["gz-a3:check"],
                                     "operation": "check", "cell": cell["cell_id"],
                                     "benchmark": "gdn", "device": 0}), ""
            )
        if argv[-5:] == ["profile", "--repeats", "3", "--round", "3"]:
            profile = {"status": "ok", "repeats": 3,
                       "handles": [f"gz-a3:p{i}" for i in range(15)],
                       "cases": [{"case": i, "samples_us": [1.0, 1.1, 1.2]}
                                 for i in cell["development_cases"]],
                       "operation": "profile", "cell": cell["cell_id"],
                       "benchmark": "gdn", "device": 0}
            return subprocess.CompletedProcess(argv, 0, json.dumps(profile), "")
        output = json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(launcher, "BudgetController", FakeBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, cell)
    codex_calls = [call for call in calls if "/runtime/node/bin/codex" in call[0]]
    assert len(codex_calls) == 3
    assert codex_calls[0][1]["input"] == "identical prompt\n"
    assert all("gpt-5.6-sol" in call[0] for call in codex_calls)
    assert all('model_reasoning_effort="low"' in call[0] for call in codex_calls)
    assert all("thread-123" in call[0] for call in codex_calls[1:])
    assert all("--ignore-rules" not in call[0] for call in codex_calls)
    assert result["rounds_completed"] == 3
    assert result["agent_controller_requests"] == 18
    assert result["controller_requests"] == 20
    assert result["status"] == "complete"
    assert budget_limits == [18]
    assert result["terminal_evidence"]["check"]["result"]["handles"] == ["gz-a3:check"]
    terminal_calls = [call for call in calls if call[0][0] == "controller"]
    assert terminal_calls[1][1]["timeout"] < terminal_calls[0][1]["timeout"]
    assert terminal_calls[1][0][-5:] == ["profile", "--repeats", "3", "--round", "3"]


def test_controller_binds_cells_for_allowed_agent_commands(tmp_path: Path):
    backend = executable(
        tmp_path / "backend",
        "#!/usr/bin/python3\nimport json,os,sys\nprint(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd()}))\n",
    )
    for cell_id, device in (("gdn-project-only", 0), ("bsa-cannbot", 1)):
        workspace = tmp_path / cell_id / "workspace"
        workspace.mkdir(parents=True)
        socket_path = tmp_path / cell_id / "controller.sock"
        command = [str(backend), "--cell", "{cell_id}", "--device", "{device}"]
        with launcher.BudgetController(socket_path, command, 2, workspace,
                                       {"cell_id": cell_id, "device": device}):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                request = {"arguments": ["profile", "--repeats", "3", "--round", "1"]}
                client.sendall(json.dumps(request).encode() + b"\n")
                response = json.loads(client.makefile("rb").readline())
        observed = json.loads(response["stdout"])
        assert observed["argv"] == ["--cell", cell_id, "--device", str(device),
                                     "profile", "--repeats", "3", "--round", "1"]
        assert observed["cwd"] == str(workspace)


def test_controller_rejects_nonworkspace_absolute_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    controller = launcher.BudgetController(tmp_path / "sock", ["true"], 1, workspace,
                                           {"cell_id": "gdn", "device": 0})
    with pytest.raises(ValueError, match="outside /workspace"):
        controller.map_arguments(["--candidate=/tmp/leak.py"])


def test_launch_rejects_noncanonical_limits_before_codex(tmp_path: Path):
    instance = launcher_fixture(tmp_path)
    with pytest.raises(launcher.LaunchError, match="three rounds"):
        instance.launch(sandbox(tmp_path), {"rounds": 2, "request_budget": 18})


def test_dry_run_builds_plan_after_real_isolation_and_dns_preflight(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path, dry_run=True)
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(sandbox(tmp_path), {"rounds": 3, "request_budget": 18})
    assert result["dry_run"] is True
    assert result["rounds"] == 3 and result["request_budget"] == 18
    assert "secret-token" not in json.dumps(result)
    assert len(calls) == 1
    assert "api.openai.com" in calls[0][-1]


def test_dry_run_does_not_create_attempt_state(tmp_path: Path):
    instance = launcher_fixture(tmp_path, dry_run=True)
    root = sandbox(tmp_path)
    first = instance.launch(root, {"rounds": 3, "request_budget": 18})
    second = instance.launch(root, {"rounds": 3, "request_budget": 18})
    assert first["attempt_id"] != second["attempt_id"]
    assert not (root / ".launcher-attempts").exists()


def test_nonzero_codex_turn_is_retained_as_infrastructure_error(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)

    class FakeBudget:
        used = 1
        rejected = 0
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    calls = 0
    def fake_run(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:  # isolation preflight
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 17, "partial output", "transport lost")

    monkeypatch.setattr(launcher, "BudgetController", FakeBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, {"cell_id": "gdn-project-only", "device": 0,
                                    "rounds": 3, "request_budget": 18})
    assert result["status"] == "infrastructure_error"
    assert result["rounds_completed"] == 0
    assert result["turns"] == [{"round": 1, "exit_code": 17,
                                "stdout": "partial output", "stderr": "transport lost"}]


@pytest.mark.parametrize(
    ("check_document", "profile_document", "expected"),
    [
        ({"status": "candidate_error", "diagnostics": "wrong answer", "handles": ["gz-a3:c"]},
         {"status": "ok", "repeats": 3, "handles": [f"gz-a3:p{i}" for i in range(15)],
          "cases": [{"case": i, "samples_us": [1, 2, 3]} for i in range(5)]}, "candidate_error"),
        ({"status": "ok", "passed": True, "cases": list(range(50)), "handles": ["gz-a3:c"]},
         {"status": "ok", "repeats": 3, "cases": []}, "infrastructure_error"),
        ({"status": "infrastructure_error", "diagnostics": "device busy",
          "handles": ["gz-a3:observe-this"]},
         {"status": "ok", "repeats": 3, "handles": [f"gz-a3:p{i}" for i in range(15)],
          "cases": [{"case": i, "samples_us": [1, 2, 3]} for i in range(5)]}, "infrastructure_error"),
    ],
)
def test_noop_agent_requires_host_terminal_gates(
    tmp_path: Path, monkeypatch, check_document, profile_document, expected,
):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)
    cell = production_cell("gdn")
    check_document.setdefault("cases", cell["all_cases"])
    check_document = identified(check_document, cell, "check")
    profile_document = identified(profile_document, cell, "profile")

    class FakeBudget:
        used = 0
        rejected = 0
        command = ("controller",)
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def fake_run(argv, **kwargs):
        if "sh" in argv[-3:]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-3:] == ["check", "--scope", "full"]:
            return subprocess.CompletedProcess(argv, 2, json.dumps(check_document), "")
        if argv[-5:] == ["profile", "--repeats", "3", "--round", "3"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(profile_document), "")
        event = json.dumps({"type": "thread.started", "thread_id": "noop-thread"}) + "\n"
        return subprocess.CompletedProcess(argv, 0, event, "")

    monkeypatch.setattr(launcher, "BudgetController", FakeBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, cell)
    assert result["rounds_completed"] == 3
    assert result["status"] == expected
    assert (root / ".launcher-attempts" / result["attempt_id"]
            / "terminal-check.json").is_file()
    if check_document.get("handles"):
        assert result["terminal_evidence"]["check"]["result"]["handles"] == check_document["handles"]
    if check_document["status"] != "ok":
        assert "profile" not in result["terminal_evidence"]
        assert result["controller_requests"] == 1
    else:
        assert result["controller_requests"] == 2


def test_budget_exhaustion_is_recorded_but_terminal_gates_still_decide(tmp_path: Path, monkeypatch):
    instance = launcher_fixture(tmp_path)
    root = sandbox(tmp_path)

    class ExhaustedBudget:
        used = 18
        rejected = 1
        invalid_requests = 0
        over_budget_requests = 1
        limit = 18
        command = ("controller",)
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    codex_calls = 0

    def fake_run(argv, **kwargs):
        nonlocal codex_calls
        if "sh" in argv[-3:]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[-3:] == ["check", "--scope", "full"]:
            document = identified({"status": "ok", "passed": True,
                                   "cases": list(range(50)), "handles": ["gz-a3:c"]},
                                  production_cell("gdn"), "check")
            return subprocess.CompletedProcess(argv, 0, json.dumps(document), "")
        if argv[-5:] == ["profile", "--repeats", "3", "--round", "3"]:
            document = identified({
                "status": "ok", "repeats": 3,
                "handles": [f"gz-a3:p{i}" for i in range(15)],
                "cases": [{"case": case, "samples_us": [1, 2, 3]}
                          for case in [40, 49, 47, 46, 45]],
            }, production_cell("gdn"), "profile")
            return subprocess.CompletedProcess(argv, 0, json.dumps(document), "")
        codex_calls += 1
        event = json.dumps({"type": "thread.started", "thread_id": "budget-thread"}) + "\n"
        if codex_calls == 2:
            return subprocess.CompletedProcess(argv, 17, event, "agent process failed")
        return subprocess.CompletedProcess(argv, 0, event, "")

    monkeypatch.setattr(launcher, "BudgetController", ExhaustedBudget)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = instance.launch(root, {"cell_id": "gdn", "device": 0,
                                    "rounds": 3, "request_budget": 18,
                                    "benchmark": "gdn", "development_cases": [40, 49, 47, 46, 45],
                                    "all_cases": list(range(50))})
    assert result["status"] == "complete"
    assert result["budget_exhausted"] is True
    assert result["controller_usage"]["over_budget_requests"] == 1
    assert result["controller_requests"] == 20
    assert set(result["terminal_evidence"]) == {"check", "profile"}
    assert result["rounds_completed"] == 1
    assert result["exit_code"] == 17
    assert result["turns"][-1]["stderr"] == "agent process failed"


def test_terminal_gate_requires_exact_identity_and_case_sequences():
    cell = production_cell()
    check = identified({"status": "ok", "passed": True, "handles": ["gz-a3:c"],
                        "cases": cell["all_cases"]}, cell, "check")
    assert launcher.ProductionLauncher._gate_status(check, cell, "check") == ("complete", "")
    wrong_identity = {**check, "device": 1}
    assert launcher.ProductionLauncher._gate_status(wrong_identity, cell, "check")[0] == "infrastructure_error"
    reordered = {**check, "cases": list(reversed(cell["all_cases"]))}
    assert launcher.ProductionLauncher._gate_status(reordered, cell, "check")[0] == "infrastructure_error"

    profile = identified({
        "status": "ok", "repeats": 3,
        "handles": [f"gz-a3:p{i}" for i in range(15)],
        "cases": [{"case": case, "samples_us": [1, 2, 3]}
                  for case in cell["development_cases"]],
    }, cell, "profile")
    assert launcher.ProductionLauncher._gate_status(profile, cell, "profile") == ("complete", "")
    duplicate = {**profile, "cases": [*profile["cases"][:-1], profile["cases"][0]]}
    assert launcher.ProductionLauncher._gate_status(duplicate, cell, "profile")[0] == "infrastructure_error"
    malformed_samples = {
        **profile,
        "cases": [{**row, "samples_us": 1.0} if index == 0 else row
                  for index, row in enumerate(profile["cases"])],
    }
    assert launcher.ProductionLauncher._gate_status(
        malformed_samples, cell, "profile"
    )[0] == "infrastructure_error"
    duplicate_handles = {**profile, "handles": ["gz-a3:same"] * 15}
    assert launcher.ProductionLauncher._gate_status(
        duplicate_handles, cell, "profile"
    )[0] == "infrastructure_error"


def test_matmul_terminal_profile_accepts_three_cases_and_nine_unique_handles():
    cell = {
        "cell_id": "matmul-project-guarded", "benchmark": "matmul", "device": 0,
        "development_cases": [7, 8, 9], "all_cases": list(range(10)),
    }
    profile = identified({
        "status": "ok", "repeats": 3,
        "handles": [f"gz-a3:matmul-{index}" for index in range(9)],
        "cases": [{"case": case, "samples_us": [1, 2, 3]}
                  for case in cell["development_cases"]],
    }, cell, "profile")
    assert launcher.ProductionLauncher._gate_status(profile, cell, "profile") == (
        "complete", ""
    )
    assert launcher.ProductionLauncher._gate_status(
        {**profile, "handles": profile["handles"] + ["gz-a3:extra"]}, cell, "profile"
    )[0] == "infrastructure_error"


def test_missing_auth_or_bwrap_fails_without_reading_credentials(tmp_path: Path):
    runtime = tmp_path / "node"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(runtime / "bin" / "codex", "#!/bin/sh\nexit 0\n")
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"
    auth.mkdir()
    with pytest.raises(launcher.LaunchError, match="Bubblewrap"):
        launcher.ProductionLauncher(["ctl"], codex=str(codex), bwrap=str(tmp_path / "missing"), auth_home=auth)
    bwrap = executable(tmp_path / "bwrap", "#!/bin/sh\nexit 0\n")
    with pytest.raises(launcher.LaunchError, match="authenticated"):
        launcher.ProductionLauncher(["ctl"], codex=str(codex), bwrap=str(bwrap), auth_home=auth)


def test_real_bwrap_with_functional_fake_codex_runs_persistent_rounds(tmp_path: Path):
    bwrap = __import__("shutil").which("bwrap")
    if not bwrap:
        pytest.skip("Bubblewrap is unavailable")
    runtime = tmp_path / "runtime"
    executable(runtime / "bin" / "node", "#!/bin/sh\nexit 0\n")
    codex = executable(
        runtime / "bin" / "codex",
        """#!/bin/sh
test "$CODEX_HOME" = /codex-home || exit 21
test -d /workspace/.agents/skills/ascend-profiling || exit 22
test ! -e /codex-home/skills || exit 23
printf '%s\n' '{"type":"thread.started","thread_id":"fake-thread"}'
""",
    )
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}\n")
    root = sandbox(tmp_path)
    instance = launcher.ProductionLauncher(
        ["/bin/true"], codex=str(codex), bwrap=bwrap, auth_home=auth,
        forbidden_paths=[ROOT, Path.home() / ".codex" / "skills"],
    )
    try:
        result = instance.launch(root, {"cell_id": "fake-project-only", "device": 0,
                                        "rounds": 3, "request_budget": 18})
    except launcher.LaunchError as error:
        if "outer isolation preflight failed" in str(error) and "Operation not permitted" in str(error):
            pytest.skip("user namespaces are disabled")
        raise
    assert result["exit_code"] == 0
    assert result["session_id"] == "fake-thread"
    assert result["rounds_completed"] == 3
