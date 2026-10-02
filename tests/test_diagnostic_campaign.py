from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from scripts import campaign as production_campaign


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_campaign", ROOT / "scripts" / "diagnostic_campaign.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(diagnostic)


def manifest(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("write the kernel\n")
    model = {"name": "test-model", "reasoning_effort": "low"}
    treatments = {name: list(diagnostic.TREATMENT_SKILLS[name])
                  for name in diagnostic.TREATMENTS}
    return {
        "prompt": str(prompt),
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "model": model,
        "model_sha256": hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "treatments": {
            name: {"skills": skills, "skill_sha256": {skill: f"hash-{skill}" for skill in skills}}
            for name, skills in treatments.items()
        },
    }


class RecordingLauncher:
    def __init__(self, outcomes: dict[tuple[int, str, int], str] | None = None):
        self.requests = []
        self.outcomes = outcomes or {}
        self.lock = threading.Lock()

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        with self.lock:
            self.requests.append((request, timeout_seconds))
        status = self.outcomes.get(
            (request["wave"], request["treatment"], request["attempt"]), "ok"
        )
        if status == "ok":
            workspace = Path(request["workspace"])
            (workspace / "candidate.py").write_text("def kernel(): pass\n")
            (workspace / "candidate.manifest.json").write_text('{"kernel":"kernel"}\n')
        return {
            "status": status,
            "controller_usage": {"billed": 1, "calls": [
                {"arguments": ["check", "--scope", "development", "--round", "1"]}
            ]},
            "milestones": [{"name": "agent-finished", "at": 1}],
        }


class RecordingTerminal:
    def __init__(self, status: str = "ok"):
        self.requests = []
        self.status = status

    def check(self, request: dict, timeout_seconds: int) -> dict:
        self.requests.append((request, timeout_seconds))
        return {"status": self.status, "passed": self.status == "ok",
                "diagnostics": "complete traceback"}


def test_runs_four_sequential_waves_of_three_one_shot_cells(tmp_path: Path):
    launcher, terminal = RecordingLauncher(), RecordingTerminal()
    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", launcher, terminal,
        agent_timeout=360, cell_timeout=600,
    ).run()

    assert result["status"] == "complete"
    assert [wave["wave"] for wave in result["waves"]] == [1, 2, 3, 4]
    assert all([cell["treatment"] for cell in wave["cells"]]
               == list(diagnostic.TREATMENTS) for wave in result["waves"])
    assert len(launcher.requests) == len(terminal.requests) == 12
    assert all(timeout == 360 for _, timeout in launcher.requests)
    assert all(request["controller_contract"] == {
        "billed_limit": 1,
        "command": ["check", "--scope", "development", "--round", "1"],
    } for request, _ in launcher.requests)
    assert all(request["cases"] == list(range(7)) for request, _ in terminal.requests)
    assert {cell["outcome"] for wave in result["waves"] for cell in wave["cells"]} == {"success"}
    assert all(cell["candidate_sha256"] for wave in result["waves"] for cell in wave["cells"])
    assert all(cell["skill_sha256"] for wave in result["waves"] for cell in wave["cells"])


def test_cells_receive_identical_prompt_and_only_declared_treatment_skills(tmp_path: Path):
    config = manifest(tmp_path)
    launcher = RecordingLauncher()
    diagnostic.DiagnosticCampaign(config, tmp_path / "run", launcher,
                                  RecordingTerminal()).run()

    assert {request["prompt_sha256"] for request, _ in launcher.requests} == {
        config["prompt_sha256"]
    }
    for request, _ in launcher.requests:
        assert request["protocol_version"] == 2
        assert diagnostic.request_prompt_bytes(request) == Path(config["prompt"]).read_bytes()
        declared = config["treatments"][request["treatment"]]
        assert request["skills"] == declared["skills"]
        assert request["skill_sha256"] == declared["skill_sha256"]
    cannbot = next(request for request, _ in launcher.requests
                   if request["treatment"] == "cannbot")
    guarded = next(request for request, _ in launcher.requests
                   if request["treatment"] == "project-guarded")
    assert "ascend-profiling" not in cannbot["skills"]
    assert "triton-op-coding" not in guarded["skills"]


def test_infrastructure_is_retried_once_but_counted_failure_is_not(tmp_path: Path):
    outcomes = {
        (1, "cannbot", 1): "device_or_runtime_infra",
        (1, "cannbot", 2): "device_or_runtime_infra",
    }
    launcher = RecordingLauncher(outcomes)

    class MissingSubmission(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            result = super().launch(request, timeout_seconds)
            if request["wave"] == 1 and request["treatment"] == "project-cannbot":
                Path(request["workspace"], "candidate.py").unlink()
                Path(request["workspace"], "candidate.manifest.json").unlink()
            return result

    launcher = MissingSubmission(outcomes)
    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", launcher, RecordingTerminal()
    ).run()
    wave = result["waves"][0]
    infra = wave["cells"][0]
    missing = wave["cells"][1]
    assert infra["category"] == "infrastructure"
    assert infra["retry"]["attempt"] == 2
    assert result["reschedule"] == ["wave-1-cannbot"]
    assert result["status"] == "reschedule_pending"
    assert missing["outcome"] == "no_submission" and "retry" not in missing
    assert len([request for request, _ in launcher.requests
                if request["wave"] == 1 and request["treatment"] == "project-cannbot"]) == 1


@pytest.mark.parametrize(
    ("agent", "terminal", "files", "expected", "category"),
    [
        ({"status": "ok", "controller_usage": {"billed": 0}}, None, False,
         "protocol_error", "counted"),
        ({"status": "ok", "controller_usage": {"billed": 1, "calls": [{"arguments":
          ["check", "--scope", "development", "--round", "1"]}]}}, None, False,
         "no_submission", "counted"),
        ({"status": "timeout"}, None, False, "agent_timeout", "observed"),
        ({"status": "model_service_error"}, None, False,
         "model_service_error", "infrastructure"),
        ({"status": "protocol_error"}, None, False,
         "protocol_error", "counted"),
        ({"status": "ok", "controller_usage": {"billed": 1, "calls": [{"arguments":
          ["check", "--scope", "development", "--round", "1"]}]}},
         {"status": "compile_error"}, True, "compile_error", "counted"),
        ({"status": "ok", "controller_usage": {"billed": 1, "calls": [{"arguments":
          ["check", "--scope", "development", "--round", "1"]}]}},
         {"status": "timeout"}, True, "candidate_timeout", "counted"),
        ({"status": "ok", "controller_usage": {"billed": 1, "calls": [{"arguments":
          ["check", "--scope", "development", "--round", "1"]}]}},
         {"status": "ok", "passed": True}, True, "success", "counted"),
    ],
)
def test_outcome_taxonomy(tmp_path, agent, terminal, files, expected, category):
    if files:
        (tmp_path / "candidate.py").write_text("x")
        (tmp_path / "candidate.manifest.json").write_text("{}")
    assert diagnostic.classify(agent, terminal, tmp_path) == (expected, category)


def test_manifest_drift_and_skill_inventory_are_rejected(tmp_path: Path):
    config = manifest(tmp_path)
    Path(config["prompt"]).write_text("changed")
    with pytest.raises(diagnostic.DiagnosticError, match="prompt"):
        diagnostic.validate_manifest(config)

    config = manifest(tmp_path / "second")
    config["treatments"]["cannbot"]["skills"].append("undeclared")
    with pytest.raises(diagnostic.DiagnosticError, match="inventory"):
        diagnostic.validate_manifest(config)


@pytest.mark.parametrize("mutation", ["omitted", "extra", "wrong", "reordered"])
def test_manifest_rejects_noncanonical_cannbot_skill_inventory(tmp_path: Path, mutation):
    config = manifest(tmp_path)
    skills = list(diagnostic.TREATMENT_SKILLS["cannbot"])
    if mutation == "omitted":
        skills.pop()
    elif mutation == "extra":
        skills.append("ascend-profiling")
    elif mutation == "wrong":
        skills[-1] = "ascend-profiling"
    else:
        skills.reverse()
    config["treatments"]["cannbot"] = {
        "skills": skills,
        "skill_sha256": {skill: f"hash-{skill}" for skill in skills},
    }

    with pytest.raises(diagnostic.DiagnosticError, match="noncanonical"):
        diagnostic.validate_manifest(config)


def test_manifest_accepts_exact_ordered_canonical_skill_inventories(tmp_path: Path):
    config = manifest(tmp_path)

    diagnostic.validate_manifest(config)

    assert all(
        config["treatments"][name]["skills"] == list(diagnostic.TREATMENT_SKILLS[name])
        for name in diagnostic.TREATMENTS
    )
    assert diagnostic.TREATMENT_SKILLS is production_campaign.TREATMENT_SKILLS
    assert diagnostic.TREATMENT_SKILLS["cannbot"] == (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ops-profiling",
    )
    assert diagnostic.TREATMENT_SKILLS["project-cannbot"][-1] == "ascend-profiling"
    assert diagnostic.TREATMENT_SKILLS["project-guarded"] == (
        "ascend-profiling", "triton-guarded-kernel",
    )


def test_ledger_is_valid_at_every_atomic_checkpoint(tmp_path: Path, monkeypatch):
    observed = []
    real_replace = diagnostic.os.replace

    def replace(source, destination):
        is_ledger = Path(destination).name == "ledger.json"
        if is_ledger:
            json.loads(Path(source).read_text())
        real_replace(source, destination)
        if is_ledger:
            observed.append(json.loads(Path(destination).read_text()))

    monkeypatch.setattr(diagnostic.os, "replace", replace)
    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", RecordingLauncher(), RecordingTerminal()
    ).run()
    assert len(observed) >= 14
    assert any(sum(len(wave["cells"]) for wave in state["waves"]) == 1
               for state in observed)
    assert observed[-1] == result
    assert json.loads((tmp_path / "run" / "ledger.json").read_text()) == result


def test_timeout_bytes_are_serializable_and_preserve_diagnostics(tmp_path: Path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=kwargs["timeout"], output=b"partial-\xff",
            stderr=b"diagnostic-\xfe",
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", timeout)
    result = diagnostic._invoke(["agent"], {"operation": "one_shot"}, 1)

    assert result["status"] == "timeout"
    assert result["stdout"] == "partial-\ufffd"
    assert result["stderr"] == "diagnostic-\ufffd"
    json.dumps(result)


def test_nonzero_process_cannot_claim_success(tmp_path: Path, monkeypatch):
    completed = subprocess.CompletedProcess(
        ["agent"], 7, stdout='{"status":"ok","passed":true}', stderr="cleanup failed",
    )
    monkeypatch.setattr(diagnostic.subprocess, "run", lambda *args, **kwargs: completed)

    result = diagnostic._invoke(["agent"], {"operation": "terminal_check"}, 1)

    assert result["status"] == "transport_or_observer_error"
    assert result["exit_code"] == 7
    assert "claimed ok" in result["diagnostics"]


def test_cancelled_command_registry_cannot_start_a_late_process(monkeypatch):
    registry = diagnostic._ProcessRegistry()
    registry.cancel()

    def unexpected_process(*args, **kwargs):
        raise AssertionError("cancelled registry started a process")

    monkeypatch.setattr(diagnostic.subprocess, "Popen", unexpected_process)
    result = diagnostic._invoke(
        ["agent"], {"operation": "one_shot"}, 1, registry,
    )

    assert result["status"] == "transport_or_observer_error"
    assert "cancelled" in result["diagnostics"]


def test_timeout_kills_descendant_after_command_leader_exits(tmp_path: Path):
    child_pid = tmp_path / "descendant.pid"
    program = (
        "import json, os, pathlib, sys, time\n"
        "request = json.load(sys.stdin)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    pathlib.Path(request['pid_file']).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "os._exit(0)\n"
    )
    started = time.monotonic()
    result = diagnostic._invoke(
        [sys.executable, "-c", program], {"pid_file": str(child_pid)}, 0.2,
        diagnostic._ProcessRegistry(),
    )

    assert result["status"] == "timeout"
    assert time.monotonic() - started < 2
    assert child_pid.is_file()
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 1
    while True:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists():
            break
        # A container PID 1 may defer reaping an orphaned zombie. It has
        # exited and cannot execute or retain the command pipes.
        if status.read_text().split()[2] == "Z":
            break
        assert time.monotonic() < deadline, f"descendant {pid} survived group termination"
        time.sleep(0.01)


def test_primary_infrastructure_attempt_is_checkpointed_before_retry(tmp_path: Path,
                                                                    monkeypatch):
    observed = []
    real_replace = diagnostic.os.replace

    def replace(source, destination):
        real_replace(source, destination)
        if Path(destination).name == "ledger.json":
            observed.append(json.loads(Path(destination).read_text()))

    monkeypatch.setattr(diagnostic.os, "replace", replace)
    outcomes = {(1, "cannbot", 1): "device_or_runtime_infra"}
    diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", RecordingLauncher(outcomes),
        RecordingTerminal(),
    ).run()

    primary_only = [
        cell
        for state in observed
        for wave in state["waves"] if wave["wave"] == 1
        for cell in wave["cells"]
        if cell["treatment"] == "cannbot" and "retry" not in cell
    ]
    retried = [
        cell
        for state in observed
        for wave in state["waves"] if wave["wave"] == 1
        for cell in wave["cells"]
        if cell["treatment"] == "cannbot" and "retry" in cell
    ]
    assert primary_only and retried
    assert primary_only[0]["outcome"] == "device_or_runtime_infra"
    assert retried[0]["retry"]["outcome"] == "success"


def test_infrastructure_then_observed_retry_remains_pending(tmp_path: Path):
    outcomes = {
        (1, "cannbot", 1): "device_or_runtime_infra",
        (1, "cannbot", 2): "request_budget_exhausted",
    }
    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", RecordingLauncher(outcomes),
        RecordingTerminal(),
    ).run()

    cell = result["waves"][0]["cells"][0]
    assert cell["category"] == "infrastructure"
    assert cell["retry"]["category"] == "observed"
    assert result["reschedule"] == ["wave-1-cannbot"]
    assert result["status"] == "reschedule_pending"


def test_retry_that_consumes_remaining_wave_budget_stays_pending(tmp_path: Path):
    class BudgetConsumingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            if request["wave"] == 1 and request["treatment"] == "cannbot":
                if request["attempt"] == 1:
                    return {"status": "device_or_runtime_infra"}
                workspace = Path(request["workspace"])
                (workspace / "candidate.py").write_text("def kernel(): pass\n")
                (workspace / "candidate.manifest.json").write_text("{}\n")
                time.sleep(0.06)
                return {"status": "ok", "controller_usage": {"billed": 1, "calls": [
                    {"arguments": ["check", "--scope", "development", "--round", "1"]}
                ]}}
            return super().launch(request, timeout_seconds)

    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", BudgetConsumingLauncher(),
        RecordingTerminal(), wave_timeout=0.05,
    ).run()

    retry = result["waves"][0]["cells"][0]["retry"]
    assert retry["outcome"] == "wave_budget_exhausted"
    assert retry["category"] == "infrastructure"
    assert result["reschedule"] == ["wave-1-cannbot"]
    assert result["status"] == "reschedule_pending"


@pytest.mark.parametrize("malformed", ["no_submission", "protocol_error"])
def test_over_budget_retry_overrides_incomplete_agent_result(tmp_path: Path, monkeypatch,
                                                            malformed: str):
    clock = [0.0]
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: clock[0])

    class OverBudgetLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            if request["wave"] == 1 and request["treatment"] == "cannbot":
                if request["attempt"] == 1:
                    return {"status": "device_or_runtime_infra"}
                clock[0] += 0.06
                result = {"status": "ok", "controller_usage": {"billed": 1, "calls": [
                    {"arguments": ["check", "--scope", "development", "--round", "1"]}
                ]}}
                if malformed == "protocol_error":
                    result["controller_usage"]["billed"] = 0
                return result
            return super().launch(request, timeout_seconds)

    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", OverBudgetLauncher(), RecordingTerminal(),
        wave_timeout=0.05,
    ).run()

    retry = result["waves"][0]["cells"][0]["retry"]
    assert retry["outcome"] == "wave_budget_exhausted"
    assert retry["category"] == "infrastructure"
    assert retry["agent"]["reported_agent"]["status"] == "ok"
    assert result["reschedule"] == ["wave-1-cannbot"]
    assert result["status"] == "reschedule_pending"


def test_retry_terminal_completion_after_deadline_stays_pending(tmp_path: Path,
                                                               monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: clock[0])

    class RetryLauncher(RecordingLauncher):
        def __init__(self):
            super().__init__({(1, "cannbot", 1): "device_or_runtime_infra"})

    class DeadlineCrossingTerminal(RecordingTerminal):
        crossed = False

        def check(self, request, timeout_seconds):
            if request["cell_id"] == "wave-1-cannbot" and not self.crossed:
                self.crossed = True
                clock[0] += 0.06
            return super().check(request, timeout_seconds)

    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", RetryLauncher(),
        DeadlineCrossingTerminal(), wave_timeout=0.05,
    ).run()

    retry = result["waves"][0]["cells"][0]["retry"]
    assert retry["agent"]["status"] == "ok"
    assert retry["terminal"]["failure_type"] == "wave_budget_exhausted"
    assert retry["outcome"] == "wave_budget_exhausted"
    assert retry["category"] == "infrastructure"
    assert result["reschedule"] == ["wave-1-cannbot"]
    assert result["status"] == "reschedule_pending"


@pytest.mark.parametrize("component", ["launcher", "terminal"])
def test_hook_exception_is_checkpointed_and_sibling_cells_finish(tmp_path: Path, component: str):
    class RaisingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            if request["treatment"] == "cannbot":
                raise RuntimeError("agent transport disconnected")
            return super().launch(request, timeout_seconds)

    class RaisingTerminal(RecordingTerminal):
        def check(self, request, timeout_seconds):
            if request["cell_id"].split("-", 2)[2] == "cannbot":
                raise RuntimeError("terminal transport disconnected")
            return super().check(request, timeout_seconds)

    launcher = RaisingLauncher() if component == "launcher" else RecordingLauncher()
    terminal = RaisingTerminal() if component == "terminal" else RecordingTerminal()
    root = tmp_path / "run"
    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), root, launcher, terminal,
    ).run()

    cell = result["waves"][0]["cells"][0]
    evidence = cell["agent"] if component == "launcher" else cell["terminal"]
    assert cell["outcome"] == "transport_or_observer_error"
    assert cell["retry"]["outcome"] == "transport_or_observer_error"
    assert "RuntimeError" in evidence["diagnostics"]
    assert [entry["outcome"] for entry in result["waves"][0]["cells"][1:]] == [
        "success", "success",
    ]
    assert result["status"] == "reschedule_pending"
    assert json.loads((root / "ledger.json").read_text()) == result


@pytest.mark.parametrize("component", ["agent", "terminal"])
def test_bz_client_structured_infrastructure_error_is_interoperable(tmp_path: Path,
                                                                   component: str):
    (tmp_path / "candidate.py").write_text("x")
    (tmp_path / "candidate.manifest.json").write_text("{}")
    valid_agent = {"status": "ok", "controller_usage": {"billed": 1, "calls": [
        {"arguments": ["check", "--scope", "development", "--round", "1"]}
    ]}}
    infrastructure = {
        "status": "infrastructure_error", "failure_type": "runner_setup_error",
        "diagnostics": "torch import failed", "handle": "bz-a3-1:job-1",
    }

    if component == "agent":
        result = diagnostic.classify(infrastructure, None, tmp_path)
    else:
        result = diagnostic.classify(valid_agent, infrastructure, tmp_path)

    assert result == ("runner_setup_error", "infrastructure")


def test_submission_error_is_counted_candidate_failure(tmp_path: Path):
    (tmp_path / "candidate.py").write_text("x")
    (tmp_path / "candidate.manifest.json").write_text("{}")
    agent = {"status": "ok", "controller_usage": {"billed": 1, "calls": [
        {"arguments": ["check", "--scope", "development", "--round", "1"]}
    ]}}

    assert diagnostic.classify(
        agent, {"status": "submission_error", "diagnostics": "malformed manifest"},
        tmp_path,
    ) == ("submission_error", "counted")


@pytest.mark.parametrize("status", [None, "unexpected"])
def test_unknown_or_missing_agent_status_cannot_inherit_terminal_success(tmp_path: Path,
                                                                        status):
    (tmp_path / "candidate.py").write_text("x")
    (tmp_path / "candidate.manifest.json").write_text("{}")
    agent = {"controller_usage": {"billed": 1, "calls": [
        {"arguments": ["check", "--scope", "development", "--round", "1"]}
    ]}}
    if status is not None:
        agent["status"] = status

    assert diagnostic.classify(
        agent, {"status": "ok", "passed": True}, tmp_path,
    ) == ("protocol_error", "counted")


def test_all_primary_completions_are_checkpointed_before_retry(tmp_path: Path):
    class DelayedPrimaryLauncher(RecordingLauncher):
        def __init__(self):
            super().__init__({(1, "cannbot", 1): "device_or_runtime_infra"})
            self.completed = set()

        def launch(self, request, timeout_seconds):
            if request["wave"] == 1 and request["attempt"] == 1 \
                    and request["treatment"] != "cannbot":
                time.sleep(0.03)
            if request["wave"] == 1 and request["attempt"] == 2:
                assert self.completed == set(diagnostic.TREATMENTS)
            result = super().launch(request, timeout_seconds)
            if request["wave"] == 1 and request["attempt"] == 1:
                self.completed.add(request["treatment"])
            return result

    result = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", DelayedPrimaryLauncher(),
        RecordingTerminal(),
    ).run()

    assert result["status"] == "complete"
    assert result["waves"][0]["cells"][0]["retry"]["outcome"] == "success"


def test_unexpected_cell_exception_is_durable_and_does_not_abort_siblings(tmp_path: Path,
                                                                         monkeypatch):
    campaign = diagnostic.DiagnosticCampaign(
        manifest(tmp_path), tmp_path / "run", RecordingLauncher(), RecordingTerminal(),
    )
    original = campaign._cell

    def broken_cell(wave, treatment, attempt, available_seconds=None):
        if treatment == "cannbot":
            raise RuntimeError("unexpected hook failure")
        return original(wave, treatment, attempt, available_seconds)

    monkeypatch.setattr(campaign, "_cell", broken_cell)
    result = campaign.run()

    failed = result["waves"][0]["cells"][0]
    assert failed["outcome"] == "transport_or_observer_error"
    assert failed["retry"]["outcome"] == "transport_or_observer_error"
    assert [cell["outcome"] for cell in result["waves"][0]["cells"][1:]] == [
        "success", "success",
    ]
    assert result["status"] == "reschedule_pending"
    assert json.loads((tmp_path / "run" / "ledger.json").read_text()) == result


def test_keyboard_interrupt_checkpoints_terminal_state_and_sibling_evidence(tmp_path: Path,
                                                                            monkeypatch):
    sibling_checkpointed = threading.Event()
    real_atomic = diagnostic._atomic_json

    def observing_atomic(path, value):
        real_atomic(path, value)
        if any(cell["treatment"] == "cannbot"
               for wave in value["waves"] for cell in wave["cells"]):
            sibling_checkpointed.set()

    monkeypatch.setattr(diagnostic, "_atomic_json", observing_atomic)

    class InterruptingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            if request["wave"] == 1 and request["treatment"] == "project-cannbot":
                assert sibling_checkpointed.wait(timeout=1)
                raise KeyboardInterrupt("operator interrupted")
            return super().launch(request, timeout_seconds)

    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
        diagnostic.DiagnosticCampaign(
            manifest(tmp_path), root, InterruptingLauncher(), RecordingTerminal(),
        ).run()

    ledger = json.loads((root / "ledger.json").read_text())
    assert ledger["status"] == "interrupted"
    assert ledger["failure"] == {
        "type": "KeyboardInterrupt", "message": "operator interrupted",
    }
    completed = {cell["treatment"] for cell in ledger["waves"][0]["cells"]}
    assert "cannbot" in completed
    assert "wave-1-project-cannbot" in ledger["reschedule"]
    assert {
        f"wave-{wave}-{treatment}"
        for wave in range(2, 5)
        for treatment in diagnostic.TREATMENTS
    }.issubset(ledger["reschedule"])
    assert "wave-1-cannbot" not in ledger["reschedule"]


def test_interrupt_checkpoints_without_waiting_for_blocked_sibling(tmp_path: Path):
    sibling_entered = threading.Event()
    release_sibling = threading.Event()
    campaign_finished = threading.Event()
    raised = []

    class BlockingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            treatment = request["treatment"]
            if treatment == "cannbot":
                assert release_sibling.wait(timeout=5)
                return super().launch(request, timeout_seconds)
            if treatment == "project-guarded":
                sibling_entered.set()
                # Deliberately ignore the launcher's advisory timeout.
                assert release_sibling.wait(timeout=5)
                return super().launch(request, timeout_seconds)
            if treatment == "project-cannbot":
                assert sibling_entered.wait(timeout=1)
                raise KeyboardInterrupt("operator interrupted")
            return super().launch(request, timeout_seconds)

    root = tmp_path / "run"

    def run_campaign():
        try:
            diagnostic.DiagnosticCampaign(
                manifest(tmp_path), root, BlockingLauncher(), RecordingTerminal(),
            ).run()
        except BaseException as error:
            raised.append(error)
        finally:
            campaign_finished.set()

    runner = threading.Thread(target=run_campaign)
    runner.start()
    try:
        assert campaign_finished.wait(timeout=1), (
            "campaign waited for a blocked sibling before checkpointing interruption"
        )
        assert len(raised) == 1
        assert isinstance(raised[0], KeyboardInterrupt)
        ledger = json.loads((root / "ledger.json").read_text())
        assert ledger["status"] == "interrupted"
        assert set(ledger["reschedule"]) == {
            f"wave-{wave}-{treatment}"
            for wave in range(1, 5)
            for treatment in diagnostic.TREATMENTS
        }
    finally:
        release_sibling.set()
        runner.join(timeout=2)


def test_interrupt_during_worker_submission_is_checkpointed_and_cancelled(tmp_path: Path,
                                                                          monkeypatch):
    released = threading.Event()

    class CancellableLauncher(RecordingLauncher):
        def __init__(self):
            super().__init__()
            self.cancelled = False

        def launch(self, request, timeout_seconds):
            assert released.wait(timeout=2)
            return super().launch(request, timeout_seconds)

        def cancel(self):
            self.cancelled = True
            released.set()

    real_submit = diagnostic._submit_daemon
    submissions = 0

    def interrupting_submit(function, *args):
        nonlocal submissions
        submissions += 1
        if submissions == 2:
            raise KeyboardInterrupt("interrupted during submission")
        return real_submit(function, *args)

    monkeypatch.setattr(diagnostic, "_submit_daemon", interrupting_submit)
    launcher = CancellableLauncher()
    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt, match="during submission"):
        diagnostic.DiagnosticCampaign(
            manifest(tmp_path), root, launcher, RecordingTerminal(),
        ).run()

    ledger = json.loads((root / "ledger.json").read_text())
    assert ledger["status"] == "interrupted"
    assert len(ledger["reschedule"]) == 12
    assert launcher.cancelled is True


def test_checkpoint_failure_preserves_interrupt_and_cancels_subprocesses(tmp_path: Path,
                                                                         monkeypatch):
    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        "import json, os, pathlib, sys, time\n"
        "request = json.load(sys.stdin)\n"
        "pathlib.Path(request['workspace'], 'child.pid').write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    command = diagnostic.CommandLauncher([sys.executable, str(blocker)])

    class InterruptingCommandLauncher:
        def launch(self, request, timeout_seconds):
            if request["treatment"] == "project-cannbot":
                pid_files = [
                    tmp_path / "run" / "cells" / f"wave-1-{treatment}" /
                    "attempt-1" / "workspace" / "child.pid"
                    for treatment in ("cannbot", "project-guarded")
                ]
                deadline = time.monotonic() + 1
                while not all(path.is_file() for path in pid_files):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                raise KeyboardInterrupt("original interruption")
            return command.launch(request, timeout_seconds)

        def cancel(self):
            command.cancel()

    real_atomic = diagnostic._atomic_json

    def fail_interrupted_checkpoint(path, value):
        if value.get("status") == "interrupted":
            raise OSError("checkpoint storage unavailable")
        real_atomic(path, value)

    monkeypatch.setattr(diagnostic, "_atomic_json", fail_interrupted_checkpoint)
    root = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        diagnostic.DiagnosticCampaign(
            manifest(tmp_path), root, InterruptingCommandLauncher(), RecordingTerminal(),
        ).run()

    pid_files = root.glob("cells/wave-1-*/attempt-1/workspace/child.pid")
    pids = [int(path.read_text()) for path in pid_files]
    assert len(pids) == 2
    deadline = time.monotonic() + 1
    for pid in pids:
        while True:
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            assert time.monotonic() < deadline, f"launcher child {pid} survived cleanup"
            time.sleep(0.01)


def test_interrupted_campaign_process_exits_with_permanently_blocked_siblings(tmp_path: Path):
    if os.environ.get("DIAGNOSTIC_BLOCKED_SIBLING_CHILD") == "1":
        blocker = tmp_path / "blocker.py"
        blocker.write_text(
            "import json, os, pathlib, sys, time\n"
            "request = json.load(sys.stdin)\n"
            "pathlib.Path(request['workspace'], 'child.pid').write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        command = diagnostic.CommandLauncher([sys.executable, str(blocker)])

        class InterruptingCommandLauncher:
            def launch(self, request, timeout_seconds):
                if request["treatment"] == "project-cannbot":
                    pid_files = [
                        tmp_path / "run" / "cells" / f"wave-1-{treatment}" /
                        "attempt-1" / "workspace" / "child.pid"
                        for treatment in ("cannbot", "project-guarded")
                    ]
                    deadline = time.monotonic() + 1
                    while not all(path.is_file() for path in pid_files):
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    raise KeyboardInterrupt("operator interrupted")
                return command.launch(request, timeout_seconds)

            def cancel(self):
                command.cancel()

        root = tmp_path / "run"
        with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
            diagnostic.DiagnosticCampaign(
                manifest(tmp_path), root,
                InterruptingCommandLauncher(), RecordingTerminal(),
            ).run()
        ledger = json.loads((root / "ledger.json").read_text())
        assert ledger["status"] == "interrupted"
        assert set(ledger["reschedule"]) == {
            f"wave-{wave}-{treatment}"
            for wave in range(1, 5)
            for treatment in diagnostic.TREATMENTS
        }
        pid_files = root.glob("cells/wave-1-*/attempt-1/workspace/child.pid")
        pids = [int(path.read_text()) for path in pid_files]
        assert len(pids) == 2
        deadline = time.monotonic() + 1
        for pid in pids:
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                assert time.monotonic() < deadline, f"launcher child {pid} survived cancellation"
                time.sleep(0.01)
        return

    environment = dict(os.environ, DIAGNOSTIC_BLOCKED_SIBLING_CHILD="1")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{Path(__file__).resolve()}::{test_interrupted_campaign_process_exits_with_permanently_blocked_siblings.__name__}"],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=3,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_reused_output_root_is_rejected_without_changing_ledger(tmp_path: Path):
    root = tmp_path / "run"
    config = manifest(tmp_path)
    diagnostic.DiagnosticCampaign(
        config, root, RecordingLauncher(), RecordingTerminal(),
    ).run()
    retained = (root / "ledger.json").read_bytes()

    with pytest.raises(diagnostic.DiagnosticError, match="not fresh"):
        diagnostic.DiagnosticCampaign(
            config, root, RecordingLauncher(), RecordingTerminal(),
        ).run()

    assert (root / "ledger.json").read_bytes() == retained


@pytest.mark.parametrize("drift", ["prompt", "manifest"])
def test_inputs_are_revalidated_immediately_before_initial_ledger(tmp_path: Path, drift: str):
    config = manifest(tmp_path)
    root = tmp_path / "run"
    campaign = diagnostic.DiagnosticCampaign(
        config, root, RecordingLauncher(), RecordingTerminal(),
    )
    if drift == "prompt":
        Path(config["prompt"]).write_text("changed after construction\n")
    else:
        config["treatments"]["cannbot"]["skills"] = ["triton-op-coding"]

    with pytest.raises(diagnostic.DiagnosticError):
        campaign.run()

    assert not (root / "ledger.json").exists()


def test_run_uses_manifest_snapshot_after_revalidation(tmp_path: Path):
    config = manifest(tmp_path)

    class MutatingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            config["treatments"]["cannbot"]["skills"] = ["wrong-after-run-started"]
            return super().launch(request, timeout_seconds)

    launcher = MutatingLauncher()
    diagnostic.DiagnosticCampaign(
        config, tmp_path / "run", launcher, RecordingTerminal(),
    ).run()

    cannbot_requests = [request for request, _ in launcher.requests
                        if request["treatment"] == "cannbot"]
    assert cannbot_requests
    assert all(request["skills"] == list(diagnostic.TREATMENT_SKILLS["cannbot"])
               for request in cannbot_requests)


def test_run_freezes_prompt_bytes_before_launch(tmp_path: Path):
    config = manifest(tmp_path)
    source = Path(config["prompt"])
    original = source.read_bytes()

    class PromptMutatingLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            source.write_text("mutated after run started\n")
            assert diagnostic.request_prompt_bytes(request) == original
            return super().launch(request, timeout_seconds)

    launcher = PromptMutatingLauncher()
    result = diagnostic.DiagnosticCampaign(
        config, tmp_path / "run", launcher, RecordingTerminal(),
    ).run()

    assert result["status"] == "complete"
    assert all(
        diagnostic.request_prompt_bytes(request) == original
        for request, _ in launcher.requests
    )


def test_swap_and_restore_cannot_drift_in_band_prompts(tmp_path: Path):
    config = manifest(tmp_path)
    original = Path(config["prompt"]).read_bytes()
    barrier = threading.Barrier(3)
    consumed = []
    consumed_lock = threading.Lock()

    class DestructiveLauncher(RecordingLauncher):
        def launch(self, request, timeout_seconds):
            shared = tmp_path / "run" / "inputs" / "prompt.md"
            if request["treatment"] == "project-cannbot":
                shared.unlink()
                shared.write_text("temporary sibling-controlled bytes\n")
            barrier.wait(timeout=5)
            content = diagnostic.request_prompt_bytes(request)
            with consumed_lock:
                consumed.append(content)
            barrier.wait(timeout=5)
            if request["treatment"] == "project-cannbot":
                shared.unlink()
                shared.write_bytes(original)
            return super().launch(request, timeout_seconds)

    launcher = DestructiveLauncher()
    result = diagnostic.DiagnosticCampaign(
        config, tmp_path / "run", launcher, RecordingTerminal(),
    ).run()

    outcomes = [
        (cell["outcome"], cell["agent"].get("diagnostics"), cell.get("retry"))
        for wave in result["waves"] for cell in wave["cells"]
    ]
    assert result["status"] == "complete", outcomes
    assert consumed == [original] * 12
    assert len(launcher.requests) == 12


def test_command_launcher_rejects_legacy_path_prompt_before_spawning(monkeypatch):
    launcher = diagnostic.CommandLauncher(["agent"])
    request = {"protocol_version": 2, "prompt": "/writable/prompt.md",
               "prompt_sha256": "digest"}

    def unexpected_process(*args, **kwargs):
        raise AssertionError("legacy prompt request reached the command consumer")

    monkeypatch.setattr(diagnostic.subprocess, "Popen", unexpected_process)
    with pytest.raises(diagnostic.DiagnosticError, match="in-band"):
        launcher.launch(request, 1)


def test_command_launcher_stops_detached_same_group_writer(tmp_path: Path):
    marker = tmp_path / "late-write"
    child = (
        "import pathlib,time; time.sleep(0.3); "
        f"pathlib.Path({str(marker)!r}).write_text('mutated')"
    )
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json,subprocess,sys\n"
        "json.load(sys.stdin)\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(json.dumps({'status': 'ok'}), flush=True)\n"
    )
    content = b"prompt\n"
    digest = hashlib.sha256(content).hexdigest()
    request = {
        "protocol_version": 2,
        "prompt": {"encoding": "base64", "data": "cHJvbXB0Cg==", "sha256": digest},
        "prompt_sha256": digest,
    }
    result = diagnostic.CommandLauncher([sys.executable, str(agent)]).launch(request, 2)
    assert result["status"] == "ok"
    time.sleep(0.4)
    assert not marker.exists()


@pytest.mark.parametrize("version", [None, 1])
def test_command_launcher_rejects_non_v2_prompt_envelope_before_spawning(
        monkeypatch, version):
    launcher = diagnostic.CommandLauncher(["agent"])
    digest = hashlib.sha256(b"x").hexdigest()
    request = {
        "prompt": {"encoding": "base64", "data": "eA==", "sha256": digest},
        "prompt_sha256": digest,
    }
    if version is not None:
        request["protocol_version"] = version

    def unexpected_process(*args, **kwargs):
        raise AssertionError("non-v2 request reached the command consumer")

    monkeypatch.setattr(diagnostic.subprocess, "Popen", unexpected_process)
    with pytest.raises(diagnostic.DiagnosticError, match="protocol version 2"):
        launcher.launch(request, 1)


def test_concurrent_launcher_mutation_cannot_cross_cell_or_change_evidence(tmp_path: Path):
    config = manifest(tmp_path)
    expected_model = dict(config["model"])
    expected_skills = {
        name: list(config["treatments"][name]["skills"])
        for name in diagnostic.TREATMENTS
    }
    expected_hashes = {
        name: dict(config["treatments"][name]["skill_sha256"])
        for name in diagnostic.TREATMENTS
    }

    class MutatingConcurrentLauncher:
        def __init__(self):
            self.barrier = threading.Barrier(3)
            self.observed = []
            self.lock = threading.Lock()

        def launch(self, request, timeout_seconds):
            with self.lock:
                self.observed.append((
                    request["treatment"], dict(request["model"]),
                    list(request["skills"]), dict(request["skill_sha256"]),
                ))
            request["model"]["name"] = "mutated"
            request["skills"].append("cross-cell-leak")
            request["skill_sha256"].clear()
            self.barrier.wait()
            workspace = Path(request["workspace"])
            (workspace / "candidate.py").write_text("def kernel(): pass\n")
            (workspace / "candidate.manifest.json").write_text("{}\n")
            return {"status": "ok", "controller_usage": {"billed": 1, "calls": [
                {"arguments": ["check", "--scope", "development", "--round", "1"]}
            ]}}

    launcher = MutatingConcurrentLauncher()
    result = diagnostic.DiagnosticCampaign(
        config, tmp_path / "run", launcher, RecordingTerminal(),
    ).run()

    assert all(model == expected_model and skills == expected_skills[treatment]
               and hashes == expected_hashes[treatment]
               for treatment, model, skills, hashes in launcher.observed)
    assert all(cell["skill_sha256"] == expected_hashes[cell["treatment"]]
               for wave in result["waves"] for cell in wave["cells"])
    assert result["status"] == "complete"
