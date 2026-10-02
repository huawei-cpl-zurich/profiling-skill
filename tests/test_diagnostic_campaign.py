from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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
            frozen = Path(request["prompt"])
            assert frozen != source
            assert frozen.read_bytes() == original
            assert frozen.stat().st_mode & 0o222 == 0
            assert diagnostic.sha256_file(frozen) == request["prompt_sha256"]
            return super().launch(request, timeout_seconds)

    launcher = PromptMutatingLauncher()
    result = diagnostic.DiagnosticCampaign(
        config, tmp_path / "run", launcher, RecordingTerminal(),
    ).run()

    assert result["status"] == "complete"
    frozen_paths = {request["prompt"] for request, _ in launcher.requests}
    assert frozen_paths == {str(tmp_path / "run" / "inputs" / "prompt.md")}


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
