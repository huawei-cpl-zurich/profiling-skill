from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import threading
from pathlib import Path

import pytest


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
    treatments = {
        "cannbot": ["triton-op-coding", "ops-profiling"],
        "project-cannbot": ["triton-op-coding", "ascend-profiling"],
        "project-guarded": ["ascend-profiling", "triton-guarded-kernel"],
    }
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


def test_ledger_is_valid_at_every_atomic_checkpoint(tmp_path: Path, monkeypatch):
    observed = []
    real_replace = diagnostic.os.replace

    def replace(source, destination):
        json.loads(Path(source).read_text())
        real_replace(source, destination)
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


def test_primary_infrastructure_attempt_is_checkpointed_before_retry(tmp_path: Path,
                                                                    monkeypatch):
    observed = []
    real_replace = diagnostic.os.replace

    def replace(source, destination):
        real_replace(source, destination)
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
