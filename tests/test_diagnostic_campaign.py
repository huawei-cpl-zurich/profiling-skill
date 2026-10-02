from __future__ import annotations

import hashlib
import importlib.util
import json
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


def test_terminal_command_timeout_distinguishes_workload_from_observer(
    tmp_path: Path, monkeypatch,
):
    def timeout(*_args, **_kwargs):
        raise diagnostic.subprocess.TimeoutExpired(["terminal"], 5)

    monkeypatch.setattr(diagnostic.subprocess, "run", timeout)
    hook = diagnostic.CommandTerminalHook(["terminal"])
    request = {"operation": "terminal_check", "cell_id": "wave-1-cannbot"}
    handle = "bz-a3-1:retained-1"

    workload = hook.check(request, 5)
    observer = hook.resume(request, handle, 5)

    assert workload["status"] == "timeout" and workload["invocation_timeout"] is True
    assert "handle" not in workload
    assert observer["status"] == "transport_or_observer_error"
    assert observer["invocation_timeout"] is True
    assert observer["handle"] == handle

    (tmp_path / "candidate.py").write_text("candidate\n")
    (tmp_path / "candidate.manifest.json").write_text("{}\n")
    agent = {"status": "ok", "controller_usage": {"billed": 1, "calls": [
        {"arguments": ["check", "--scope", "development", "--round", "1"]}
    ]}}
    assert diagnostic.classify(agent, workload, tmp_path) == (
        "candidate_timeout", "counted")
    assert diagnostic.classify(agent, observer, tmp_path) == (
        "transport_or_observer_error", "infrastructure")


def test_resume_preserves_parsed_terminal_workload_timeout(tmp_path: Path, monkeypatch):
    def completed(*_args, **_kwargs):
        return diagnostic.subprocess.CompletedProcess(
            ["terminal"], 0, stdout=json.dumps({"status": "timeout"}), stderr="")

    monkeypatch.setattr(diagnostic.subprocess, "run", completed)
    hook = diagnostic.CommandTerminalHook(["terminal"])
    terminal = hook.resume(
        {"operation": "terminal_check", "cell_id": "wave-1-cannbot"},
        "bz-a3-1:retained-1", 5,
    )

    assert terminal["status"] == "timeout"
    assert "invocation_timeout" not in terminal
    (tmp_path / "candidate.py").write_text("candidate\n")
    (tmp_path / "candidate.manifest.json").write_text("{}\n")
    agent = {"status": "ok", "controller_usage": {"billed": 1, "calls": [
        {"arguments": ["check", "--scope", "development", "--round", "1"]}
    ]}}
    assert diagnostic.classify(agent, terminal, tmp_path) == (
        "candidate_timeout", "counted")


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


def receipt(wave: int) -> dict:
    return {
        "wave": wave,
        "accepted": True,
        "stable_ref_citations": [f"ref://profiling-skill/triton-ascend/debugging/wave-{wave}"],
        "librarian_query_ids": [f"query-{wave}"],
    }


def test_adaptive_wave_pauses_then_resumes_with_only_prompt_revision(tmp_path: Path):
    first = manifest(tmp_path / "inputs-1")
    launcher, terminal = RecordingLauncher(), RecordingTerminal()
    root = tmp_path / "run"
    campaign = diagnostic.DiagnosticCampaign(
        first, root, launcher, terminal, campaign_config_sha256="config-v1"
    )

    paused = campaign.run_wave(1)
    assert paused["status"] == "awaiting_curation"
    assert len(paused["waves"]) == 1 and len(launcher.requests) == 3
    assert paused["waves"][0]["prompt_sha256"] == first["prompt_sha256"]
    ready = campaign.acknowledge_curation(receipt(1))
    assert ready["status"] == "ready_for_next"

    second = manifest(tmp_path / "inputs-2")
    Path(second["prompt"]).write_text("revised after curated evidence\n")
    second["prompt_sha256"] = diagnostic.sha256_file(Path(second["prompt"]))
    resumed = diagnostic.DiagnosticCampaign(
        second, root, launcher, terminal, campaign_config_sha256="config-v1"
    ).run_wave(2)

    assert resumed["status"] == "awaiting_curation"
    assert [wave["prompt_sha256"] for wave in resumed["waves"]] == [
        first["prompt_sha256"], second["prompt_sha256"]
    ]
    assert len(launcher.requests) == 6
    assert len([request for request, _ in launcher.requests if request["wave"] == 1]) == 3


def test_adaptive_requires_valid_curation_receipt(tmp_path: Path):
    campaign = diagnostic.DiagnosticCampaign(
        manifest(tmp_path / "inputs"), tmp_path / "run", RecordingLauncher(),
        RecordingTerminal(), campaign_config_sha256="config-v1"
    )
    campaign.run_wave(1)
    with pytest.raises(diagnostic.DiagnosticError, match="invalid curation"):
        campaign.acknowledge_curation({"wave": 1, "accepted": True,
                                      "stable_ref_citations": ["/raw/path"],
                                      "librarian_query_ids": []})
    with pytest.raises(diagnostic.DiagnosticError, match="not ready"):
        campaign.run_wave(2)


@pytest.mark.parametrize("drift", ["model", "skills", "config"])
def test_adaptive_rejects_non_prompt_drift(tmp_path: Path, drift: str):
    original = manifest(tmp_path / "inputs")
    root = tmp_path / "run"
    campaign = diagnostic.DiagnosticCampaign(
        original, root, RecordingLauncher(), RecordingTerminal(),
        campaign_config_sha256="config-v1"
    )
    campaign.run_wave(1)
    campaign.acknowledge_curation(receipt(1))
    changed = dict(original)
    config_hash = "config-v1"
    if drift == "model":
        changed["model"] = {"name": "different-model", "reasoning_effort": "low"}
        changed["model_sha256"] = hashlib.sha256(json.dumps(
            changed["model"], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
    elif drift == "skills":
        changed["treatments"] = json.loads(json.dumps(original["treatments"]))
        changed["treatments"]["cannbot"]["skill_sha256"]["ops-profiling"] = "changed"
    else:
        config_hash = "config-v2"
    with pytest.raises(diagnostic.DiagnosticError, match="drift"):
        diagnostic.DiagnosticCampaign(
            changed, root, RecordingLauncher(), RecordingTerminal(),
            campaign_config_sha256=config_hash
        ).run_wave(2)


def test_four_adaptive_waves_finish_only_after_final_curation(tmp_path: Path):
    launcher, terminal = RecordingLauncher(), RecordingTerminal()
    campaign = diagnostic.DiagnosticCampaign(
        manifest(tmp_path / "inputs"), tmp_path / "run", launcher, terminal,
        campaign_config_sha256="config-v1"
    )
    for wave in range(1, 5):
        paused = campaign.run_wave(wave)
        assert paused["status"] == "awaiting_curation"
        final = campaign.acknowledge_curation(receipt(wave))
    assert final["status"] == "complete"
    assert len(final["waves"]) == len(final["curation_receipts"]) == 4
    assert len(launcher.requests) == 12
    with pytest.raises(diagnostic.DiagnosticError, match="not ready"):
        campaign.run_wave(4)


def test_adaptive_checkpoints_are_atomic(tmp_path: Path, monkeypatch):
    observed = []
    real_replace = diagnostic.os.replace

    def replace(source, destination):
        state = json.loads(Path(source).read_text())
        real_replace(source, destination)
        observed.append(state)

    monkeypatch.setattr(diagnostic.os, "replace", replace)
    campaign = diagnostic.DiagnosticCampaign(
        manifest(tmp_path / "inputs"), tmp_path / "run", RecordingLauncher(),
        RecordingTerminal(), campaign_config_sha256="config-v1"
    )
    result = campaign.run_wave(1)
    assert observed[-1] == result
    assert any(state["status"] == "running" for state in observed)
    assert any(len(state["waves"][0]["cells"]) == 1 for state in observed if state["waves"])
