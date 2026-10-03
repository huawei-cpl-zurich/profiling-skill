from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
diagnostic_campaign = importlib.import_module("diagnostic_campaign")
smoke = importlib.import_module("two_shot_smoke_campaign")


def manifest(tmp_path: Path) -> dict:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("write and repair the kernel\n")
    model = {"name": "test", "reasoning_effort": "low"}
    return {
        "prompt": str(prompt), "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "model": model, "model_sha256": hashlib.sha256(json.dumps(
            model, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "treatments": {name: {
            "skills": list(diagnostic_campaign.TREATMENT_SKILLS[name]),
            "skill_sha256": {skill: f"hash-{skill}"
                             for skill in diagnostic_campaign.TREATMENT_SKILLS[name]},
        } for name in smoke.TREATMENTS},
    }


class Launcher:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.requests = []
        self.lock = threading.Lock()

    def launch(self, request, timeout_seconds):
        with self.lock:
            self.requests.append(request)
        outcome = self.outcomes.get((request["wave"], request["treatment"], request["attempt"]), "ok")
        if outcome == "ok":
            workspace = Path(request["workspace"])
            (workspace / "candidate.py").write_text("round two\n")
            (workspace / "candidate.manifest.json").write_text("{}\n")
        return {"status": outcome, "rounds_completed": 2,
                "candidate_sha256": {"1": "a" * 64, "2": "b" * 64},
                "controller_usage": {"billed": 2, "invalid": 0, "over_budget": 0,
                                     "calls": [
                    {"arguments": smoke.check_command(1)},
                    {"arguments": smoke.check_command(2)},
                ]}}


class Terminal:
    def __init__(self):
        self.requests = []

    def check(self, request, timeout_seconds):
        self.requests.append(request)
        return {"status": "ok", "passed": True, "handle": "bz-a3-1:job"}


def test_two_fresh_waves_run_three_isolated_treatments(tmp_path: Path):
    launcher, terminal = Launcher(), Terminal()
    result = smoke.TwoShotSmokeCampaign(
        manifest(tmp_path), tmp_path / "run", launcher, terminal,
        benchmark="matmul", cases=list(range(7)),
    ).run()

    assert result["status"] == "complete"
    assert len(launcher.requests) == len(terminal.requests) == 6
    assert all(request["operation"] == "two_shot" for request in launcher.requests)
    assert all(request["controller_contract"] == {
        "billed_limit": 2,
        "commands": [smoke.check_command(1), smoke.check_command(2)],
    } for request in launcher.requests)
    assert len({request["workspace"] for request in launcher.requests}) == 6
    assert all(request["cases"] == list(range(7)) for request in terminal.requests)
    assert all((Path(request["workspace"]).parent / "frozen-submission" /
                "candidate.py").stat().st_mode & 0o777 == 0o444
               for request in terminal.requests)
    assert {cell["outcome"] for wave in result["waves"] for cell in wave["cells"]} == {"success"}


def test_only_infrastructure_failure_is_retried(tmp_path: Path):
    launcher = Launcher({(1, "cannbot", 1): "device_or_runtime_infra",
                         (1, "project-cannbot", 1): "compile_error"})
    result = smoke.TwoShotSmokeCampaign(
        manifest(tmp_path), tmp_path / "run", launcher, Terminal(),
        benchmark="bsa", cases=[47, 46, 49, 44, 43],
    ).run()

    assert len([request for request in launcher.requests
                if request["wave"] == 1 and request["treatment"] == "cannbot"]) == 2
    assert len([request for request in launcher.requests
                if request["wave"] == 1 and request["treatment"] == "project-cannbot"]) == 1
    cells = {cell["treatment"]: cell for cell in result["waves"][0]["cells"]}
    assert cells["cannbot"]["outcome"] == "success"
    assert cells["project-cannbot"]["outcome"] == "compile_error"


def test_checkpoints_each_cell_before_slow_siblings_finish(tmp_path: Path):
    root = tmp_path / "run"

    class DelayedLauncher(Launcher):
        def launch(self, request, timeout_seconds):
            if request["treatment"] != "cannbot":
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if (root / "ledger.json").is_file():
                        ledger = json.loads((root / "ledger.json").read_text())
                        if ledger["waves"] and ledger["waves"][0]["cells"]:
                            break
                    time.sleep(.01)
                else:
                    raise AssertionError("fast cell was not checkpointed")
            return super().launch(request, timeout_seconds)

    result = smoke.TwoShotSmokeCampaign(
        manifest(tmp_path), root, DelayedLauncher(), Terminal(),
        benchmark="matmul", cases=list(range(7))).run()
    assert result["status"] == "complete"


def test_resume_partial_wave_preserves_completed_cells(tmp_path: Path):
    root = tmp_path / "run"; config = manifest(tmp_path)
    first = smoke.TwoShotSmokeCampaign(
        config, root, Launcher(), Terminal(), benchmark="matmul", cases=list(range(7)))
    original = first._cell

    def interrupted(wave, treatment, *args, **kwargs):
        if wave == 1 and treatment == "project-cannbot":
            time.sleep(.1)
            raise KeyboardInterrupt
        if treatment == "project-guarded":
            time.sleep(.2)
        return original(wave, treatment, *args, **kwargs)

    first._cell = interrupted
    with pytest.raises(KeyboardInterrupt):
        first.run()
    partial = json.loads((root / "ledger.json").read_text())
    assert {cell["treatment"] for cell in partial["waves"][0]["cells"]} == {
        "cannbot", "project-guarded"}

    launcher = Launcher()
    result = smoke.TwoShotSmokeCampaign(
        config, root, launcher, Terminal(), benchmark="matmul",
        cases=list(range(7)), resume=True).run()
    assert result["status"] == "complete"
    assert not any(request["wave"] == 1 and request["treatment"] == "cannbot"
                   for request in launcher.requests)
    assert not any(request["wave"] == 1 and request["treatment"] == "project-guarded"
                   for request in launcher.requests)


def test_resume_retries_only_infrastructure_and_preserves_counted(tmp_path: Path):
    root = tmp_path / "run"; config = manifest(tmp_path)
    outcomes = {(1, "cannbot", attempt): "device_or_runtime_infra"
                for attempt in (1, 2)}
    outcomes[(1, "project-cannbot", 1)] = "compile_error"
    first = smoke.TwoShotSmokeCampaign(
        config, root, Launcher(outcomes), Terminal(), benchmark="matmul",
        cases=list(range(7))).run()
    assert first["status"] == "infrastructure_pending"

    launcher = Launcher()
    resumed = smoke.TwoShotSmokeCampaign(
        config, root, launcher, Terminal(), benchmark="matmul",
        cases=list(range(7)), resume=True).run()
    assert resumed["status"] == "smoke_failures"
    assert {(request["wave"], request["treatment"]) for request in launcher.requests} == {
        (1, "cannbot")}
    cell = next(cell for cell in resumed["waves"][0]["cells"]
                if cell["treatment"] == "project-cannbot")
    assert cell["outcome"] == "compile_error" and len(cell["attempts"]) == 1


def test_uncertain_terminal_requires_reconciliation_without_redispatch(tmp_path: Path):
    class UncertainTerminal(Terminal):
        def check(self, request, timeout_seconds):
            self.requests.append(request)
            if request["wave"] == 1 and request["treatment"] == "cannbot":
                return {"status": "transport_or_observer_error",
                        "manual_reconciliation_required": True,
                        "handle": "bz-a3-1:kept"}
            return {"status": "ok", "passed": True, "handle": "bz-a3-1:job"}

    root = tmp_path / "run"; config = manifest(tmp_path)
    first_launcher = Launcher()
    first = smoke.TwoShotSmokeCampaign(
        config, root, first_launcher, UncertainTerminal(), benchmark="matmul",
        cases=list(range(7))).run()
    assert first["status"] == "reconciliation_required"
    assert len([request for request in first_launcher.requests
                if request["wave"] == 1 and request["treatment"] == "cannbot"]) == 1

    resumed_launcher = Launcher()
    resumed = smoke.TwoShotSmokeCampaign(
        config, root, resumed_launcher, Terminal(), benchmark="matmul",
        cases=list(range(7)), resume=True).run()
    assert resumed["status"] == "reconciliation_required"
    assert not any(request["wave"] == 1 and request["treatment"] == "cannbot"
                   for request in resumed_launcher.requests)


def test_bsa_requires_complete_matmul_gate(tmp_path: Path):
    ledger = tmp_path / "matmul.json"
    ledger.write_text(json.dumps({"benchmark": "matmul", "waves": [
        {"cells": [{"outcome": "success"} for _ in smoke.TREATMENTS]},
    ]}))
    with pytest.raises(smoke.SmokeError, match="six successful"):
        smoke.validate_matmul_gate(ledger)


@pytest.mark.parametrize("usage", [
    {"billed": 1, "invalid": 0, "over_budget": 0},
    {"billed": 2, "invalid": 1, "over_budget": 0},
    {"billed": 2, "invalid": 0, "over_budget": 1},
])
def test_protocol_rejects_non_exact_controller_usage(tmp_path: Path, usage: dict):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    usage["calls"] = [{"arguments": smoke.check_command(1)},
                      {"arguments": smoke.check_command(2)}]
    agent = {"status": "ok", "rounds_completed": 2,
             "candidate_sha256": {"1": "a" * 64, "2": "b" * 64},
             "controller_usage": usage}
    assert smoke._classification(
        agent, {"status": "ok", "passed": True}, workspace) == (
            "protocol_error", "counted")


@pytest.mark.parametrize("hashes", [
    {"1": "a" * 64},
    {"1": "not-a-digest", "2": "b" * 64},
    {"1": "A" * 64, "2": "b" * 64},
])
def test_protocol_rejects_incomplete_or_malformed_candidate_hashes(
        tmp_path: Path, hashes: dict):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "candidate.py").write_text("candidate\n")
    (workspace / "candidate.manifest.json").write_text("{}\n")
    agent = {"status": "ok", "rounds_completed": 2,
             "candidate_sha256": hashes,
             "controller_usage": {"billed": 2, "invalid": 0, "over_budget": 0,
                                  "calls": [{"arguments": smoke.check_command(1)},
                                            {"arguments": smoke.check_command(2)}]}}
    assert smoke._classification(
        agent, {"status": "ok", "passed": True}, workspace) == (
            "protocol_error", "counted")


def test_matmul_gate_requires_exact_two_wave_matrix(tmp_path: Path):
    valid = {"status": "complete", "benchmark": "matmul", "cases": list(range(7)),
             "prompt_sha256": "frozen", "waves": [
                 {"wave": wave, "cells": [
                     {"treatment": treatment, "outcome": "success"}
                     for treatment in smoke.TREATMENTS]}
                 for wave in (1, 2)]}
    path = tmp_path / "ledger.json"; path.write_text(json.dumps(valid))
    assert smoke.validate_matmul_gate(path)["prompt_sha256"] == "frozen"
    with pytest.raises(smoke.SmokeError, match="six successful"):
        smoke.validate_matmul_gate(path, prompt_sha256="different")
    for mutate in (
        lambda value: value.update(status="smoke_failures"),
        lambda value: value.update(cases=[1]),
        lambda value: value["waves"][1].update(wave=3),
        lambda value: value["waves"][0]["cells"].__setitem__(
            2, {"treatment": "cannbot", "outcome": "success"}),
    ):
        changed = json.loads(json.dumps(valid)); mutate(changed); path.write_text(json.dumps(changed))
        with pytest.raises(smoke.SmokeError, match="six successful"):
            smoke.validate_matmul_gate(path)
