from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("campaign", ROOT / "scripts" / "campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def tree(path: Path, content: str = "content") -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(content)
    return path


def fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("same prompt\n")
    gdn = tree(tmp_path / "gdn", "gdn")
    bsa = tree(tmp_path / "bsa", "bsa")
    project = tree(tmp_path / "project", "project")
    frozen = tmp_path / "frozen"
    for skill in (
        *campaign.CANNBOT_TRITON_SKILLS,
        *campaign.CANNBOT_DEPENDENCIES,
        "ops-profiling",
    ):
        tree(frozen / "skills" / skill, skill)
    tree(frozen / "support" / "triton-op-generator", "support")
    (frozen / "support" / "triton-op-generator" / "AGENTS.md").write_text("cannbot rules")
    (frozen / "support" / "triton-op-generator" / "config.json").write_text("{}")
    designer = frozen / "skills" / "triton-op-designer"
    (designer / "SKILL.md").write_text(
        "@../npu-arch/references/hardware.md\n"
        "@../../plugins-official/triton-op-generator/template/linear.md\n"
    )
    npu_references = frozen / "skills" / "npu-arch" / "references"
    npu_references.mkdir()
    (npu_references / "hardware.md").write_text("hardware")
    templates = frozen / "support" / "triton-op-generator" / "template"
    templates.mkdir()
    (templates / "linear.md").write_text("linear")
    record = {
        "repository": "https://example.invalid/cannbot.git",
        "commit": "a" * 40,
        "skills": {
            path.name: campaign.digest_tree(path)
            for path in (frozen / "skills").iterdir()
        },
        "support": {
            "triton-op-generator": campaign.digest_tree(
                frozen / "support" / "triton-op-generator"
            )
        },
    }
    (frozen / "freeze.json").write_text(json.dumps(record, sort_keys=True))
    controller_config = tmp_path / "controller.json"
    controller_config.write_text('{"version":1}\n')
    controller_command = [
        "python", "/opt/campaign/scripts/experimentctl.py", "--config",
        str(controller_config.resolve()), "--cell", "{cell_id}",
    ]
    manifest_path = tmp_path / "campaign.json"
    manifest = campaign.write_manifest(
        manifest_path, prompt, {"gdn": gdn, "bsa": bsa}, project, frozen,
        controller_config, controller_command,
    )
    return manifest, manifest_path


def run_campaign(manifest, root, launcher, on_wave=None, resume=False):
    binding = manifest["controller"]
    config = Path(binding["config"]["path"])
    command = [argument.replace("{controller_config}", str(config))
               for argument in binding["command_argv"]]
    return campaign.run_campaign(
        manifest, root, launcher, on_wave, resume,
        controller_config=config, controller_command=command,
    )


def test_fixed_schedule_has_six_fresh_balanced_cells():
    cells = campaign.cells()
    assert len(cells) == 6
    assert [[c.benchmark for c in cells if c.wave == wave] for wave in range(1, 4)] == [
        ["gdn", "bsa"], ["gdn", "bsa"], ["gdn", "bsa"]
    ]
    assert {c.treatment for c in cells if c.benchmark == "gdn"} == set(
        campaign.TREATMENT_SKILLS
    )
    assert all(c.device == (0 if c.benchmark == "gdn" else 1) for c in cells)
    assert all(c.rounds == 3 and c.request_budget == 12 for c in cells)


def test_prepare_cell_copies_exact_inputs_and_treatment_skills(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = next(c for c in manifest["cells"] if c["treatment"] == "project-cannbot")
    sandbox = campaign.prepare_cell(manifest, cell, tmp_path / "runs")
    metadata = campaign.preflight(manifest, sandbox)
    assert metadata["prompt_sha256"] == manifest["prompt"]["sha256"]
    assert metadata["baseline_sha256"] == manifest["baselines"][cell["benchmark"]]["sha256"]
    assert set(metadata["skills"]) == {
        *campaign.CANNBOT_TRITON_SKILLS,
        *campaign.CANNBOT_DEPENDENCIES,
        "ascend-profiling",
    }
    assert "ops-profiling" not in metadata["skills"]
    assert (
        sandbox / "workspace" / ".agents" / "plugins-official"
        / "triton-op-generator"
    ).is_dir()
    assert (sandbox / "workspace" / ".agents" / "skills").is_dir()
    assert (sandbox / "workspace" / "AGENTS.md").read_text() == "cannbot rules"
    assert (sandbox / "workspace" / "config.json").read_text() == "{}"
    assert not (sandbox / ".agents").exists()
    assert not any(path.is_symlink() for path in sandbox.rglob("*"))


def test_prepare_rejects_reused_session_folder(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = manifest["cells"][0]
    campaign.prepare_cell(manifest, cell, tmp_path / "runs")
    with pytest.raises(campaign.CampaignError, match="fresh sandbox"):
        campaign.prepare_cell(manifest, cell, tmp_path / "runs")


def test_preflight_rejects_wrong_skill_and_mutated_inputs(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = next(c for c in manifest["cells"] if c["treatment"] == "project-only")
    sandbox = campaign.prepare_cell(manifest, cell, tmp_path / "runs")
    tree(sandbox / "workspace" / ".agents" / "skills" / "ops-profiling", "leak")
    with pytest.raises(campaign.CampaignError, match="skill isolation mismatch"):
        campaign.preflight(manifest, sandbox)
    (sandbox / "workspace" / ".agents" / "skills" / "ops-profiling" / "SKILL.md").unlink()
    (sandbox / "workspace" / ".agents" / "skills" / "ops-profiling").rmdir()
    (sandbox / "PROMPT.md").write_text("different")
    with pytest.raises(campaign.CampaignError, match="prompt hash mismatch"):
        campaign.preflight(manifest, sandbox)


def test_copy_regular_tree_rejects_symlinks(tmp_path: Path):
    source = tree(tmp_path / "source")
    (source / "escape").symlink_to(tmp_path)
    with pytest.raises(campaign.CampaignError, match="symlink forbidden"):
        campaign.copy_regular_tree(source, tmp_path / "target")


def test_copy_preserves_executable_mode_and_digest_detects_mode_drift(tmp_path: Path):
    source = tree(tmp_path / "source")
    script = source / "collect_profile.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    expected = campaign.digest_tree(source)
    destination = tmp_path / "destination"
    campaign.copy_regular_tree(source, destination)
    copied = destination / script.name
    assert os.access(copied, os.X_OK)
    assert campaign.digest_tree(destination) == expected
    copied.chmod(0o644)
    assert campaign.digest_tree(destination) != expected


class RecordingLauncher:
    def __init__(self):
        self.calls = []

    def launch(self, sandbox, cell):
        self.calls.append((sandbox, cell.copy()))
        return {"exit_code": 0, "rounds_completed": 3,
                "session_id": f"session-{cell['cell_id']}"}


def test_campaign_runs_fixed_waves_and_writes_structured_ledger(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    launcher = RecordingLauncher()
    waves = []
    run_root = tmp_path / "runs"
    run_root.mkdir()
    ledger = run_campaign(manifest, run_root, launcher, waves.append)
    assert waves == [1, 2, 3]
    assert sorted(call[1]["wave"] for call in launcher.calls) == [1, 1, 2, 2, 3, 3]
    assert len({call[0] for call in launcher.calls}) == 6
    assert ledger["status"] == "complete"
    assert json.loads((run_root / "ledger.json").read_text()) == ledger


def test_ledger_checkpoints_completed_cells_and_failure(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class FailingLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            if cell["cell_id"] == "gdn-project-cannbot":
                raise RuntimeError("launcher failed")
            return super().launch(sandbox, cell)

    run_root = tmp_path / "runs"
    run_root.mkdir()
    result = run_campaign(manifest, run_root, FailingLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert result["status"] == "needs_reschedule"
    failed = next(entry for entry in ledger["cells"]
                  if entry["cell"]["cell_id"] == "gdn-project-cannbot")
    assert failed["result"]["failure_type"] == "launcher_exception"
    assert "RuntimeError: launcher failed" in failed["result"]["diagnostics"]
    assert ledger["reschedule"] == ["gdn-project-cannbot"]
    assert len(ledger["cells"]) == 6


def test_incomplete_launcher_result_is_evidence_then_fails_ledger(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class IncompleteLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            return {"exit_code": 9, "rounds_completed": 1, "stderr": "transport failed"}

    run_root = tmp_path / "runs"
    result = run_campaign(manifest, run_root, IncompleteLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert result["status"] == "needs_reschedule"
    assert len(ledger["cells"]) == 6
    assert all(entry["result"]["stderr"] == "transport failed" for entry in ledger["cells"])


def test_ledger_persists_interrupted_status(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class InterruptedLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            if cell["cell_id"] == "gdn-cannbot":
                raise KeyboardInterrupt()
            return super().launch(sandbox, cell)

    run_root = tmp_path / "runs"
    with pytest.raises(KeyboardInterrupt):
        run_campaign(manifest, run_root, InterruptedLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert ledger["status"] == "interrupted"
    assert ledger["failure"]["type"] == "KeyboardInterrupt"


def test_between_wave_source_drift_is_rejected_and_checkpointed(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    project = Path(manifest["skill_sources"]["ascend-profiling"]["path"])

    def mutate_before_wave(wave):
        if wave == 2:
            (project / "SKILL.md").write_text("between-wave drift")

    run_root = tmp_path / "runs"
    run_root.mkdir()
    with pytest.raises(campaign.CampaignError, match="project profiling skill drifted"):
        run_campaign(manifest, run_root, RecordingLauncher(), mutate_before_wave)
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert ledger["status"] == "failed"
    assert len(ledger["cells"]) == 2
    assert all(entry["cell"]["wave"] == 1 for entry in ledger["cells"])


def test_all_treatments_expose_exact_skill_manifests(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    for cell in manifest["cells"]:
        sandbox = campaign.prepare_cell(manifest, cell, tmp_path / cell["cell_id"])
        visible = {
            path.name
            for path in (sandbox / "workspace" / ".agents" / "skills").iterdir()
        }
        assert visible == set(campaign.TREATMENT_SKILLS[cell["treatment"]])
        profilers = visible & {"ops-profiling", "ascend-profiling"}
        expected = "ops-profiling" if cell["treatment"] == "cannbot" else "ascend-profiling"
        assert profilers == {expected}


def test_cannbot_relative_skill_dependencies_resolve_in_workspace(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = next(c for c in manifest["cells"] if c["treatment"] == "cannbot")
    sandbox = campaign.prepare_cell(manifest, cell, tmp_path / "runs")
    designer = sandbox / "workspace" / ".agents" / "skills" / "triton-op-designer"
    assert (designer / "../npu-arch/references/hardware.md").resolve().is_file()
    assert (
        designer
        / "../../plugins-official/triton-op-generator/template/linear.md"
    ).resolve().is_file()


def test_prepare_rejects_project_and_cannbot_drift(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    project = Path(manifest["skill_sources"]["ascend-profiling"]["path"])
    (project / "SKILL.md").write_text("drift")
    with pytest.raises(campaign.CampaignError, match="project profiling skill drifted"):
        campaign.prepare_cell(manifest, manifest["cells"][0], tmp_path / "project-drift")

    manifest, _ = fixture(tmp_path / "second")
    frozen = Path(manifest["skill_sources"]["cannbot"]["path"])
    (frozen / "skills" / "triton-op-coding" / "SKILL.md").write_text("drift")
    with pytest.raises(campaign.CampaignError, match="frozen CANNBot skill drifted"):
        campaign.prepare_cell(manifest, manifest["cells"][0], tmp_path / "cannbot-drift")


def test_command_launcher_preserves_codex_home_and_uses_workspace(tmp_path: Path, monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("CODEX_HOME", "/real/authenticated/codex-home")
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    sandbox = tmp_path / "cell"
    (sandbox / "workspace").mkdir(parents=True)
    (sandbox / "PROMPT.md").write_text("identical experiment prompt\n")
    campaign.CommandLauncher(["codex", "exec"]).launch(sandbox, campaign.cells()[0].__dict__)
    assert observed["cwd"] == sandbox / "workspace"
    assert observed["env"]["CODEX_HOME"] == "/real/authenticated/codex-home"
    assert observed["input"] == (sandbox / "PROMPT.md").read_text()


def test_freeze_resolves_then_copies_only_regular_selected_skills(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    for name, relative in campaign.CANNBOT_SKILL_SOURCES.items():
        tree(source / relative, name)
    tree(source / campaign.CANNBOT_SUPPORT_SOURCE, "support")
    commit = "a" * 40
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["git", "clone"]:
            checkout = Path(argv[-1])
            campaign.copy_regular_tree(source, checkout)
        return type("Result", (), {"stdout": f"{commit}\trefs/heads/master\n"})()

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    output = tmp_path / "freeze"
    record = campaign.freeze_cannbot("https://example.invalid/cannbot.git", output)
    assert record["commit"] == commit
    assert set(record["skills"]) == set(campaign.CANNBOT_SKILL_SOURCES)
    assert (output / "skills" / "triton-op-coding" / "SKILL.md").read_text() == "triton-op-coding"
    assert (output / "support" / "triton-op-generator" / "SKILL.md").read_text() == "support"
    assert calls[0][0:2] == ["git", "ls-remote"]
    assert any("checkout" in call for call in calls)


def test_run_cli_accepts_structured_experimentctl_command_without_swallowing_options(
    tmp_path: Path, monkeypatch, capsys,
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    observed = {}

    class FakeProductionLauncher:
        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "production_launcher",
                        types.SimpleNamespace(ProductionLauncher=FakeProductionLauncher))
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda document, output, value, **kwargs: {"status": "dry-run"})
    command = ["python", "scripts/experimentctl.py", "--config", "/host/cells.json",
               "--cell", "{cell_id}"]
    monkeypatch.setattr(sys, "argv", ["campaign.py", "run", "--manifest", str(manifest),
                        "--output", str(tmp_path / "out"), "--controller-json",
                        json.dumps(command), "--codex", "fake-codex", "--dry-run"])
    assert campaign.main() == 0
    assert observed["command"] == command
    assert observed["kwargs"]["codex"] == "fake-codex"
    assert observed["kwargs"]["dry_run"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"


def test_generate_manifest_cli_writes_reproducible_inputs(tmp_path: Path, monkeypatch, capsys):
    manifest, _ = fixture(tmp_path / "source")
    output = tmp_path / "nested" / "campaign.json"
    controller_config = manifest["controller"]["config"]["path"]
    controller_command = [
        argument.replace("{controller_config}", controller_config)
        for argument in manifest["controller"]["command_argv"]
    ]
    monkeypatch.setattr(sys, "argv", [
        "campaign.py", "generate-manifest",
        "--prompt", manifest["prompt"]["path"],
        "--gdn-baseline", manifest["baselines"]["gdn"]["path"],
        "--bsa-baseline", manifest["baselines"]["bsa"]["path"],
        "--project-skill", manifest["skill_sources"]["ascend-profiling"]["path"],
        "--cannbot-freeze", manifest["skill_sources"]["cannbot"]["path"],
        "--controller-config", controller_config,
        "--controller-json", json.dumps(controller_command),
        "--output", str(output), "--rounds", "3", "--request-budget", "12",
    ])
    assert campaign.main() == 0
    written = json.loads(output.read_text())
    assert written == json.loads(capsys.readouterr().out)
    assert len(written["cells"]) == 6
    sandbox = campaign.prepare_cell(written, written["cells"][0], tmp_path / "runs")
    assert campaign.preflight(written, sandbox)["cell"]["cell_id"] == "gdn-cannbot"


def test_manifest_rejects_nonpositive_campaign_limits(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    with pytest.raises(campaign.CampaignError, match="must be positive"):
        campaign.write_manifest(
            tmp_path / "bad.json", Path(manifest["prompt"]["path"]),
            {name: Path(value["path"]) for name, value in manifest["baselines"].items()},
            Path(manifest["skill_sources"]["ascend-profiling"]["path"]),
            Path(manifest["skill_sources"]["cannbot"]["path"]),
            Path(manifest["controller"]["config"]["path"]),
            [argument.replace("{controller_config}", manifest["controller"]["config"]["path"])
             for argument in manifest["controller"]["command_argv"]], rounds=0,
        )


def test_infrastructure_cell_is_retained_for_explicit_reschedule(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class InfrastructureLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            return {"status": "infrastructure_error", "attempt_id": "kept",
                    "terminal_evidence": {"check": {"handles": ["gz-a3:durable"]}}}

    result = run_campaign(manifest, tmp_path / "runs", InfrastructureLauncher())
    ledger = json.loads((tmp_path / "runs" / "ledger.json").read_text())
    assert result["status"] == "needs_reschedule"
    assert set(ledger["reschedule"]) == {cell["cell_id"] for cell in manifest["cells"]}
    assert all(entry["result"]["attempt_id"] == "kept" for entry in ledger["cells"])


def test_dry_run_checkpoints_all_cells_without_claiming_completion(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class DryRunLauncher(RecordingLauncher):
        dry_run = True
        def launch(self, sandbox, cell):
            return {"status": "dry_run", "dry_run": True, "rounds_completed": 0,
                    "attempt_id": f"plan-{cell['cell_id']}"}

    ledger = run_campaign(manifest, tmp_path / "runs", DryRunLauncher())
    assert ledger["status"] == "dry_run"
    assert len(ledger["cells"]) == 6
    assert all(entry["result"]["dry_run"] for entry in ledger["cells"])
    assert not (tmp_path / "runs" / "attempts").exists()
    production = run_campaign(manifest, tmp_path / "runs", RecordingLauncher())
    assert production["status"] == "complete"
    assert (tmp_path / "runs" / "attempts").is_dir()


def test_candidate_failure_is_counted_and_does_not_abort_later_waves(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class CandidateLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            if cell["cell_id"] == "gdn-cannbot":
                return {"status": "candidate_error", "failure_type": "compile_error"}
            return {"status": "complete", "rounds_completed": 3}

    ledger = run_campaign(manifest, tmp_path / "runs", CandidateLauncher())
    assert ledger["status"] == "completed_with_candidate_failures"
    assert len(ledger["cells"]) == 6
    assert "reschedule" not in ledger or not ledger["reschedule"]


def test_resume_runs_only_infrastructure_cells_in_fresh_sandbox(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    failed_id = "gdn-project-cannbot"

    class FirstLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            self.calls.append((sandbox, cell.copy()))
            if cell["cell_id"] == failed_id:
                return {"status": "infrastructure_error", "diagnostics": "flaky"}
            return {"status": "complete"}

    first = FirstLauncher()
    root = tmp_path / "runs"
    ledger = run_campaign(manifest, root, first)
    assert ledger["status"] == "needs_reschedule"
    original_sandbox = next(path for path, cell in first.calls if cell["cell_id"] == failed_id)

    second = RecordingLauncher()
    ledger = run_campaign(manifest, root, second, resume=True)
    assert ledger["status"] == "complete"
    assert [cell["cell_id"] for _, cell in second.calls] == [failed_id]
    assert second.calls[0][0] != original_sandbox
    assert len(ledger["cells"]) == 7
    assert not ledger["reschedule"]


@pytest.mark.parametrize("drift", ["config", "command"])
def test_controller_drift_is_rejected_before_prepare_or_launch(tmp_path: Path, drift: str):
    manifest, _ = fixture(tmp_path)
    binding = manifest["controller"]
    config = Path(binding["config"]["path"])
    command = [argument.replace("{controller_config}", str(config))
               for argument in binding["command_argv"]]
    if drift == "config":
        config.write_text('{"version":2}\n')
    else:
        command[0] = "python-different"
    launcher = RecordingLauncher()
    root = tmp_path / "runs"
    with pytest.raises(campaign.CampaignError, match="controller (config drifted|command)"):
        campaign.run_campaign(
            manifest, root, launcher, controller_config=config,
            controller_command=command,
        )
    assert launcher.calls == []
    assert not (root / "attempts").exists()


def test_controller_binding_replays_from_relocated_config_and_rejects_resume_drift(tmp_path: Path):
    manifest, _ = fixture(tmp_path / "source")
    source = Path(manifest["controller"]["config"]["path"])
    relocated = tmp_path / "relocated" / "cells.json"
    relocated.parent.mkdir()
    relocated.write_bytes(source.read_bytes())
    command = [argument.replace("{controller_config}", str(relocated.resolve()))
               for argument in manifest["controller"]["command_argv"]]

    class OneFlake(RecordingLauncher):
        def launch(self, sandbox, cell):
            result = super().launch(sandbox, cell)
            if cell["cell_id"] == "gdn-cannbot":
                result["status"] = "infrastructure_error"
            return result

    root = tmp_path / "runs"
    ledger = campaign.run_campaign(
        manifest, root, OneFlake(), controller_config=relocated,
        controller_command=command,
    )
    assert ledger["controller"]["config_sha256"] == campaign.digest_file(relocated)
    assert ledger["controller"]["command_argv"] == manifest["controller"]["command_argv"]
    assert str(relocated) not in json.dumps(ledger["controller"])

    relocated.write_text("drift\n")
    launcher = RecordingLauncher()
    with pytest.raises(campaign.CampaignError, match="controller config drifted"):
        campaign.run_campaign(
            manifest, root, launcher, resume=True, controller_config=relocated,
            controller_command=command,
        )
    assert launcher.calls == []
