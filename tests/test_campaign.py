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
    manifest_path = tmp_path / "campaign.json"
    manifest = campaign.write_manifest(
        manifest_path, prompt, {"gdn": gdn, "bsa": bsa}, project, frozen
    )
    return manifest, manifest_path


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
    ledger = campaign.run_campaign(manifest, run_root, launcher, waves.append)
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
    with pytest.raises(RuntimeError, match="launcher failed"):
        campaign.run_campaign(manifest, run_root, FailingLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert ledger["status"] == "failed"
    assert ledger["failure"]["type"] == "RuntimeError"
    assert {entry["cell"]["wave"] for entry in ledger["cells"]} == {1, 2}
    assert len(ledger["cells"]) == 3


def test_incomplete_launcher_result_is_evidence_then_fails_ledger(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class IncompleteLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            return {"exit_code": 9, "rounds_completed": 1, "stderr": "transport failed"}

    run_root = tmp_path / "runs"
    with pytest.raises(campaign.CampaignError, match="did not complete three"):
        campaign.run_campaign(manifest, run_root, IncompleteLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert ledger["status"] == "failed"
    assert len(ledger["cells"]) == 2
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
        campaign.run_campaign(manifest, run_root, InterruptedLauncher())
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
        campaign.run_campaign(manifest, run_root, RecordingLauncher(), mutate_before_wave)
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
    monkeypatch.setattr(campaign, "run_campaign", lambda document, output, value: {"status": "dry-run"})
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
