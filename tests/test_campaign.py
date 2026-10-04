from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
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


def fixture(tmp_path: Path, request_budget: int = 18,
            benchmarks: tuple[str, ...] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_bytes((ROOT / "prompts/kernel-optimization.md").read_bytes())
    gdn = tree(tmp_path / "gdn", "gdn")
    bsa = tree(tmp_path / "bsa", "bsa")
    matmul = tree(tmp_path / "matmul", "matmul")
    project = tree(tmp_path / "project", "project")
    guarded = tree(tmp_path / "guarded", "guarded")
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
    controller_config.write_text('{"schema_version":1,"cells":{}}\n')
    controller_command = [
        sys.executable, str(ROOT / "scripts" / "experimentctl.py"), "--config",
        str(controller_config.resolve()), "--cell", "{cell_id}",
    ]
    manifest_path = tmp_path / "campaign.json"
    manifest = campaign.write_manifest(
        manifest_path, prompt, {"gdn": gdn, "bsa": bsa, "matmul": matmul},
        project, guarded, frozen,
        controller_config, controller_command,
        request_budget=request_budget,
        guarded_revision="b" * 40,
        benchmarks=benchmarks,
    )
    return manifest, manifest_path


def run_campaign(manifest, root, launcher, on_wave=None, resume=False):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest["prompt"]["path"]).parent / "campaign.json"
    command, evidence = campaign.stage_controller(
        manifest, manifest_path, root, sys.executable, resume=resume,
    )
    return campaign.run_campaign(
        manifest, root, launcher, on_wave, resume,
        controller_evidence=evidence, controller_bundle=root / "controller",
    )


def test_fixed_schedule_has_nine_cells_in_collision_free_four_four_one_waves():
    cells = campaign.cells()
    assert len(cells) == 9
    assert [sum(c.wave == wave for c in cells) for wave in range(1, 4)] == [4, 4, 1]
    for wave in range(1, 4):
        devices = [c.device for c in cells if c.wave == wave]
        assert len(devices) == len(set(devices))
    assert all({c.treatment for c in cells if c.benchmark == benchmark}
               == set(campaign.TREATMENT_SKILLS)
               for benchmark in campaign.BENCHMARK_DEVICE)
    assert {(c.benchmark, c.treatment): c.device for c in cells} == campaign.CELL_DEVICE
    assert all(c.rounds == 3 and c.request_budget == 18 for c in cells)


def test_gdn_slice_is_one_concurrent_wave_with_three_treatments():
    selected = campaign.cells(benchmarks=("gdn",))
    assert [cell.cell_id for cell in selected] == [
        "gdn-cannbot", "gdn-project-cannbot", "gdn-project-guarded",
    ]
    assert [cell.device for cell in selected] == [0, 1, 2]
    assert {cell.wave for cell in selected} == {1}
    assert {cell.treatment for cell in selected} == set(campaign.TREATMENT_SKILLS)


def test_gdn_manifest_calibrates_only_participating_devices(tmp_path: Path):
    manifest, _ = fixture(tmp_path, benchmarks=("gdn",))
    assert [cell["cell_id"] for cell in manifest["cells"]] == [
        "gdn-cannbot", "gdn-project-cannbot", "gdn-project-guarded",
    ]
    assert manifest["calibration"] == {
        "devices": [0, 1, 2], "case": 7,
        "selector": "streaming_matmul_add_kernel_mix_aic",
        "max_drift_fraction": 0.10,
    }


def test_manifest_binds_agent_help_and_budget_and_rejects_drift(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    interface = manifest["controller"]["agent_interface"]
    assert interface["request_budget"] == 18
    assert interface["help"]["operation"] == "help"
    assert interface["help"]["billed"] is False
    assert len(interface["help_sha256"]) == 64
    assert manifest["controller"]["agent_model"] == {
        "name": "gpt-5.6-sol", "reasoning_effort": "low",
    }
    campaign.verify_controller_schema(manifest)

    interface["help"]["usage"] += "drift"
    with pytest.raises(campaign.CampaignError, match="help hash"):
        campaign.verify_controller_schema(manifest)


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


def test_canonical_prompt_is_identical_in_every_prepared_cell_and_manifest_bound(
    tmp_path: Path,
):
    manifest, _ = fixture(tmp_path / "source", request_budget=18)
    canonical = ROOT / "prompts/kernel-optimization.md"

    expected = canonical.read_bytes()
    assert manifest["prompt"]["sha256"] == campaign.digest_file(canonical)
    assert {cell["request_budget"] for cell in manifest["cells"]} == {18}
    interface = manifest["controller"]["agent_interface"]
    encoded_help = json.dumps(
        interface["help"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert interface["request_budget"] == 18
    assert interface["help"]["billed"] is False
    assert interface["help_sha256"] == hashlib.sha256(encoded_help).hexdigest()

    delivered = []
    for cell in manifest["cells"]:
        sandbox = campaign.prepare_cell(manifest, cell, tmp_path / "runs")
        metadata = campaign.preflight(manifest, sandbox)
        delivered.append((sandbox / "PROMPT.md").read_bytes())
        assert metadata["prompt_sha256"] == manifest["prompt"]["sha256"]

    assert delivered == [expected] * 9


def test_prepare_rejects_reused_session_folder(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = manifest["cells"][0]
    campaign.prepare_cell(manifest, cell, tmp_path / "runs")
    with pytest.raises(campaign.CampaignError, match="fresh sandbox"):
        campaign.prepare_cell(manifest, cell, tmp_path / "runs")


def test_preflight_rejects_wrong_skill_and_mutated_inputs(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    cell = next(c for c in manifest["cells"] if c["treatment"] == "project-guarded")
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
    assert sorted(call[1]["wave"] for call in launcher.calls) == [1, 1, 1, 1, 2, 2, 2, 2, 3]
    assert len({call[0] for call in launcher.calls}) == 9
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
    assert len(ledger["cells"]) == 9


def test_incomplete_launcher_result_is_evidence_then_fails_ledger(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class IncompleteLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            return {"exit_code": 9, "rounds_completed": 1, "stderr": "transport failed"}

    run_root = tmp_path / "runs"
    result = run_campaign(manifest, run_root, IncompleteLauncher())
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert result["status"] == "needs_reschedule"
    assert len(ledger["cells"]) == 9
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
    with pytest.raises(campaign.CampaignError, match="project skill drifted.*ascend-profiling"):
        run_campaign(manifest, run_root, RecordingLauncher(), mutate_before_wave)
    ledger = json.loads((run_root / "ledger.json").read_text())
    assert ledger["status"] == "failed"
    assert len(ledger["cells"]) == 4
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
    with pytest.raises(campaign.CampaignError, match="project skill drifted.*ascend-profiling"):
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


def test_run_cli_uses_private_frozen_controller_bundle(
    tmp_path: Path, monkeypatch, capsys,
):
    document, manifest = fixture(tmp_path / "source")
    observed = {}

    class FakeProductionLauncher:
        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "production_launcher",
                        types.SimpleNamespace(ProductionLauncher=FakeProductionLauncher))
    monkeypatch.setattr(campaign, "run_campaign",
                        lambda document, output, value, **kwargs: {"status": "dry-run"})
    monkeypatch.setattr(sys, "argv", ["campaign.py", "run", "--manifest", str(manifest),
                        "--output", str(tmp_path / "out"), "--python", sys.executable,
                        "--codex", "fake-codex", "--dry-run"])
    assert campaign.main() == 0
    assert observed["command"][1] == str((tmp_path / "out/controller/scripts/experimentctl.py").resolve())
    assert observed["command"][3] == str((tmp_path / "out/controller/controller.json").resolve())
    assert observed["command"][0] == campaign.shutil.which(sys.executable)
    assert observed["kwargs"]["codex"] == "fake-codex"
    assert observed["kwargs"]["dry_run"] is True
    assert observed["kwargs"]["agent_interface"] == document["controller"]["agent_interface"]
    assert observed["kwargs"]["agent_model"] == document["controller"]["agent_model"]
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"


def test_generate_manifest_cli_writes_reproducible_inputs(tmp_path: Path, monkeypatch, capsys):
    manifest, _ = fixture(tmp_path / "source")
    output = tmp_path / "nested" / "campaign.json"
    controller_config = tmp_path / "source/campaign.controller/controller.json"
    controller_command = [sys.executable, str(ROOT / "scripts/experimentctl.py"),
                          "--config", str(controller_config.resolve()),
                          "--cell", "{cell_id}"]
    monkeypatch.setattr(sys, "argv", [
        "campaign.py", "generate-manifest",
        "--prompt", manifest["prompt"]["path"],
        "--gdn-baseline", manifest["baselines"]["gdn"]["path"],
        "--bsa-baseline", manifest["baselines"]["bsa"]["path"],
        "--matmul-baseline", manifest["baselines"]["matmul"]["path"],
        "--project-skill", manifest["skill_sources"]["ascend-profiling"]["path"],
        "--guarded-skill", manifest["skill_sources"]["triton-guarded-kernel"]["path"],
        "--guarded-skill-revision", "b" * 40,
        "--cannbot-freeze", manifest["skill_sources"]["cannbot"]["path"],
        "--controller-config", str(controller_config),
        "--controller-json", json.dumps(controller_command),
        "--output", str(output), "--rounds", "3", "--request-budget", "18",
    ])
    assert campaign.main() == 0
    written = json.loads(output.read_text())
    assert written == json.loads(capsys.readouterr().out)
    assert written["version"] == 2
    assert not any(str(tmp_path) in json.dumps(value)
                   for value in [written["controller"]])
    assert (output.parent / written["controller"]["bundle"] / "scripts/benchmark_backend.py").is_file()
    assert (output.parent / written["controller"]["bundle"] / "benchmarks/gdn/baseline.py").is_file()
    assert (output.parent / written["controller"]["bundle"] / "benchmarks/matmul/cases.jsonl").is_file()
    assert len(written["cells"]) == 9
    sandbox = campaign.prepare_cell(written, written["cells"][0], tmp_path / "runs")
    assert campaign.preflight(written, sandbox)["cell"]["cell_id"] == "gdn-cannbot"


@pytest.mark.parametrize("value", [{}, 7, "command", [], ["python", 3], [""]])
def test_generate_manifest_cli_rejects_non_string_array_controller_json(
    tmp_path: Path, monkeypatch, capsys, value,
):
    monkeypatch.setattr(sys, "argv", [
        "campaign.py", "generate-manifest", "--prompt", "missing-prompt",
        "--gdn-baseline", "missing-gdn", "--bsa-baseline", "missing-bsa",
        "--matmul-baseline", "missing-matmul",
        "--project-skill", "missing-skill", "--guarded-skill", "missing-guarded",
        "--guarded-skill-revision", "b" * 40,
        "--cannbot-freeze", "missing-freeze",
        "--controller-config", "missing-config", "--controller-json", json.dumps(value),
        "--output", str(tmp_path / "campaign.json"),
    ])
    with pytest.raises(SystemExit) as raised:
        campaign.main()
    assert raised.value.code == 2
    assert "--controller-json must be a JSON string array" in capsys.readouterr().err
    assert not (tmp_path / "campaign.json").exists()


@pytest.mark.parametrize(("rounds", "budget"), [(0, 18), (3, 0), (3, 12), (2, 18)])
def test_manifest_rejects_nonproduction_campaign_limits(tmp_path: Path, rounds, budget):
    manifest, _ = fixture(tmp_path)
    with pytest.raises(campaign.CampaignError, match="three rounds and an 18-request budget"):
        campaign.write_manifest(
            tmp_path / "bad.json", Path(manifest["prompt"]["path"]),
            {name: Path(value["path"]) for name, value in manifest["baselines"].items()},
            Path(manifest["skill_sources"]["ascend-profiling"]["path"]),
            Path(manifest["skill_sources"]["triton-guarded-kernel"]["path"]),
            Path(manifest["skill_sources"]["cannbot"]["path"]),
            tmp_path / "controller.json", [], rounds=rounds, request_budget=budget,
            guarded_revision="b" * 40,
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


def test_each_concurrent_infrastructure_result_checkpoints_its_reschedule(
        tmp_path: Path, monkeypatch):
    manifest, _ = fixture(tmp_path)
    snapshots = []
    original_replace = campaign.Path.replace

    def capture_replace(path, target):
        snapshots.append(json.loads(path.read_text()))
        return original_replace(path, target)

    class InfrastructureLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            return {"status": "infrastructure_error", "attempt_id": cell["cell_id"]}

    monkeypatch.setattr(campaign.Path, "replace", capture_replace)
    run_campaign(manifest, tmp_path / "runs", InfrastructureLauncher())

    first_wave = {cell["cell_id"] for cell in manifest["cells"] if cell["wave"] == 1}
    after_last_future = next(
        snapshot for snapshot in snapshots
        if snapshot["status"] == "running"
        and {entry["cell"]["cell_id"] for entry in snapshot["cells"]} == first_wave
    )
    assert set(after_last_future["reschedule"]) == first_wave


def test_dry_run_checkpoints_all_cells_without_claiming_completion(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class DryRunLauncher(RecordingLauncher):
        dry_run = True
        def launch(self, sandbox, cell):
            return {"status": "dry_run", "dry_run": True, "rounds_completed": 0,
                    "attempt_id": f"plan-{cell['cell_id']}"}

    ledger = run_campaign(manifest, tmp_path / "runs", DryRunLauncher())
    assert ledger["status"] == "dry_run"
    assert len(ledger["cells"]) == 9
    assert all(entry["result"]["dry_run"] for entry in ledger["cells"])
    assert not (tmp_path / "runs" / "attempts").exists()
    production = run_campaign(manifest, tmp_path / "runs", RecordingLauncher())
    assert production["status"] == "complete"
    assert (tmp_path / "runs" / "attempts").is_dir()


def test_dry_run_infrastructure_failure_takes_precedence(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    failed_id = "bsa-project-cannbot"

    class MixedDryRunLauncher(RecordingLauncher):
        dry_run = True
        def launch(self, sandbox, cell):
            if cell["cell_id"] == failed_id:
                return {"status": "infrastructure_error", "diagnostics": "DNS failed"}
            return {"status": "dry_run", "dry_run": True, "rounds_completed": 0}

    ledger = run_campaign(manifest, tmp_path / "runs", MixedDryRunLauncher())

    assert ledger["status"] == "needs_reschedule"
    assert ledger["reschedule"] == [failed_id]


def test_candidate_failure_is_counted_and_does_not_abort_later_waves(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class CandidateLauncher(RecordingLauncher):
        def launch(self, sandbox, cell):
            if cell["cell_id"] == "gdn-cannbot":
                return {"status": "candidate_error", "failure_type": "compile_error"}
            return {"status": "complete", "rounds_completed": 3}

    ledger = run_campaign(manifest, tmp_path / "runs", CandidateLauncher())
    assert ledger["status"] == "completed_with_candidate_failures"
    assert len(ledger["cells"]) == 9
    assert "reschedule" not in ledger or not ledger["reschedule"]
    failed = next(entry for entry in ledger["cells"]
                  if entry["cell"]["cell_id"] == "gdn-cannbot")
    assert failed["outcome"] == "compile_error"


def test_performance_is_normalized_by_bracketing_calibration(tmp_path: Path):
    result = {
        "status": "complete", "device": 1,
        "calibration": {"devices": {
            "0": {"before": {"latency_us": 16.0}, "after": {"latency_us": 25.0}},
            "1": {"before": {"latency_us": 4.0}, "after": {"latency_us": 9.0}},
        }},
        "terminal_evidence": {"profile": {"result": {"geomean_us": 12.0}}},
    }
    normalized = campaign.normalize_performance(result)
    assert normalized is not None
    assert normalized["reference_us"] == pytest.approx(6.0)
    assert normalized["canonical_reference_us"] == pytest.approx(20.0)
    assert normalized["normalized_latency_us"] == pytest.approx(40.0)


def test_wave_calibrates_all_devices_around_agents_and_retains_records(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    events = []

    class CalibratingLauncher(RecordingLauncher):
        def calibrate(self, sandbox, cell, phase, wave, attempt_id):
            events.append((phase, wave, cell["device"], attempt_id))
            latency = 10.0 + cell["device"]
            if phase == "after":
                latency *= 1.05
            return {
                "status": "complete", "timestamp": f"{phase}-{wave}-{cell['device']}",
                "evidence_path": f"/{phase}-{wave}-{cell['device']}.json",
                "result": {"latency_us": latency,
                           "handles": [f"gz-a3:{phase}-{wave}-{cell['device']}"],
                           "selector": "streaming_matmul_add_kernel_mix_aic"},
            }

        def launch(self, sandbox, cell):
            events.append(("launch", cell["wave"], cell["device"]))
            return {
                "status": "complete", "rounds_completed": 3,
                "terminal_evidence": {
                    "profile": {"result": {"geomean_us": 20.0}}
                },
            }

    ledger = run_campaign(manifest, tmp_path / "runs", CalibratingLauncher())
    for wave in (1, 2, 3):
        positions = [index for index, event in enumerate(events) if event[1] == wave]
        wave_events = [events[index][0] for index in positions]
        assert wave_events[:4] == ["before"] * 4
        assert wave_events[-4:] == ["after"] * 4
        assert {event[2] for event in events if event[:2] == ("before", wave)} == set(range(4))
    assert set(ledger["calibrations"]) == {"1", "2", "3"}
    assert all(entry["outcome"] == "success" for entry in ledger["cells"])
    assert all("normalized_performance" in entry for entry in ledger["cells"])


def test_calibration_drift_invalidates_whole_wave_as_infrastructure(tmp_path: Path):
    manifest, _ = fixture(tmp_path)

    class DriftingLauncher(RecordingLauncher):
        def calibrate(self, sandbox, cell, phase, wave, attempt_id):
            latency = 10.0 if phase == "before" else 12.0
            return {"status": "complete", "timestamp": phase, "evidence_path": "/evidence",
                    "result": {"latency_us": latency, "handles": ["gz-a3:cal"],
                               "selector": "streaming_matmul_add_kernel_mix_aic"}}

    ledger = run_campaign(manifest, tmp_path / "runs", DriftingLauncher())
    assert ledger["status"] == "needs_reschedule"
    assert set(ledger["reschedule"]) == {cell["cell_id"] for cell in manifest["cells"]}
    assert all(entry["outcome"] == "infra_discarded" for entry in ledger["cells"])
    assert all("reused a durable handle" in entry["result"]["diagnostics"]
               for entry in ledger["cells"])


def test_resume_uses_fresh_calibration_attempt_identity(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    failed_id = "gdn-project-cannbot"
    identities = []

    class CalibratingLauncher(RecordingLauncher):
        def __init__(self, fail=False):
            super().__init__()
            self.fail = fail

        def calibrate(self, sandbox, cell, phase, wave, attempt_id):
            identities.append((phase, wave, attempt_id))
            return {
                "status": "complete", "timestamp": phase,
                "evidence_path": f"/{attempt_id}-{phase}.json",
                "result": {
                    "latency_us": 10.0,
                    "handles": [f"gz-a3:{attempt_id}-{phase}-{cell['device']}"],
                    "selector": "streaming_matmul_add_kernel_mix_aic",
                },
            }

        def launch(self, sandbox, cell):
            if self.fail and cell["cell_id"] == failed_id:
                return {"status": "infrastructure_error", "diagnostics": "flaky"}
            return {"status": "complete", "rounds_completed": 3}

    root = tmp_path / "runs"
    first = run_campaign(manifest, root, CalibratingLauncher(fail=True))
    assert first["reschedule"] == [failed_id]
    first_ids = {item[2] for item in identities}
    identities.clear()

    second = run_campaign(manifest, root, CalibratingLauncher(), resume=True)
    assert second["status"] == "complete"
    second_ids = {item[2] for item in identities}
    assert first_ids.isdisjoint(second_ids)
    assert len(second_ids) == 1
    assert {item[0] for item in identities} == {"before", "after"}


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        ({"status": "complete"}, "success"),
        ({"status": "infrastructure_error"}, "infra_discarded"),
        ({"status": "candidate_error", "failure_type": "runtime_error"},
         "runtime_error"),
        ({"status": "candidate_error", "terminal_evidence": {
            "check": {"result": {"failure_type": "correctness_error"}}
        }}, "correctness_error"),
    ],
)
def test_outcome_taxonomy(result, outcome):
    assert campaign.classify_outcome(result) == outcome


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
    assert len(ledger["cells"]) == 10
    assert not ledger["reschedule"]


def test_production_rejects_legacy_unbound_manifest_before_launch(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    manifest.pop("controller")
    manifest["version"] = 1
    launcher = RecordingLauncher()
    with pytest.raises(campaign.CampaignError, match="version 2 controller-bound"):
        campaign.run_campaign(manifest, tmp_path / "runs", launcher)
    assert launcher.calls == []


def test_private_bundle_executes_after_relocation_and_ignores_source_mutation(tmp_path: Path):
    manifest, manifest_path = fixture(tmp_path / "source")
    root = tmp_path / "relocated-run"
    root.mkdir()
    command, evidence = campaign.stage_controller(
        manifest, manifest_path, root, sys.executable,
    )
    help_result = __import__("subprocess").run(
        [command[0], command[1], "--help"], capture_output=True, text=True,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout

    source_config = manifest_path.parent / manifest["controller"]["bundle"] / "controller.json"
    def mutate_source(wave):
        if wave == 2:
            source_config.write_text("source changed after staging\n")

    ledger = campaign.run_campaign(
        manifest, root, RecordingLauncher(), mutate_source,
        controller_evidence=evidence, controller_bundle=root / "controller",
    )
    assert ledger["status"] == "complete"
    assert ledger["controller"]["command_argv"] == manifest["controller"]["command_argv"]
    assert ledger["controller"]["agent_interface"] == manifest["controller"]["agent_interface"]
    assert ledger["controller"]["agent_model"] == manifest["controller"]["agent_model"]
    assert str(tmp_path) not in json.dumps(ledger["controller"])


def test_controller_bundle_rewrites_backend_to_private_runtime(tmp_path: Path):
    client = [sys.executable, str(ROOT / "scripts/gz_a3_job_client.py"),
              "--adapter-json", '["/approved/adapter"]',
              "--state-dir", "/private/job-state"]
    config = tmp_path / "cells.json"
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": [
        sys.executable, str(ROOT / "scripts/benchmark_backend.py"), "--benchmark", "gdn",
        "--job-client-json", json.dumps(client),
    ]}}}}))
    output = tmp_path / "campaign.json"
    binding = campaign.freeze_controller_bundle(
        output, config,
        [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
         str(config.resolve()), "--cell", "{cell_id}"],
    )
    bundled = json.loads((tmp_path / binding["bundle"] / "controller.json").read_text())
    assert bundled["cells"]["cell"]["backend"]["command"][:2] == [
        "{python}", "{bundle}/scripts/benchmark_backend.py"
    ]
    command = bundled["cells"]["cell"]["backend"]["command"]
    frozen_client = json.loads(command[command.index("--job-client-json") + 1])
    assert frozen_client[:2] == ["{python}", "{bundle}/scripts/gz_a3_job_client.py"]
    assert frozen_client[2:] == client[2:]
    assert str(ROOT) not in json.dumps(bundled)


def test_controller_bundle_freezes_bz_client_and_placements(tmp_path: Path):
    placements = tmp_path / "placements.json"
    placements.write_text(json.dumps({
        "0": {"profile": "bz-a3-1", "device": 0},
        "1": {"profile": "bz-a3-1", "device": 1},
        "2": {"profile": "bz-a3-2", "device": 2},
    }))
    client = [
        sys.executable, str(ROOT / "scripts/bz_a3_job_client.py"),
        "--state-dir", str(tmp_path / "job-state"),
        "--placements-json", str(placements),
        "--adapter-json", '["/approved/adapter"]',
        "--remote-root", "/frozen/remote/root",
    ]
    config = tmp_path / "cells.json"
    config.write_text(json.dumps({"cells": {"gdn-cannbot": {"backend": {"command": [
        sys.executable, str(ROOT / "scripts/benchmark_backend.py"), "--benchmark", "gdn",
        "--job-client-json", json.dumps(client),
    ]}}}}))
    output = tmp_path / "campaign.json"
    binding = campaign.freeze_controller_bundle(
        output, config,
        [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
         str(config.resolve()), "--cell", "{cell_id}"],
    )
    bundle = tmp_path / binding["bundle"]
    bundled = json.loads((bundle / "controller.json").read_text())
    command = bundled["cells"]["gdn-cannbot"]["backend"]["command"]
    frozen_client = json.loads(command[command.index("--job-client-json") + 1])
    assert frozen_client[1] == "{bundle}/scripts/bz_a3_job_client.py"
    assert frozen_client[frozen_client.index("--placements-json") + 1] == \
        "{bundle}/placements.json"
    assert frozen_client[frozen_client.index("--state-dir") + 1] == \
        "{bundle}/../job-state"
    assert frozen_client[frozen_client.index("--remote-root") + 1] == \
        "/frozen/remote/root"
    assert json.loads((bundle / "placements.json").read_text()) == json.loads(
        placements.read_text())
    assert str(placements) not in json.dumps(bundled)
    assert "placements.json" in binding["files"]


@pytest.mark.parametrize("flag", ["--placements-json", "--state-dir", "--remote-root"])
def test_controller_bundle_rejects_duplicate_bz_frozen_flags(tmp_path: Path, flag: str):
    placements = tmp_path / "placements.json"
    placements.write_text(json.dumps({"0": {"profile": "bz-a3-1", "device": 0}}))
    values = {
        "--placements-json": str(placements),
        "--state-dir": str(tmp_path / "state"),
        "--remote-root": "/approved/root",
    }
    client = [sys.executable, str(ROOT / "scripts/bz_a3_job_client.py")]
    for name, value in values.items():
        client.extend([name, value])
    client.extend(["--adapter-json", '["/approved/adapter"]', flag, values[flag]])
    config = tmp_path / "cells.json"
    backend = [
        sys.executable, str(ROOT / "scripts/benchmark_backend.py"), "--benchmark", "gdn",
        "--job-client-json", json.dumps(client),
    ]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match=f"exactly one {flag}"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


@pytest.mark.parametrize("flag", ["--placements-json", "--state-dir", "--remote-root"])
@pytest.mark.parametrize("form", ["equals", "abbreviation"])
def test_controller_bundle_rejects_noncanonical_bz_frozen_flags(
        tmp_path: Path, flag: str, form: str):
    placements = tmp_path / "placements.json"
    placements.write_text(json.dumps({"0": {"profile": "bz-a3-1", "device": 0}}))
    values = {
        "--placements-json": str(placements),
        "--state-dir": str(tmp_path / "state"),
        "--remote-root": "/approved/root",
    }
    client = [sys.executable, str(ROOT / "scripts/bz_a3_job_client.py")]
    for name, value in values.items():
        if name != flag:
            client.extend([name, value])
        elif form == "equals":
            client.append(f"{name}={value}")
        else:
            client.extend([name[:-2], value])
    client.extend(["--adapter-json", '["/approved/adapter"]'])
    config = tmp_path / "cells.json"
    backend = [sys.executable, str(ROOT / "scripts/benchmark_backend.py"),
               "--benchmark", "gdn", "--job-client-json", json.dumps(client)]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match=f"must use exact {flag}"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


@pytest.mark.parametrize(("missing", "following"), [
    ("--state-dir", "--remote-root"),
    ("--remote-root", "--adapter-json"),
])
def test_controller_bundle_rejects_bz_flag_followed_by_another_option(
        tmp_path: Path, missing: str, following: str):
    placements = tmp_path / "placements.json"
    placements.write_text(json.dumps({"0": {"profile": "bz-a3-1", "device": 0}}))
    values = {
        "--placements-json": str(placements),
        "--state-dir": str(tmp_path / "state"),
        "--remote-root": "/approved/root",
        "--adapter-json": '["/approved/adapter"]',
    }
    order = ["--placements-json", "--state-dir", "--remote-root", "--adapter-json"]
    order.remove(following)
    order.insert(order.index(missing) + 1, following)
    client = [sys.executable, str(ROOT / "scripts/bz_a3_job_client.py")]
    for flag in order:
        client.append(flag)
        if flag != missing:
            client.append(values[flag])
    config = tmp_path / "cells.json"
    backend = [sys.executable, str(ROOT / "scripts/benchmark_backend.py"),
               "--benchmark", "gdn", "--job-client-json", json.dumps(client)]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match=f"exactly one {missing}"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


def test_controller_bundle_rejects_duplicate_job_client_marker(tmp_path: Path):
    config = tmp_path / "cells.json"
    client = json.dumps([sys.executable, str(ROOT / "scripts/gz_a3_job_client.py")])
    backend = [
        sys.executable, str(ROOT / "scripts/benchmark_backend.py"), "--benchmark", "gdn",
        "--job-client-json", client, "--job-client-json", client,
    ]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match="exactly one --job-client-json"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


@pytest.mark.parametrize("form", ["equals", "abbreviation"])
def test_controller_bundle_rejects_noncanonical_job_client_marker(
        tmp_path: Path, form: str):
    config = tmp_path / "cells.json"
    client = json.dumps([sys.executable, str(ROOT / "scripts/gz_a3_job_client.py")])
    marker = (f"--job-client-json={client}" if form == "equals"
              else "--job-client-j")
    backend = [sys.executable, str(ROOT / "scripts/benchmark_backend.py"),
               "--benchmark", "gdn", marker]
    if form == "abbreviation":
        backend.append(client)
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError,
                       match="must use exact --job-client-json"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


def test_controller_bundle_rejects_job_client_marker_followed_by_option(tmp_path: Path):
    config = tmp_path / "cells.json"
    backend = [sys.executable, str(ROOT / "scripts/benchmark_backend.py"),
               "--benchmark", "gdn", "--job-client-json", "--timeout", "30"]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match="exactly one --job-client-json"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()


def test_controller_freeze_failure_cleans_private_build_and_allows_retry(tmp_path: Path):
    config = tmp_path / "cells.json"
    backend = [sys.executable, str(ROOT / "scripts/benchmark_backend.py"),
               "--benchmark", "gdn"]
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    output = tmp_path / "campaign.json"
    controller = [sys.executable, str(ROOT / "scripts/experimentctl.py"), "--config",
                  str(config.resolve()), "--cell", "{cell_id}"]
    with pytest.raises(campaign.CampaignError, match="exactly one --job-client-json"):
        campaign.freeze_controller_bundle(output, config, controller)
    assert not (tmp_path / "campaign.controller").exists()
    assert not list(tmp_path.glob(".campaign.controller.*"))

    client = [sys.executable, str(ROOT / "scripts/gz_a3_job_client.py"),
              "--adapter-json", '["/approved/adapter"]',
              "--state-dir", "/private/job-state"]
    backend.extend(["--job-client-json", json.dumps(client)])
    config.write_text(json.dumps({"cells": {"cell": {"backend": {"command": backend}}}}))
    binding = campaign.freeze_controller_bundle(output, config, controller)
    assert (tmp_path / binding["bundle"] / "controller.json").is_file()


def test_staged_matmul_backend_uses_complete_closure_after_source_is_removed(tmp_path: Path):
    source = tmp_path / "disposable-source"
    shutil.copytree(ROOT / "scripts", source / "scripts")
    shutil.copytree(ROOT / "benchmarks", source / "benchmarks")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("#!/usr/bin/env python3\nimport sys\nprint('controlled adapter failure')\nsys.exit(9)\n")
    adapter.chmod(0o755)
    client = [sys.executable, str(source / "scripts/gz_a3_job_client.py"),
              "--adapter-json", json.dumps([str(adapter)]),
              "--state-dir", str(tmp_path / "job-state")]
    config = tmp_path / "controller.json"
    config.write_text(json.dumps({"cells": {"matmul-cannbot": {
        "device": 2, "benchmark": "matmul", "development_cases": [7, 8, 9],
        "all_cases": list(range(10)), "backend": {"command": [
            sys.executable, str(source / "scripts/benchmark_backend.py"),
            "--benchmark", "matmul", "--candidate", "candidate.py",
            "--job-client-json", json.dumps(client),
        ], "timeout_seconds": 10},
    }}}))
    manifest_path = tmp_path / "campaign.json"
    binding = campaign.freeze_controller_bundle(
        manifest_path, config,
        [sys.executable, str(source / "scripts/experimentctl.py"), "--config",
         str(config.resolve()), "--cell", "{cell_id}"],
    )
    manifest = {"version": 2, "controller": binding}
    manifest_path.write_text(json.dumps(manifest))
    run_root = tmp_path / "run"
    command, _ = campaign.stage_controller(manifest, manifest_path, run_root, sys.executable)
    shutil.rmtree(source)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "candidate.py").write_text("# candidate\n")
    result = subprocess.run(
        [*(value.replace("{cell_id}", "matmul-cannbot") for value in command),
         "rank", "--benchmark", "matmul", "--warmups", "0", "--repeats", "1"],
        cwd=workspace, text=True, capture_output=True, check=False,
    )
    response = json.loads(result.stdout)
    assert response["status"] == "infrastructure_error", response
    assert "controlled adapter failure" in response["diagnostics"]
    assert "required candidate or frozen benchmark file is missing" not in response["diagnostics"]


def test_resume_rejects_private_bundle_drift_before_launch(tmp_path: Path):
    manifest, _ = fixture(tmp_path)
    root = tmp_path / "runs"
    run_campaign(manifest, root, type("Flaky", (RecordingLauncher,), {
        "launch": lambda self, sandbox, cell: {"status": "infrastructure_error"}
    })())
    private_config = root / "controller/controller.json"
    private_config.chmod(0o644)
    private_config.write_text("drift\n")
    launcher = RecordingLauncher()
    with pytest.raises(campaign.CampaignError, match="bundle drifted"):
        run_campaign(manifest, root, launcher, resume=True)
    assert launcher.calls == []


@pytest.mark.parametrize("mode", ["missing", "drift"])
def test_stage_rejects_missing_or_drifted_frozen_benchmark_asset(tmp_path: Path, mode: str):
    manifest, manifest_path = fixture(tmp_path)
    asset = manifest_path.parent / manifest["controller"]["bundle"] / "benchmarks/gdn/cases.jsonl"
    if mode == "missing":
        asset.unlink()
    else:
        asset.write_text("changed after freeze\n")
    with pytest.raises(campaign.CampaignError, match="bundle (file set )?drifted"):
        campaign.stage_controller(manifest, manifest_path, tmp_path / "run", sys.executable)
