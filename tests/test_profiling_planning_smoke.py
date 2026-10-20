"""Behavioral isolation and planning receipts; no prompt-text assertions."""
import json
from pathlib import Path
import shutil
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import profiling_planning_smoke as smoke
from planning_tool_client import response


def launcher(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "API.md").write_text("Public API\n")
    (bundle / "candidate.py").write_text("raise NotImplementedError\n")
    (bundle / "secret.py").write_text("hidden oracle")
    (bundle / "task.json").write_text(json.dumps({"public_files": ["API.md", "candidate.py"]}))
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    for name in ("coding", "profiling"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text("Guidance\n")
    return smoke.PlanningLauncher(
        tmp_path / "output", bundle, smoke.PROMPTS / "revised.md",
        {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        [sys.executable, str(SCRIPTS / "planning_tool_client.py"), "serve"],
        coding=tmp_path / "coding", skill=tmp_path / "profiling", auth_home=auth,
        timeout=5)


def fake_model(instance, code):
    runtime = instance.root.parent / "fake-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib" / "node_modules").mkdir(parents=True)
    (runtime / "bin" / "node").touch()
    instance.codex = runtime / "bin" / "codex"
    instance.codex.write_text("#!/usr/bin/python3\n" + code)
    instance.codex.chmod(0o755)


def test_help_capabilities_and_all_execution_denied():
    assert json.loads(response(["help"])["stdout"])["execution_enabled"] is False
    assert response(["capabilities"])["returncode"] == 0
    for args in (["benchmark"], ["profile", "--kernel", "foo"], ["run"],
                 ["python", "candidate.py"], [], ["--help", "benchmark"]):
        result = response(args)
        assert result["returncode"] == 77
        assert result["stdout"] == ""


def test_preparation_freezes_only_public_inputs_and_reproducible_content(tmp_path):
    instance = launcher(tmp_path)
    instance.prepare()
    manifest = json.loads((instance.root / "manifest.json").read_text())
    assert not (instance.workspace / "secret.py").exists()
    assert manifest["frozen"] == smoke.hashes(instance.frozen)
    original = smoke.hashes(instance.frozen)
    (tmp_path / "profiling" / "SKILL.md").write_text("later source mutation")
    assert original == smoke.hashes(instance.frozen)
    with pytest.raises(smoke.LaunchError, match="already exists"):
        instance.prepare()


@pytest.mark.parametrize("plan,exit_code", [(True, 0), (False, 3)])
def test_planning_completion_needs_plan_not_kernel_commits(tmp_path, plan, exit_code):
    if not shutil.which("bwrap"):
        pytest.skip("Bubblewrap required")
    instance = launcher(tmp_path)
    fake_model(instance, """import subprocess
from pathlib import Path
for args in (['help'], ['capabilities']):
    subprocess.run(['experiment-tool', *args], check=True)
for args in (['benchmark'], ['profile', '--kernel', 'foo'], ['run']):
    p = subprocess.run(['experiment-tool', *args], capture_output=True, text=True)
    assert p.returncode == 77, p
    assert 'no workload dispatched' in p.stderr
""" + ("Path('/workspace/PLAN.md').write_text('Proposed workflow')\n" if plan else ""))
    result = instance.launch()
    assert result["exit_code"] == exit_code
    assert result["candidate_commits"] == 0
    assert json.loads((instance.root / "receipt.json").read_text())["plan_present"] == plan
    calls = [json.loads(line) for line in instance.journal.read_text().splitlines()]
    assert len(calls) == 5
    assert all("serve" in row["command"] for row in calls)


def test_real_boundary_immutable_context_and_private_paths_absent(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("Bubblewrap required")
    instance = launcher(tmp_path)
    fake_model(instance, """from pathlib import Path
import importlib.util
assert not Path('/experiment/production_launcher.py').exists()
assert not Path('/workspace/secret.py').exists()
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('triton') is None
assert not list(Path('/dev').glob('davinci*'))
for name in ['candidate.py', 'API.md', '.agents/skills/ascend-profiling/SKILL.md']:
    try:
        Path('/workspace', name).write_text('tampered')
    except OSError:
        pass
    else:
        raise AssertionError(name + ' writable')
Path('/workspace/PLAN.md').write_text('Read context, then formulate hypotheses')
""")
    assert instance.launch()["exit_code"] == 0


def test_timed_out_model_keeps_partial_transcript_and_receipt(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("Bubblewrap required")
    instance = launcher(tmp_path)
    instance.timeout = 1
    fake_model(instance, "import time\nprint('partial planning', flush=True)\ntime.sleep(20)\n")
    result = instance.launch()
    assert result["exit_code"] == 124
    assert "partial planning" in (instance.root / "agent.jsonl").read_text()
    assert (instance.root / "receipt.json").is_file()


def test_forbidden_extra_public_exports_rejected_before_launch(tmp_path):
    instance = launcher(tmp_path)
    (instance.bundle / "task.json").write_text(json.dumps({"public_files": ["API.md", "candidate.py", "secret.py"]}))
    from argparse import Namespace
    args = Namespace(bundle=instance.bundle, output=tmp_path / "campaign", version="both")
    with pytest.raises(smoke.LaunchError, match="only API"):
        smoke.run(args)
    assert not args.output.exists()


def campaign_args(tmp_path):
    from argparse import Namespace
    instance = launcher(tmp_path)
    model = tmp_path / "model.json"
    model.write_text(json.dumps(instance.model))
    return Namespace(bundle=instance.bundle, output=tmp_path / "campaign",
                     version="previous", model=model, coding=tmp_path / "coding",
                     profiling=tmp_path / "profiling", replicates=3, timeout=5,
                     parallel=3, preflight_only=False)


@pytest.mark.parametrize("invalid", ["{bad", "{}", "[]", '{"name":"x","reasoning_effort":""}'])
def test_invalid_model_leaves_output_unreserved(tmp_path, invalid):
    args = campaign_args(tmp_path)
    args.model.write_text(invalid)
    with pytest.raises((ValueError, smoke.LaunchError)):
        smoke.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("missing", ["coding", "profiling", "prompt", "public"])
def test_missing_campaign_input_leaves_output_unreserved(tmp_path, monkeypatch, missing):
    args = campaign_args(tmp_path)
    if missing in ("coding", "profiling"):
        (getattr(args, missing) / "SKILL.md").unlink()
    elif missing == "prompt":
        monkeypatch.setattr(smoke, "PROMPTS", tmp_path / "missing-prompts")
    else:
        (args.bundle / "candidate.py").unlink()
    with pytest.raises(smoke.LaunchError):
        smoke.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("failure", ["construction", "prepare", "preflight"])
def test_cell_setup_failure_retains_receipt_and_completed_siblings(tmp_path, monkeypatch, failure):
    args = campaign_args(tmp_path)

    class FailingLauncher:
        def __init__(self, output, *unused, **kwargs):
            self.output = output
            if output.name == "previous-1" and failure == "construction":
                raise smoke.LaunchError("construction unavailable")

        def launch(self, preflight_only):
            self.output.mkdir()
            if self.output.name == "previous-1":
                if failure == "preflight":
                    smoke.write_json(self.output / "isolation.json", {"returncode": 1})
                raise smoke.LaunchError(failure + " unavailable")
            result = {"exit_code": 0, "plan_present": True}
            smoke.write_json(self.output / "receipt.json", result)
            return result

    monkeypatch.setattr(smoke, "PlanningLauncher", FailingLauncher)
    assert smoke.run(args) == 1
    results = json.loads((args.output / "results.json").read_text())
    assert [row["exit_code"] for row in results] == [1, 0, 0]
    failed = json.loads((args.output / "previous-1" / "receipt.json").read_text())
    assert failed["error"] == failure + " unavailable"
    for number in (2, 3):
        receipt = args.output / f"previous-{number}" / "receipt.json"
        assert json.loads(receipt.read_text())["exit_code"] == 0
