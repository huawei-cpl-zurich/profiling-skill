from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audited_contract as contract  # noqa: E402
import audited_experiment as cli  # noqa: E402
import audited_lifecycle  # noqa: E402,F401
import audited_runtime  # noqa: E402,F401
import audited_verifier  # noqa: E402,F401
import run_audited_toy_battery as toy  # noqa: E402


class Runner:
    def __init__(self, repo: Path, prompt: bytes = b"prompt\n", task: bytes = b"task\n"):
        self.repo = repo
        self.prompt_bytes = prompt
        self.task_bytes = task
        self.round_count = 3
        self.calls = []

    def run(self, run_id: str, agent_id: str):
        self.calls.append((run_id, agent_id))
        return cli.RunResult("complete", f"experiment/{run_id}/{agent_id}",
                             f"thread-{agent_id}", "seed", ("1", "2", "3"))


def test_acceptance_requires_exactly_three_isolated_identical_inputs(tmp_path: Path):
    runners = {name: Runner(tmp_path / name) for name in ("a", "b", "c")}
    result = cli.run_acceptance_battery("toy", runners, verify=lambda repo: {"status": "valid"})
    assert set(result) == {"a", "b", "c"}
    assert [r.calls for r in runners.values()] == [[("toy", "a")], [("toy", "b")], [("toy", "c")]]

    with pytest.raises(contract.AuditError, match="exactly three"):
        cli.run_acceptance_battery("toy", dict(list(runners.items())[:2]))
    duplicate = {"a": Runner(tmp_path / "same"), "b": Runner(tmp_path / "same"),
                 "c": Runner(tmp_path / "other")}
    with pytest.raises(contract.AuditError, match="isolated"):
        cli.run_acceptance_battery("toy", duplicate)
    mismatch = {"a": Runner(tmp_path / "1"), "b": Runner(tmp_path / "2", task=b"other"),
                "c": Runner(tmp_path / "3")}
    with pytest.raises(contract.AuditError, match="byte-identical"):
        cli.run_acceptance_battery("toy", mismatch)

    mixed_rounds = {
        name: Runner(tmp_path / f"round-{name}") for name in ("a", "b", "c")
    }
    mixed_rounds["c"].round_count = 4
    with pytest.raises(contract.AuditError, match="one immutable round count"):
        cli.run_acceptance_battery("toy", mixed_rounds)


def test_acceptance_rejects_incomplete_or_unverified_result(tmp_path: Path):
    runners = {name: Runner(tmp_path / name) for name in ("a", "b", "c")}
    runners["b"].run = lambda *_: cli.RunResult(
        "blocked", "experiment/toy/b", "thread-b", "seed", ("1", "2")
    )
    with pytest.raises(contract.AuditError, match="did not complete"):
        cli.run_acceptance_battery("toy", runners, verify=lambda repo: {"status": "valid"})

    runners = {name: Runner(tmp_path / f"ok-{name}") for name in ("a", "b", "c")}
    with pytest.raises(contract.AuditError, match="offline verification"):
        cli.run_acceptance_battery("toy", runners, verify=lambda repo: {"status": "bad"})


def test_cli_passes_agent_identity_and_always_scrubs_auth(tmp_path: Path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    prompt, task = tmp_path / "prompt", tmp_path / "task"
    prompt.write_text("prompt\n")
    task.write_text("task\n")
    seen = {}

    class Invoker:
        def __init__(self, repo, **kwargs):
            seen["agent_id"] = kwargs["agent_id"]
            seen["invoker"] = self
            self.scrubbed = False

        def scrub_auth(self):
            self.scrubbed = True

    class Lifecycle:
        def __init__(self, repo, prompt, task, invoke, controller, *, round_count,
                     max_candidate_repairs, required_manifest_schema):
            seen["round_count"] = round_count
            seen["max_candidate_repairs"] = max_candidate_repairs
            seen["required_manifest_schema"] = required_manifest_schema

        def run(self, *args, **kwargs):
            raise contract.AuditError("synthetic failure")

    monkeypatch.setattr(cli, "CodexInvoker", Invoker)
    monkeypatch.setattr(cli, "AuditedExperimentRunner", Lifecycle)
    with pytest.raises(SystemExit):
        cli.main(["--repo", str(repo), "--prompt", str(prompt), "--task", str(task),
                  "--run-id", "run", "--agent-id", "agent-7", "--controller", "true"])
    assert seen["agent_id"] == "agent-7"
    assert seen["max_candidate_repairs"] == 2
    assert seen["required_manifest_schema"] == "profiling-skill/candidate-kernel/v2"
    assert seen["invoker"].scrubbed is True
    assert "synthetic failure" in capsys.readouterr().err


def test_cli_passes_declared_round_count_to_lifecycle(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    prompt, task = tmp_path / "prompt", tmp_path / "task"
    prompt.write_text("prompt\n")
    task.write_text("task\n")
    seen = {}

    class Invoker:
        def __init__(self, *args, **kwargs):
            pass

        def scrub_auth(self):
            pass

    class Lifecycle:
        def __init__(self, repo, prompt, task, invoke, controller, *, round_count,
                     max_candidate_repairs, required_manifest_schema):
            seen["round_count"] = round_count
            seen["max_candidate_repairs"] = max_candidate_repairs
            seen["required_manifest_schema"] = required_manifest_schema

        def run(self, *args, **kwargs):
            return cli.RunResult("complete", "branch", "session", "seed", tuple("1234"))

    monkeypatch.setattr(cli, "CodexInvoker", Invoker)
    monkeypatch.setattr(cli, "AuditedExperimentRunner", Lifecycle)

    assert cli.main([
        "--repo", str(repo), "--prompt", str(prompt), "--task", str(task),
        "--run-id", "run", "--agent-id", "agent", "--controller", "true",
        "--rounds", "4", "--max-candidate-repairs", "1",
    ]) == 0
    assert seen["round_count"] == 4
    assert seen["max_candidate_repairs"] == 1
    assert seen["required_manifest_schema"] == "profiling-skill/candidate-kernel/v2"

    assert cli.main([
        "--repo", str(repo), "--prompt", str(prompt), "--task", str(task),
        "--run-id", "replay", "--agent-id", "agent", "--controller", "true",
        "--rounds", "4", "--allow-legacy-v1",
    ]) == 0
    assert seen["required_manifest_schema"] == "profiling-skill/candidate-kernel/v1"


def test_toy_controller_receipt_is_deterministic_and_policy_valid():
    digest, manifest = "a" * 64, "b" * 64
    first = toy.local_receipt("agent-a", 2, digest, manifest)
    second = toy.local_receipt("agent-a", 2, digest, manifest)
    assert first == second
    assert first["device"] == "local-no-npu"
    assert first["policy"]["observed_handles"] == [first["handle"]]
    contract.validate_controller_receipt(first, digest, manifest)


def test_toy_repo_initialization_is_reproducible(tmp_path: Path):
    repo = tmp_path / "repo"
    toy.initialize_repo(repo)
    assert (repo / "candidate.py").read_text() == "VALUE = 0\n"
    assert json.loads((repo / "candidate.manifest.json").read_text()) == {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "toy_kernel", "entrypoint": "toy_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }
    assert subprocess.check_output(
        ["git", "log", "-1", "--format=%s"], cwd=repo, text=True,
    ).strip() == "toy baseline"
    with pytest.raises(contract.AuditError, match="must not already exist"):
        toy.initialize_repo(repo)


def test_toy_cli_binds_v2_by_default_and_v1_only_for_explicit_replay(
        tmp_path: Path, monkeypatch):
    schemas = []

    class Invoker:
        def __init__(self, *args, **kwargs): pass
        def scrub_auth(self): pass

    class Lifecycle:
        def __init__(self, repo, prompt, task, invoke, controller, *,
                     required_manifest_schema):
            self.repo = repo
            schemas.append(required_manifest_schema)

    def battery(_run_id, runners):
        return {agent: cli.RunResult(
            "complete", "branch", f"thread-{agent}", "seed", ("1", "2", "3"),
        ) for agent in runners}

    monkeypatch.setattr(toy, "CodexInvoker", Invoker)
    monkeypatch.setattr(toy, "AuditedExperimentRunner", Lifecycle)
    monkeypatch.setattr(toy, "run_acceptance_battery", battery)

    assert toy.main([str(tmp_path / "current")]) == 0
    assert schemas == ["profiling-skill/candidate-kernel/v2"] * 3
    assert {json.loads(path.read_text())["schema"] for path in
            (tmp_path / "current").glob("agent-*/candidate.manifest.json")} == {
                "profiling-skill/candidate-kernel/v2"
            }

    schemas.clear()
    assert toy.main([str(tmp_path / "historical"), "--allow-legacy-v1"]) == 0
    assert schemas == ["profiling-skill/candidate-kernel/v1"] * 3
    assert {json.loads(path.read_text())["schema"] for path in
            (tmp_path / "historical").glob("agent-*/candidate.manifest.json")} == {
                "profiling-skill/candidate-kernel/v1"
            }


def test_real_composition_creates_three_verified_isolated_histories(tmp_path: Path):
    prompt, task = tmp_path / "prompt.md", tmp_path / "task.md"
    prompt.write_text("invariant\n")
    task.write_text("increment three times\n")
    runners = {}

    for agent_id in ("a", "b", "c"):
        repo = tmp_path / agent_id
        toy.initialize_repo(repo)

        def invoke(number, session, instruction, *, repo=repo, agent_id=agent_id):
            thread = f"thread-{agent_id}"
            events = [{"type": "thread.started", "thread_id": thread}]
            if instruction is None:
                (repo / "candidate.py").write_text(f"VALUE = {number}\n")
                events.append({"type": "item.completed", "item": {
                    "type": "command_execution", "command": "python candidate.py",
                    "exit_code": 0, "aggregated_output": f"{number}\n",
                }})
            else:
                candidate = contract.sha256_bytes((repo / "candidate.py").read_bytes())
                manifest = contract.sha256_bytes(
                    (repo / "candidate.manifest.json").read_bytes()
                )
                result = toy.local_receipt(agent_id, number, candidate, manifest)
                report = {
                    "hypothesis": "the next integer is valid",
                    "expected_result": f"VALUE equals {number}",
                    "change": f"set VALUE to {number}",
                    "evidence": "the local command and controller succeeded",
                    "observed_result": f"VALUE is {number}", "decision": "retain",
                    "postmortem": "the result matched", "next_experiment": "increment again",
                    "candidate_sha256": candidate, "manifest_sha256": manifest,
                    "controller_handle": result["handle"],
                    "controller_receipt_sha256": contract.sha256_json(result),
                    "sources": [], "no_sources_reason": "self-contained toy task",
                }
                events.append({"type": "item.completed", "item": {
                    "type": "agent_message", "text": json.dumps(report),
                }})
            return "\n".join(json.dumps(event) for event in events)

        def controller(number, candidate, manifest, agent_id=agent_id):
            return toy.local_receipt(agent_id, number, candidate, manifest)

        runners[agent_id] = cli.AuditedExperimentRunner(
            repo, prompt, task, invoke, controller,
            required_manifest_schema="profiling-skill/candidate-kernel/v2",
        )

    results = cli.run_acceptance_battery("composition", runners)
    assert len(results) == 3
    for agent_id, runner in runners.items():
        assert cli.validate_branch(runner.repo)["session_id"] == f"thread-{agent_id}"
        assert subprocess.check_output(
            ["git", "rev-list", "--count", "main..HEAD"], cwd=runner.repo, text=True,
        ).strip() == "4"
