from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audited_contract as contract  # noqa: E402
import audited_lifecycle as lifecycle  # noqa: E402


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", path], check=True)
    subprocess.run(["git", "config", "user.name", "Host Runner"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "host@example.test"], cwd=path, check=True)
    (path / "candidate.py").write_text("VALUE = 0\n")
    (path / "candidate.manifest.json").write_text('{"candidate":"candidate.py"}\n')
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture base"], cwd=path, check=True)


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    prompt, task = tmp_path / "input-prompt.md", tmp_path / "input-task.md"
    prompt.write_text("Invariant instructions; read TASK.md.\n")
    task.write_text("Increment VALUE once per experiment.\n")
    return prompt, task


def receipt(candidate: str, manifest: str, number: int) -> dict:
    handle = f"local:{number}"
    samples = [number + 0.0, number + 0.1, number + 0.2]
    return {
        "status": "ok", "handle": handle, "candidate_sha256": candidate,
        "manifest_sha256": manifest, "device": f"dynamic-{number}",
        "samples_us": samples, "median_us": number + 0.1,
        "policy": {
            "schema": contract.CONTROLLER_POLICY_SCHEMA,
            "selected_device": f"dynamic-{number}",
            "admission_controls": [{"device": f"dynamic-{number}", "status": "pass",
                                      "healthy": True, "idle": True, "warmed": True}],
            "submission_candidate_sha256": candidate,
            "submitted_handles": [handle], "observed_handles": [handle],
            "infra_retries": 0, "retry_budget": 1, "quarantined_devices": [],
            "quarantine_controls": {}, "sample_count": 3,
            "variability_threshold": 0.25,
            "variability_ratio": 0.2 / (number + 0.1),
            "confirmation_count": 0, "post_control": "stable",
        },
    }


def report(number: int, candidate: str, manifest: str, decision: str = "retain") -> dict:
    controller = receipt(candidate, manifest, number)
    return {
        "hypothesis": f"change {number} improves the toy",
        "expected_result": "the deterministic score increases",
        "change": f"set VALUE to {number}",
        "evidence": "local check and retained controller receipt",
        "observed_result": f"score={number}", "decision": decision,
        "postmortem": "the observation explains the decision",
        "next_experiment": "make one more independent change" if number < 3 else "done",
        "candidate_sha256": candidate, "manifest_sha256": manifest,
        "controller_handle": controller["handle"],
        "controller_receipt_sha256": contract.sha256_json(controller),
        "sources": [], "no_sources_reason": "The toy task is self-contained.",
    }


def events(thread: str, document: dict, *, command: bool = True) -> str:
    stream = [{"type": "thread.started", "thread_id": thread}]
    if command:
        stream.append({"type": "item.completed", "item": {
            "type": "command_execution", "command": "python check.py",
            "exit_code": 0, "aggregated_output": "ok\n",
        }})
    stream.append({"type": "item.completed", "item": {
        "type": "agent_message", "text": json.dumps(document),
    }})
    return "\n".join(json.dumps(item) for item in stream) + "\n"


def invalid_command_events(thread: str, document: dict) -> str:
    return "\n".join(json.dumps(item) for item in [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {
            "type": "command_execution", "command": "python check.py",
            "exit_code": "not-an-integer", "aggregated_output": "bad event\n",
        }},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(document),
        }},
    ]) + "\n"


def updating_invoker(repo: Path, *, revert: int | None = None, malformed: bool = False):
    calls: list[tuple[int, str | None, str | None]] = []
    needs_repair = malformed

    def invoke(number: int, session: str | None, instruction: str | None) -> str:
        nonlocal needs_repair
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-fixed", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest,
                          "revert" if number == revert else "retain")
        if needs_repair and instruction.startswith("Finalize"):
            needs_repair = False
            document.pop("postmortem")
        return events("thread-fixed", document, command=False)

    return invoke, calls


def test_three_host_commits_use_one_session_and_immutable_seed(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, calls = updating_invoker(repo, malformed=True)
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    )

    result = runner.run("smoke", "agent-a")

    assert result.status == "complete" and result.session_id == "thread-fixed"
    assert len(result.commits) == 3
    assert all(call[1] == "thread-fixed" for call in calls[1:])
    assert any(instruction and instruction.startswith("Repair only")
               for _, _, instruction in calls)
    subjects = subprocess.check_output(
        ["git", "log", "--format=%s", "--reverse", "main..HEAD"], cwd=repo, text=True
    ).splitlines()
    assert subjects == ["experiment seed: smoke/agent-a", "experiment 1: agent-a",
                        "experiment 2: agent-a", "experiment 3: agent-a"]
    assert (repo / "PROMPT.md").read_bytes() == prompt.read_bytes()
    assert (repo / "TASK.md").read_bytes() == task.read_bytes()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)


def test_host_declared_four_rounds_are_seeded_and_completed(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, calls = updating_invoker(repo)
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
        round_count=4,
    )

    result = runner.run("four", "agent-a")

    seed = json.loads((repo / ".experiment/seed.json").read_text())
    assert seed["schema"] == "profiling-skill/audited-seed/v2"
    assert seed["round_count"] == 4
    assert len(result.commits) == 4
    assert [
        number for number, _, instruction in calls if instruction is None
    ] == [1, 2, 3, 4]
    assert (repo / "experiments/04/evidence.json").is_file()


def test_seed_accepts_and_commits_only_materialized_starter_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    (repo / "candidate.py").write_text("TASK = 'gdn'\nVALUE = 0\n")
    (repo / "candidate.manifest.json").write_text(
        '{"candidate":"candidate.py","kernel_name":"gdn"}\n'
    )
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, lambda *args: "", lambda *args: {}, round_count=4,
    )

    _, seed_commit, _ = runner._initialize("starters", "gdn-agent")

    assert subprocess.check_output(
        ["git", "show", f"{seed_commit}:candidate.py"], cwd=repo, text=True
    ) == "TASK = 'gdn'\nVALUE = 0\n"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )


def test_seed_rejects_materialized_starter_with_unrelated_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    (repo / "candidate.py").write_text("VALUE = 2\n")
    (repo / "unexpected.txt").write_text("not starter materialization\n")
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, lambda *args: "", lambda *args: {}, round_count=4,
    )

    with pytest.raises(contract.AuditError, match="starter materialization"):
        runner._initialize("starters", "agent")


def test_resume_rejects_changed_host_round_count(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    def blocked_controller(number, candidate, manifest):
        return {"status": "infrastructure_error", "reason": "not admitted"}

    first = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, blocked_controller, round_count=4,
    )
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        first.run("immutable-rounds", "agent")

    changed = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, blocked_controller, round_count=3,
    )
    with pytest.raises(contract.AuditError, match="round count"):
        changed.run("immutable-rounds", "agent", resume=True)


def test_revert_archives_tested_source_and_restores_prior_best(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo, revert=2)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    ).run("revert", "agent")

    evidence = json.loads((repo / "experiments/02/evidence.json").read_text())
    assert evidence["tested_candidate_sha256"] == sha("VALUE = 2\n")
    assert evidence["restored_candidate_sha256"] == sha("VALUE = 1\n")
    assert (repo / "experiments/02/tested_candidate.py").read_text() == "VALUE = 2\n"
    assert subprocess.check_output(
        ["git", "show", "HEAD~1:candidate.py"], cwd=repo, text=True
    ) == "VALUE = 1\n"


@pytest.mark.parametrize("target", ["candidate.py", "candidate.manifest.json"])
def test_finalize_freezes_candidate_and_manifest(tmp_path: Path, target: str):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-frozen", {"prepared": True})
        (repo / target).write_text("mutated during finalize\n")
        return events("thread-frozen", {"finalized": True}, command=False)

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    )
    with pytest.raises(contract.AuditError, match="changed during finalize"):
        runner.run("frozen", "agent")
    state = json.loads((repo / ".experiment/blocked.json").read_text())
    assert state["stage"] == "finalize" and state["session_id"] == "thread-frozen"


def test_stale_manifest_is_repaired_before_controller_submission(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    calls = []

    def invoke(number, session, instruction):
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            if number == 1:
                (repo / "candidate.manifest.json").write_text(
                    json.dumps({"candidate_sha256": "0" * 64}) + "\n"
                )
            return events("thread-repair", {"prepared": True})
        if instruction.startswith("Repair the prepared"):
            candidate = sha((repo / "candidate.py").read_bytes())
            (repo / "candidate.manifest.json").write_text(
                json.dumps({"candidate_sha256": candidate}) + "\n"
            )
            return events("thread-repair", {"repaired": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-repair", report(number, candidate, manifest), command=False)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    ).run("repair", "agent")

    assert any(instruction and instruction.startswith("Repair the prepared")
               for _, _, instruction in calls)
    assert all(session == "thread-repair" for _, session, _ in calls[1:])
    assert len((repo / "experiments/01/commands.jsonl").read_text().splitlines()) == 2


def test_missing_preparation_commands_repair_in_reported_session(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    calls = []

    def invoke(number, session, instruction):
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-early", {"prepared": True}, command=False)
        if instruction.startswith("Repair preparation evidence"):
            return events("thread-early", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-early", report(number, candidate, manifest), command=False)

    result = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    ).run("missing-command", "agent")

    assert result.session_id == "thread-early"
    assert calls[1][1] == "thread-early"
    assert calls[1][2].startswith("Repair preparation evidence")


def test_exhausted_preparation_evidence_checkpoints_reported_session(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        (repo / "candidate.py").write_text("VALUE = 1\n")
        return events("thread-checkpoint", {"prepared": True}, command=False)

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
        max_repairs=1,
    )
    with pytest.raises(contract.AuditError, match="no command evidence"):
        runner.run("missing-command", "agent")

    state = json.loads((repo / ".experiment/blocked.json").read_text())
    assert state["session_id"] == "thread-checkpoint"
    assert state["stage"] == "prepare"


def test_controller_resume_observes_durable_handle_without_resubmission(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        def __init__(self):
            self.submissions = 0
            self.observations = []

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            if self.submissions == 1:
                return {
                    "status": "infrastructure_error", "handle": "local:1",
                    "candidate_sha256": candidate, "manifest_sha256": manifest,
                    "reason": "observer disconnected",
                }
            return receipt(candidate, manifest, number)

        def observe(self, number, candidate, manifest, handle):
            self.observations.append((number, handle, candidate, manifest))
            document = receipt(candidate, manifest, number)
            document["handle"] = handle
            document["policy"]["submitted_handles"] = [handle]
            document["policy"]["observed_handles"] = [handle]
            return document

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, controller)
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("observe", "agent")

    result = runner.run("observe", "agent", resume=True)

    assert result.status == "complete"
    assert controller.submissions == 3
    assert len(controller.observations) == 1
    assert controller.observations[0][1] == "local:1"


def test_controller_resume_without_observer_fails_without_resubmission(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)
    submissions = 0

    def controller(number, candidate, manifest):
        nonlocal submissions
        submissions += 1
        return {"status": "infrastructure_error", "handle": "durable:1",
                "candidate_sha256": candidate, "manifest_sha256": manifest,
                "reason": "observer disconnected"}

    runner = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, controller)
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("no-observer", "agent")
    with pytest.raises(contract.AuditError, match="does not support durable-handle observation"):
        runner.run("no-observer", "agent", resume=True)
    assert submissions == 1
    assert (repo / ".experiment/blocked.json").is_file()


def test_finalize_retains_commands_from_malformed_report_attempt(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    malformed = True

    def invoke(number, session, instruction):
        nonlocal malformed
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-finalize", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        if malformed:
            malformed = False
            document.pop("postmortem")
            return events("thread-finalize", document)
        return events("thread-finalize", document, command=False)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    ).run("finalize-command", "agent")

    # Preparation and the rejected finalize attempt both remain auditable.
    assert len((repo / "experiments/01/commands.jsonl").read_text().splitlines()) == 2


def test_invalid_exit_events_repair_in_session_during_prepare_and_finalize(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    bad_prepare = bad_finalize = True
    calls = []

    def invoke(number, session, instruction):
        nonlocal bad_prepare, bad_finalize
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            if bad_prepare:
                bad_prepare = False
                return invalid_command_events("thread-invalid", {"prepared": True})
            return events("thread-invalid", {"prepared": True})
        if instruction.startswith("Repair preparation evidence"):
            return events("thread-invalid", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        if instruction.startswith("Finalize") and bad_finalize:
            bad_finalize = False
            return invalid_command_events("thread-invalid", document)
        return events("thread-invalid", document, command=False)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    ).run("invalid-exit", "agent")

    assert all(session == "thread-invalid" for _, session, _ in calls[1:])
    commands = [json.loads(line) for line in
                (repo / "experiments/01/commands.jsonl").read_text().splitlines()]
    markers = [item for item in commands
               if item["command"] == "<invalid structured command event>"]
    assert len(markers) == 2  # malformed preparation and malformed finalization
    command_fields = {
        "command", "exit_code", "output_sha256", "output_excerpt", "output_truncated",
    }
    assert all(set(item) == command_fields for item in commands)
    assert all(len(item["output_sha256"]) == 64 for item in markers)
    assert all(item["output_excerpt"].startswith("structured event stream rejected:")
               for item in markers)


def test_invocation_timeout_checkpoints_and_resumes_finalization_session(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    timed_out = True

    class TurnError(RuntimeError):
        def __init__(self, message, structured_stdout):
            super().__init__(message)
            self.structured_stdout = structured_stdout

    class Controller:
        def __init__(self):
            self.submissions = 0
            self.observations = 0

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            return receipt(candidate, manifest, number)

        def observe(self, number, candidate, manifest, handle):
            self.observations += 1
            raise AssertionError("a terminal receipt must not be re-observed")

    def invoke(number, session, instruction):
        nonlocal timed_out
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-timeout", {"prepared": True})
        if number == 1 and timed_out:
            timed_out = False
            partial = json.dumps({
                "type": "thread.started", "thread_id": "thread-timeout",
            }) + "\n"
            raise TurnError("turn timed out", partial)
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-timeout", report(number, candidate, manifest), command=False)

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, controller)
    with pytest.raises(contract.AuditError, match="checkpointed session thread-timeout"):
        runner.run("timeout", "agent")

    state = json.loads((repo / ".experiment/blocked.json").read_text())
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    assert state["stage"] == "finalize" and state["session_id"] == "thread-timeout"
    assert state["receipt"]["handle"] == "local:1"

    result = runner.run("timeout", "agent", resume=True)

    assert result.branch == branch and result.session_id == "thread-timeout"
    assert controller.submissions == 3 and controller.observations == 0
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "main..HEAD"], cwd=repo, text=True
    ).strip() == "4"


def test_failure_before_thread_resumes_initial_turn_on_same_seed_branch(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    unavailable = True
    calls = []

    def invoke(number, session, instruction):
        nonlocal unavailable
        calls.append((number, session, instruction))
        if unavailable:
            unavailable = False
            raise OSError("container runtime unavailable")
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-after-infra", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events(
            "thread-after-infra", report(number, candidate, manifest), command=False,
        )

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
    )
    with pytest.raises(contract.AuditError, match="checkpointed before session start"):
        runner.run("pre-session", "agent")

    state = json.loads((repo / ".experiment/blocked.json").read_text())
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
    ).strip()
    assert state["session_id"] is None and state["pre_session"] is True
    assert state["stage"] == "prepare" and state["experiment"] == 1

    result = runner.run("pre-session", "agent", resume=True)

    assert result.branch == branch and result.session_id == "thread-after-infra"
    assert calls[:2] == [(1, None, None), (1, None, None)]
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "main..HEAD"], cwd=repo, text=True,
    ).strip() == "4"


def test_prehandle_controller_failure_resubmits_identical_candidate_once(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        def __init__(self):
            self.submissions = []
            self.observations = 0

        def __call__(self, number, candidate, manifest):
            self.submissions.append((number, candidate, manifest))
            if len(self.submissions) == 1:
                return {"status": "infrastructure_error", "reason": "no handle allocated"}
            return receipt(candidate, manifest, number)

        def observe(self, *args):
            self.observations += 1
            raise AssertionError("pre-handle failure must resubmit, not observe")

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, controller)
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("prehandle", "agent")
    state = json.loads((repo / ".experiment/blocked.json").read_text())
    assert state["stage"] == "controller" and state["receipt"].get("handle") is None

    result = runner.run("prehandle", "agent", resume=True)

    assert result.status == "complete" and controller.observations == 0
    assert controller.submissions[0] == controller.submissions[1]
    assert len(controller.submissions) == 4  # retry round 1, then rounds 2 and 3
    assert len(result.commits) == 3


def test_measurement_pending_remeasures_without_consuming_experiment(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        def __init__(self):
            self.submissions = 0
            self.remeasurements = []

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            document = receipt(candidate, manifest, number)
            if number == 1:
                document["status"] = "measurement_pending"
            return document

        def remeasure(self, number, candidate, manifest, handle):
            self.remeasurements.append((number, candidate, manifest, handle))
            return receipt(candidate, manifest, number)

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, controller)
    with pytest.raises(contract.AuditError, match="measurement pending"):
        runner.run("remeasure", "agent")
    state = json.loads((repo / ".experiment/blocked.json").read_text())
    assert state["stage"] == "measurement" and state["experiment"] == 1
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "main..HEAD"], cwd=repo, text=True,
    ).strip() == "2"  # seed plus replaceable checkpoint, no experiment commit

    result = runner.run("remeasure", "agent", resume=True)

    assert len(result.commits) == 3 and len(controller.remeasurements) == 1
    assert controller.remeasurements[0][1:3] == (
        state["candidate_sha256"], state["manifest_sha256"],
    )
    assert controller.submissions == 3


@pytest.mark.parametrize("helper_name,tracked", [("helper.py", True), ("scratch.py", False)])
def test_undeclared_helper_changes_never_reach_controller(
    tmp_path: Path, helper_name: str, tracked: bool,
):
    repo = tmp_path / "repo"
    init_repo(repo)
    if tracked:
        (repo / helper_name).write_text("ORIGINAL = True\n")
        subprocess.run(["git", "add", helper_name], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "add helper"], cwd=repo, check=True)
    prompt, task = inputs(tmp_path)
    controller_calls = 0

    def invoke(number, session, instruction):
        (repo / "candidate.py").write_text("VALUE = 1\n")
        (repo / helper_name).write_text("CHANGED = True\n")
        return events("thread-helper", {"prepared": True})

    def controller(number, candidate, manifest):
        nonlocal controller_calls
        controller_calls += 1
        return receipt(candidate, manifest, number)

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, max_repairs=0,
    )
    with pytest.raises(contract.AuditError, match="undeclared worktree changes"):
        runner.run("helper", "agent")
    assert controller_calls == 0
    assert json.loads((repo / ".experiment/blocked.json").read_text())["stage"] == "prepare"


def test_resume_rejects_reproducibility_drift_from_seed(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        def __init__(self, model):
            self.model = model

        def reproducibility_metadata(self):
            return {"adapter": "test-controller", "model": self.model,
                    "reasoning_effort": "low", "image": "image@sha256:fixed"}

        def __call__(self, number, candidate, manifest):
            return {"status": "infrastructure_error", "reason": "no handle allocated"}

    first = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, Controller("model-a"))
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        first.run("provenance", "agent")

    changed = lifecycle.AuditedExperimentRunner(repo, prompt, task, invoke, Controller("model-b"))
    with pytest.raises(contract.AuditError, match="reproducibility metadata"):
        changed.run("provenance", "agent", resume=True)
    assert (repo / ".experiment/blocked.json").is_file()
