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
import audited_runtime as runtime  # noqa: E402
import audited_verifier as verifier  # noqa: E402


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", path], check=True)
    subprocess.run(["git", "config", "user.name", "Host Runner"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "host@example.test"], cwd=path, check=True)
    (path / "candidate.py").write_text("VALUE = 0\n")
    (path / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"kernel"}\n'
    )
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


def candidate_error_receipt(candidate: str, manifest: str, number: int,
                            reason: str = "compile failed") -> dict:
    result = receipt(candidate, manifest, number)
    result.update(status="candidate_error", reason=reason)
    result.pop("samples_us")
    result.pop("median_us")
    result["policy"].pop("sample_count")
    result["policy"].pop("variability_ratio")
    result["policy"]["post_control"] = "not_run"
    return result


def observed_transaction_receipt(candidate: str, manifest: str, number: int,
                                 retained: str = "remote:retained",
                                 final_handle: str | None = None) -> dict:
    result = receipt(candidate, manifest, number)
    final_handle = final_handle or f"remote:final:{number}"
    result["handle"] = final_handle
    request = sha(f"retained-request:{number}")
    final_request = sha(f"final-request:{number}")
    handles = list(dict.fromkeys((retained, final_handle)))
    result["policy"].update(
        submitted_handles=handles,
        observed_handles=handles,
        infra_retries=1,
        measurement_generation=0,
        operation_history=[
            {"request_sha256": request, "mode": "submit",
             "status": "infrastructure_error", "terminal": False,
             "handle": retained, "action": "calibrate",
             "attempt_id": f"experiment-{number}-before"},
            {"request_sha256": request, "mode": "observe", "status": "ok",
             "terminal": True, "handle": retained, "action": "calibrate",
             "attempt_id": f"experiment-{number}-before"},
            {"request_sha256": final_request, "mode": "submit", "status": "ok",
             "terminal": True, "handle": final_handle, "action": "calibrate",
             "attempt_id": f"experiment-{number}-after"},
        ],
    )
    return result


def observe_transition_receipt(candidate: str, manifest: str, number: int,
                               observed: str, pending: str) -> dict:
    request = sha(f"observed-request:{number}")
    pending_request = sha(f"pending-request:{number}")
    history = [
        {"request_sha256": request, "mode": "submit",
         "status": "infrastructure_error", "terminal": False,
         "handle": observed, "action": "check", "attempt_id": None},
        {"request_sha256": request, "mode": "observe", "status": "ok",
         "terminal": True, "handle": observed, "action": "check",
         "attempt_id": None},
        {"request_sha256": pending_request, "mode": "submit",
         "status": "infrastructure_error", "terminal": False,
         "handle": pending, "action": "profile",
         "attempt_id": f"experiment-{number}-measurement-0-primary"},
    ]
    return {
        "status": "infrastructure_error", "terminal": False,
        "reason": "profile observer disconnected", "handle": pending,
        "experiment": number, "candidate_sha256": candidate,
        "manifest_sha256": manifest,
        "observe_transition": {
            "schema": contract.OBSERVE_TRANSITION_SCHEMA,
            "observed_handle": observed, "pending_handle": pending,
            "operation_history": history,
        },
    }


def chained_observed_receipt(candidate: str, manifest: str, number: int,
                             first: str, second: str) -> dict:
    result = receipt(candidate, manifest, number)
    final = f"remote:final:{number}"
    history = observe_transition_receipt(
        candidate, manifest, number, first, second,
    )["observe_transition"]["operation_history"]
    history.extend([
        {**history[-1], "mode": "observe", "status": "ok", "terminal": True},
        {"request_sha256": sha(f"final-request:{number}"), "mode": "submit",
         "status": "ok", "terminal": True, "handle": final,
         "action": "calibrate", "attempt_id": f"experiment-{number}-after"},
    ])
    result["handle"] = final
    result["policy"].update(
        submitted_handles=[first, second, final],
        observed_handles=[first, second, final], infra_retries=2,
        retry_budget=2, measurement_generation=0, operation_history=history,
    )
    return result


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


def retry_evidence(action: str, attempt: int) -> dict:
    return {
        "schema": "profiling-skill/codex-transient-retry/v1",
        "terminal_error": "server_overloaded", "action": action,
        "attempt": attempt, "stdout_sha256": sha(f"stdout-{attempt}"),
        "stdout_bytes": attempt, "stderr_sha256": sha(f"stderr-{attempt}"),
        "stderr_bytes": attempt + 1,
    }


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


def test_seed_only_resume_retains_prepared_work_and_starts_new_session(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    controller_calls = []
    invocations = []

    def invoke(number, session, instruction):
        invocations.append((session, instruction, (repo / "candidate.py").read_text()))
        if instruction and instruction.startswith("Recover seed-only"):
            return events("thread-recovered", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-recovered", report(number, candidate, manifest), command=False)

    def controller(number, candidate, manifest):
        controller_calls.append((number, candidate, manifest))
        return receipt(candidate, manifest, number)

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    )
    runner._initialize("seed-crash", "agent")
    (repo / "candidate.py").write_text("VALUE = 7\n")
    (repo / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"recovered"}\n'
    )

    result = runner.run("seed-crash", "agent", resume=True)

    assert result.status == "complete" and len(result.commits) == 1
    assert invocations[0][0] is None
    assert invocations[0][1].startswith("Recover seed-only")
    assert invocations[0][2] == "VALUE = 7\n"
    assert len(controller_calls) == 1


@pytest.mark.parametrize("mutation", ["extra-commit", "unrelated-dirt"])
def test_seed_only_resume_rejects_nonseed_history_or_unrelated_dirt(
        tmp_path: Path, mutation: str):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, lambda *args: "", lambda *args: {}, round_count=1,
    )
    runner._initialize("seed-crash", "agent")
    if mutation == "extra-commit":
        (repo / "extra.txt").write_text("committed\n")
        subprocess.run(["git", "add", "extra.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "unexpected"], cwd=repo, check=True)
    else:
        (repo / "extra.txt").write_text("dirty\n")

    with pytest.raises(contract.AuditError, match="seed-only"):
        runner.run("seed-crash", "agent", resume=True)


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
                    json.dumps({
                        "schema": "profiling-skill/candidate-kernel/v1",
                        "kernel_name": "kernel", "candidate_sha256": "0" * 64,
                    }) + "\n"
                )
            return events("thread-repair", {"prepared": True})
        if instruction.startswith("Repair the prepared"):
            candidate = sha((repo / "candidate.py").read_bytes())
            (repo / "candidate.manifest.json").write_text(
                json.dumps({
                    "schema": "profiling-skill/candidate-kernel/v1",
                    "kernel_name": "kernel", "candidate_sha256": candidate,
                }) + "\n"
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


def test_unresolved_selector_is_repaired_before_controller_submission(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    calls = []
    controller_calls = []

    def invoke(number, session, instruction):
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 2\n")
            (repo / "candidate.manifest.json").write_text(json.dumps({
                "schema": "profiling-skill/candidate-kernel/v1",
                "kernel_name": "REPLACE_WITH_EXACT_EXPORTED_KERNEL",
            }) + "\n")
            return events("thread-selector", {"prepared": True})
        if instruction.startswith("Repair the prepared"):
            (repo / "candidate.manifest.json").write_text(json.dumps({
                "schema": "profiling-skill/candidate-kernel/v1",
                "kernel_name": "actual_exported_kernel",
            }) + "\n")
            return events("thread-selector", {"repaired": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-selector", report(number, candidate, manifest), command=False)

    def controller(number, candidate, manifest):
        controller_calls.append((number, json.loads(
            (repo / "candidate.manifest.json").read_text())["kernel_name"]))
        return receipt(candidate, manifest, number)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    ).run("selector", "agent")

    assert any(instruction and "unresolved starter selector" in instruction
               for _, _, instruction in calls)
    assert controller_calls == [(1, "actual_exported_kernel")]


def test_candidate_validation_accepts_and_checks_mandatory_v2_manifest(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    prior = sha((repo / "candidate.py").read_bytes())
    (repo / "candidate.py").write_text("VALUE = 1\n")
    manifest = {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": "complete_kernel_mix_aiv",
        "entrypoint": "complete_kernel",
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }
    (repo / "candidate.manifest.json").write_text(json.dumps(manifest))
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, lambda *args: "", lambda *args: {}, round_count=4,
        required_manifest_schema="profiling-skill/candidate-kernel/v2",
    )
    candidate_hash, selected = runner._validate_candidate(prior)
    assert candidate_hash == sha((repo / "candidate.py").read_bytes())
    assert selected.name == "candidate.manifest.json"

    manifest["fusion"]["complete_operator"] = False
    (repo / "candidate.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(contract.AuditError, match="fusion"):
        runner._validate_candidate(prior)

    (repo / "candidate.manifest.json").write_text(json.dumps({
        "schema": "profiling-skill/candidate-kernel/v1", "kernel_name": "legacy",
    }))
    with pytest.raises(contract.AuditError, match="candidate-kernel/v2"):
        runner._validate_candidate(prior)


@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_candidate_errors_are_repaired_inside_one_experiment(
    tmp_path: Path, failed_attempts: int,
):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    calls = []
    controller_calls = []

    def invoke(number, session, instruction):
        calls.append((number, session, instruction))
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-attempts", {"prepared": True})
        if instruction.startswith("Repair candidate attempt"):
            attempt = len([call for call in calls if call[2] and
                           call[2].startswith("Repair candidate attempt")])
            (repo / "candidate.py").write_text(f"VALUE = {attempt + 1}\n")
            return events("thread-attempts", {"repair": attempt})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-attempts", report(number, candidate, manifest), command=False)

    def controller(number, candidate, manifest):
        controller_calls.append((number, candidate))
        return (candidate_error_receipt(candidate, manifest, number)
                if len(controller_calls) <= failed_attempts
                else receipt(candidate, manifest, number))

    result = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    ).run("attempts", "agent")

    assert len(result.commits) == 1
    assert [number for number, _ in controller_calls] == [1] * (failed_attempts + 1)
    assert {session for _, session, _ in calls[1:]} == {"thread-attempts"}
    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert [item["status"] for item in evidence["candidate_attempts"]] == (
        ["candidate_error"] * failed_attempts + ["ok"]
    )
    assert evidence["final_attempt"] == failed_attempts + 1
    for attempt in range(1, failed_attempts + 2):
        root = repo / f"experiments/01/attempts/{attempt:02d}"
        assert {path.name for path in root.iterdir()} == {
            "candidate.py", "candidate.manifest.json", "controller.json",
            "commands.jsonl", "reasoning.txt",
        }


def test_candidate_repair_exhaustion_commits_final_attempt(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-exhausted", {"prepared": True})
        if instruction.startswith("Repair candidate attempt"):
            value = int((repo / "candidate.py").read_text().split()[-1]) + 1
            (repo / "candidate.py").write_text(f"VALUE = {value}\n")
            return events("thread-exhausted", {"repair": value})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        failed = candidate_error_receipt(candidate, manifest, number, "runtime failed")
        document = report(number, candidate, manifest, "revert")
        document["controller_receipt_sha256"] = contract.sha256_json(failed)
        return events(
            "thread-exhausted", document,
            command=False,
        )

    def controller(number, candidate, manifest):
        return candidate_error_receipt(candidate, manifest, number, "runtime failed")

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    ).run("exhausted", "agent")

    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert evidence["final_attempt"] == 3
    assert [item["status"] for item in evidence["candidate_attempts"]] == [
        "candidate_error", "candidate_error", "candidate_error",
    ]
    assert (repo / "candidate.py").read_text() == "VALUE = 0\n"


def test_unchanged_candidate_repair_is_rejected(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
        return events("thread-unchanged", {"prepared": True})

    def controller(number, candidate, manifest):
        return candidate_error_receipt(candidate, manifest, number)

    with pytest.raises(contract.AuditError, match="repair did not change"):
        lifecycle.AuditedExperimentRunner(
            repo, prompt, task, invoke, controller, round_count=1,
        ).run("unchanged", "agent")


@pytest.mark.parametrize("value", [-1, 3, True])
def test_candidate_repair_public_limit_is_zero_to_two(tmp_path: Path, value):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    with pytest.raises(contract.AuditError, match="candidate repair count"):
        lifecycle.AuditedExperimentRunner(
            repo, prompt, task, lambda *args: "", lambda *args: {},
            max_candidate_repairs=value,
        )


@pytest.mark.parametrize("value", [0, 1, 2])
def test_candidate_repair_public_limit_accepts_bounded_values(tmp_path: Path, value: int):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, lambda *args: "", lambda *args: {},
        max_candidate_repairs=value,
    )
    assert runner.max_candidate_repairs == value


def test_interrupted_candidate_repair_resumes_exact_attempt_and_session(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    controller_calls = []

    class RepairInterrupted(RuntimeError):
        structured_stdout = events("thread-resume", {"interrupted": True})

    class Invoker:
        fail_repair = True

        def __call__(self, number, session, instruction):
            if instruction is None:
                (repo / "candidate.py").write_text("VALUE = 1\n")
                return events("thread-resume", {"prepared": True})
            if instruction.startswith("Repair candidate attempt") and self.fail_repair:
                self.fail_repair = False
                raise RepairInterrupted("transport ended")
            if instruction.startswith("Resume blocked experiment"):
                (repo / "candidate.py").write_text("VALUE = 2\n")
                return events("thread-resume", {"repaired": True})
            candidate = sha((repo / "candidate.py").read_bytes())
            manifest = sha((repo / "candidate.manifest.json").read_bytes())
            return events("thread-resume", report(number, candidate, manifest), command=False)

    invoker = Invoker()

    def controller(number, candidate, manifest):
        controller_calls.append(candidate)
        return (candidate_error_receipt(candidate, manifest, number)
                if len(controller_calls) == 1 else receipt(candidate, manifest, number))

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoker, controller, round_count=1,
    )
    with pytest.raises(contract.AuditError, match="checkpointed session thread-resume"):
        runner.run("resume-attempt", "agent")

    checkpoint = json.loads((repo / ".experiment/blocked.json").read_text())
    assert checkpoint["experiment"] == 1 and checkpoint["stage"] == "prepare"
    assert len(checkpoint["candidate_attempts"]) == 1

    result = runner.run("resume-attempt", "agent", resume=True)
    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert result.session_id == "thread-resume"
    assert len(controller_calls) == 2
    assert evidence["final_attempt"] == 2


def test_resumed_repair_must_differ_from_last_failed_attempt(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    interrupted = True

    class TurnError(RuntimeError):
        structured_stdout = events("thread-same", {"interrupted": True})

    def invoke(number, session, instruction):
        nonlocal interrupted
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-same", {"prepared": True})
        if instruction.startswith("Repair candidate attempt") and interrupted:
            interrupted = False
            raise TurnError("transport ended")
        return events("thread-same", {"prepared": True})

    def controller(number, candidate, manifest):
        return candidate_error_receipt(candidate, manifest, number)

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    )
    with pytest.raises(contract.AuditError, match="checkpointed session"):
        runner.run("same-resume", "agent")
    with pytest.raises(contract.AuditError, match="repair did not change"):
        runner.run("same-resume", "agent", resume=True)


def test_repaired_candidate_observer_interruption_resumes_exact_handle(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    repair_turns = 0

    terminal_receipts = {}

    def invoke(number, session, instruction):
        nonlocal repair_turns
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-infra", {"prepared": True})
        if instruction.startswith("Repair candidate attempt"):
            repair_turns += 1
            (repo / "candidate.py").write_text("VALUE = 2\n")
            return events("thread-infra", {"repaired": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        observed = terminal_receipts[number]
        document = report(number, candidate, manifest)
        document["controller_handle"] = observed["handle"]
        document["controller_receipt_sha256"] = contract.sha256_json(observed)
        return events("thread-infra", document, command=False)

    class Controller:
        calls = 0
        observations = 0

        def __call__(self, number, candidate, manifest):
            self.calls += 1
            if self.calls == 1:
                return candidate_error_receipt(candidate, manifest, number)
            return {
                "status": "infrastructure_error", "handle": "remote:retained",
                "terminal": False,
                "reason": "observer disconnected",
            }

        def observe(self, number, candidate, manifest, handle):
            self.observations += 1
            result = observed_transaction_receipt(candidate, manifest, number, handle)
            terminal_receipts[number] = result
            return result

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    )
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("infra-repair", "agent")
    checkpoint = json.loads((repo / ".experiment/blocked.json").read_text())
    assert len(checkpoint["candidate_attempts"]) == 1

    result = runner.run("infra-repair", "agent", resume=True)
    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert result.status == "complete" and len(result.commits) == 1
    assert repair_turns == 1
    assert controller.calls == 2 and controller.observations == 1
    assert evidence["controller"]["handle"] == "remote:final:1"
    assert evidence["final_attempt"] == 2
    assert [item["status"] for item in evidence["candidate_attempts"]] == [
        "candidate_error", "ok",
    ]


@pytest.mark.parametrize("mutation", [
    lambda history: history[1].update(request_sha256="0" * 64),
    lambda history: history[1].update(handle="remote:wrong"),
    lambda history: history[1].update(mode="retry_submit"),
    lambda history: history[0].update(terminal=True),
    lambda history: history[0].update(action="check"),
])
def test_observe_transaction_rejects_tampered_request_proof(mutation):
    candidate, manifest = sha("candidate"), sha("manifest")
    retained = "remote:retained"
    result = observed_transaction_receipt(candidate, manifest, 1, retained)
    mutation(result["policy"]["operation_history"])

    with pytest.raises(contract.AuditError):
        contract.validate_controller_receipt(result, candidate, manifest)
        contract.validate_observe_transaction(result, retained)


def test_observe_transaction_rejects_unrelated_interleaved_operation():
    candidate, manifest = sha("candidate"), sha("manifest")
    retained = "remote:retained"
    result = observed_transaction_receipt(candidate, manifest, 1, retained)
    unrelated = dict(result["policy"]["operation_history"][-1])
    unrelated.update(handle="remote:unrelated", terminal=True)
    result["policy"]["operation_history"].insert(1, unrelated)
    result["policy"]["submitted_handles"] = [
        retained, "remote:unrelated", result["handle"],
    ]
    result["policy"]["observed_handles"] = [
        "remote:unrelated", retained, result["handle"],
    ]

    contract.validate_controller_receipt(result, candidate, manifest)
    with pytest.raises(contract.AuditError, match="observe transaction proof"):
        contract.validate_observe_transaction(result, retained)


def test_observe_transaction_accepts_repeated_observer_transport_failure():
    candidate, manifest = sha("candidate"), sha("manifest")
    retained = "remote:retained"
    result = observed_transaction_receipt(candidate, manifest, 1, retained)
    failed_observe = dict(result["policy"]["operation_history"][1])
    failed_observe.update(status="infrastructure_error", terminal=False)
    result["policy"]["operation_history"].insert(1, failed_observe)
    result["policy"]["infra_retries"] = 2
    result["policy"]["retry_budget"] = 2

    contract.validate_controller_receipt(result, candidate, manifest)
    assert contract.validate_observe_transaction(result, retained) is result


@pytest.mark.parametrize(("field", "value"), [
    ("terminal", True),
    ("status", "ok"),
    ("candidate_sha256", "0" * 64),
    ("manifest_sha256", "1" * 64),
    ("experiment", True),
    ("experiment", 1.0),
    ("experiment", 2),
])
def test_repaired_candidate_resume_rejects_tampered_observation_receipt(
        tmp_path: Path, field: str, value: object):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-tamper", {"prepared": True})
        if instruction.startswith("Repair candidate attempt"):
            (repo / "candidate.py").write_text("VALUE = 2\n")
            return events("thread-tamper", {"repaired": True})
        raise AssertionError("resume must reject the checkpoint before another agent turn")

    class Controller:
        calls = 0

        def __call__(self, number, candidate, manifest):
            self.calls += 1
            if self.calls == 1:
                return candidate_error_receipt(candidate, manifest, number)
            return {
                "status": "infrastructure_error", "terminal": False,
                "handle": "remote:retained", "reason": "observer disconnected",
            }

        def observe(self, *args):
            raise AssertionError("tampered checkpoint must not observe a remote handle")

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, Controller(), round_count=1,
    )
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("tampered-repair", "agent")

    path = repo / ".experiment/blocked.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["receipt"][field] = value
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "add", str(path.relative_to(repo))], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--amend", "--no-edit", "-q"], cwd=repo, check=True)

    with pytest.raises(contract.AuditError, match="blocked checkpoint receipt is invalid"):
        runner.run("tampered-repair", "agent", resume=True)


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
    terminal_receipts = {}

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-fixed", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        observed = terminal_receipts.get(number)
        if observed is not None:
            document["controller_handle"] = observed["handle"]
            document["controller_receipt_sha256"] = contract.sha256_json(observed)
        return events("thread-fixed", document, command=False)

    class Controller:
        def __init__(self):
            self.submissions = 0
            self.observations = []

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            if self.submissions == 1:
                return {
                    "status": "infrastructure_error", "handle": "local:1",
                    "terminal": False,
                    "candidate_sha256": candidate, "manifest_sha256": manifest,
                    "reason": "observer disconnected",
                }
            return receipt(candidate, manifest, number)

        def observe(self, number, candidate, manifest, handle):
            self.observations.append((number, handle, candidate, manifest))
            document = observed_transaction_receipt(
                candidate, manifest, number, handle, handle,
            )
            terminal_receipts[number] = document
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


def test_chained_observer_interruption_checkpoints_new_handle_without_redispatch(
        tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    terminal_receipts = {}
    preparation_turns = 0

    def invoke(number, session, instruction):
        nonlocal preparation_turns
        if instruction is None:
            preparation_turns += 1
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-chain", {"prepared": True})
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        terminal = terminal_receipts[number]
        document["controller_handle"] = terminal["handle"]
        document["controller_receipt_sha256"] = contract.sha256_json(terminal)
        return events("thread-chain", document, command=False)

    class Controller:
        submissions = 0
        observations = []

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            return {
                "status": "infrastructure_error", "terminal": False,
                "handle": "remote:check", "reason": "check observer disconnected",
                "experiment": number, "candidate_sha256": candidate,
                "manifest_sha256": manifest,
            }

        def observe(self, number, candidate, manifest, handle):
            self.observations.append(handle)
            if handle == "remote:check":
                return observe_transition_receipt(
                    candidate, manifest, number, handle, "remote:profile",
                )
            result = chained_observed_receipt(
                candidate, manifest, number, "remote:check", handle,
            )
            terminal_receipts[number] = result
            return result

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    )
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("observe-chain", "agent")
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("observe-chain", "agent", resume=True)
    checkpoint = json.loads((repo / ".experiment/blocked.json").read_text())
    assert checkpoint["receipt"]["handle"] == "remote:profile"

    result = runner.run("observe-chain", "agent", resume=True)

    assert result.status == "complete" and len(result.commits) == 1
    assert controller.submissions == 1
    assert controller.observations == ["remote:check", "remote:profile"]
    assert preparation_turns == 1


def test_lifecycle_rejects_same_handle_retry_submit_as_observation(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        submissions = 0

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            return {"status": "infrastructure_error", "terminal": False,
                    "handle": "remote:same", "reason": "disconnected",
                    "candidate_sha256": candidate, "manifest_sha256": manifest}

        def observe(self, number, candidate, manifest, handle):
            result = observed_transaction_receipt(
                candidate, manifest, number, handle, handle,
            )
            result["policy"]["operation_history"][1]["mode"] = "retry_submit"
            return result

    controller = Controller()
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=1,
    )
    with pytest.raises(contract.AuditError, match="blocked by controller"):
        runner.run("same-handle-tamper", "agent")
    with pytest.raises(contract.AuditError, match="observe transaction proof"):
        runner.run("same-handle-tamper", "agent", resume=True)
    assert controller.submissions == 1


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
                "terminal": False,
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
    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert evidence["final_attempt"] == 1
    assert verifier.validate_branch(repo)["status"] == "valid"
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "main..HEAD"], cwd=repo, text=True
    ).strip() == "4"


def test_exhausted_candidate_error_repairs_retain_report_to_revert(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    finalizations = 0

    def invoke(number, session, instruction):
        nonlocal finalizations
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-revert", {"prepared": True})
        if instruction.startswith("Finalize") or instruction.startswith("Repair only"):
            finalizations += 1
            candidate = sha((repo / "candidate.py").read_bytes())
            manifest = sha((repo / "candidate.manifest.json").read_bytes())
            failed = candidate_error_receipt(candidate, manifest, number)
            document = report(
                number, candidate, manifest, "retain" if finalizations == 1 else "revert"
            )
            document["controller_receipt_sha256"] = contract.sha256_json(failed)
            return events("thread-revert", document, command=False)
        value = int((repo / "candidate.py").read_text().split()[-1]) + 1
        (repo / "candidate.py").write_text(f"VALUE = {value}\n")
        return events("thread-revert", {"repaired": True})

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: candidate_error_receipt(
            candidate, manifest, number
        ), round_count=1,
    ).run("forced-revert", "agent")

    assert finalizations == 2
    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert evidence["decision"] == "revert"


def test_successful_transient_retry_is_committed_as_distinct_audit_evidence(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    class Invoker:
        last_retry_evidence = ()

        def __call__(self, number, session, instruction):
            self.last_retry_evidence = ()
            if instruction is None:
                (repo / "candidate.py").write_text("VALUE = 1\n")
                self.last_retry_evidence = (retry_evidence("retry", 1),)
                return events("thread-retry", {"prepared": True})
            candidate = sha((repo / "candidate.py").read_bytes())
            manifest = sha((repo / "candidate.manifest.json").read_bytes())
            return events(
                "thread-retry", report(number, candidate, manifest), command=False,
            )

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, Invoker(),
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
        round_count=1,
    ).run("retry-success", "agent")

    retained = json.loads((repo / "experiments/01/evidence.json").read_text())
    assert retained["agent_retries"] == [{
        **retry_evidence("retry", 1), "experiment": 1, "stage": "prepare",
    }]
    assert "server_overloaded" not in (repo / "experiments/01/commands.jsonl").read_text()
    assert verifier.validate_branch(repo)["status"] == "valid"


def test_attempt_retains_bounded_sanitized_reasoning_content(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text("VALUE = 1\n")
            return events("thread-reasoning", {
                "prepared": True, "api_key": "must-not-survive", "detail": "x" * 9000,
            })
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        return events("thread-reasoning", report(number, candidate, manifest), command=False)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        lambda number, candidate, manifest: receipt(candidate, manifest, number),
        round_count=1,
    ).run("reasoning", "agent")

    evidence = json.loads((repo / "experiments/01/evidence.json").read_text())
    retained = (repo / "experiments/01/attempts/01/reasoning.txt").read_bytes()
    assert b"must-not-survive" not in retained and b"<redacted>" in retained
    assert len(retained) <= lifecycle.MAX_REASONING_OUTPUT
    assert sha(retained) == evidence["candidate_attempts"][0]["reasoning_sha256"]
    assert verifier.validate_branch(repo)["status"] == "valid"


def test_exhausted_transient_retry_is_retained_in_lifecycle_checkpoint(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)

    class Exhausted(RuntimeError):
        structured_stdout = json.dumps({
            "type": "thread.started", "thread_id": "thread-exhausted",
        }) + "\n"
        retry_evidence = (
            retry_evidence("retry", 1), retry_evidence("exhausted", 2),
        )

    def invoke(number, session, instruction):
        raise Exhausted("safe classified failure")

    with pytest.raises(contract.AuditError, match="checkpointed session"):
        lifecycle.AuditedExperimentRunner(
            repo, prompt, task, invoke,
            lambda number, candidate, manifest: receipt(candidate, manifest, number),
            round_count=1,
        ).run("retry-exhausted", "agent")

    retained = json.loads((repo / ".experiment/blocked.json").read_text())
    assert [entry["action"] for entry in retained["agent_retries"]] == [
        "retry", "exhausted",
    ]
    assert all(entry["stage"] == "prepare" for entry in retained["agent_retries"])
    assert "safe classified failure" not in json.dumps(retained["agent_retries"])


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
    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-fixed", {"prepared": True})
        encoded = instruction.split("receipt:\n", 1)[1].split(
            "\nThe frozen candidate", 1,
        )[0]
        controller_receipt = json.loads(encoded)
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        document["controller_handle"] = controller_receipt["handle"]
        document["controller_receipt_sha256"] = contract.sha256_json(
            controller_receipt
        )
        return events("thread-fixed", document, command=False)

    class Controller:
        def __init__(self):
            self.submissions = 0
            self.remeasurements = []

        def __call__(self, number, candidate, manifest):
            self.submissions += 1
            document = receipt(candidate, manifest, number)
            if number == 1:
                document["status"] = "measurement_pending"
                document["experiment"] = number
                document["handle"] = "local:1:old"
                document["policy"].update(
                    submitted_handles=[document["handle"]],
                    observed_handles=[document["handle"]],
                    measurement_generation=0,
                    operation_history=[{
                        "request_sha256": sha("measurement-0"), "mode": "submit",
                        "status": "ok", "terminal": True,
                        "handle": document["handle"], "action": "profile",
                        "attempt_id": "experiment-1-measurement-0-primary",
                    }],
                )
            return document

        def remeasure(self, number, candidate, manifest, handle, pending_receipt):
            self.remeasurements.append(
                (number, candidate, manifest, handle, pending_receipt)
            )
            document = receipt(candidate, manifest, number)
            document["experiment"] = number
            history = pending_receipt["policy"]["operation_history"] + [
                {"request_sha256": sha("measurement-1"), "mode": "submit",
                 "status": "ok", "terminal": True, "handle": document["handle"],
                 "action": "profile",
                 "attempt_id": "experiment-1-measurement-1-primary"},
                {"request_sha256": sha("measurement-1-after"), "mode": "submit",
                 "status": "ok", "terminal": True, "handle": "local:1:after",
                 "action": "calibrate",
                 "attempt_id": "experiment-1-measurement-1-after"},
            ]
            document["policy"].update(
                submitted_handles=[handle, document["handle"], "local:1:after"],
                observed_handles=[handle, document["handle"], "local:1:after"],
                measurement_generation=1, operation_history=history,
            )
            document["remeasure_transition"] = {
                "schema": contract.REMEASURE_TRANSITION_SCHEMA,
                "pending_receipt_sha256": contract.sha256_json(pending_receipt),
                "pending_handle": handle, "candidate_sha256": candidate,
                "manifest_sha256": manifest, "experiment": number,
                "from_measurement_generation": 0,
                "to_measurement_generation": 1,
                "profile_handle": document["handle"],
                "post_control_handle": "local:1:after",
                "operation_history_sha256": contract.sha256_json(history),
            }
            return document

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
    assert controller.remeasurements[0][4] == state["receipt"]
    assert controller.submissions == 3


def test_measurement_resume_rejects_unproved_custom_controller_transition(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    invoke, _ = updating_invoker(repo)

    class Controller:
        def __call__(self, number, candidate, manifest):
            document = receipt(candidate, manifest, number)
            if number == 1:
                document.update(status="measurement_pending", experiment=number)
                document["policy"].update(
                    measurement_generation=0,
                    operation_history=[{
                        "request_sha256": sha("measurement-0"), "mode": "submit",
                        "status": "ok", "terminal": True,
                        "handle": document["handle"], "action": "profile",
                        "attempt_id": "experiment-1-measurement-0-primary",
                    }],
                )
            return document

        def remeasure(self, number, candidate, manifest, handle, pending_receipt):
            document = receipt(candidate, manifest, number)
            document["handle"] = "local:1:unproved-fresh"
            document["policy"].update(
                submitted_handles=[document["handle"]],
                observed_handles=[document["handle"]],
            )
            return document

    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, Controller(),
    )
    with pytest.raises(contract.AuditError, match="measurement pending"):
        runner.run("unproved-remeasure", "agent")

    with pytest.raises(contract.AuditError, match="remeasure transition"):
        runner.run("unproved-remeasure", "agent", resume=True)
    blocked = json.loads((repo / ".experiment/blocked.json").read_text())
    assert blocked["stage"] == "controller"
    assert blocked["reason"] == "controller remeasure transition proof is invalid"


def test_real_lifecycle_remeasure_accepts_authenticated_new_handles(tmp_path: Path):
    repo = tmp_path / "repo"
    init_repo(repo)
    prompt, task = inputs(tmp_path)
    def invoke(number, session, instruction):
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            return events("thread-real", {"prepared": True})
        encoded = instruction.split(
            "receipt:\n", 1,
        )[1].split("\nThe frozen candidate", 1)[0]
        controller_receipt = json.loads(encoded)
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        document = report(number, candidate, manifest)
        document["controller_handle"] = controller_receipt["handle"]
        document["controller_receipt_sha256"] = contract.sha256_json(
            controller_receipt
        )
        return events("thread-real", document, command=False)
    backend = tmp_path / "backend.py"
    backend.write_text(
        """import hashlib, json, statistics, sys
from pathlib import Path
request=json.load(sys.stdin)
attempt=request.get('attempt_id') or request['action']
handle='job:'+hashlib.sha256(attempt.encode()).hexdigest()[:12]
if request['action'] == 'calibrate':
 result={'status':'ok','handle':handle,'latency_us':10.0,
         'samples_us':[9.0,10.0,11.0],'median_us':10.0}
elif request['action'] == 'check':
 result={'status':'ok','handle':handle,'passed':True}
else:
 samples=([1.0,10.0,100.0] if 'measurement-0' in attempt else [5.0,5.0,5.0])
 marker=Path(__file__).with_suffix('.measurement-1-seen')
 if 'measurement-1' in attempt and not marker.exists():
  marker.write_text('seen')
  result={'status':'infrastructure_error','handle':handle,
          'failure_type':'transport','diagnostics':'observer disconnected'}
 else:
  result={'status':'ok','handle':handle,'kernel_name':'kernel',
          'cases':[{'case':case,'samples_us':samples,
                    'median_us':statistics.median(samples)}
                   for case in request['cases']]}
print(json.dumps(result))
"""
    )
    baseline = {
        "schema": "profiling-skill/baseline-timing/v1", "benchmark": "matmul",
        "case_medians_us": [
            {"case": case, "median_us": value}
            for case, value in zip([7, 8, 9], [30.0, 33.0, 36.0])
        ],
        "control_median_us": 10.0,
    }
    baseline["sha256"] = contract.sha256_json(baseline)
    config = tmp_path / "controller.json"
    config.write_text(json.dumps({
        "schema": "profiling-skill/audited-bz-controller-config/v1",
        "benchmark": "matmul", "round_count": 1, "request_budget": 24,
        "profile_repeats": 3, "variability_threshold": 0.1,
        "control_drift_threshold": 0.2, "baseline": baseline,
        "devices": [{"id": "local/device-0", "device": 0}],
        "development_cases": [7, 8, 9], "all_cases": list(range(10)),
        "backend_command": [sys.executable, str(backend)],
    }))
    state = tmp_path / "controller-state"
    state.mkdir()
    command = [
        sys.executable, str(ROOT / "scripts/audited_bz_controller.py"),
        "--config", str(config), "--state-dir", str(state),
    ]
    runner = lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke,
        runtime.CommandController(command, repo, timeout=30), round_count=1,
    )

    with pytest.raises(contract.AuditError, match="measurement pending"):
        runner.run("real-remeasure", "agent")
    blocked = json.loads((repo / ".experiment/blocked.json").read_text())
    pending = blocked["receipt"]
    assert blocked["stage"] == "measurement"

    with pytest.raises(contract.AuditError, match="infrastructure"):
        runner.run("real-remeasure", "agent", resume=True)
    redirected = json.loads((repo / ".experiment/blocked.json").read_text())
    assert redirected["stage"] == "controller"
    assert redirected["receipt"]["handle"] != pending["handle"]
    contract.validate_remeasure_redirect(
        redirected["receipt"], pending["handle"], blocked["candidate_sha256"],
        blocked["manifest_sha256"], 1, pending,
    )

    result = runner.run("real-remeasure", "agent", resume=True)

    assert result.status == "complete" and len(result.commits) == 1
    controller_state = json.loads(next(state.glob("*/state.json")).read_text())
    terminal = controller_state["terminal"]
    contract.validate_remeasure_transition(
        terminal, pending, blocked["candidate_sha256"], blocked["manifest_sha256"], 1,
    )
    assert terminal["status"] == "ok"
    assert terminal["handle"] != pending["handle"]
    assert terminal["remeasure_transition"]["post_control_handle"] == \
        terminal["calibration"]["after"]["handle"]


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
