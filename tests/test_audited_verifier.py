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
import audited_verifier as verifier  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


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
            "admission_controls": [{"device": f"dynamic-{number}",
                                     "status": "pass", "healthy": True,
                                     "idle": True, "warmed": True}],
            "submission_candidate_sha256": candidate,
            "submitted_handles": [handle], "observed_handles": [handle],
            "infra_retries": 0, "retry_budget": 1, "quarantined_devices": [],
            "quarantine_controls": {}, "sample_count": 3,
            "variability_threshold": 0.25,
            "variability_ratio": 0.2 / (number + 0.1),
            "confirmation_count": 0, "post_control": "stable",
        },
    }


def events(thread: str, report: dict | None = None) -> str:
    stream = [{"type": "thread.started", "thread_id": thread}]
    if report is None:
        stream.append({"type": "item.completed", "item": {
            "type": "command_execution", "command": "python check.py",
            "exit_code": 0, "aggregated_output": "ok\n",
        }})
    else:
        stream.append({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(report),
        }})
    return "\n".join(json.dumps(item) for item in stream) + "\n"


def invalid_command_events(thread: str, report: dict) -> str:
    return "\n".join(json.dumps(item) for item in [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {
            "type": "command_execution", "command": "python broken.py",
            "exit_code": "not-an-integer", "aggregated_output": "malformed\n",
        }},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(report),
        }},
    ]) + "\n"


def build_branch(tmp_path: Path, *, revert: int | None = 2, round_count: int = 3,
                 invalid_identity: str | None = None,
                 repaired_attempts: bool = False) -> Path:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    git(repo, "config", "user.name", "Host Runner")
    git(repo, "config", "user.email", "host@example.test")
    (repo / "candidate.py").write_text("VALUE = 0\n")
    (repo / "candidate.manifest.json").write_text('{"candidate":"candidate.py"}\n')
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    prompt, task = tmp_path / "prompt.md", tmp_path / "task.md"
    prompt.write_text("Invariant prompt.\n")
    task.write_text("Make three toy changes.\n")
    bad_prepare = bad_finalize = repaired_attempts

    def invoke(number: int, session: str | None, instruction: str | None) -> str:
        nonlocal bad_prepare, bad_finalize
        if instruction is None:
            (repo / "candidate.py").write_text(f"VALUE = {number}\n")
            if number == 3 and bad_prepare:
                bad_prepare = False
                return invalid_command_events("one-session", {"prepared": True})
            return events("one-session")
        if instruction.startswith("Repair preparation evidence"):
            return events("one-session")
        candidate = sha((repo / "candidate.py").read_bytes())
        manifest = sha((repo / "candidate.manifest.json").read_bytes())
        controller = receipt(candidate, manifest, number)
        report = {
            "hypothesis": f"change {number} improves the toy",
            "expected_result": "the deterministic score changes",
            "change": f"set VALUE to {number}",
            "evidence": "local check and retained controller receipt",
            "observed_result": f"score={number}",
            "decision": "revert" if number == revert else "retain",
            "postmortem": "the evidence supports the decision",
            "next_experiment": "continue" if number < 3 else "done",
            "candidate_sha256": candidate, "manifest_sha256": manifest,
            "controller_handle": controller["handle"],
            "controller_receipt_sha256": contract.sha256_json(controller),
            "sources": [], "no_sources_reason": "The toy is self-contained.",
        }
        if number == 3 and instruction.startswith("Finalize") and bad_finalize:
            bad_finalize = False
            return invalid_command_events("one-session", report)
        return events("one-session", report)

    def identity(document: dict, invalid: str | None = None) -> dict:
        metadata = dict(document)
        metadata["identity_sha256"] = contract.sha256_json(metadata)
        if invalid == "stale-model":
            metadata["model"] = "mutated-after-hashing"
        elif invalid == "stale-image":
            metadata["docker_image_id"] = "sha256:mutated-after-hashing"
        elif invalid == "stale-controller":
            metadata["argv"] = ["fixture-controller", "--different-mode"]
        elif invalid == "wrong-scope":
            metadata["identity_sha256"] = contract.sha256_json({
                "adapter": metadata["adapter"],
            })
        return metadata

    invoke.reproducibility_metadata = lambda: identity({  # type: ignore[attr-defined]
        "adapter": "FixtureAgent", "model": "fixture-model",
        "docker_image_id": "sha256:fixture-image",
    }, invalid_identity if invalid_identity != "stale-controller" else None)

    def controller(number: int, candidate: str, manifest: str) -> dict:
        return receipt(candidate, manifest, number)

    controller.reproducibility_metadata = lambda: identity({  # type: ignore[attr-defined]
        "adapter": "FixtureController", "argv": ["fixture-controller", "--compact"],
        "executable_sha256": "1" * 64,
    }, invalid_identity if invalid_identity == "stale-controller" else None)

    lifecycle.AuditedExperimentRunner(
        repo, prompt, task, invoke, controller, round_count=round_count,
    ).run("validation", "agent-a")
    return repo


def amend(repo: Path, path: str, transform) -> None:
    target = repo / path
    transform(target)
    git(repo, "add", path)
    git(repo, "commit", "--amend", "--no-edit")


def rewrite_json(path: Path, transform) -> None:
    document = json.loads(path.read_text())
    transform(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def rewrite_first_command(path: Path, **changes) -> None:
    commands = [json.loads(line) for line in path.read_text().splitlines()]
    commands[0].update(changes)
    path.write_text("".join(json.dumps(item) + "\n" for item in commands))


def test_validates_complete_unmerged_branch_and_revert(tmp_path: Path):
    repo = build_branch(tmp_path)

    result = verifier.validate_branch(repo)

    assert result["status"] == "valid"
    assert result["branch"] == "experiment/validation/agent-a"
    assert result["session_id"] == "one-session"
    assert [item["decision"] for item in result["experiments"]] == [
        "retain", "revert", "retain",
    ]


def test_validates_v2_seed_with_four_rounds(tmp_path: Path):
    repo = build_branch(tmp_path, round_count=4)

    result = verifier.validate_branch(repo)

    assert result["round_count"] == 4
    assert len(result["experiments"]) == 4
    assert [item["experiment"] for item in result["experiments"]] == [1, 2, 3, 4]


def test_legacy_v1_seed_still_means_three_rounds(tmp_path: Path):
    repo = build_branch(tmp_path)
    seed = json.loads((repo / ".experiment/seed.json").read_text())

    assert seed["schema"] == "profiling-skill/audited-seed/v1"
    result = verifier.validate_branch(repo)
    assert result["status"] == "valid" and result["round_count"] == 3


def test_repaired_malformed_attempt_markers_validate_but_extra_fields_reject(
    tmp_path: Path,
):
    repo = build_branch(tmp_path, repaired_attempts=True)
    commands_path = repo / "experiments/03/commands.jsonl"
    commands = [json.loads(line) for line in commands_path.read_text().splitlines()]
    markers = [item for item in commands
               if item["command"] == "<invalid structured command event>"]

    assert len(markers) == 2
    assert all(set(marker) == verifier.COMMAND_FIELDS for marker in markers)
    assert verifier.validate_branch(repo)["status"] == "valid"

    commands[0]["untrusted_detail"] = "must not enter the retained schema"
    commands_path.write_text("".join(json.dumps(item) + "\n" for item in commands))
    git(repo, "add", "experiments/03/commands.jsonl")
    git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(contract.AuditError, match="command evidence"):
        verifier.validate_branch(repo)


@pytest.mark.parametrize("author", [
    "Other Agent <agent-a@experiment.invalid>",
    "Experiment Agent agent-a <other@experiment.invalid>",
])
def test_rejects_experiment_commit_author_not_derived_from_seed(
    tmp_path: Path, author: str,
):
    repo = build_branch(tmp_path)
    git(repo, "commit", "--amend", "--no-edit", "--author", author)

    with pytest.raises(contract.AuditError, match="author identity"):
        verifier.validate_branch(repo)


@pytest.mark.parametrize("exit_code", [None, True, -1, 256])
def test_rejects_command_exit_code_outside_producer_domain(
    tmp_path: Path, exit_code,
):
    repo = build_branch(tmp_path)
    amend(repo, "experiments/03/commands.jsonl",
          lambda path: rewrite_first_command(path, exit_code=exit_code))

    with pytest.raises(contract.AuditError, match="command evidence"):
        verifier.validate_branch(repo)


@pytest.mark.parametrize("invalid_identity", [
    "stale-model", "stale-image", "stale-controller", "wrong-scope",
])
def test_rejects_provenance_hash_not_bound_to_all_identity_content(
    tmp_path: Path, invalid_identity: str,
):
    repo = build_branch(tmp_path, invalid_identity=invalid_identity)

    with pytest.raises(contract.AuditError, match="identity hash"):
        verifier.validate_branch(repo)


@pytest.mark.parametrize(("path", "transform", "match"), [
    ("experiments/03/commands.jsonl",
     lambda p: p.write_text('{}\n'), "command evidence"),
    ("experiments/03/sources.json",
     lambda p: rewrite_json(p, lambda d: d.update(no_sources_reason="")), "sources"),
    ("experiments/03/evidence.json",
     lambda p: rewrite_json(p, lambda d: d.update(session_id="other")), "session"),
    ("experiments/03/results.json",
     lambda p: rewrite_json(p, lambda d: d.update(candidate_sha256="0" * 64)),
     "different candidate"),
    ("experiments/03/report.md",
     lambda p: p.write_text("# Experiment 3\n"), "report"),
    ("experiments/03/evidence.json",
     lambda p: rewrite_json(p, lambda d: d.update(controller_receipt_sha256="0" * 64)),
     "receipt hash"),
])
def test_rejects_tampered_artifact(tmp_path: Path, path: str, transform, match: str):
    repo = build_branch(tmp_path)
    amend(repo, path, transform)

    with pytest.raises(contract.AuditError, match=match):
        verifier.validate_branch(repo)


def test_rejects_changed_prior_evidence_tree(tmp_path: Path):
    repo = build_branch(tmp_path)
    amend(repo, "experiments/01/report.md", lambda p: p.write_text(p.read_text() + "later\n"))

    with pytest.raises(contract.AuditError, match="evidence changed"):
        verifier.validate_branch(repo)


def test_rejects_revert_archive_or_restored_manifest_tampering(tmp_path: Path):
    archive_repo = build_branch(tmp_path / "archive")
    amend(archive_repo, "experiments/02/tested_candidate.py",
          lambda p: p.write_text("VALUE = 999\n"))
    with pytest.raises(contract.AuditError, match="revert evidence|evidence changed"):
        verifier.validate_branch(archive_repo)

    manifest_repo = build_branch(tmp_path / "manifest", revert=3)
    (manifest_repo / "candidate.manifest.json").write_text('{"changed":true}\n')
    git(manifest_repo, "add", "candidate.manifest.json")
    git(manifest_repo, "commit", "--amend", "--no-edit")
    with pytest.raises(contract.AuditError, match="restoration"):
        verifier.validate_branch(manifest_repo)


def test_rejects_dirty_merged_or_non_linear_history(tmp_path: Path):
    dirty = build_branch(tmp_path / "dirty")
    (dirty / "scratch").write_text("dirty\n")
    with pytest.raises(contract.AuditError, match="clean"):
        verifier.validate_branch(dirty)

    merged = build_branch(tmp_path / "merged")
    git(merged, "branch", "-f", "main", "HEAD")
    with pytest.raises(contract.AuditError, match="unmerged"):
        verifier.validate_branch(merged)

    extra = build_branch(tmp_path / "extra")
    (extra / "extra").write_text("unexpected\n")
    git(extra, "add", "extra")
    git(extra, "commit", "-m", "extra")
    with pytest.raises(contract.AuditError, match="one seed and 3"):
        verifier.validate_branch(extra)


def test_cli_prints_machine_readable_summary(tmp_path: Path):
    repo = build_branch(tmp_path)
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_audited_experiment.py"), str(repo)],
        text=True, capture_output=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "valid"
