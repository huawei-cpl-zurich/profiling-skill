from __future__ import annotations

import hashlib
import json
import shutil
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
                 repaired_attempts: bool = False,
                 candidate_failures: int = 0) -> Path:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    git(repo, "config", "user.name", "Host Runner")
    git(repo, "config", "user.email", "host@example.test")
    (repo / "candidate.py").write_text("VALUE = 0\n")
    (repo / "candidate.manifest.json").write_text(
        '{"schema":"profiling-skill/candidate-kernel/v1","kernel_name":"toy_kernel"}\n'
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    prompt, task = tmp_path / "prompt.md", tmp_path / "task.md"
    prompt.write_text("Invariant prompt.\n")
    task.write_text("Make three toy changes.\n")
    bad_prepare = bad_finalize = repaired_attempts
    controller_counts: dict[int, int] = {}

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
        if instruction.startswith("Repair candidate attempt"):
            value = int((repo / "candidate.py").read_text().split()[-1]) + 10
            (repo / "candidate.py").write_text(f"VALUE = {value}\n")
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
        controller_counts[number] = controller_counts.get(number, 0) + 1
        result = receipt(candidate, manifest, number)
        if controller_counts[number] <= candidate_failures:
            result.update(status="candidate_error", reason="compile failed")
            result.pop("samples_us")
            result.pop("median_us")
            result["policy"].pop("sample_count")
            result["policy"].pop("variability_ratio")
            result["policy"]["post_control"] = "not_run"
        return result

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


def migrated_branch(tmp_path: Path, boundary: int) -> tuple[Path, dict, Path]:
    repo = build_branch(tmp_path, round_count=4)
    commits = git(repo, "rev-list", "--first-parent", "--reverse", "main..HEAD").splitlines()
    seed_commit, experiments = commits[0], commits[1:]
    seed = json.loads((repo / ".experiment/seed.json").read_text())
    old_identity = seed["reproducibility"]["controller"]
    new_identity = {
        "adapter": "FixtureController",
        "argv": ["fixture-controller", "--migration-aware"],
        "executable_sha256": "2" * 64,
    }
    new_identity["identity_sha256"] = contract.sha256_json(new_identity)

    git(repo, "reset", "--hard", seed_commit)
    rewritten = []
    for number, commit in enumerate(experiments, 1):
        git(repo, "cherry-pick", commit)
        if number >= boundary:
            evidence_path = repo / f"experiments/{number:02d}/evidence.json"
            rewrite_json(
                evidence_path,
                lambda value: value["reproducibility"].update(
                    {"controller": new_identity}
                ),
            )
            git(repo, "add", str(evidence_path.relative_to(repo)))
            git(repo, "commit", "--amend", "--no-edit")
        rewritten.append(git(repo, "rev-parse", "HEAD"))

    cell = {
        "cell_id": "agent-a", "experiment": boundary,
        "branch": "experiment/validation/agent-a", "seed_commit": seed_commit,
        "resume_parent": seed_commit if boundary == 1 else rewritten[boundary - 2],
        "old_controller_identity": old_identity,
        "new_controller_identity": new_identity,
    }
    attestation = {
        "schema": "profiling-skill/audited-runtime-preflight/v1",
        "migration_id": f"fixture-round-{boundary}", "plan_sha256": "3" * 64,
        "cells": [cell], "runtimes": {"old": {}, "new": {}},
    }
    attestation["attestation_sha256"] = contract.sha256_json(attestation)
    path = tmp_path / f"attestation-{boundary}.json"
    path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
    proof = {
        "schema": "profiling-skill/audited-runtime-migration-trust/v1",
        "attestation_path": str(path.resolve()),
        "attestation_file_sha256": sha(path.read_bytes()),
        "attestation_sha256": attestation["attestation_sha256"],
    }
    return repo, proof, path


@pytest.mark.parametrize("boundary", [1, 3])
def test_validates_sealed_controller_migration_at_exact_round(
    tmp_path: Path, boundary: int,
):
    repo, proof, _ = migrated_branch(tmp_path, boundary)

    with pytest.raises(contract.AuditError, match="provenance diverges"):
        verifier.validate_branch(repo)

    result = verifier.validate_branch(repo, migration_proofs=[proof])

    assert result["status"] == "valid"
    assert result["controller_migrations"] == [{
        "migration_id": f"fixture-round-{boundary}",
        "experiment": boundary,
        "attestation_sha256": proof["attestation_sha256"],
    }]


@pytest.mark.parametrize("defect", [
    "missing", "tampered", "wrong-cell", "wrong-boundary", "identity", "nonmonotonic",
])
def test_rejects_invalid_controller_migration_proof(tmp_path: Path, defect: str):
    repo, proof, path = migrated_branch(tmp_path, 3)
    proofs = [proof]
    if defect == "missing":
        proofs = []
    elif defect == "tampered":
        path.write_text(path.read_text() + " ")
    else:
        attestation = json.loads(path.read_text())
        cell = attestation["cells"][0]
        if defect == "wrong-cell":
            cell["cell_id"] = "other-agent"
        elif defect == "wrong-boundary":
            cell["experiment"] = 2
        elif defect == "identity":
            cell["old_controller_identity"] = cell["new_controller_identity"]
        else:
            second = json.loads(json.dumps(attestation))
            second["migration_id"] = "earlier-after-later"
            second["cells"][0]["experiment"] = 2
            second["attestation_sha256"] = contract.sha256_json({
                key: value for key, value in second.items()
                if key != "attestation_sha256"
            })
            second_path = tmp_path / "attestation-2.json"
            second_path.write_text(json.dumps(second, sort_keys=True) + "\n")
            proofs.append({
                "schema": proof["schema"],
                "attestation_path": str(second_path.resolve()),
                "attestation_file_sha256": sha(second_path.read_bytes()),
                "attestation_sha256": second["attestation_sha256"],
            })
            with pytest.raises(contract.AuditError, match="migration proof"):
                verifier.validate_branch(repo, migration_proofs=proofs)
            return
        attestation["attestation_sha256"] = contract.sha256_json({
            key: value for key, value in attestation.items()
            if key != "attestation_sha256"
        })
        path.write_text(json.dumps(attestation, sort_keys=True) + "\n")
        proof["attestation_file_sha256"] = sha(path.read_bytes())
        proof["attestation_sha256"] = attestation["attestation_sha256"]

    with pytest.raises(contract.AuditError, match="migration proof|provenance diverges"):
        verifier.validate_branch(repo, migration_proofs=proofs)


def test_cli_accepts_an_ordered_pinned_migration_proof(tmp_path: Path):
    repo, proof, _ = migrated_branch(tmp_path, 3)

    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/validate_audited_experiment.py"),
        str(repo), "--migration-proof", proof["attestation_path"],
        proof["attestation_file_sha256"], proof["attestation_sha256"],
    ], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["controller_migrations"][0]["experiment"] == 3


def test_validates_ordered_controller_migration_chain(tmp_path: Path):
    repo, first, first_path = migrated_branch(tmp_path, 3)
    first_attestation = json.loads(first_path.read_text())
    first_cell = first_attestation["cells"][0]
    second_identity = {
        "adapter": "FixtureController", "argv": ["fixture-controller", "--v5"],
        "executable_sha256": "4" * 64,
    }
    second_identity["identity_sha256"] = contract.sha256_json(second_identity)
    evidence = repo / "experiments/04/evidence.json"
    rewrite_json(
        evidence,
        lambda value: value["reproducibility"].update(
            {"controller": second_identity}
        ),
    )
    git(repo, "add", "experiments/04/evidence.json")
    git(repo, "commit", "--amend", "--no-edit")
    second_attestation = {
        "schema": "profiling-skill/audited-runtime-preflight/v1",
        "migration_id": "fixture-round-4", "plan_sha256": "5" * 64,
        "runtimes": {"old": {}, "new": {}},
        "cells": [{
            "cell_id": "agent-a", "experiment": 4,
            "branch": "experiment/validation/agent-a",
            "seed_commit": first_cell["seed_commit"],
            "resume_parent": git(repo, "rev-parse", "HEAD^"),
            "old_controller_identity": first_cell["new_controller_identity"],
            "new_controller_identity": second_identity,
        }],
    }
    second_attestation["attestation_sha256"] = contract.sha256_json(
        second_attestation
    )
    second_path = tmp_path / "attestation-4.json"
    second_path.write_text(json.dumps(second_attestation, sort_keys=True) + "\n")
    second = {
        "schema": first["schema"], "attestation_path": str(second_path.resolve()),
        "attestation_file_sha256": sha(second_path.read_bytes()),
        "attestation_sha256": second_attestation["attestation_sha256"],
    }

    result = verifier.validate_branch(repo, migration_proofs=[first, second])

    assert [item["experiment"] for item in result["controller_migrations"]] == [3, 4]


def test_validates_complete_unmerged_branch_and_revert(tmp_path: Path):
    repo = build_branch(tmp_path)

    result = verifier.validate_branch(repo)

    assert result["status"] == "valid"
    assert result["branch"] == "experiment/validation/agent-a"
    assert result["run_id"] == "validation"
    assert result["agent_id"] == "agent-a"
    assert result["session_id"] == "one-session"
    assert [item["decision"] for item in result["experiments"]] == [
        "retain", "revert", "retain",
    ]


def test_rejects_retry_evidence_with_raw_service_message(tmp_path: Path):
    repo = build_branch(tmp_path)

    def inject(document: dict) -> None:
        document["agent_retries"] = [{
            "schema": "profiling-skill/codex-transient-retry/v1",
            "terminal_error": "server_overloaded", "action": "retry",
            "attempt": 1, "stdout_sha256": "1" * 64, "stdout_bytes": 1,
            "stderr_sha256": "2" * 64, "stderr_bytes": 2,
            "experiment": 3, "stage": "prepare",
            "raw_message": "must not be retained",
        }]

    amend(repo, "experiments/03/evidence.json", lambda path: rewrite_json(path, inject))

    with pytest.raises(contract.AuditError, match="agent retry evidence"):
        verifier.validate_branch(repo)


def test_accepts_legacy_evidence_without_retry_field(tmp_path: Path):
    repo = build_branch(tmp_path)
    amend(
        repo, "experiments/03/evidence.json",
        lambda path: rewrite_json(path, lambda document: document.pop("agent_retries")),
    )

    assert verifier.validate_branch(repo)["status"] == "valid"


def test_rejects_actual_candidate_attempt_reordering(
    tmp_path: Path,
):
    repo = build_branch(tmp_path, candidate_failures=2)

    assert verifier.validate_branch(repo)["status"] == "valid"
    evidence_path = repo / "experiments/03/evidence.json"
    rewrite_json(evidence_path, lambda document: document["candidate_attempts"].reverse())
    attempts = repo / "experiments/03/attempts"
    (attempts / "01").rename(attempts / "tmp")
    (attempts / "02").rename(attempts / "01")
    (attempts / "tmp").rename(attempts / "02")
    git(repo, "add", "-A", "experiments/03")
    git(repo, "commit", "--amend", "--no-edit")

    with pytest.raises(contract.AuditError, match="candidate attempt"):
        verifier.validate_branch(repo)


def test_rejects_cross_round_candidate_attempt_substitution(tmp_path: Path):
    repo = build_branch(tmp_path, candidate_failures=1)
    first = json.loads((repo / "experiments/01/evidence.json").read_text())
    second_path = repo / "experiments/03/evidence.json"
    second = json.loads(second_path.read_text())
    second["candidate_attempts"][0] = first["candidate_attempts"][0]
    second_path.write_text(json.dumps(second, indent=2, sort_keys=True) + "\n")
    target = repo / "experiments/03/attempts/01"
    shutil.rmtree(target)
    shutil.copytree(repo / "experiments/01/attempts/01", target)
    git(repo, "add", "-A", "experiments/03")
    git(repo, "commit", "--amend", "--no-edit")

    with pytest.raises(contract.AuditError, match="candidate attempt"):
        verifier.validate_branch(repo)


def test_accepts_pre_attempt_legacy_evidence(tmp_path: Path):
    repo = build_branch(tmp_path)
    commits = git(repo, "rev-list", "--first-parent", "--reverse", "main..HEAD").splitlines()
    seed, experiments = commits[0], commits[1:]
    git(repo, "reset", "--hard", seed)
    for number, commit in enumerate(experiments, 1):
        git(repo, "cherry-pick", commit)
        root = repo / f"experiments/{number:02d}"
        subprocess.run(
            ["git", "rm", "-qr", f"experiments/{number:02d}/attempts"],
            cwd=repo, check=True,
        )
        rewrite_json(
            root / "evidence.json",
            lambda document: (
                document.pop("candidate_attempts"), document.pop("final_attempt")
            ),
        )
        git(repo, "add", str(root / "evidence.json"))
        git(repo, "commit", "--amend", "--no-edit")

    assert verifier.validate_branch(repo)["status"] == "valid"


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
