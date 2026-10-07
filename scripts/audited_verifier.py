#!/usr/bin/env python3
"""Offline validation for a completed audited-experiment Git branch."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

from audited_contract import (
    AuditError,
    MAX_EXCERPT,
    MAX_REPORT_FIELD,
    REPORT_FIELDS,
    sha256_bytes,
    sha256_json,
    validate_controller_receipt,
    validate_sources,
)

SEED_SCHEMA_V1 = "profiling-skill/audited-seed/v1"
SEED_SCHEMA_V2 = "profiling-skill/audited-seed/v2"
EVIDENCE_SCHEMA = "profiling-skill/audited-evidence/v1"
STANDARD_ARTIFACTS = {
    "commands.jsonl", "evidence.json", "report.md", "results.json", "sources.json",
}
REVERT_ARTIFACTS = {"tested_candidate.py", "tested_candidate.manifest.json"}
EVIDENCE_FIELDS = {
    "schema", "session_id", "candidate_sha256", "tested_candidate_sha256",
    "committed_candidate_sha256", "restored_candidate_sha256", "decision",
    "controller_receipt_sha256", "manifest_sha256", "prompt_sha256",
    "task_sha256", "reproducibility", "controller",
}
ATTEMPT_FIELDS = {
    "schema", "attempt", "status", "candidate_sha256", "manifest_sha256",
    "controller_receipt_sha256", "commands_sha256", "reasoning_sha256",
}
RETRY_FIELDS = {
    "schema", "terminal_error", "action", "attempt", "stdout_sha256",
    "stdout_bytes", "stderr_sha256", "stderr_bytes", "experiment", "stage",
}
COMMAND_FIELDS = {
    "command", "exit_code", "output_sha256", "output_excerpt", "output_truncated",
}
_HASH = re.compile(r"[0-9a-f]{64}")
MIGRATION_ATTESTATION_SCHEMA = "profiling-skill/audited-runtime-preflight/v1"
MIGRATION_TRUST_SCHEMA = "profiling-skill/audited-runtime-migration-trust/v1"
MIGRATION_TRUST_FIELDS = {
    "schema", "attestation_path", "attestation_file_sha256", "attestation_sha256",
}


def _run(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, text=True, capture_output=True, check=False,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise AuditError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def _git(repo: Path, *arguments: str) -> str:
    return _run(repo, *arguments).stdout.strip()


def _blob(repo: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=False,
    )
    if result.returncode:
        raise AuditError(f"missing retained artifact {path} at {commit}")
    return result.stdout


def _json_blob(repo: Path, commit: str, path: str) -> dict:
    try:
        document = json.loads(_blob(repo, commit, path))
    except json.JSONDecodeError as failure:
        raise AuditError(f"retained artifact {path} is invalid JSON") from failure
    if not isinstance(document, dict):
        raise AuditError(f"retained artifact {path} must be a JSON object")
    return document


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _validate_retries(value: object, experiment: int) -> None:
    if not isinstance(value, list) or len(value) > 64:
        raise AuditError(f"experiment {experiment} agent retry evidence is invalid")
    for record in value:
        if (not isinstance(record, dict) or set(record) != RETRY_FIELDS
                or record.get("schema") != "profiling-skill/codex-transient-retry/v1"
                or record.get("terminal_error") != "server_overloaded"
                or record.get("action") not in {"retry", "exhausted"}
                or type(record.get("attempt")) is not int
                or not 1 <= record["attempt"] <= 6
                or not _hash(record.get("stdout_sha256"))
                or not _hash(record.get("stderr_sha256"))
                or any(type(record.get(key)) is not int or record[key] < 0
                       for key in ("stdout_bytes", "stderr_bytes"))
                or record.get("experiment") != experiment
                or record.get("stage") not in {"prepare", "finalize"}):
            raise AuditError(f"experiment {experiment} agent retry evidence is invalid")


def _validate_provenance(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"agent", "controller"}:
        raise AuditError("seed provenance must identify the agent and controller")
    for name in ("agent", "controller"):
        identity = value[name]
        if (not isinstance(identity, dict)
                or not isinstance(identity.get("adapter"), str)
                or not identity["adapter"]
                or not _hash(identity.get("identity_sha256"))):
            raise AuditError(f"seed {name} provenance is incomplete")
        canonical = {
            key: item for key, item in identity.items() if key != "identity_sha256"
        }
        # The lifecycle's callable fallback predates structured identities and
        # hashes its sole adapter name directly. Runtime adapters hash the full
        # canonical JSON identity before adding identity_sha256.
        expected = (
            sha256_bytes(identity["adapter"].encode())
            if set(canonical) == {"adapter"}
            else sha256_json(canonical)
        )
        if identity["identity_sha256"] != expected:
            raise AuditError(f"seed {name} identity hash does not bind its metadata")
        if (identity["adapter"] == "CodexInvoker:docker"
                and (not isinstance(identity.get("docker_image_id"), str)
                     or not identity["docker_image_id"].startswith("sha256:"))):
            raise AuditError("seed Docker image provenance is not immutable")
    return value


def _parse_report(data: bytes, number: int) -> dict[str, str]:
    try:
        text = data.decode()
    except UnicodeDecodeError as failure:
        raise AuditError(f"experiment {number} report is not UTF-8") from failure
    title = f"# Experiment {number}\n\n"
    if not text.startswith(title):
        raise AuditError(f"experiment {number} report has an invalid title")
    remaining = text[len(title):]
    values: dict[str, str] = {}
    for index, field in enumerate(REPORT_FIELDS):
        heading = f"## {field.replace('_', ' ').title()}\n\n"
        if not remaining.startswith(heading):
            raise AuditError(f"experiment {number} report is missing {field}")
        remaining = remaining[len(heading):]
        if index + 1 < len(REPORT_FIELDS):
            next_heading = f"\n\n## {REPORT_FIELDS[index + 1].replace('_', ' ').title()}\n\n"
            value, separator, remaining = remaining.partition(next_heading)
            if not separator:
                raise AuditError(f"experiment {number} report is missing "
                                 f"{REPORT_FIELDS[index + 1]}")
            # The next iteration expects its heading, so put it back.
            remaining = next_heading[2:] + remaining
        else:
            value, remaining = remaining.rstrip("\n"), ""
        if not value.strip() or len(value) > MAX_REPORT_FIELD:
            raise AuditError(f"experiment {number} report has invalid {field}")
        values[field] = value
    if remaining:
        raise AuditError(f"experiment {number} report has unexpected content")
    return values


def _validate_commands(data: bytes, number: int) -> list[dict]:
    try:
        lines = data.decode().splitlines()
        commands = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise AuditError(f"experiment {number} command evidence is invalid") from failure
    if not commands:
        raise AuditError(f"experiment {number} command evidence is empty")
    for command in commands:
        if (not isinstance(command, dict) or set(command) != COMMAND_FIELDS
                or not isinstance(command["command"], str) or not command["command"]
                or type(command["exit_code"]) is not int
                or not 0 <= command["exit_code"] <= 255
                or not _hash(command["output_sha256"])
                or not isinstance(command["output_excerpt"], str)
                or len(command["output_excerpt"]) > MAX_EXCERPT
                or type(command["output_truncated"]) is not bool):
            raise AuditError(f"experiment {number} command evidence is invalid")
    return commands


def _validate_candidate_attempts(repo: Path, commit: str, root: str,
                                 evidence: dict, results: dict,
                                 number: int) -> set[str]:
    attempts = evidence.get("candidate_attempts")
    if (not isinstance(attempts, list) or not 1 <= len(attempts) <= 3
            or evidence.get("final_attempt") != len(attempts)):
        raise AuditError(f"experiment {number} candidate attempt history is invalid")
    artifacts: set[str] = set()
    previous_identity = None
    for index, attempt in enumerate(attempts, 1):
        prefix = f"attempts/{index:02d}"
        artifacts |= {
            f"{prefix}/candidate.py", f"{prefix}/candidate.manifest.json",
            f"{prefix}/controller.json", f"{prefix}/commands.jsonl",
            f"{prefix}/reasoning.sha256",
        }
        if (not isinstance(attempt, dict) or set(attempt) != ATTEMPT_FIELDS
                or attempt.get("schema") != "profiling-skill/candidate-attempt/v1"
                or attempt.get("attempt") != index
                or attempt.get("status") not in {"ok", "candidate_error"}
                or any(not _hash(attempt.get(field)) for field in (
                    "candidate_sha256", "manifest_sha256",
                    "controller_receipt_sha256", "commands_sha256",
                    "reasoning_sha256",
                ))):
            raise AuditError(f"experiment {number} candidate attempt {index} is invalid")
        candidate = _blob(repo, commit, f"{root}/{prefix}/candidate.py")
        manifest = _blob(repo, commit, f"{root}/{prefix}/candidate.manifest.json")
        receipt = _json_blob(repo, commit, f"{root}/{prefix}/controller.json")
        commands = _validate_commands(
            _blob(repo, commit, f"{root}/{prefix}/commands.jsonl"), number
        )
        reasoning = _blob(repo, commit, f"{root}/{prefix}/reasoning.sha256")
        if (sha256_bytes(candidate) != attempt["candidate_sha256"]
                or sha256_bytes(manifest) != attempt["manifest_sha256"]
                or sha256_json(receipt) != attempt["controller_receipt_sha256"]
                or sha256_json(commands) != attempt["commands_sha256"]
                or reasoning != (attempt["reasoning_sha256"] + "\n").encode()
                or receipt.get("status") != attempt["status"]):
            raise AuditError(f"experiment {number} candidate attempt {index} hash is invalid")
        validate_controller_receipt(
            receipt, attempt["candidate_sha256"], attempt["manifest_sha256"]
        )
        identity = (attempt["candidate_sha256"], attempt["manifest_sha256"])
        if identity == previous_identity:
            raise AuditError(f"experiment {number} candidate repair did not change inputs")
        previous_identity = identity
        if index < len(attempts) and attempt["status"] != "candidate_error":
            raise AuditError(f"experiment {number} candidate attempt ordering is invalid")
    last = attempts[-1]
    if (last["candidate_sha256"] != evidence["tested_candidate_sha256"]
            or last["manifest_sha256"] != evidence["manifest_sha256"]
            or last["controller_receipt_sha256"] != sha256_json(results)):
        raise AuditError(f"experiment {number} final candidate attempt is not bound")
    return artifacts


def _immutable(repo: Path, commits: list[str], path: str, expected: bytes) -> None:
    if any(_blob(repo, commit, path) != expected for commit in commits):
        raise AuditError(f"immutable seed artifact {path} changed")


def _validate_commit_paths(repo: Path, parent: str, commit: str, root: str,
                           expected_artifacts: set[str]) -> None:
    changed = set(_git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r",
                       parent, commit).splitlines())
    allowed = {"candidate.py", "candidate.manifest.json"} | {
        f"{root}/{name}" for name in expected_artifacts
    }
    if not changed.issubset(allowed):
        raise AuditError("experiment commit changed files outside its retained evidence")


def _migration_chain(repo: Path, proofs: Sequence[dict], *, branch: str,
                     agent_id: str, seed_commit: str, seed_provenance: dict,
                     experiment_commits: list[str], round_count: int) -> tuple[list[dict], list[dict]]:
    """Authenticate ordered controller transitions and derive per-round provenance."""
    expected = json.loads(json.dumps(seed_provenance))
    per_round = [json.loads(json.dumps(expected)) for _ in range(round_count)]
    summaries: list[dict] = []
    previous_boundary = 0
    for proof in proofs:
        try:
            if (not isinstance(proof, dict) or set(proof) != MIGRATION_TRUST_FIELDS
                    or proof.get("schema") != MIGRATION_TRUST_SCHEMA
                    or not _hash(proof.get("attestation_file_sha256"))
                    or not _hash(proof.get("attestation_sha256"))):
                raise AuditError("migration proof trust binding is invalid")
            path_text = proof.get("attestation_path")
            path = Path(path_text)
            if (not isinstance(path_text, str) or not path.is_absolute()
                    or path.resolve().is_relative_to(repo) or not path.is_file()):
                raise AuditError("migration proof attestation path is invalid")
            raw = path.read_bytes()
            attestation = json.loads(raw)
            if not isinstance(attestation, dict):
                raise AuditError("migration proof attestation must be an object")
            seal = attestation.get("attestation_sha256")
            unsealed = {key: value for key, value in attestation.items()
                        if key != "attestation_sha256"}
            cells = [cell for cell in attestation.get("cells", [])
                     if isinstance(cell, dict) and cell.get("cell_id") == agent_id]
            if (attestation.get("schema") != MIGRATION_ATTESTATION_SCHEMA
                    or not isinstance(attestation.get("migration_id"), str)
                    or not attestation["migration_id"]
                    or not _hash(attestation.get("plan_sha256"))
                    or sha256_bytes(raw) != proof["attestation_file_sha256"]
                    or seal != proof["attestation_sha256"] or not _hash(seal)
                    or sha256_json(unsealed) != seal or len(cells) != 1):
                raise AuditError("migration proof attestation is invalid")
            cell = cells[0]
            boundary = cell.get("experiment")
            expected_parent = (seed_commit if boundary == 1
                               else experiment_commits[boundary - 2]
                               if type(boundary) is int and 1 < boundary <= round_count
                               else None)
            old_identity, new_identity = (
                cell.get("old_controller_identity"), cell.get("new_controller_identity")
            )
            _validate_provenance({"agent": expected["agent"], "controller": new_identity})
            if (type(boundary) is not int or not 1 <= boundary <= round_count
                    or boundary <= previous_boundary
                    or cell.get("branch") != branch
                    or cell.get("seed_commit") != seed_commit
                    or cell.get("resume_parent") != expected_parent
                    or old_identity != expected["controller"]
                    or new_identity == old_identity):
                raise AuditError("migration proof cell or transition boundary is invalid")
            expected = {"agent": expected["agent"], "controller": new_identity}
            for index in range(boundary - 1, round_count):
                per_round[index] = json.loads(json.dumps(expected))
            summaries.append({
                "migration_id": attestation.get("migration_id"),
                "experiment": boundary, "attestation_sha256": seal,
            })
            previous_boundary = boundary
        except AuditError:
            raise
        except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError,
                json.JSONDecodeError) as failure:
            raise AuditError("migration proof is malformed") from failure
    return per_round, summaries


def validate_branch(repo: Path, base: str = "main", *,
                    migration_proofs: Sequence[dict] = ()) -> dict:
    """Validate retained evidence without importing or invoking the experiment runner."""
    repo = repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise AuditError("audited branch must be clean")
    branch = _git(repo, "branch", "--show-current")
    if not branch.startswith("experiment/"):
        raise AuditError("audited branch name must start with experiment/")
    _git(repo, "rev-parse", "--verify", base)
    if _run(repo, "merge-base", "--is-ancestor", "HEAD", base, check=False).returncode == 0:
        raise AuditError("audited branch must remain unmerged")

    commits = _git(repo, "rev-list", "--first-parent", "--reverse",
                   f"{base}..HEAD").splitlines()
    all_commits = _git(repo, "rev-list", f"{base}..HEAD").splitlines()
    if not commits:
        raise AuditError("audited branch must contain a seed commit")
    seed_commit, experiment_commits = commits[0], commits[1:]
    seed = _json_blob(repo, seed_commit, ".experiment/seed.json")
    seed_fields = {"schema", "run_id", "agent_id", "prompt_sha256", "task_sha256",
                   "reproducibility"}
    schema = seed.get("schema")
    round_count = 3 if schema == SEED_SCHEMA_V1 else seed.get("round_count")
    if schema == SEED_SCHEMA_V2:
        seed_fields.add("round_count")
    if (set(seed) != seed_fields or schema not in {SEED_SCHEMA_V1, SEED_SCHEMA_V2}
            or type(round_count) is not int or round_count < 1
            or not all(isinstance(seed.get(field), str) and seed[field]
                       for field in ("run_id", "agent_id"))
            or not _hash(seed.get("prompt_sha256")) or not _hash(seed.get("task_sha256"))):
        raise AuditError("seed contract schema is invalid")
    expected_commits = round_count + 1
    if len(commits) != expected_commits or len(all_commits) != expected_commits:
        raise AuditError(
            f"audited branch must contain exactly one seed and {round_count} linear commits"
        )
    provenance = _validate_provenance(seed["reproducibility"])
    round_provenance, migrations = _migration_chain(
        repo, migration_proofs, branch=branch, seed_commit=seed_commit,
        agent_id=seed["agent_id"], seed_provenance=provenance,
        experiment_commits=experiment_commits,
        round_count=round_count,
    )
    prompt, task = _blob(repo, seed_commit, "PROMPT.md"), _blob(repo, seed_commit, "TASK.md")
    seed_bytes = _blob(repo, seed_commit, ".experiment/seed.json")
    if (sha256_bytes(prompt) != seed["prompt_sha256"]
            or sha256_bytes(task) != seed["task_sha256"]):
        raise AuditError("seed contract hashes are invalid")
    _immutable(repo, commits, "PROMPT.md", prompt)
    _immutable(repo, commits, "TASK.md", task)
    _immutable(repo, commits, ".experiment/seed.json", seed_bytes)

    sessions: set[str] = set()
    summaries = []
    prior = seed_commit
    prior_candidate = sha256_bytes(_blob(repo, prior, "candidate.py"))
    prior_manifest = _blob(repo, prior, "candidate.manifest.json")
    for number, commit in enumerate(experiment_commits, 1):
        root = f"experiments/{number:02d}"
        author = _git(repo, "show", "-s", "--format=%an%x00%ae", commit).split("\0")
        expected_author = [f"Experiment Agent {seed['agent_id']}",
                           f"{seed['agent_id']}@experiment.invalid"]
        if author != expected_author:
            raise AuditError(f"experiment {number} author identity is invalid")
        evidence = _json_blob(repo, commit, f"{root}/evidence.json")
        results = _json_blob(repo, commit, f"{root}/results.json")
        sources = _json_blob(repo, commit, f"{root}/sources.json")
        report = _parse_report(_blob(repo, commit, f"{root}/report.md"), number)
        _validate_commands(_blob(repo, commit, f"{root}/commands.jsonl"), number)
        if set(sources) != {"sources", "no_sources_reason"}:
            raise AuditError(f"experiment {number} sources schema is invalid")
        validate_sources(sources["sources"], sources["no_sources_reason"],
                         f"experiment {number}")
        fields = frozenset(evidence)
        legacy_fields = {frozenset(EVIDENCE_FIELDS),
                         frozenset(EVIDENCE_FIELDS | {"agent_retries"})}
        attempt_fields = {
            frozenset(EVIDENCE_FIELDS | {"candidate_attempts", "final_attempt"}),
            frozenset(EVIDENCE_FIELDS | {
                "agent_retries", "candidate_attempts", "final_attempt",
            }),
        }
        if (fields not in legacy_fields | attempt_fields
                or evidence.get("schema") != EVIDENCE_SCHEMA
                or any(not _hash(evidence.get(field)) for field in (
                    "candidate_sha256", "tested_candidate_sha256",
                    "committed_candidate_sha256", "controller_receipt_sha256",
                    "manifest_sha256", "prompt_sha256", "task_sha256"))
                or evidence.get("restored_candidate_sha256") is not None
                and not _hash(evidence["restored_candidate_sha256"])
                or not isinstance(evidence.get("session_id"), str)
                or not evidence["session_id"]):
            raise AuditError(f"experiment {number} evidence schema is invalid")
        _validate_retries(evidence.get("agent_retries", []), number)
        if (evidence["candidate_sha256"] != evidence["tested_candidate_sha256"]
                or evidence["prompt_sha256"] != seed["prompt_sha256"]
                or evidence["task_sha256"] != seed["task_sha256"]
                or evidence["reproducibility"] != round_provenance[number - 1]):
            raise AuditError(f"experiment {number} evidence provenance diverges")
        tested = evidence["tested_candidate_sha256"]
        if tested == prior_candidate:
            raise AuditError(f"experiment {number} did not test a material candidate change")
        validate_controller_receipt(results, tested, evidence["manifest_sha256"])
        if evidence["controller"] != results:
            raise AuditError(f"experiment {number} controller evidence diverges")
        if evidence["controller_receipt_sha256"] != sha256_json(results):
            raise AuditError(f"experiment {number} controller receipt hash is invalid")
        decision = evidence.get("decision")
        if report["decision"] != decision:
            raise AuditError(f"experiment {number} report decision diverges")
        committed = _blob(repo, commit, "candidate.py")
        committed_manifest = _blob(repo, commit, "candidate.manifest.json")
        if sha256_bytes(committed) != evidence["committed_candidate_sha256"]:
            raise AuditError(f"experiment {number} committed candidate hash is invalid")
        expected_artifacts = set(STANDARD_ARTIFACTS)
        if "candidate_attempts" in evidence:
            expected_artifacts |= _validate_candidate_attempts(
                repo, commit, root, evidence, results, number,
            )
            if results["status"] == "candidate_error" and decision != "revert":
                raise AuditError(
                    f"experiment {number} exhausted candidate repairs must revert"
                )
        if decision == "revert":
            expected_artifacts |= REVERT_ARTIFACTS
            archived = _blob(repo, commit, f"{root}/tested_candidate.py")
            archived_manifest = _blob(repo, commit, f"{root}/tested_candidate.manifest.json")
            if (sha256_bytes(archived) != tested
                    or sha256_bytes(archived_manifest) != evidence["manifest_sha256"]):
                raise AuditError(f"experiment {number} revert evidence is invalid")
            if (committed != _blob(repo, prior, "candidate.py")
                    or committed_manifest != prior_manifest
                    or evidence["restored_candidate_sha256"] != prior_candidate):
                raise AuditError(f"experiment {number} revert restoration is invalid")
        elif decision == "retain":
            if (evidence["restored_candidate_sha256"] is not None
                    or sha256_bytes(committed) != tested
                    or sha256_bytes(committed_manifest) != evidence["manifest_sha256"]):
                raise AuditError(f"experiment {number} retain evidence is invalid")
        else:
            raise AuditError(f"experiment {number} decision is invalid")
        # `git ls-tree <commit> <dir>` prints the directory itself; inspect its tree.
        names = set(_git(
            repo, "ls-tree", "-r", "--name-only", f"{commit}:{root}"
        ).splitlines())
        if names != expected_artifacts:
            raise AuditError(f"experiment {number} artifact set is invalid")
        _validate_commit_paths(repo, prior, commit, root, expected_artifacts)
        original_tree = _git(repo, "rev-parse", f"{commit}:{root}")
        for later in experiment_commits[number:]:
            if _git(repo, "rev-parse", f"{later}:{root}") != original_tree:
                raise AuditError(f"experiment {number} evidence changed in a later commit")
        sessions.add(evidence["session_id"])
        summaries.append({"experiment": number, "commit": commit, "decision": decision,
                          "tested_candidate_sha256": tested})
        prior, prior_candidate, prior_manifest = commit, sha256_bytes(committed), committed_manifest
    if len(sessions) != 1:
        raise AuditError("experiments did not retain one persistent session")
    return {"status": "valid", "branch": branch, "run_id": seed["run_id"],
            "agent_id": seed["agent_id"],
            "seed_commit": seed_commit,
            "round_count": round_count,
            "session_id": sessions.pop(), "experiments": summaries,
            "controller_migrations": migrations}
