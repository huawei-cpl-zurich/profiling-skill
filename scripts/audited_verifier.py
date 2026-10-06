#!/usr/bin/env python3
"""Offline validation for a completed audited-experiment Git branch."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

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
COMMAND_FIELDS = {
    "command", "exit_code", "output_sha256", "output_excerpt", "output_truncated",
}
_HASH = re.compile(r"[0-9a-f]{64}")


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


def _validate_commands(data: bytes, number: int) -> None:
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


def validate_branch(repo: Path, base: str = "main") -> dict:
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
        if (set(evidence) != EVIDENCE_FIELDS or evidence.get("schema") != EVIDENCE_SCHEMA
                or any(not _hash(evidence.get(field)) for field in (
                    "candidate_sha256", "tested_candidate_sha256",
                    "committed_candidate_sha256", "controller_receipt_sha256",
                    "manifest_sha256", "prompt_sha256", "task_sha256"))
                or evidence.get("restored_candidate_sha256") is not None
                and not _hash(evidence["restored_candidate_sha256"])
                or not isinstance(evidence.get("session_id"), str)
                or not evidence["session_id"]):
            raise AuditError(f"experiment {number} evidence schema is invalid")
        if (evidence["candidate_sha256"] != evidence["tested_candidate_sha256"]
                or evidence["prompt_sha256"] != seed["prompt_sha256"]
                or evidence["task_sha256"] != seed["task_sha256"]
                or evidence["reproducibility"] != provenance):
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
        names = set(_git(repo, "ls-tree", "--name-only", f"{commit}:{root}").splitlines())
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
    return {"status": "valid", "branch": branch, "seed_commit": seed_commit,
            "round_count": round_count,
            "session_id": sessions.pop(), "experiments": summaries}
