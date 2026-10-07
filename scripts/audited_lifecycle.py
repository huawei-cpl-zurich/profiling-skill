#!/usr/bin/env python3
"""Host-controlled lifecycle for audited multi-round experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from audited_contract import (
    AuditError,
    ParsedEvents,
    REPORT_FIELDS,
    extract_thread_id,
    parse_agent_events,
    parse_command_events,
    sha256_bytes,
    sha256_json,
    validate_controller_receipt,
)

MAX_PARTIAL_OUTPUT = 65536
MANIFEST_SCHEMA = "profiling-skill/candidate-kernel/v1"
STARTER_SELECTOR = "REPLACE_WITH_EXACT_EXPORTED_KERNEL"
MIGRATION_CITATION_SCHEMA = "profiling-skill/audited-runtime-migration-citation/v1"
MIGRATION_ATTESTATION_SCHEMA = "profiling-skill/audited-runtime-preflight/v1"
MIGRATION_TRUST_SCHEMA = "profiling-skill/audited-runtime-migration-trust/v1"
MIGRATION_RETRY_REASON = "audited selector-runtime migration prepared a fresh profile attempt"
CODEX_RETRY_SCHEMA = "profiling-skill/codex-transient-retry/v1"
_RETRY_KEYS = {
    "schema", "terminal_error", "action", "attempt", "stdout_sha256",
    "stdout_bytes", "stderr_sha256", "stderr_bytes",
}


def _retry_records(value: object, *, contextual: bool = False) -> tuple[dict, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise AuditError("agent retry evidence is invalid")
    required = _RETRY_KEYS | ({"experiment", "stage"} if contextual else set())
    retained = []
    for record in value:
        if (not isinstance(record, dict) or set(record) != required
                or record.get("schema") != CODEX_RETRY_SCHEMA
                or record.get("terminal_error") != "server_overloaded"
                or record.get("action") not in {"retry", "exhausted"}
                or type(record.get("attempt")) is not int
                or not 1 <= record["attempt"] <= 6
                or any(not isinstance(record.get(key), str)
                       or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None
                       for key in ("stdout_sha256", "stderr_sha256"))
                or any(type(record.get(key)) is not int or record[key] < 0
                       for key in ("stdout_bytes", "stderr_bytes"))
                or (contextual and (
                    type(record.get("experiment")) is not int
                    or record["experiment"] < 1
                    or record.get("stage") not in {"prepare", "finalize"}
                ))):
            raise AuditError("agent retry evidence is invalid")
        retained.append(dict(record))
    return tuple(retained)


def _git(repo: Path, *arguments: str, env: dict | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, text=True, capture_output=True, env=env,
        check=False,
    )
    if result.returncode:
        raise AuditError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_blob(repo: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=repo, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AuditError(f"{path} is not tracked at {revision}")
    return result.stdout


def _identity(component: object) -> dict:
    metadata = getattr(component, "reproducibility_metadata", None)
    if callable(metadata):
        return metadata()
    name = f"{getattr(component, '__module__', 'unknown')}." \
           f"{getattr(component, '__qualname__', type(component).__name__)}"
    return {"adapter": name, "identity_sha256": sha256_bytes(name.encode())}


def _tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise AuditError("runtime migration closure is unavailable")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        mode = path.lstat().st_mode
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(mode)):
            raise AuditError("runtime migration closure has an unsupported entry")
        if path.is_file():
            digest.update(relative.encode() + b"\0")
            digest.update(oct(stat.S_IMODE(mode)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_runtime_migration(repo: Path, state: dict, seed: dict,
                                current: dict, agent_id: str,
                                trusted: dict | list[dict] | None) -> dict:
    try:
        citation = state["runtime_migration"]
        if not isinstance(citation, dict) or citation.get("schema") != MIGRATION_CITATION_SCHEMA:
            raise AuditError("runtime migration citation is invalid")
        proofs = [trusted] if isinstance(trusted, dict) else trusted
        if not isinstance(proofs, list) or not proofs:
            raise AuditError("runtime migration attestation is invalid")
        expected_controller = seed.get("reproducibility", {}).get("controller")
        previous_boundary = 0
        latest = None
        for proof in proofs:
            path_text = proof.get("attestation_path") if isinstance(proof, dict) else None
            path = Path(path_text) if isinstance(path_text, str) else Path("")
            if (not isinstance(proof, dict) or proof.get("schema") != MIGRATION_TRUST_SCHEMA
                    or not path.is_absolute() or path.resolve().is_relative_to(repo)
                    or not path.is_file()):
                raise AuditError("runtime migration attestation is invalid")
            raw = path.read_bytes()
            attestation = json.loads(raw)
            seal = attestation.get("attestation_sha256")
            unsealed = {key: value for key, value in attestation.items()
                        if key != "attestation_sha256"}
            cells = [item for item in attestation.get("cells", [])
                     if isinstance(item, dict) and item.get("cell_id") == agent_id]
            if (attestation.get("schema") != MIGRATION_ATTESTATION_SCHEMA
                    or proof.get("attestation_file_sha256") != sha256_bytes(raw)
                    or proof.get("attestation_sha256") != seal
                    or not isinstance(seal, str) or sha256_json(unsealed) != seal
                    or len(cells) != 1):
                raise AuditError("runtime migration attestation is invalid")
            cell = cells[0]
            boundary = cell.get("experiment")
            if (type(boundary) is not int or boundary <= previous_boundary
                    or boundary > state.get("experiment", 0)
                    or cell.get("branch") != state.get("branch")
                    or cell.get("seed_commit") != state.get("seed_commit")
                    or cell.get("old_controller_identity") != expected_controller):
                raise AuditError("runtime migration cell or controller identity is invalid")
            expected_controller = cell.get("new_controller_identity")
            latest = (proof, attestation, cell)
            previous_boundary = boundary
            for label in ("old", "new"):
                binding = attestation["runtimes"][label]
                config_path = Path(binding["config_path"])
                runtime_path = Path(binding["runtime_path"])
                config_raw = config_path.read_bytes()
                config = json.loads(config_raw)
                runtime = config["runtime_scripts"]
                if (not config_path.is_absolute() or not runtime_path.is_absolute()
                        or sha256_bytes(config_raw) != binding.get("config_sha256")
                        or Path(runtime["path"]).resolve() != runtime_path.resolve()
                        or _tree_sha256(runtime_path) != binding.get("closure_sha256")
                        or runtime.get("sha256") != binding.get("closure_sha256")):
                    raise AuditError("runtime migration runtime artifacts changed")
        proof, attestation, cell = latest
        checkpoint = citation.get("checkpoint_commit")
        original_raw = citation.get("original_blocked_json")
        original = json.loads(original_raw)
        current_without_citation = {key: value for key, value in state.items()
                                    if key != "runtime_migration"}
        if (citation.get("attestation_path") != proof.get("attestation_path")
                or citation.get("attestation_file_sha256")
                != proof.get("attestation_file_sha256")
                or citation.get("attestation_sha256") != proof.get("attestation_sha256")
                or citation.get("plan_sha256") != attestation.get("plan_sha256")
                or citation.get("cell_id") != agent_id
                or citation.get("experiment") != cell.get("experiment")
                or checkpoint != cell.get("checkpoint_commit")
                or sha256_bytes(original_raw.encode()) != cell.get("blocked_sha256")
                or citation.get("old_controller_identity") != cell.get("old_controller_identity")
                or citation.get("new_controller_identity") != cell.get("new_controller_identity")
                or current["controller"] != expected_controller):
            raise AuditError("runtime migration cell or controller identity is invalid")
        continuation = citation.get("continuation")
        if continuation is None:
            original_without_citation = {
                key: value for key, value in original.items() if key != "runtime_migration"
            }
            migrated = {**original_without_citation,
                        "reason": MIGRATION_RETRY_REASON, "receipt": {},
                        "controller_submissions": 0, "measurement_attempts": 0}
            if (state.get("experiment") != cell.get("experiment")
                    or current_without_citation not in (original_without_citation, migrated)):
                raise AuditError("runtime migration initial checkpoint is invalid")
        else:
            state_sha = sha256_json(current_without_citation)
            expected = {
                "schema": "profiling-skill/audited-runtime-migration-continuation/v1",
                "source_checkpoint_commit": checkpoint,
                "source_experiment": cell["experiment"],
                "experiment": state.get("experiment"), "stage": state.get("stage"),
                "resume_parent": state.get("resume_parent"),
                "candidate_sha256": state.get("candidate_sha256"),
                "manifest_sha256": state.get("manifest_sha256"),
                "state_sha256": state_sha,
            }
            if continuation != expected or state.get("experiment", 0) < cell["experiment"]:
                raise AuditError("runtime migration continuation is invalid")
        return json.loads(json.dumps(citation))
    except AuditError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OSError,
            json.JSONDecodeError) as failure:
        raise AuditError("runtime migration citation is malformed") from failure


@dataclass(frozen=True)
class RunResult:
    status: str
    branch: str
    session_id: str
    seed_commit: str
    commits: tuple[str, ...]


class AuditedExperimentRunner:
    """Drive one persistent agent and make exactly one host commit per round."""

    def __init__(self, repo: Path, prompt: Path, task: Path,
                 invoke: Callable[[int, str | None, str | None], str],
                 controller: Callable[[int, str, str], dict], *, max_repairs: int = 2,
                 max_candidate_repairs: int = 2,
                 max_controller_resubmits: int = 2, max_remeasurements: int = 1,
                 round_count: int = 3,
                 trusted_runtime_migration: dict | list[dict] | None = None):
        if type(round_count) is not int or round_count < 1:
            raise AuditError("round count must be a positive integer")
        if type(max_candidate_repairs) is not int or max_candidate_repairs < 0:
            raise AuditError("candidate repair count must be a nonnegative integer")
        self.repo = repo.resolve()
        self.prompt_bytes = prompt.resolve().read_bytes()
        self.task_bytes = task.resolve().read_bytes()
        self.prompt_hash = sha256_bytes(self.prompt_bytes)
        self.task_hash = sha256_bytes(self.task_bytes)
        self.invoke = invoke
        self.controller = controller
        self.max_repairs = max_repairs
        self.max_candidate_repairs = max_candidate_repairs
        self.max_controller_resubmits = max_controller_resubmits
        self.max_remeasurements = max_remeasurements
        self.round_count = round_count
        self.trusted_runtime_migration = trusted_runtime_migration
        self.runtime_migration: dict | None = None
        self.agent_retry_evidence: tuple[dict, ...] = ()
        self.candidate_attempts: list[dict] = []
        self.reasoning_sha256 = sha256_bytes(b"")
        self.reproducibility = {
            "agent": _identity(invoke), "controller": _identity(controller),
        }

    def run(self, run_id: str, agent_id: str, *, resume: bool = False) -> RunResult:
        state = self._resume(run_id, agent_id) if resume else None
        if state:
            branch = state["branch"]
            seed_commit = state["seed_commit"]
            seed_hash = state["seed_hash"]
            session = state["session_id"]
            start = state["experiment"]
            prior_hash = state["prior_candidate_sha256"]
            commits = _git(
                self.repo, "rev-list", "--first-parent", "--reverse",
                f"{seed_commit}..HEAD",
            ).splitlines()
        else:
            branch, seed_commit, seed_hash = self._initialize(run_id, agent_id)
            session = None
            start = 1
            prior_hash = sha256_bytes((self.repo / "candidate.py").read_bytes())
            commits = []
        self.agent_retry_evidence = tuple(state.get("agent_retries", ())) if state else ()

        for number in range(start, self.round_count + 1):
            prior_candidate = _git_blob(self.repo, "HEAD", "candidate.py")
            prior_manifest = _git_blob(self.repo, "HEAD", "candidate.manifest.json")
            continuing = state if state and number == start else None
            commands = tuple(continuing.get("commands", ())) if continuing else ()
            stage = continuing.get("stage") if continuing else "prepare"
            self.candidate_attempts = list(
                continuing.get("candidate_attempts", ())) if continuing else []
            self.reasoning_sha256 = (
                continuing.get("reasoning_sha256", sha256_bytes(b""))
                if continuing else sha256_bytes(b"")
            )

            if not continuing:
                session, commands = self._preparation_turn(
                    number, session, None, commands, branch, seed_commit, seed_hash,
                    prior_hash,
                )
            elif stage == "prepare":
                session, commands = self._preparation_turn(
                    number, session,
                    (
                        "Recover seed-only experiment 1 using the candidate and manifest "
                        "already present. Preserve that work, inspect it, run a local check, "
                        "and stop without starting another experiment."
                    ) if continuing.get("seed_only_recovery") else
                    None if continuing.get("pre_session") else (
                        f"Resume blocked experiment {number}: repair the prepared candidate. "
                        "Do not start another experiment; stop after a local check."
                    ),
                    commands, branch, seed_commit, seed_hash, prior_hash,
                )

            command_offset = sum(len(item["commands"]) for item in self.candidate_attempts)
            while True:
                candidate_hash, manifest, commands, session = self._repair_candidate(
                    number, session, prior_hash, commands, branch, seed_commit, seed_hash,
                )
                manifest_hash = sha256_bytes(manifest.read_bytes())
                receipt = self._controller_turn(
                    number, session, candidate_hash, manifest_hash, continuing, stage,
                    branch, seed_commit, seed_hash, prior_hash, commands,
                )
                self._record_candidate_attempt(
                    candidate_hash, manifest_hash, receipt,
                    commands[command_offset:], self.reasoning_sha256,
                )
                command_offset = len(commands)
                continuing = None
                stage = "prepare"
                if (receipt["status"] != "candidate_error"
                        or len(self.candidate_attempts) > self.max_candidate_repairs):
                    break
                failed_candidate, failed_manifest = candidate_hash, manifest_hash
                session, commands = self._preparation_turn(
                    number, session,
                    f"Repair candidate attempt {len(self.candidate_attempts)} for experiment "
                    f"{number} after this candidate failure:\n"
                    f"{json.dumps(receipt, sort_keys=True)}\n"
                    "Stay in the same experiment and session. Change candidate.py or "
                    "candidate.manifest.json, run a local smoke check, and stop.",
                    commands, branch, seed_commit, seed_hash, prior_hash,
                )
                repaired_candidate, repaired_manifest = self._validate_candidate(prior_hash)
                if (repaired_candidate == failed_candidate
                        and sha256_bytes(repaired_manifest.read_bytes()) == failed_manifest):
                    self._checkpoint(
                        number, session, "candidate repair did not change candidate or manifest",
                        "prepare", branch, seed_commit, seed_hash, prior_hash, commands,
                    )
                    raise AuditError("candidate repair did not change candidate or manifest")

            tested_candidate = (self.repo / "candidate.py").read_bytes()
            tested_manifest = manifest.read_bytes()
            parsed, session = self._finalize(
                number, session, candidate_hash, manifest_hash, receipt, commands,
                tested_candidate, tested_manifest, branch, seed_commit, seed_hash,
                prior_hash,
            )

            restored_hash = None
            committed_hash = candidate_hash
            if parsed.report["decision"] == "revert":
                (self.repo / "candidate.py").write_bytes(prior_candidate)
                manifest.write_bytes(prior_manifest)
                restored_hash = sha256_bytes(prior_candidate)
                committed_hash = restored_hash

            self._write_evidence(
                number, session, candidate_hash, tested_manifest, parsed, receipt,
                committed_hash, restored_hash, tested_candidate,
            )
            self.agent_retry_evidence = ()
            self.candidate_attempts = []
            _git(self.repo, "add", "candidate.py", "candidate.manifest.json",
                 f"experiments/{number:02d}")
            author = dict(os.environ)
            author.update({
                "GIT_AUTHOR_NAME": f"Experiment Agent {agent_id}",
                "GIT_AUTHOR_EMAIL": f"{agent_id}@experiment.invalid",
            })
            _git(self.repo, "commit", "-m", f"experiment {number}: {agent_id}", env=author)
            commits.append(_git(self.repo, "rev-parse", "HEAD"))
            prior_hash = committed_hash

        assert session is not None
        self._verify_history(seed_commit, seed_hash, commits)
        return RunResult("complete", branch, session, seed_commit, tuple(commits))

    def _controller_turn(self, number: int, session: str, candidate_hash: str,
                         manifest_hash: str, continuing: dict | None, stage: str,
                         branch: str, seed_commit: str, seed_hash: str,
                         prior_hash: str, commands: tuple[dict, ...]) -> dict:
        submissions = int(continuing.get("controller_submissions", 0)) if continuing else 0
        measurements = int(continuing.get("measurement_attempts", 0)) if continuing else 0
        previous = continuing.get("receipt") if continuing else None

        if continuing and stage == "finalize":
            return previous
        if continuing and stage == "controller":
            handle = previous.get("handle") if isinstance(previous, dict) else None
            if isinstance(handle, str) and handle:
                observe = getattr(self.controller, "observe", None)
                if not callable(observe):
                    self._controller_checkpoint(
                        number, session, "controller lacks durable-handle observation",
                        branch, seed_commit, seed_hash, prior_hash, commands, previous,
                        submissions, measurements,
                    )
                    raise AuditError("controller does not support durable-handle observation")
                try:
                    receipt = observe(number, candidate_hash, manifest_hash, handle)
                except Exception as failure:
                    self._controller_checkpoint(
                        number, session, f"controller observation failed: {failure}",
                        branch, seed_commit, seed_hash, prior_hash, commands, previous,
                        submissions, measurements,
                    )
                    raise AuditError(
                        f"controller observation failed for durable handle {handle}"
                    ) from failure
                if not isinstance(receipt, dict) or receipt.get("handle") != handle:
                    self._controller_checkpoint(
                        number, session, "controller observation changed durable handle",
                        branch, seed_commit, seed_hash, prior_hash, commands, previous,
                        submissions, measurements,
                    )
                    raise AuditError("controller observation changed the durable handle")
            else:
                if submissions > self.max_controller_resubmits:
                    self._controller_checkpoint(
                        number, session, "pre-handle resubmission budget exhausted",
                        branch, seed_commit, seed_hash, prior_hash, commands, previous,
                        submissions, measurements,
                    )
                    raise AuditError("pre-handle controller resubmission budget exhausted")
                submissions += 1
                try:
                    receipt = self.controller(number, candidate_hash, manifest_hash)
                except Exception as failure:
                    self._controller_checkpoint(
                        number, session, f"controller submission failed: {failure}",
                        branch, seed_commit, seed_hash, prior_hash, commands, {},
                        submissions, measurements,
                    )
                    raise AuditError("pre-handle controller submission failed") from failure
        elif continuing and stage == "measurement":
            if measurements >= self.max_remeasurements:
                self._measurement_checkpoint(
                    number, session, "remeasurement budget exhausted", branch,
                    seed_commit, seed_hash, prior_hash, commands, previous,
                    submissions, measurements,
                )
                raise AuditError("remeasurement budget exhausted")
            remeasure = getattr(self.controller, "remeasure", None)
            if not callable(remeasure):
                self._measurement_checkpoint(
                    number, session, "controller lacks explicit remeasurement", branch,
                    seed_commit, seed_hash, prior_hash, commands, previous,
                    submissions, measurements,
                )
                raise AuditError("controller does not support explicit remeasurement")
            handle = previous.get("handle") if isinstance(previous, dict) else None
            measurements += 1
            try:
                receipt = remeasure(number, candidate_hash, manifest_hash, handle)
            except Exception as failure:
                self._measurement_checkpoint(
                    number, session, f"controller remeasurement failed: {failure}",
                    branch, seed_commit, seed_hash, prior_hash, commands, previous,
                    submissions, measurements,
                )
                raise AuditError("controller remeasurement failed") from failure
        else:
            submissions = 1
            try:
                receipt = self.controller(number, candidate_hash, manifest_hash)
            except Exception as failure:
                self._controller_checkpoint(
                    number, session, f"controller submission failed: {failure}", branch,
                    seed_commit, seed_hash, prior_hash, commands, {}, submissions,
                    measurements,
                )
                raise AuditError("pre-handle controller submission failed") from failure

        terminal = {"ok", "candidate_error", "measurement_pending"}
        if not isinstance(receipt, dict) or receipt.get("status") not in terminal:
            reason = receipt.get("reason", receipt) if isinstance(receipt, dict) else receipt
            self._controller_checkpoint(
                number, session, str(reason), branch, seed_commit, seed_hash, prior_hash,
                commands, receipt if isinstance(receipt, dict) else {}, submissions,
                measurements,
            )
            status = receipt.get("status") if isinstance(receipt, dict) else None
            raise AuditError(f"experiment {number} blocked by controller status {status!r}")
        try:
            receipt = validate_controller_receipt(receipt, candidate_hash, manifest_hash)
        except AuditError as failure:
            self._controller_checkpoint(
                number, session, str(failure), branch, seed_commit, seed_hash,
                prior_hash, commands, receipt, submissions, measurements,
            )
            raise
        if receipt["status"] == "measurement_pending":
            self._measurement_checkpoint(
                number, session, "measurement remains pending", branch, seed_commit,
                seed_hash, prior_hash, commands, receipt, submissions, measurements,
            )
            raise AuditError(f"experiment {number} measurement pending; resume to remeasure")
        return receipt

    def _controller_checkpoint(self, number: int, session: str, reason: str,
                               branch: str, seed_commit: str, seed_hash: str,
                               prior_hash: str, commands: tuple[dict, ...], receipt: dict,
                               submissions: int, measurements: int) -> None:
        self._checkpoint(
            number, session, reason, "controller", branch, seed_commit, seed_hash,
            prior_hash, commands, receipt, controller_submissions=submissions,
            measurement_attempts=measurements,
        )

    def _measurement_checkpoint(self, number: int, session: str, reason: str,
                                branch: str, seed_commit: str, seed_hash: str,
                                prior_hash: str, commands: tuple[dict, ...], receipt: dict,
                                submissions: int, measurements: int) -> None:
        self._checkpoint(
            number, session, reason, "measurement", branch, seed_commit, seed_hash,
            prior_hash, commands, receipt, controller_submissions=submissions,
            measurement_attempts=measurements,
        )

    def _initialize(self, run_id: str, agent_id: str) -> tuple[str, str, str]:
        changed = set(_git(self.repo, "diff", "--name-only", "HEAD").splitlines())
        changed.update(_git(
            self.repo, "ls-files", "--others", "--exclude-standard",
        ).splitlines())
        allowed = {"candidate.py", "candidate.manifest.json"}
        if changed - allowed or any(not (self.repo / path).is_file() for path in allowed):
            raise AuditError(
                "experiment repository may contain only pinned starter materialization"
            )
        branch = f"experiment/{run_id}/{agent_id}"
        _git(self.repo, "switch", "-c", branch)
        directory = self.repo / ".experiment"
        directory.mkdir()
        (self.repo / "PROMPT.md").write_bytes(self.prompt_bytes)
        (self.repo / "TASK.md").write_bytes(self.task_bytes)
        seed = directory / "seed.json"
        seed_document = {
            "schema": ("profiling-skill/audited-seed/v1" if self.round_count == 3
                       else "profiling-skill/audited-seed/v2"), "run_id": run_id,
            "agent_id": agent_id, "prompt_sha256": self.prompt_hash,
            "task_sha256": self.task_hash, "reproducibility": self.reproducibility,
        }
        if self.round_count != 3:
            seed_document["round_count"] = self.round_count
        seed.write_text(json.dumps(seed_document, indent=2, sort_keys=True) + "\n")
        _git(self.repo, "add", ".experiment/seed.json", "PROMPT.md", "TASK.md",
             "candidate.py", "candidate.manifest.json")
        _git(self.repo, "commit", "-m", f"experiment seed: {run_id}/{agent_id}")
        return branch, _git(self.repo, "rev-parse", "HEAD"), sha256_bytes(seed.read_bytes())

    def _preparation_turn(self, number: int, session: str | None,
                          instruction: str | None, commands: tuple[dict, ...],
                          branch: str, seed_commit: str, seed_hash: str,
                          prior_hash: str) -> tuple[str, tuple[dict, ...]]:
        failure = ""
        for attempt in range(self.max_repairs + 1):
            turn_instruction = instruction if attempt == 0 else (
                f"Repair preparation evidence for experiment {number}: {failure}. "
                "Stay in this experiment and session; run a local check and stop."
            )
            output = self._invoke_or_checkpoint(
                number, session, turn_instruction, commands, "prepare", branch,
                seed_commit, seed_hash, prior_hash,
            )
            self.reasoning_sha256 = sha256_bytes(output.encode())
            session = extract_thread_id(output, session)
            try:
                _, new_commands = parse_command_events(output, session)
                return session, commands + new_commands
            except AuditError as error:
                failure = str(error)
                if failure != "structured preparation contains no command evidence":
                    commands += (self._invalid_attempt(output, failure),)
        assert session is not None
        self._checkpoint(number, session, failure, "prepare", branch, seed_commit,
                         seed_hash, prior_hash, commands)
        raise AuditError(failure)

    def _record_candidate_attempt(self, candidate_hash: str, manifest_hash: str,
                                  receipt: dict, commands: tuple[dict, ...],
                                  reasoning_sha256: str) -> None:
        candidate = (self.repo / "candidate.py").read_bytes()
        manifest = (self.repo / "candidate.manifest.json").read_bytes()
        self.candidate_attempts.append({
            "schema": "profiling-skill/candidate-attempt/v1",
            "attempt": len(self.candidate_attempts) + 1,
            "status": receipt["status"],
            "candidate_sha256": candidate_hash,
            "manifest_sha256": manifest_hash,
            "controller_receipt_sha256": sha256_json(receipt),
            "commands_sha256": sha256_json(list(commands)),
            "reasoning_sha256": reasoning_sha256,
            "candidate": candidate.decode("utf-8"),
            "manifest": manifest.decode("utf-8"),
            "receipt": receipt, "commands": list(commands),
        })

    def _repair_candidate(self, number: int, session: str, prior_hash: str,
                          commands: tuple[dict, ...], branch: str, seed_commit: str,
                          seed_hash: str) -> tuple[str, Path, tuple[dict, ...], str]:
        failure = ""
        for attempt in range(self.max_repairs + 1):
            try:
                candidate_hash, manifest = self._validate_candidate(prior_hash)
                return candidate_hash, manifest, commands, session
            except AuditError as error:
                failure = str(error)
                if attempt == self.max_repairs:
                    self._checkpoint(number, session, failure, "prepare", branch,
                                     seed_commit, seed_hash, prior_hash, commands)
                    raise
                session, commands = self._preparation_turn(
                    number, session,
                    f"Repair the prepared candidate for experiment {number}: {error}. "
                    "Do not start another experiment; stop after a local check.",
                    commands, branch, seed_commit, seed_hash, prior_hash,
                )
        raise AuditError(failure)

    def _validate_candidate(self, prior_hash: str) -> tuple[str, Path]:
        candidate = self.repo / "candidate.py"
        manifest = self.repo / "candidate.manifest.json"
        allowed = {"candidate.py", "candidate.manifest.json"}
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=self.repo,
            text=True, capture_output=True, check=True,
        ).stdout
        undeclared = [line[3:] for line in status.splitlines()
                      if len(line) >= 4 and line[3:] not in allowed]
        if undeclared:
            raise AuditError(
                "prepared experiment has undeclared worktree changes: "
                + ", ".join(sorted(undeclared))
            )
        if not candidate.is_file() or not manifest.is_file():
            raise AuditError("candidate.py and candidate.manifest.json are required")
        try:
            document = json.loads(manifest.read_text())
        except json.JSONDecodeError as failure:
            raise AuditError(f"candidate manifest is invalid JSON: {failure}") from failure
        if not isinstance(document, dict):
            raise AuditError("candidate manifest must be a JSON object")
        kernel_name = document.get("kernel_name")
        if document.get("schema") != MANIFEST_SCHEMA:
            raise AuditError(f"candidate manifest requires schema {MANIFEST_SCHEMA}")
        if (not isinstance(kernel_name, str) or not kernel_name.strip()
                or kernel_name != kernel_name.strip()):
            raise AuditError("candidate manifest requires a nonempty exact kernel selector")
        if kernel_name == STARTER_SELECTOR:
            raise AuditError("candidate manifest has the unresolved starter selector")
        candidate_hash = sha256_bytes(candidate.read_bytes())
        if candidate_hash == prior_hash:
            raise AuditError("prepared candidate is unchanged")
        if document.get("candidate_sha256") not in {None, candidate_hash}:
            raise AuditError("candidate manifest hash is stale")
        return candidate_hash, manifest

    def _finalize(self, number: int, session: str, candidate_hash: str,
                  manifest_hash: str, receipt: dict, commands: tuple[dict, ...],
                  candidate_bytes: bytes, manifest_bytes: bytes, branch: str,
                  seed_commit: str, seed_hash: str, prior_hash: str
                  ) -> tuple[ParsedEvents, str]:
        instruction = (
            f"Finalize experiment {number} from this exact host controller receipt:\n"
            f"{json.dumps(receipt, sort_keys=True)}\n"
            f"The frozen candidate SHA-256 is {candidate_hash}; manifest SHA-256 is "
            f"{manifest_hash}; controller receipt SHA-256 is {sha256_json(receipt)}. "
            "Return the required JSON report citing those values. Do not edit files."
        )
        failure = ""
        for attempt in range(self.max_repairs + 1):
            turn_instruction = instruction if attempt == 0 else (
                f"Repair only the experiment {number} JSON report: {failure}. "
                "Do not edit files; cite the same controller receipt."
            )
            output = self._invoke_or_checkpoint(
                number, session, turn_instruction, commands, "finalize", branch,
                seed_commit, seed_hash, prior_hash, receipt,
            )
            session = extract_thread_id(output, session)
            command_error = ""
            try:
                _, attempt_commands = parse_command_events(output, session)
                commands += attempt_commands
            except AuditError as error:
                command_error = str(error)
                if command_error != "structured preparation contains no command evidence":
                    commands += (self._invalid_attempt(output, command_error),)
            candidate_changed = sha256_bytes((self.repo / "candidate.py").read_bytes()) != candidate_hash
            manifest_changed = sha256_bytes((self.repo / "candidate.manifest.json").read_bytes()) != manifest_hash
            if candidate_changed or manifest_changed:
                (self.repo / "candidate.py").write_bytes(candidate_bytes)
                (self.repo / "candidate.manifest.json").write_bytes(manifest_bytes)
                self._checkpoint(number, session, "candidate or manifest changed during finalize",
                                 "finalize", branch, seed_commit, seed_hash, prior_hash,
                                 commands, receipt)
                raise AuditError("candidate or manifest changed during finalize")
            try:
                parsed = parse_agent_events(output, session, require_commands=False)
                expected = {
                    "controller_handle": receipt["handle"],
                    "candidate_sha256": candidate_hash,
                    "manifest_sha256": manifest_hash,
                    "controller_receipt_sha256": sha256_json(receipt),
                }
                if any(parsed.report.get(key) != value for key, value in expected.items()):
                    raise AuditError("final report does not cite the frozen receipt and candidate")
                if not commands:
                    raise AuditError("completed experiment contains no command evidence")
                return ParsedEvents(parsed.thread_id, parsed.report, commands), session
            except AuditError as error:
                failure = command_error or str(error)
        self._checkpoint(number, session, failure, "finalize", branch, seed_commit,
                         seed_hash, prior_hash, commands, receipt)
        raise AuditError(failure)

    def _invoke_or_checkpoint(self, number: int, session: str | None,
                              instruction: str | None, commands: tuple[dict, ...],
                              stage: str, branch: str, seed_commit: str,
                              seed_hash: str, prior_hash: str,
                              receipt: dict | None = None) -> str:
        """Honor the invoker's duck-typed ``structured_stdout`` failure field."""
        try:
            output = self.invoke(number, session, instruction)
        except Exception as failure:
            self._retain_agent_retries(
                number, stage, getattr(failure, "retry_evidence", ()),
            )
            partial = getattr(failure, "structured_stdout", "")
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            if not isinstance(partial, str):
                partial = ""
            partial = partial[:MAX_PARTIAL_OUTPUT]
            recovered = session
            if partial:
                try:
                    recovered = extract_thread_id(partial, session)
                    try:
                        _, partial_commands = parse_command_events(partial, recovered)
                        commands += partial_commands
                    except AuditError as parse_failure:
                        if str(parse_failure) != "structured preparation contains no command evidence":
                            commands += (self._invalid_attempt(partial, str(parse_failure)),)
                except AuditError:
                    pass
            reason = f"agent invocation failed: {failure}"
            if recovered:
                self._checkpoint(number, recovered, reason, stage, branch, seed_commit,
                                 seed_hash, prior_hash, commands, receipt)
                raise AuditError(
                    f"{reason}; checkpointed session {recovered} for resume"
                ) from failure
            if stage == "prepare" and number == 1:
                self._checkpoint(number, None, reason, stage, branch, seed_commit,
                                 seed_hash, prior_hash, commands, receipt)
                raise AuditError(
                    f"{reason}; checkpointed before session start for resume"
                ) from failure
            raise AuditError(f"{reason}; no durable session id was available") from failure
        self._retain_agent_retries(
            number, stage, getattr(self.invoke, "last_retry_evidence", ()),
        )
        return output

    def _retain_agent_retries(self, number: int, stage: str, records: object) -> None:
        retained = _retry_records(records)
        self.agent_retry_evidence += tuple({
            **record, "experiment": number, "stage": stage,
        } for record in retained)
        if len(self.agent_retry_evidence) > 64:
            raise AuditError("agent retry evidence exceeds the per-experiment limit")

    @staticmethod
    def _invalid_attempt(output: str, error: str) -> dict:
        """Retain bounded proof of rejected command events without trusting their fields."""
        excerpt = f"structured event stream rejected: {error}"
        return {
            "command": "<invalid structured command event>", "exit_code": 255,
            "output_sha256": sha256_bytes(output.encode()),
            "output_excerpt": excerpt[:512],
            "output_truncated": len(excerpt) > 512,
        }

    def _write_evidence(self, number: int, session: str, candidate_hash: str,
                        manifest_bytes: bytes, parsed: ParsedEvents, receipt: dict,
                        committed_hash: str, restored_hash: str | None,
                        tested_candidate: bytes) -> None:
        directory = self.repo / "experiments" / f"{number:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        if restored_hash:
            (directory / "tested_candidate.py").write_bytes(tested_candidate)
            (directory / "tested_candidate.manifest.json").write_bytes(manifest_bytes)
        report = parsed.report
        sections = [f"# Experiment {number}"] + [
            f"## {field.replace('_', ' ').title()}\n\n{report[field]}"
            for field in REPORT_FIELDS
        ]
        (directory / "report.md").write_text("\n\n".join(sections) + "\n")
        evidence = {
            "schema": "profiling-skill/audited-evidence/v1", "session_id": session,
            "candidate_sha256": candidate_hash, "tested_candidate_sha256": candidate_hash,
            "committed_candidate_sha256": committed_hash,
            "restored_candidate_sha256": restored_hash, "decision": report["decision"],
            "controller_receipt_sha256": sha256_json(receipt),
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "prompt_sha256": self.prompt_hash, "task_sha256": self.task_hash,
            "reproducibility": self.reproducibility, "controller": receipt,
            "agent_retries": list(self.agent_retry_evidence),
            "candidate_attempts": [
                {key: value for key, value in attempt.items()
                 if key not in {"candidate", "manifest", "receipt", "commands"}}
                for attempt in self.candidate_attempts
            ],
            "final_attempt": len(self.candidate_attempts),
        }
        (directory / "evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        (directory / "results.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        (directory / "sources.json").write_text(json.dumps({
            "sources": report["sources"],
            "no_sources_reason": report.get("no_sources_reason", ""),
        }, indent=2, sort_keys=True) + "\n")
        (directory / "commands.jsonl").write_text("".join(
            json.dumps(command, sort_keys=True) + "\n" for command in parsed.commands
        ))
        attempts = directory / "attempts"
        for attempt in self.candidate_attempts:
            target = attempts / f"{attempt['attempt']:02d}"
            target.mkdir(parents=True)
            (target / "candidate.py").write_text(attempt["candidate"])
            (target / "candidate.manifest.json").write_text(attempt["manifest"])
            (target / "controller.json").write_text(
                json.dumps(attempt["receipt"], indent=2, sort_keys=True) + "\n"
            )
            (target / "commands.jsonl").write_text("".join(
                json.dumps(command, sort_keys=True) + "\n"
                for command in attempt["commands"]
            ))
            (target / "reasoning.sha256").write_text(attempt["reasoning_sha256"] + "\n")

    def _checkpoint(self, number: int, session: str | None, reason: str, stage: str,
                    branch: str, seed_commit: str, seed_hash: str, prior_hash: str,
                    commands: tuple[dict, ...], receipt: dict | None = None, *,
                    controller_submissions: int = 0,
                    measurement_attempts: int = 0) -> None:
        if session is None and (stage != "prepare" or number != 1):
            raise AuditError("only the initial preparation may checkpoint before a session")
        path = self.repo / ".experiment" / "blocked.json"
        candidate = self.repo / "candidate.py"
        manifest = self.repo / "candidate.manifest.json"
        checkpoint = {
            "schema": ("profiling-skill/audited-blocked/v1" if self.round_count == 3
                       else "profiling-skill/audited-blocked/v2"), "experiment": number,
            "session_id": session, "reason": reason, "stage": stage, "branch": branch,
            "pre_session": session is None,
            "controller_submissions": controller_submissions,
            "measurement_attempts": measurement_attempts,
            "seed_commit": seed_commit, "seed_hash": seed_hash,
            "resume_parent": _git(self.repo, "rev-parse", "HEAD"),
            "prior_candidate_sha256": prior_hash, "commands": commands,
            "agent_retries": list(self.agent_retry_evidence),
            "candidate_attempts": self.candidate_attempts,
            "reasoning_sha256": self.reasoning_sha256,
            "receipt": receipt,
            "candidate_sha256": sha256_bytes(candidate.read_bytes()) if candidate.is_file() else None,
            "manifest_sha256": sha256_bytes(manifest.read_bytes()) if manifest.is_file() else None,
        }
        if self.round_count != 3:
            checkpoint["round_count"] = self.round_count
        if self.runtime_migration is not None:
            citation = {key: value for key, value in self.runtime_migration.items()
                        if key != "continuation"}
            citation["continuation"] = {
                "schema": "profiling-skill/audited-runtime-migration-continuation/v1",
                "source_checkpoint_commit": citation["checkpoint_commit"],
                "source_experiment": citation["experiment"],
                "experiment": number, "stage": stage,
                "resume_parent": checkpoint["resume_parent"],
                "candidate_sha256": checkpoint["candidate_sha256"],
                "manifest_sha256": checkpoint["manifest_sha256"],
                "state_sha256": sha256_json(checkpoint),
            }
            checkpoint["runtime_migration"] = citation
        path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
        paths = [".experiment/blocked.json"] + [
            name for name in ("candidate.py", "candidate.manifest.json")
            if (self.repo / name).is_file()
        ]
        _git(self.repo, "add", *paths)
        _git(self.repo, "commit", "-m", f"checkpoint blocked experiment {number}")

    def _resume(self, run_id: str, agent_id: str) -> dict:
        path = self.repo / ".experiment" / "blocked.json"
        if not path.is_file():
            return self._resume_seed_only(run_id, agent_id)
        if _git(self.repo, "status", "--porcelain"):
            raise AuditError("blocked experiment repository must be clean before resume")
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError as failure:
            raise AuditError("blocked checkpoint is invalid JSON") from failure
        required = ("reason", "seed_commit", "seed_hash", "resume_parent",
                    "prior_candidate_sha256")
        expected_branch = f"experiment/{run_id}/{agent_id}"

        def valid_hash(value: object) -> bool:
            return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

        checkpoint_schema = state.get("schema") if isinstance(state, dict) else None
        checkpoint_rounds = 3 if checkpoint_schema == "profiling-skill/audited-blocked/v1" \
            else state.get("round_count") if isinstance(state, dict) else None
        if checkpoint_rounds != self.round_count:
            raise AuditError("blocked checkpoint round count differs from the host declaration")
        if (not isinstance(state, dict)
                or checkpoint_schema not in {
                    "profiling-skill/audited-blocked/v1",
                    "profiling-skill/audited-blocked/v2",
                }
                or (checkpoint_schema == "profiling-skill/audited-blocked/v2"
                    and (type(state.get("round_count")) is not int
                         or state["round_count"] < 1))
                or state.get("stage") not in {"prepare", "controller", "measurement", "finalize"}
                or type(state.get("experiment")) is not int
                or not 1 <= state["experiment"] <= self.round_count
                or any(not isinstance(state.get(key), str) or not state[key] for key in required)
                or type(state.get("pre_session")) is not bool
                or not (
                    isinstance(state.get("session_id"), str) and state["session_id"]
                    or (state.get("session_id") is None and state["pre_session"]
                        and state["stage"] == "prepare" and state["experiment"] == 1)
                )
                or (state["pre_session"] and state.get("session_id") is not None)
                or not isinstance(state.get("commands"), list)
                or type(state.get("controller_submissions")) is not int
                or state["controller_submissions"] < 0
                or type(state.get("measurement_attempts")) is not int
                or state["measurement_attempts"] < 0
                or (state["stage"] in {"controller", "measurement", "finalize"}
                    and (not valid_hash(state.get("candidate_sha256"))
                         or not valid_hash(state.get("manifest_sha256"))))
                or (state["stage"] in {"controller", "measurement", "finalize"}
                    and not isinstance(state.get("receipt"), dict))):
            raise AuditError("blocked checkpoint schema is invalid")
        agent_retries = list(_retry_records(
            state.get("agent_retries", ()), contextual=True,
        ))
        if any(record["experiment"] != state["experiment"]
               for record in agent_retries):
            raise AuditError("agent retry evidence belongs to a different experiment")
        attempts = state.get("candidate_attempts", [])
        if (not isinstance(attempts, list)
                or len(attempts) > self.max_candidate_repairs + 1
                or not valid_hash(state.get("reasoning_sha256", sha256_bytes(b"")))):
            raise AuditError("blocked checkpoint candidate attempt history is invalid")
        for index, attempt in enumerate(attempts, 1):
            try:
                candidate = attempt["candidate"].encode()
                manifest = attempt["manifest"].encode()
                commands = attempt["commands"]
                receipt = attempt["receipt"]
                valid = (
                    set(attempt) == {
                        "schema", "attempt", "status", "candidate_sha256",
                        "manifest_sha256", "controller_receipt_sha256",
                        "commands_sha256", "reasoning_sha256", "candidate",
                        "manifest", "receipt", "commands",
                    }
                    and attempt["schema"] == "profiling-skill/candidate-attempt/v1"
                    and attempt["attempt"] == index
                    and (
                        attempt["status"] == "candidate_error"
                        or (state["stage"] == "finalize" and index == len(attempts)
                            and attempt["status"] == "ok")
                    )
                    and sha256_bytes(candidate) == attempt["candidate_sha256"]
                    and sha256_bytes(manifest) == attempt["manifest_sha256"]
                    and sha256_json(receipt) == attempt["controller_receipt_sha256"]
                    and sha256_json(commands) == attempt["commands_sha256"]
                    and valid_hash(attempt["reasoning_sha256"])
                    and isinstance(commands, list)
                )
                validate_controller_receipt(
                    receipt, attempt["candidate_sha256"], attempt["manifest_sha256"]
                )
            except (AttributeError, KeyError, TypeError, UnicodeEncodeError, AuditError):
                valid = False
            if not valid:
                raise AuditError("blocked checkpoint candidate attempt history is invalid")
        if state.get("branch") != expected_branch or _git(
            self.repo, "branch", "--show-current"
        ) != expected_branch:
            raise AuditError("blocked checkpoint belongs to a different experiment branch")
        completed = _git(
            self.repo, "rev-list", "--first-parent", "--reverse",
            f"{state['seed_commit']}..HEAD^",
        ).splitlines()
        all_commits = _git(self.repo, "rev-list", f"{state['seed_commit']}..HEAD^").splitlines()
        if (len(completed) != state["experiment"] - 1 or len(all_commits) != len(completed)
                or _git(self.repo, "rev-parse", "HEAD^") != state["resume_parent"]
                or sha256_bytes(_git_blob(self.repo, state["seed_commit"], ".experiment/seed.json"))
                != state["seed_hash"]
                or sha256_bytes(_git_blob(self.repo, "HEAD^", "candidate.py"))
                != state["prior_candidate_sha256"]
                or sha256_bytes((self.repo / "PROMPT.md").read_bytes()) != self.prompt_hash
                or sha256_bytes((self.repo / "TASK.md").read_bytes()) != self.task_hash):
            raise AuditError("blocked checkpoint history is invalid")
        try:
            seed = json.loads(_git_blob(self.repo, state["seed_commit"], ".experiment/seed.json"))
        except json.JSONDecodeError as failure:
            raise AuditError("experiment seed provenance is invalid") from failure
        seed_rounds = 3 if seed.get("schema") == "profiling-skill/audited-seed/v1" \
            else seed.get("round_count")
        if seed_rounds != self.round_count or seed_rounds != checkpoint_rounds:
            raise AuditError("experiment seed round count differs from the checkpoint")
        current_reproducibility = {
            "agent": _identity(self.invoke), "controller": _identity(self.controller),
        }
        seeded = seed.get("reproducibility", {})
        if seeded.get("agent") != current_reproducibility["agent"]:
            raise AuditError("current agent identity differs from experiment seed")
        controller_changed = seeded.get("controller") != current_reproducibility["controller"]
        if controller_changed and state.get("runtime_migration") is None:
            raise AuditError("current reproducibility metadata differs from experiment seed")
        if controller_changed or state.get("runtime_migration") is not None:
            self.runtime_migration = _validate_runtime_migration(
                self.repo, state, seed, current_reproducibility, agent_id,
                self.trusted_runtime_migration,
            )
        elif seed.get("reproducibility") != current_reproducibility:
            raise AuditError("current reproducibility metadata differs from experiment seed")
        state["agent_retries"] = agent_retries
        if state["stage"] in {"controller", "measurement", "finalize"} and (
            sha256_bytes((self.repo / "candidate.py").read_bytes()) != state["candidate_sha256"]
            or sha256_bytes((self.repo / "candidate.manifest.json").read_bytes())
            != state["manifest_sha256"]
        ):
            raise AuditError("blocked checkpoint candidate is invalid")
        handle = (state.get("receipt") or {}).get("handle")
        bound_receipt = state["stage"] != "controller" or isinstance(handle, str) and handle
        if bound_receipt and state["stage"] in {"controller", "measurement", "finalize"} and (
            state["receipt"].get("candidate_sha256") != state["candidate_sha256"]
            or state["receipt"].get("manifest_sha256") != state["manifest_sha256"]
        ):
            raise AuditError("blocked checkpoint receipt is invalid")
        _git(self.repo, "reset", "--mixed", "HEAD^")
        path.unlink()
        return state

    def _resume_seed_only(self, run_id: str, agent_id: str) -> dict:
        expected_branch = f"experiment/{run_id}/{agent_id}"
        seed_commit = _git(
            self.repo, "log", "-1", "--format=%H", "--", ".experiment/seed.json",
        )
        changed = set(_git(self.repo, "diff", "--name-only", "HEAD").splitlines())
        changed.update(_git(
            self.repo, "ls-files", "--others", "--exclude-standard",
        ).splitlines())
        allowed = {"candidate.py", "candidate.manifest.json"}
        if (not seed_commit or _git(self.repo, "rev-parse", "HEAD") != seed_commit
                or _git(self.repo, "branch", "--show-current") != expected_branch
                or changed - allowed
                or any(not (self.repo / name).is_file() for name in allowed)):
            raise AuditError("seed-only resume history or worktree is invalid")
        try:
            seed_bytes = _git_blob(self.repo, seed_commit, ".experiment/seed.json")
            seed = json.loads(seed_bytes)
        except json.JSONDecodeError as failure:
            raise AuditError("seed-only resume provenance is invalid") from failure
        seed_rounds = 3 if seed.get("schema") == "profiling-skill/audited-seed/v1" \
            else seed.get("round_count")
        current_reproducibility = {
            "agent": _identity(self.invoke), "controller": _identity(self.controller),
        }
        if (seed.get("run_id") != run_id or seed.get("agent_id") != agent_id
                or seed_rounds != self.round_count
                or seed.get("prompt_sha256") != self.prompt_hash
                or seed.get("task_sha256") != self.task_hash
                or seed.get("reproducibility") != current_reproducibility
                or sha256_bytes((self.repo / "PROMPT.md").read_bytes()) != self.prompt_hash
                or sha256_bytes((self.repo / "TASK.md").read_bytes()) != self.task_hash):
            raise AuditError("seed-only resume provenance differs from current inputs")
        return {
            "branch": expected_branch, "seed_commit": seed_commit,
            "seed_hash": sha256_bytes(seed_bytes), "session_id": None,
            "experiment": 1, "stage": "prepare", "pre_session": True,
            "seed_only_recovery": True, "commands": [],
            "controller_submissions": 0, "measurement_attempts": 0,
            "prior_candidate_sha256": sha256_bytes(
                _git_blob(self.repo, seed_commit, "candidate.py")
            ),
        }

    def _verify_history(self, seed_commit: str, seed_hash: str,
                        commits: list[str]) -> None:
        history = _git(
            self.repo, "rev-list", "--first-parent", "--reverse", f"{seed_commit}..HEAD",
        ).splitlines()
        all_history = _git(self.repo, "rev-list", f"{seed_commit}..HEAD").splitlines()
        if (history != commits or len(history) != self.round_count
                or len(all_history) != self.round_count
                or sha256_bytes(_git_blob(self.repo, seed_commit, ".experiment/seed.json"))
                != seed_hash
                or sha256_bytes((self.repo / ".experiment/seed.json").read_bytes()) != seed_hash
                or sha256_bytes((self.repo / "PROMPT.md").read_bytes()) != self.prompt_hash
                or sha256_bytes((self.repo / "TASK.md").read_bytes()) != self.task_hash
                or _git(self.repo, "status", "--porcelain")):
            raise AuditError(
                f"seed plus {self.round_count}-commit linear history was not preserved"
            )
