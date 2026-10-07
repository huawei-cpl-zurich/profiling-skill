#!/usr/bin/env python3
"""Production resource and cell adapters for the audited nine-cell campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIRECTORY))
try:
    from audited_resource_admission import (
        TARGETS, AdmissionError, CplRemoteResourcePool,
        cpl_remote_closure_sha256, file_sha256,
    )
finally:
    sys.path.pop(0)

RUNTIME_SCHEMA = "profiling-skill/audited-campaign-runtime/v2"
SUPPORTED_REQUEST_BUDGETS = {24, 48}
MIGRATION_ATTESTATION_SCHEMA = "profiling-skill/audited-runtime-preflight/v1"
MIGRATION_TRUST_SCHEMA = "profiling-skill/audited-runtime-migration-trust/v1"
DEVELOPMENT_CASES = {
    "matmul": [7, 8, 9],
    "gdn": [40, 49, 47, 46, 45],
    "bsa": [47, 46, 49, 44, 43],
}
ALL_CASES = {"matmul": list(range(10)), "gdn": list(range(50)),
             "bsa": list(range(50))}
TREATMENT_SKILLS = {
    "cannbot": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ops-profiling",
    ),
    "project-cannbot": (
        "triton-task-extractor", "triton-op-designer", "triton-op-coding",
        "triton-op-verifier", "triton-latency-optimizer",
        "triton-simulator-optimizer", "npu-arch", "ascend-profiling",
    ),
    "project-guarded": ("ascend-profiling", "triton-guarded-kernel"),
}
CANNBOT_SKILLS = tuple(sorted({
    skill for treatment in ("cannbot", "project-cannbot")
    for skill in TREATMENT_SKILLS[treatment]
    if skill not in {"ascend-profiling", "triton-guarded-kernel"}
}))
RUNTIME_FILES = {
    "audited_bz_controller.py", "audited_contract.py", "audited_lifecycle.py",
    "audited_runtime.py", "audited_verifier.py", "benchmark_backend.py",
    "bz_a3_job_client.py", "validate_audited_experiment.py",
}


class ProductionError(RuntimeError):
    pass


def document_sha256(document: dict) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _positive_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def digest_tree(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ProductionError(f"pinned tree is unavailable: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        mode = path.lstat().st_mode
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(mode)):
            raise ProductionError(f"pinned tree contains unsupported entry: {relative}")
        if path.is_file():
            digest.update(relative.encode() + b"\0")
            digest.update(oct(stat.S_IMODE(mode)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def digest_skill_bundle(bindings: dict, names: list[str] | tuple[str, ...]) -> str:
    """Bind a composite manifest skill pin to each exact constituent tree."""
    digest = hashlib.sha256()
    for name in sorted(names):
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise ProductionError(f"skill bundle source is missing: {name}")
        tree = Path(binding.get("path", ""))
        actual = digest_tree(tree)
        if binding.get("sha256") != actual:
            raise ProductionError(f"skill {name} hash does not match pinned input")
        digest.update(name.encode() + b"\0" + actual.encode() + b"\0")
    return digest.hexdigest()


def _read_pinned(path: Path, expected: str, label: str) -> dict:
    if not path.is_file():
        raise ProductionError(f"{label} placement-provider input is unavailable: {path}")
    if file_sha256(path) != expected:
        raise ProductionError(f"{label} hash does not match the pinned input")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ProductionError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ProductionError(f"{label} must be a JSON object")
    return value


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ProductionError(f"isolated skill destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_canary_gate(config: dict, results_path: Path | None,
                         results_sha256: str | None) -> dict:
    """Authenticate the three declared canaries before a repair-aware campaign."""
    if results_path is None or results_sha256 is None:
        raise ProductionError("pinned canary results are required")
    binding = config.get("canary_definition")
    if (not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)):
        raise ProductionError("pinned canary definition is required")
    definition = _read_pinned(
        Path(binding["path"]), binding["sha256"], "canary definition"
    )
    expected_gate = {
        "all_canaries_terminal_ok": True,
        "all_branches_offline_valid": True,
        "all_final_timings_positive": True,
        "minimum_repaired_canaries": 2,
        "resume_canary_required": True,
    }
    declarations = definition.get("canaries")
    if (definition.get("schema") != "profiling-skill/audited-repair-canaries/v1"
            or definition.get("benchmark") != "matmul"
            or definition.get("request_budget") != 48
            or definition.get("max_candidate_repairs_per_round") != 2
            or definition.get("placement") != "dynamic-bz-a3-admission"
            or definition.get("gate") != expected_gate
            or not isinstance(declarations, list) or len(declarations) != 3):
        raise ProductionError("canary definition does not match the production gate")
    declared = {}
    for item in declarations:
        if (not isinstance(item, dict)
                or set(item) != {"id", "treatment", "required_evidence"}
                or not isinstance(item.get("id"), str) or not item["id"]
                or item.get("treatment") not in TREATMENT_SKILLS
                or not isinstance(item.get("required_evidence"), list)
                or not item["required_evidence"]
                or any(not isinstance(value, str) or not value
                       for value in item["required_evidence"])
                or item["id"] in declared):
            raise ProductionError("canary declaration is malformed")
        declared[item["id"]] = item
    if {item["treatment"] for item in declarations} != set(TREATMENT_SKILLS):
        raise ProductionError("canary declarations must cover all three treatments")

    results = _read_pinned(Path(results_path), results_sha256, "canary results")
    records = results.get("results")
    if (results.get("schema") != "profiling-skill/audited-repair-canary-results/v1"
            or results.get("definition_sha256") != binding["sha256"]
            or results.get("source_revision") != config.get("provenance", {}).get(
                "source_revision")
            or results.get("runtime_closure_sha256") != config.get(
                "runtime_scripts", {}).get("sha256")
            or not isinstance(records, list) or len(records) != 3):
        raise ProductionError("canary results do not match pinned production inputs")
    def artifact(binding: object, label: str) -> dict:
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                or not isinstance(binding.get("path"), str)
                or not Path(binding["path"]).is_absolute()
                or not isinstance(binding.get("sha256"), str)):
            raise ProductionError(f"canary {label} binding is malformed")
        return _read_pinned(Path(binding["path"]), binding["sha256"], label)

    seen = set()
    repaired = 0
    resumed = set()
    for result in records:
        if (not isinstance(result, dict) or set(result) != {
                "id", "treatment", "experiment_commit", "verifier_report",
                "cell_receipt", "resume_receipt",
        }):
            raise ProductionError("canary result is malformed")
        declaration = declared.get(result.get("id"))
        commit = result.get("experiment_commit")
        if (declaration is None or result["id"] in seen
                or result.get("treatment") != declaration["treatment"]
                or not isinstance(commit, str) or len(commit) != 40
                or any(character not in "0123456789abcdef" for character in commit)):
            raise ProductionError("canary result does not match its declaration")
        verifier = artifact(result.get("verifier_report"), "canary verifier report")
        receipt = artifact(result.get("cell_receipt"), "canary cell receipt")
        experiments = verifier.get("experiments")
        verifier_commits = ([item.get("commit") for item in experiments]
                            if isinstance(experiments, list)
                            and all(isinstance(item, dict) for item in experiments) else [])
        branch = receipt.get("branch")
        rounds = receipt.get("rounds")
        history = receipt.get("attempt_history")
        if (verifier.get("status") != "valid"
                or not isinstance(branch, str) or not branch
                or verifier.get("branch") != branch
                or not verifier_commits or verifier_commits[-1] != commit
                or receipt.get("status") != "complete"
                or receipt.get("commits") != verifier_commits
                or receipt.get("rounds_completed") != 4
                or not isinstance(receipt.get("durable_handle"), str)
                or not receipt["durable_handle"]
                or not isinstance(rounds, list) or len(rounds) != 4
                or not isinstance(history, list) or len(history) != 4):
            raise ProductionError("canary retained branch evidence is invalid")
        final = rounds[-1]
        if (not isinstance(final, dict) or final.get("round") != 4
                or final.get("status") != "ok"
                or final.get("handle") != receipt["durable_handle"]
                or not _positive_number(final.get("median_us"))
                or not isinstance(final.get("compact_artifacts"), list)
                or not final["compact_artifacts"]
                or any(not isinstance(value, str) or not value
                       for value in final["compact_artifacts"])
                or [item.get("round") if isinstance(item, dict) else None
                    for item in history] != [1, 2, 3, 4]
                or any(not isinstance(item.get("statuses"), list)
                       or not item["statuses"]
                       or item["statuses"][-1] != "ok"
                       or any(not isinstance(value, str) or not value
                              for value in item["statuses"])
                       for item in history)):
            raise ProductionError("canary retained timing evidence is invalid")
        repairs = sum(len(item["statuses"]) - 1 for item in history)
        required = set(declaration["required_evidence"])
        if "candidate-repair" in required and repairs == 0:
            raise ProductionError("canary retained repair evidence is missing")
        resume_binding = result.get("resume_receipt")
        if {"checkpoint-resume", "same-session"} & required:
            resume = artifact(resume_binding, "canary resume receipt")
            session = verifier.get("session_id")
            handles = {item.get("handle") for item in rounds if isinstance(item, dict)}
            checkpoint = resume.get("checkpoint_sha256")
            if (resume.get("schema") !=
                    "profiling-skill/audited-repair-canary-resume/v1"
                    or resume.get("canary_id") != result["id"]
                    or not isinstance(checkpoint, str) or len(checkpoint) != 64
                    or any(character not in "0123456789abcdef"
                           for character in checkpoint)
                    or not isinstance(session, str) or not session
                    or resume.get("session_id_before") != session
                    or resume.get("session_id_after") != session
                    or resume.get("durable_handle_before") not in handles
                    or resume.get("durable_handle_after") !=
                    resume.get("durable_handle_before")):
                raise ProductionError("canary retained resume evidence is invalid")
            resumed.add(result["id"])
        elif resume_binding is not None:
            raise ProductionError("undeclared canary resume evidence is not allowed")
        seen.add(result["id"])
        repaired += int(repairs > 0)
    resume_ids = {item["id"] for item in declarations
                  if "checkpoint-resume" in item["required_evidence"]}
    if (seen != set(declared)
            or repaired < expected_gate["minimum_repaired_canaries"]
            or not resume_ids or not resume_ids.issubset(resumed)):
        raise ProductionError("canary results do not satisfy the aggregate gate")
    return {
        "schema": "profiling-skill/audited-repair-canary-gate/v1",
        "status": "passed", "definition_sha256": binding["sha256"],
        "results_sha256": results_sha256,
        "source_revision": results["source_revision"],
        "runtime_closure_sha256": results["runtime_closure_sha256"],
        "canary_ids": sorted(seen), "repaired_canaries": repaired,
        "resumed_canaries": sorted(resumed),
    }


def _retain_canary_gate(config: dict, receipt: dict) -> None:
    path = Path(config["run_root"]) / "state" / "canary-gate.json"
    if path.is_file():
        try:
            if json.loads(path.read_text()) != receipt:
                raise ProductionError("retained canary gate does not match pinned results")
        except json.JSONDecodeError as error:
            raise ProductionError("retained canary gate is invalid JSON") from error
        return
    _write_json_atomic(path, receipt)


class DurableAdmissionPool:
    """Record every accepted admission refresh before exposing its slots."""

    def __init__(self, pool, evidence_path: Path, run_id: str):
        self.pool = pool
        self.evidence_path = Path(evidence_path)
        self.run_id = run_id
        self.last_receipt = None

    def _document(self) -> dict:
        if not self.evidence_path.exists():
            return {"schema": "profiling-skill/admission-evidence/v1",
                    "run_id": self.run_id, "refreshes": []}
        try:
            document = json.loads(self.evidence_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProductionError("admission evidence is unreadable") from error
        if (not isinstance(document, dict)
                or set(document) != {"schema", "run_id", "refreshes"}
                or document.get("schema") != "profiling-skill/admission-evidence/v1"
                or document.get("run_id") != self.run_id
                or not isinstance(document.get("refreshes"), list)):
            raise ProductionError("admission evidence does not match this campaign")
        return document

    def admit(self) -> list[dict]:
        snapshot = self.pool.admit_snapshot()
        document = self._document()
        document["refreshes"].append({
            "sequence": len(document["refreshes"]) + 1,
            "receipt_sha256": snapshot.receipt_sha256,
            "provider_id": snapshot.provider_id,
            "allowlist_sha256": snapshot.allowlist_sha256,
            "generated_at": snapshot.generated_at,
            "expires_at": snapshot.expires_at,
        })
        _write_json_atomic(self.evidence_path, document)
        self.last_receipt = snapshot
        return snapshot.as_slots()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise ProductionError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_blob(repo: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=repo, capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ProductionError(f"seed commit does not contain {path}")
    return result.stdout


class ProductionCellLauncher:
    """Create or resume one isolated four-round branch and its controller."""

    def __init__(self, config: dict, *, infrastructure_failure_type: type[Exception],
                 invoker_factory=None,
                 controller_factory=None, runner_factory=None,
                 verifier_invoke: Callable = subprocess.run,
                 trusted_runtime_migration: dict | list[dict] | None = None,
                 runtime_config_path: Path | None = None,
                 runtime_config_sha256: str | None = None,
                 audit_error_type: type[Exception] | None = None):
        if config.get("schema") != RUNTIME_SCHEMA:
            raise ProductionError(f"runtime config requires schema {RUNTIME_SCHEMA}")
        self.config = config
        if (not isinstance(infrastructure_failure_type, type)
                or not issubclass(infrastructure_failure_type, Exception)):
            raise ProductionError("infrastructure failure type must be an exception class")
        self.infrastructure_failure_type = infrastructure_failure_type
        runtime = config.get("runtime_scripts", {})
        self.scripts = Path(runtime.get("path", "")).resolve()
        if digest_tree(self.scripts) != runtime.get("sha256"):
            raise ProductionError("runtime scripts hash does not match pinned closure")
        self.trusted_runtime_migration = trusted_runtime_migration
        self._migration_attestations = self._validate_migration_trust(
            runtime_config_path, runtime_config_sha256,
        )
        self.verifier_invoke = verifier_invoke
        self.run_root = Path(config["run_root"]).resolve()
        if "cpl_remote" in config:
            raise ProductionError("runtime config cannot override global cpl-remote")
        self.cpl_remote = str(
            Path.home() / ".agents" / "skills" / "remote-access" / "scripts" / "cpl-remote"
        )
        try:
            actual_remote_closure = cpl_remote_closure_sha256(Path(self.cpl_remote))
        except AdmissionError as error:
            raise ProductionError("global cpl-remote closure is unavailable") from error
        if actual_remote_closure != config.get("cpl_remote_closure_sha256"):
            raise ProductionError("global cpl-remote closure hash does not match pinned input")
        if config.get("runtime_mode") != "docker":
            raise ProductionError("production agents require an isolated Docker runtime")
        if "adapter_command" in config or "remote_command" in config:
            raise ProductionError("runtime config cannot supply an adapter or remote command")
        self._validate_timeouts()
        self.max_candidate_repairs = config.get("max_candidate_repairs_per_round", 2)
        if self.max_candidate_repairs != 2 or type(self.max_candidate_repairs) is not int:
            raise ProductionError(
                "production max_candidate_repairs_per_round must be exactly two"
            )
        self._validate_files()
        if invoker_factory is None or controller_factory is None or runner_factory is None:
            sys.path.insert(0, str(self.scripts))
            try:
                from audited_contract import AuditError
                from audited_lifecycle import AuditedExperimentRunner
                from audited_runtime import CodexInvoker, CommandController
            finally:
                sys.path.pop(0)
            invoker_factory = invoker_factory or CodexInvoker
            controller_factory = controller_factory or CommandController
            runner_factory = runner_factory or AuditedExperimentRunner
            audit_error_type = audit_error_type or AuditError
        if (audit_error_type is not None
                and (not isinstance(audit_error_type, type)
                     or not issubclass(audit_error_type, Exception))):
            raise ProductionError("audit failure type must be an exception class")
        self.invoker_factory = invoker_factory
        self.controller_factory = controller_factory
        self.runner_factory = runner_factory
        self.audit_error_types = ((audit_error_type,) if audit_error_type is not None else ())

    def _validate_migration_trust(self, config_path: Path | None,
                                  config_sha256: str | None) -> tuple[dict, ...]:
        trusted = self.trusted_runtime_migration
        supplied = (trusted is not None, config_path is not None, config_sha256 is not None)
        if not any(supplied):
            return ()
        if not all(supplied):
            raise ProductionError("runtime migration trust requires its pinned runtime config")
        trusts = [trusted] if isinstance(trusted, dict) else trusted
        if not isinstance(trusts, (list, tuple)) or not trusts:
            raise ProductionError("runtime migration trust binding is invalid")
        expected_keys = {
            "schema", "attestation_path", "attestation_file_sha256",
            "attestation_sha256",
        }
        pinned_config = Path(config_path).resolve()
        if (not pinned_config.is_file()
                or file_sha256(pinned_config) != config_sha256):
            raise ProductionError("runtime migration config binding is invalid")
        attestations = []
        expected_old = self.config["provenance"].get("controller_sha256")
        try:
            for index, proof in enumerate(trusts):
                if (not isinstance(proof, dict) or set(proof) != expected_keys
                        or proof.get("schema") != MIGRATION_TRUST_SCHEMA):
                    raise ProductionError("runtime migration trust binding is invalid")
                attestation_path = Path(proof["attestation_path"])
                raw = attestation_path.read_bytes()
                attestation = json.loads(raw)
                seal = attestation["attestation_sha256"]
                unsealed = {key: value for key, value in attestation.items()
                            if key != "attestation_sha256"}
                runtimes = attestation["runtimes"]
                old_runtime, new_runtime = runtimes["old"], runtimes["new"]
                if (not attestation_path.is_absolute()
                        or attestation.get("schema") != MIGRATION_ATTESTATION_SCHEMA
                        or proof["attestation_file_sha256"]
                        != hashlib.sha256(raw).hexdigest()
                        or proof["attestation_sha256"] != seal
                        or document_sha256(unsealed) != seal
                        or old_runtime.get("closure_sha256") != expected_old):
                    raise ProductionError("runtime migration attestation is invalid")
                expected_old = new_runtime.get("closure_sha256")
                if index == len(trusts) - 1 and (
                        expected_old != self.config["runtime_scripts"].get("sha256")
                        or Path(new_runtime.get("config_path", "")).resolve() != pinned_config
                        or new_runtime.get("config_sha256") != config_sha256):
                    raise ProductionError("runtime migration attestation is invalid")
                attestations.append(attestation)
        except ProductionError:
            raise
        except (KeyError, TypeError, ValueError, OSError,
                json.JSONDecodeError) as error:
            raise ProductionError("runtime migration attestation is malformed") from error
        return tuple(attestations)

    def _validate_timeouts(self) -> None:
        names = (
            "agent_turn_timeout", "controller_transaction_timeout",
            "verifier_timeout", "backend_job_timeout", "timeout_grace",
        )
        if "timeout" in self.config:
            raise ProductionError("ambiguous legacy timeout is not allowed")
        for name in names:
            value = self.config.get(name)
            if type(value) is not int or value <= 0:
                raise ProductionError(f"{name} must be a positive integer")
        backend_outer = self.config["backend_job_timeout"] + self.config["timeout_grace"]
        if self.config["controller_transaction_timeout"] <= (
                backend_outer + self.config["timeout_grace"]):
            raise ProductionError(
                "controller transaction timeout must exceed the backend job timeout "
                "and both timeout grace intervals"
            )

    def _validate_files(self) -> None:
        assets = self.config.get("benchmark_assets", {})
        asset_path = Path(assets.get("path", "")).resolve()
        if (asset_path != self.scripts.parent / "benchmarks"
                or digest_tree(asset_path) != assets.get("sha256")):
            raise ProductionError("benchmark assets hash/path does not match pinned closure")
        missing = sorted(name for name in RUNTIME_FILES if not (self.scripts / name).is_file())
        if missing:
            raise ProductionError(f"runtime controller closure is incomplete: {', '.join(missing)}")
        for label, binding in [("prompt", self.config.get("prompt", {})),
                               *[(f"task {name}", value)
                                 for name, value in self.config.get("tasks", {}).items()]]:
            path = Path(binding.get("path", ""))
            if not path.is_file() or file_sha256(path) != binding.get("sha256"):
                raise ProductionError(f"{label} hash does not match pinned input")
        for name, binding in self.config.get("skill_sources", {}).items():
            path = Path(binding.get("path", ""))
            if digest_tree(path) != binding.get("sha256"):
                raise ProductionError(f"skill {name} hash does not match pinned input")
        self._validate_provenance()

    def _validate_provenance(self) -> None:
        provenance = self.config.get("provenance")
        if not isinstance(provenance, dict):
            raise ProductionError("runtime provenance is missing")
        manifest_sha256 = self.config.get("manifest_sha256")
        if (manifest_sha256 is not None
                and (not isinstance(manifest_sha256, str)
                     or len(manifest_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character in manifest_sha256))):
            raise ProductionError("runtime manifest identity is invalid")
        sources = self.config.get("source_repositories", {})
        expected_tasks = set(DEVELOPMENT_CASES)
        if (set(sources) != expected_tasks or set(self.config.get("tasks", {})) != expected_tasks):
            raise ProductionError("source/task provenance must cover the complete campaign")
        revisions = {binding.get("revision") for binding in sources.values()
                     if isinstance(binding, dict)}
        if revisions != {provenance.get("source_revision")}:
            raise ProductionError("source revision provenance does not match runtime inputs")
        runtime = self.config["runtime_scripts"]
        if (runtime.get("sha256") != provenance.get("controller_sha256")
                and not self._migration_attestations):
            raise ProductionError("controller provenance does not match runtime closure")
        if self.config.get("runtime_image_digest") != provenance.get("runtime_image_digest"):
            raise ProductionError("runtime image provenance does not match Docker input")
        if provenance.get("model") != {
            "name": self.config.get("model"),
            "reasoning_effort": self.config.get("reasoning_effort"),
        }:
            raise ProductionError("model provenance does not match runtime inputs")
        resource = self.config.get("resource_admission")
        resource_path = (Path(resource.get("path", "")).resolve()
                         if isinstance(resource, dict) else Path(""))
        expected_resource = _SCRIPT_DIRECTORY / "audited_resource_admission.py"
        actual_resource = (file_sha256(resource_path)
                           if resource_path.is_file() else None)
        if (resource_path != expected_resource
                or actual_resource != resource.get("sha256")
                or actual_resource != provenance.get("resource_admission_sha256")
                or self.config.get("cpl_remote_closure_sha256") !=
                provenance.get("cpl_remote_closure_sha256")
                or self.config.get("admission_provider_id") !=
                provenance.get("admission_provider_id")
                or self.config.get("admission_allowlist_sha256") !=
                provenance.get("admission_allowlist_sha256")):
            raise ProductionError(
                "resource admission provenance does not match runtime inputs"
            )
        baseline_sources = self.config.get("baseline_sources")
        baseline_pins = provenance.get("baselines")
        if (not isinstance(baseline_sources, dict) or not isinstance(baseline_pins, dict)
                or set(baseline_sources) != expected_tasks
                or set(baseline_pins) != expected_tasks):
            raise ProductionError("baseline provenance does not match runtime inputs")
        for task, binding in baseline_sources.items():
            path = Path(binding.get("path", "")) if isinstance(binding, dict) else Path("")
            actual = file_sha256(path) if path.is_file() else None
            if actual != binding.get("sha256") or actual != baseline_pins[task]:
                raise ProductionError(f"baseline provenance does not match runtime input: {task}")
            self._baseline(task)
        starter_sources = self.config.get("starter_sources")
        starter_pins = provenance.get("starters")
        if (not isinstance(starter_sources, dict) or not isinstance(starter_pins, dict)
                or set(starter_sources) != expected_tasks
                or set(starter_pins) != expected_tasks):
            raise ProductionError("starter provenance does not match runtime inputs")
        for task in expected_tasks:
            if starter_sources[task] != starter_pins[task]:
                raise ProductionError(
                    f"starter provenance does not match runtime input: {task}"
                )
            self._starter(task)
        skill_pins = provenance.get("skills")
        skill_sources = self.config.get("skill_sources", {})
        if not isinstance(skill_pins, dict):
            raise ProductionError("skill provenance does not match runtime inputs")
        if skill_pins.get("cannbot") != digest_skill_bundle(skill_sources, CANNBOT_SKILLS):
            raise ProductionError("cannbot skill provenance does not match runtime trees")
        for name in ("ascend-profiling", "triton-guarded-kernel"):
            binding = skill_sources.get(name)
            if (not isinstance(binding, dict)
                    or skill_pins.get(name) != binding.get("sha256")):
                raise ProductionError(f"{name} skill provenance does not match runtime tree")

    def _baseline(self, task: str) -> dict:
        path = Path(self.config["baseline_sources"][task]["path"])
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProductionError(f"timing baseline is invalid JSON: {task}") from error
        required = {"schema", "benchmark", "case_medians_us",
                    "control_median_us", "sha256"}
        rows = document.get("case_medians_us") if isinstance(document, dict) else None
        if (not isinstance(document, dict) or set(document) != required
                or document.get("schema") != "profiling-skill/baseline-timing/v1"
                or document.get("benchmark") != task or not isinstance(rows, list)
                or [row.get("case") if isinstance(row, dict) else None for row in rows]
                != DEVELOPMENT_CASES[task]
                or any(set(row) != {"case", "median_us"}
                       or not _positive_number(row["median_us"]) for row in rows)
                or not _positive_number(document.get("control_median_us"))
                or document.get("sha256") != document_sha256({
                    key: value for key, value in document.items() if key != "sha256"
                })):
            raise ProductionError(f"timing baseline contract is invalid: {task}")
        return document

    def _starter(self, task: str) -> dict[str, bytes]:
        binding = self.config["starter_sources"].get(task)
        if not isinstance(binding, dict) or set(binding) != {"candidate", "manifest"}:
            raise ProductionError(f"starter binding is invalid: {task}")
        contents = {}
        for kind in ("candidate", "manifest"):
            item = binding.get(kind)
            path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
            if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                    or not path.is_absolute() or not path.is_file() or path.is_symlink()):
                raise ProductionError(f"starter {kind} hash/path does not match pin: {task}")
            contents[kind] = path.read_bytes()
            if hashlib.sha256(contents[kind]).hexdigest() != item.get("sha256"):
                raise ProductionError(f"starter {kind} hash/path does not match pin: {task}")
        try:
            manifest = json.loads(contents["manifest"])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProductionError(f"starter manifest is invalid JSON: {task}") from error
        if (not isinstance(manifest, dict)
                or manifest.get("schema") != "profiling-skill/candidate-kernel/v1"
                or not isinstance(manifest.get("kernel_name"), str)
                or not manifest["kernel_name"].strip()
                or manifest["kernel_name"] != manifest["kernel_name"].strip()):
            raise ProductionError(f"starter manifest contract is invalid: {task}")
        return contents

    def _validate_materialized_starter(self, repo: Path, task: str,
                                       starter: dict[str, bytes]) -> None:
        seed = repo / ".experiment" / "seed.json"
        if seed.is_file():
            seed_commit = _git(repo, "log", "-1", "--format=%H", "--", ".experiment/seed.json")
            if (not seed_commit
                    or _git_blob(repo, seed_commit, "candidate.py") != starter["candidate"]
                    or _git_blob(repo, seed_commit, "candidate.manifest.json") !=
                    starter["manifest"]):
                raise ProductionError(
                    f"seed commit starter does not match pinned input: {task}"
                )
            return
        for kind, filename in (("candidate", "candidate.py"),
                               ("manifest", "candidate.manifest.json")):
            path = repo / filename
            if not path.is_file() or path.read_bytes() != starter[kind]:
                raise ProductionError(
                    f"unseeded materialized starter does not match pinned input: {task}"
                )

    def _prepare_repo(self, cell: dict) -> tuple[Path, bool]:
        root = self.run_root / cell["cell_id"]
        staging = self.run_root / f".{cell['cell_id']}.initializing"
        repo = root / "repo"
        source = self.config["source_repositories"].get(cell["task"])
        if not isinstance(source, dict):
            raise ProductionError(f"source repository is missing for {cell['task']}")
        source_path, revision = Path(source["path"]).resolve(), source.get("revision")
        if (not isinstance(revision, str) or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)):
            raise ProductionError("source repository revision must be a pinned commit")
        starter = self._starter(cell["task"])
        starter_binding = self.config["starter_sources"][cell["task"]]
        legacy_identity = {
            "cell_id": cell["cell_id"], "task": cell["task"],
            "treatment": cell["treatment"], "source_revision": revision,
            "starter": starter_binding,
        }
        identity = {
            "schema": "profiling-skill/cell-identity/v2", **legacy_identity,
            "run_id": self.config["run_id"],
            "branch": f"experiment/{self.config['run_id']}/{cell['cell_id']}",
            "round_count": cell.get("round_count"),
            "request_budget": cell.get("request_budget"),
            "task_sha256": cell.get("task_sha256"),
            "prompt_contract": cell.get("prompt_contract"),
            "skills": cell.get("skills"),
            "manifest_sha256": self.config.get("manifest_sha256"),
        }
        identity_path = root / "state" / "cell.json"
        expected = tuple(TREATMENT_SKILLS[cell["treatment"]])
        if tuple(cell["skills"]) != expected:
            raise ProductionError("cell treatment allowlist does not match campaign policy")
        if repo.exists():
            try:
                retained_identity = json.loads(identity_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ProductionError(
                    "existing isolated repository has an unreadable identity"
                ) from error
            if retained_identity == legacy_identity:
                controller_path = root / "state" / "controller.json"
                seed_commit = _git(
                    repo, "log", "-1", "--format=%H", "--", ".experiment/seed.json"
                )
                if not seed_commit:
                    raise ProductionError(
                        "legacy cell identity lacks immutable campaign evidence"
                    )
                try:
                    controller = json.loads(controller_path.read_text())
                    seed = json.loads(_git_blob(
                        repo, seed_commit, ".experiment/seed.json"
                    ))
                except (OSError, json.JSONDecodeError) as error:
                    raise ProductionError(
                        "legacy cell identity lacks immutable campaign evidence"
                    ) from error
                if (controller.get("benchmark") != cell["task"]
                        or controller.get("round_count") != cell.get("round_count")
                        or controller.get("request_budget") != cell.get("request_budget")
                        or seed.get("run_id") != self.config["run_id"]
                        or seed.get("agent_id") != cell["cell_id"]
                        or seed.get("round_count") != cell.get("round_count")
                        or seed.get("prompt_sha256") !=
                        self.config["prompt"]["sha256"]
                        or seed.get("task_sha256") !=
                        self.config["tasks"][cell["task"]]["sha256"]):
                    raise ProductionError(
                        "legacy cell identity does not match immutable campaign evidence"
                    )
                _write_json_atomic(identity_path, identity)
            elif retained_identity != identity:
                raise ProductionError("existing isolated repository has a different identity")
            self._validate_isolated_skills(repo, expected)
            self._validate_materialized_starter(repo, cell["task"], starter)
            return repo, (repo / ".experiment" / "seed.json").is_file()
        if root.exists():
            raise ProductionError("existing isolated cell is incomplete")
        self.run_root.mkdir(parents=True, exist_ok=True)
        bootstrap = {
            "schema": "profiling-skill/cell-bootstrap/v1",
            "identity": identity,
        }
        marker = staging / "state" / "bootstrap.json"
        if staging.exists():
            try:
                existing_bootstrap = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ProductionError("unowned partial bootstrap cannot be recreated") from error
            if existing_bootstrap != bootstrap:
                raise ProductionError("unowned partial bootstrap cannot be recreated")
            shutil.rmtree(staging)
        staging.mkdir()
        _write_json_atomic(marker, bootstrap)
        staging_repo = staging / "repo"
        result = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source_path),
             str(staging_repo)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise ProductionError(f"isolated clone failed: {result.stderr.strip()}")
        _git(staging_repo, "checkout", "--quiet", "--detach", revision)
        if _git(staging_repo, "rev-parse", "HEAD") != revision:
            raise ProductionError("isolated repository revision does not match pin")
        if _git(staging_repo, "status", "--porcelain"):
            raise ProductionError("isolated source checkout is not clean")
        (staging_repo / "candidate.py").write_bytes(starter["candidate"])
        (staging_repo / "candidate.manifest.json").write_bytes(starter["manifest"])
        exclusions = staging_repo / ".git" / "info" / "exclude"
        with exclusions.open("a") as stream:
            stream.write("\n.agents/\nAGENTS.md\n")
        skills = staging_repo / ".agents" / "skills"
        skills.mkdir(parents=True)
        for name in expected:
            binding = self.config["skill_sources"].get(name)
            if not isinstance(binding, dict):
                raise ProductionError(f"pinned skill source is missing: {name}")
            _copy_tree(Path(binding["path"]), skills / name)
        (staging_repo / "AGENTS.md").write_text(
            "Use only the repository-local skills under .agents/skills. "
            "Do not inspect host or global skills.\n"
        )
        self._validate_isolated_skills(staging_repo, expected)
        self._validate_materialized_starter(staging_repo, cell["task"], starter)
        _write_json_atomic(staging / "state" / "cell.json", identity)
        staging.rename(root)
        return repo, False

    def _validate_isolated_skills(self, repo: Path, expected: tuple[str, ...]) -> None:
        skills = repo / ".agents" / "skills"
        try:
            entries = list(skills.iterdir())
        except OSError as error:
            raise ProductionError("isolated treatment skills are unavailable") from error
        if ({entry.name for entry in entries} != set(expected)
                or any(not entry.is_dir() or entry.is_symlink() for entry in entries)):
            raise ProductionError("isolated treatment skills do not match the exact allowlist")
        for name in expected:
            binding = self.config["skill_sources"].get(name)
            if (not isinstance(binding, dict)
                    or digest_tree(skills / name) != binding.get("sha256")):
                raise ProductionError(f"isolated treatment skills have drifted: {name}")

    def _controller(self, cell: dict, slot: dict, root: Path, repo: Path):
        state = root / "state"
        state.mkdir(parents=True, exist_ok=True)
        if (slot.get("target") not in TARGETS or type(slot.get("device")) is not int
                or slot["device"] < 0):
            raise ProductionError("cell placement is not an admitted BZ-A3 device")
        placement_binding = state / "placement.json"
        placement = {"target": slot["target"], "device": slot["device"]}
        if placement_binding.is_file():
            if json.loads(placement_binding.read_text()) != placement:
                raise ProductionError("resume must retain the cell's admitted placement")
        else:
            placement_binding.write_text(json.dumps(placement, sort_keys=True) + "\n")
        placements = state / "placements.json"
        placements.write_text(json.dumps({
            "0": {"target": slot["target"], "device": slot["device"]}
        }, sort_keys=True) + "\n")
        job_client = [
            sys.executable, str(self.scripts / "bz_a3_job_client.py"),
            "--state-dir", str(state / "jobs"), "--placements-json", str(placements),
            "--remote-root", self.config["remote_root"],
            "--cpl-remote-sha256", file_sha256(Path(self.cpl_remote)),
            "--timeout", str(self.config["backend_job_timeout"]),
        ]
        backend = [
            sys.executable, str(self.scripts / "benchmark_backend.py"),
            "--benchmark", cell["task"], "--job-client-json", json.dumps(job_client),
        ]
        document = {
            "schema": "profiling-skill/audited-bz-controller-config/v1",
            "benchmark": cell["task"], "round_count": 4,
            "request_budget": cell["request_budget"],
            "profile_repeats": 3, "variability_threshold": 0.25,
            "control_drift_threshold": 0.2, "infrastructure_retry_budget": 3,
            "timeout_seconds": (
                self.config["backend_job_timeout"] + self.config["timeout_grace"]
            ),
            "devices": [{"id": f"{slot['target']}/device-{slot['device']}", "device": 0}],
            "development_cases": DEVELOPMENT_CASES[cell["task"]],
            "all_cases": ALL_CASES[cell["task"]],
            "baseline": self._baseline(cell["task"]), "backend_command": backend,
        }
        controller_config = state / "controller.json"
        controller_config.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        controller_state = state / "controller"
        controller_state.mkdir(exist_ok=True)
        command = [
            sys.executable, str(self.scripts / "audited_bz_controller.py"),
            "--config", str(controller_config), "--state-dir", str(controller_state),
        ]
        return self.controller_factory(
            command, repo, timeout=self.config["controller_transaction_timeout"]
        )

    def _verify(self, repo: Path, cell: dict) -> dict:
        command = [
            sys.executable, str(self.scripts / "validate_audited_experiment.py"),
            str(repo), "--base", self.config["provenance"]["source_revision"],
        ]
        trusts = ([self.trusted_runtime_migration]
                  if isinstance(self.trusted_runtime_migration, dict)
                  else self.trusted_runtime_migration or [])
        for attestation, trust in zip(self._migration_attestations, trusts):
            if any(isinstance(item, dict) and item.get("cell_id") == cell["cell_id"]
                   for item in attestation.get("cells", [])):
                command.extend([
                    "--migration-proof", trust["attestation_path"],
                    trust["attestation_file_sha256"], trust["attestation_sha256"],
                ])
        try:
            result = self.verifier_invoke(
                command, text=True, capture_output=True,
                timeout=self.config["verifier_timeout"], check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProductionError(f"independent audited verifier could not run: {error}") from error
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise ProductionError(f"independent audited verifier rejected branch: {detail}")
        try:
            verified = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ProductionError("independent audited verifier returned invalid JSON") from error
        if (not isinstance(verified, dict) or verified.get("status") != "valid"
                or len(verified.get("experiments", [])) != 4):
            raise ProductionError("independent audited verifier returned incomplete evidence")
        expected_branch = f"experiment/{self.config['run_id']}/{cell['cell_id']}"
        if verified.get("branch") != expected_branch:
            raise ProductionError("independent audited verifier returned the wrong branch")
        return verified

    def _receipt(self, repo: Path, verified: dict) -> dict:
        rounds = []
        attempt_history = []
        for number in range(1, 5):
            path = repo / "experiments" / f"{number:02d}" / "results.json"
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ProductionError(f"round {number} compact receipt is unavailable") from error
            if not isinstance(receipt, dict):
                raise ProductionError(f"round {number} compact receipt is invalid")
            evidence_path = path.with_name("evidence.json")
            attempt_statuses = None
            if evidence_path.is_file():
                try:
                    evidence = json.loads(evidence_path.read_text())
                    attempts = evidence.get("candidate_attempts")
                    if isinstance(attempts, list) and attempts:
                        attempt_statuses = []
                        for attempt in attempts:
                            attempt_number = attempt.get("attempt")
                            attempt_receipt = json.loads((
                                path.parent / "attempts" / f"{attempt_number:02d}"
                                / "controller.json"
                            ).read_text())
                            attempt_statuses.append(
                                attempt_receipt.get("failure_type", attempt_receipt["status"])
                            )
                except (OSError, KeyError, TypeError, ValueError,
                        json.JSONDecodeError) as error:
                    raise ProductionError(
                        f"round {number} candidate attempt summary is invalid"
                    ) from error
            if attempt_statuses is None:
                attempt_statuses = [receipt.get("failure_type", receipt.get("status"))]
            rounds.append({"round": number, **receipt})
            attempt_history.append({"round": number, "statuses": attempt_statuses})
        commits = tuple(item.get("commit") for item in verified["experiments"])
        if any(not isinstance(commit, str) or not commit for commit in commits):
            raise ProductionError("independent verifier omitted experiment commits")
        terminal = rounds[-1]
        terminal_status = (
            "candidate_failed"
            if any(round_receipt.get("status") == "candidate_error"
                   for round_receipt in rounds)
            else "complete"
        )
        result = {
            "status": terminal_status, "durable_handle": terminal["handle"],
            "rounds_completed": 4, "rounds": rounds,
            "attempt_history": attempt_history,
            "branch": verified.get("branch"), "session_id": verified.get("session_id"),
            "seed_commit": verified.get("seed_commit"), "commits": list(commits),
        }
        for field in ("baseline_median_us", "baseline", "calibration"):
            if field in terminal:
                result[field] = terminal[field]
        return result

    def _resume_checkpoint(self, repo: Path, cell: dict) -> dict | None:
        path = repo / ".experiment" / "blocked.json"
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ProductionError("blocked checkpoint is invalid JSON") from error
        expected_branch = f"experiment/{self.config['run_id']}/{cell['cell_id']}"
        if (not isinstance(state, dict)
                or state.get("schema") != "profiling-skill/audited-blocked/v2"
                or state.get("round_count") != cell["round_count"]
                or state.get("stage") not in {"prepare", "controller", "measurement", "finalize"}
                or state.get("experiment") not in {1, 2, 3, 4}
                or state.get("branch") != expected_branch):
            raise ProductionError("blocked checkpoint is invalid for this cell")
        return state

    def _is_seed_only_resume(self, repo: Path, cell: dict) -> bool:
        seed_commit = _git(repo, "log", "-1", "--format=%H", "--", ".experiment/seed.json")
        expected_branch = f"experiment/{self.config['run_id']}/{cell['cell_id']}"
        if (not seed_commit or _git(repo, "rev-parse", "HEAD") != seed_commit
                or _git(repo, "branch", "--show-current") != expected_branch):
            return False
        changed = set(_git(repo, "diff", "--name-only", "HEAD").splitlines())
        changed.update(_git(repo, "ls-files", "--others", "--exclude-standard").splitlines())
        if changed - {"candidate.py", "candidate.manifest.json"}:
            raise ProductionError("seed-only recovery contains unrelated worktree changes")
        return True

    def launch(self, cell: dict, slot: dict) -> dict:
        if (cell.get("round_count") != 4
                or cell.get("request_budget") not in SUPPORTED_REQUEST_BUDGETS):
            raise ProductionError(
                "production cells require four rounds and a 24 or 48-operation budget"
            )
        task_binding = self.config["tasks"].get(cell.get("task"), {})
        prompt_contract = cell.get("prompt_contract", {})
        if (cell.get("task_sha256") != task_binding.get("sha256")
                or prompt_contract.get("task_sha256") != task_binding.get("sha256")
                or prompt_contract.get("invariant_sha256") !=
                self.config["prompt"].get("sha256")):
            raise ProductionError("cell prompt/task hashes do not match pinned runtime inputs")
        repo, existing = self._prepare_repo(cell)
        root = repo.parent
        checkpoint = self._resume_checkpoint(repo, cell) if existing else None
        seed_only = existing and checkpoint is None and self._is_seed_only_resume(repo, cell)
        if existing and checkpoint is None and not seed_only:
            try:
                return self._receipt(repo, self._verify(repo, cell))
            except ProductionError as error:
                raise self.infrastructure_failure_type(
                    f"existing branch has no valid blocked checkpoint and is not complete: {error}"
                ) from error
        resume = checkpoint is not None or seed_only
        invoker = self.invoker_factory(
            repo, timeout=self.config["agent_turn_timeout"],
            auth_home=Path(self.config["auth_home"]),
            state_dir=root / "state" / "codex", runtime_mode=self.config["runtime_mode"],
            model=self.config["model"], reasoning_effort=self.config["reasoning_effort"],
            agent_id=cell["cell_id"],
            codex=self.config.get("codex", "codex"),
            docker=self.config.get("docker", "docker"),
            image=self.config.get("image", "python:3.10"),
            node_runtime=(Path(self.config["node_runtime"])
                          if self.config.get("node_runtime") else None),
        )
        expected_image = self.config.get("runtime_image_digest")
        actual_image = getattr(invoker, "docker_image_id", None)
        if (self.config["runtime_mode"] == "docker"
                and actual_image != expected_image):
            invoker.scrub_auth()
            raise ProductionError("Codex runtime image does not match pinned digest")
        try:
            controller = self._controller(cell, slot, root, repo)
            runner_options = {
                "round_count": 4,
                "max_candidate_repairs": self.max_candidate_repairs,
            }
            if self.trusted_runtime_migration is not None:
                runner_options["trusted_runtime_migration"] = self.trusted_runtime_migration
            runner = self.runner_factory(
                repo, Path(self.config["prompt"]["path"]),
                Path(self.config["tasks"][cell["task"]]["path"]),
                invoker, controller, **runner_options,
            )
            try:
                runner.run(
                    self.config["run_id"], cell["cell_id"], resume=resume
                )
            except self.audit_error_types as error:
                raise ProductionError(str(error)) from error
            except Exception as error:
                checkpoint = repo / ".experiment" / "blocked.json"
                try:
                    blocked = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
                except (OSError, json.JSONDecodeError):
                    blocked = {}
                retained = blocked.get("receipt") if isinstance(blocked, dict) else None
                handle = retained.get("handle") if isinstance(retained, dict) else None
                raise self.infrastructure_failure_type(str(error), handle) from error
        finally:
            invoker.scrub_auth()
        try:
            return self._receipt(repo, self._verify(repo, cell))
        except ProductionError as error:
            raise self.infrastructure_failure_type(str(error)) from error

    def observe(self, cell: dict, placement: dict, durable_handle: str) -> dict:
        # AuditedExperimentRunner reads its checkpoint and CommandController
        # reobserves the exact retained handle recorded there.
        blocked_path = self.run_root / cell["cell_id"] / "repo" / ".experiment" / "blocked.json"
        blocked = json.loads(blocked_path.read_text()) if blocked_path.is_file() else {}
        receipt = blocked.get("receipt") if isinstance(blocked, dict) else None
        if not isinstance(receipt, dict) or receipt.get("handle") != durable_handle:
            raise ProductionError("retained handle does not match the cell checkpoint")
        return self.launch(cell, placement)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--runtime-config-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--canary-results", type=Path)
    parser.add_argument("--canary-results-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migration-attestation", type=Path)
    parser.add_argument("--migration-attestation-file-sha256")
    parser.add_argument("--migration-attestation-sha256")
    parser.add_argument(
        "--migration-proof", nargs=3, action="append", default=[],
        metavar=("ATTESTATION", "FILE_SHA256", "ATTESTATION_SHA256"),
        help="ordered, repeatable migration proof (preferred for sequential migrations)",
    )
    args = parser.parse_args(argv)
    migration_arguments = (
        args.migration_attestation, args.migration_attestation_file_sha256,
        args.migration_attestation_sha256,
    )
    if any(value is not None for value in migration_arguments) \
            and not all(value is not None for value in migration_arguments):
        parser.error("runtime migration trust arguments must be supplied together")
    if args.migration_attestation is not None and args.migration_proof:
        parser.error("legacy migration trust arguments cannot be mixed with migration proofs")
    if (args.migration_attestation is not None or args.migration_proof) and not args.resume:
        parser.error("runtime migration trust requires --resume")
    migration_trusts = [
        {"schema": MIGRATION_TRUST_SCHEMA,
         "attestation_path": str(Path(path).resolve()),
         "attestation_file_sha256": file_hash,
         "attestation_sha256": seal}
        for path, file_hash, seal in args.migration_proof
    ]
    if args.migration_attestation is not None:
        migration_trusts = [{
            "schema": MIGRATION_TRUST_SCHEMA,
            "attestation_path": str(args.migration_attestation.resolve()),
            "attestation_file_sha256": args.migration_attestation_file_sha256,
            "attestation_sha256": args.migration_attestation_sha256,
        }]
    config = _read_pinned(args.runtime_config, args.runtime_config_sha256, "runtime config")
    if config.get("schema") != RUNTIME_SCHEMA:
        parser.error(f"runtime config requires schema {RUNTIME_SCHEMA}")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2:
        parser.error("production campaign requires starter-bound manifest schema_version 2")
    config["run_id"] = manifest["run_id"]
    config["manifest_sha256"] = manifest["manifest_sha256"]
    if config.get("provenance") != manifest.get("provenance"):
        parser.error("runtime provenance does not exactly match manifest provenance")
    model = manifest["provenance"]["model"]
    if (config.get("model"), config.get("reasoning_effort")) != (
            model["name"], model["reasoning_effort"]):
        parser.error("runtime model does not match manifest provenance")
    if config.get("runtime_image_digest") != manifest["provenance"]["runtime_image_digest"]:
        parser.error("runtime image does not match manifest provenance")
    if manifest.get("request_budget") == 48:
        gate = validate_canary_gate(
            config, args.canary_results, args.canary_results_sha256,
        )
        _retain_canary_gate(config, gate)
    from audited_campaign import InfrastructureFailure, run_campaign
    ledger = run_campaign(
        manifest, args.ledger,
        DurableAdmissionPool(
            CplRemoteResourcePool(
                args.admission, args.admission_sha256,
                provider_id=config["admission_provider_id"],
                allowlist_sha256=config["admission_allowlist_sha256"],
                cpl_remote_closure_sha256=config["cpl_remote_closure_sha256"],
            ),
            Path(config["run_root"]) / "state" / "admission-evidence.json",
            config["run_id"],
        ),
        ProductionCellLauncher(
            config, infrastructure_failure_type=InfrastructureFailure,
            **({
                "trusted_runtime_migration": (migration_trusts[0]
                                              if len(migration_trusts) == 1
                                              else migration_trusts),
                "runtime_config_path": args.runtime_config,
                "runtime_config_sha256": args.runtime_config_sha256,
            } if migration_trusts else {}),
        ), resume=args.resume,
    )
    print(json.dumps({"status": ledger["status"], "ledger": str(args.ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
