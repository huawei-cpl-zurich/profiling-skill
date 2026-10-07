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
                 trusted_runtime_migration: dict | None = None,
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
        self._migration_attestation = self._validate_migration_trust(
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
                                  config_sha256: str | None) -> dict | None:
        trusted = self.trusted_runtime_migration
        supplied = (trusted is not None, config_path is not None, config_sha256 is not None)
        if not any(supplied):
            return None
        if not all(supplied) or not isinstance(trusted, dict):
            raise ProductionError("runtime migration trust requires its pinned runtime config")
        expected_keys = {
            "schema", "attestation_path", "attestation_file_sha256",
            "attestation_sha256",
        }
        if (set(trusted) != expected_keys
                or trusted.get("schema") != MIGRATION_TRUST_SCHEMA):
            raise ProductionError("runtime migration trust binding is invalid")
        pinned_config = Path(config_path).resolve()
        if (not pinned_config.is_file()
                or file_sha256(pinned_config) != config_sha256):
            raise ProductionError("runtime migration config binding is invalid")
        try:
            attestation_path = Path(trusted["attestation_path"])
            raw = attestation_path.read_bytes()
            attestation = json.loads(raw)
            seal = attestation["attestation_sha256"]
            unsealed = {key: value for key, value in attestation.items()
                        if key != "attestation_sha256"}
            runtimes = attestation["runtimes"]
            old_runtime, new_runtime = runtimes["old"], runtimes["new"]
            provenance = self.config["provenance"]
            runtime = self.config["runtime_scripts"]
            if (not attestation_path.is_absolute()
                    or attestation.get("schema") != MIGRATION_ATTESTATION_SCHEMA
                    or trusted["attestation_file_sha256"]
                    != hashlib.sha256(raw).hexdigest()
                    or trusted["attestation_sha256"] != seal
                    or document_sha256(unsealed) != seal
                    or old_runtime.get("closure_sha256")
                    != provenance.get("controller_sha256")
                    or new_runtime.get("closure_sha256") != runtime.get("sha256")
                    or Path(new_runtime.get("config_path", "")).resolve() != pinned_config
                    or new_runtime.get("config_sha256") != config_sha256):
                raise ProductionError("runtime migration attestation is invalid")
        except ProductionError:
            raise
        except (KeyError, TypeError, ValueError, OSError,
                json.JSONDecodeError) as error:
            raise ProductionError("runtime migration attestation is malformed") from error
        return attestation

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
                and self._migration_attestation is None):
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
        identity = {"cell_id": cell["cell_id"], "task": cell["task"],
                    "treatment": cell["treatment"], "source_revision": revision,
                    "starter": starter_binding}
        identity_path = root / "state" / "cell.json"
        expected = tuple(TREATMENT_SKILLS[cell["treatment"]])
        if tuple(cell["skills"]) != expected:
            raise ProductionError("cell treatment allowlist does not match campaign policy")
        if repo.exists():
            if (not identity_path.is_file()
                    or json.loads(identity_path.read_text()) != identity):
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
            "benchmark": cell["task"], "round_count": 4, "request_budget": 24,
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
        for number in range(1, 5):
            path = repo / "experiments" / f"{number:02d}" / "results.json"
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ProductionError(f"round {number} compact receipt is unavailable") from error
            if not isinstance(receipt, dict):
                raise ProductionError(f"round {number} compact receipt is invalid")
            rounds.append({"round": number, **receipt})
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
        if (cell.get("round_count"), cell.get("request_budget")) != (4, 24):
            raise ProductionError("production cells require four rounds and 24 requests")
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
            runner_options = {"round_count": 4}
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migration-attestation", type=Path)
    parser.add_argument("--migration-attestation-file-sha256")
    parser.add_argument("--migration-attestation-sha256")
    args = parser.parse_args(argv)
    migration_arguments = (
        args.migration_attestation, args.migration_attestation_file_sha256,
        args.migration_attestation_sha256,
    )
    if any(value is not None for value in migration_arguments) \
            and not all(value is not None for value in migration_arguments):
        parser.error("runtime migration trust arguments must be supplied together")
    if args.migration_attestation is not None and not args.resume:
        parser.error("runtime migration trust requires --resume")
    config = _read_pinned(args.runtime_config, args.runtime_config_sha256, "runtime config")
    if config.get("schema") != RUNTIME_SCHEMA:
        parser.error(f"runtime config requires schema {RUNTIME_SCHEMA}")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 2:
        parser.error("production campaign requires starter-bound manifest schema_version 2")
    config["run_id"] = manifest["run_id"]
    if config.get("provenance") != manifest.get("provenance"):
        parser.error("runtime provenance does not exactly match manifest provenance")
    model = manifest["provenance"]["model"]
    if (config.get("model"), config.get("reasoning_effort")) != (
            model["name"], model["reasoning_effort"]):
        parser.error("runtime model does not match manifest provenance")
    if config.get("runtime_image_digest") != manifest["provenance"]["runtime_image_digest"]:
        parser.error("runtime image does not match manifest provenance")
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
                "trusted_runtime_migration": {
                    "schema": MIGRATION_TRUST_SCHEMA,
                    "attestation_path": str(args.migration_attestation.resolve()),
                    "attestation_file_sha256": args.migration_attestation_file_sha256,
                    "attestation_sha256": args.migration_attestation_sha256,
                },
                "runtime_config_path": args.runtime_config,
                "runtime_config_sha256": args.runtime_config_sha256,
            } if args.migration_attestation is not None else {}),
        ), resume=args.resume,
    )
    print(json.dumps({"status": ledger["status"], "ledger": str(args.ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
