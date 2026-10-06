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


TARGETS = ("bz-a3-1", "bz-a3-2")
ADMISSION_SCHEMA = "profiling-skill/bz-a3-admission/v1"
RUNTIME_SCHEMA = "profiling-skill/audited-campaign-runtime/v1"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class CplRemoteResourcePool:
    """Health-check approved targets and consume a pinned occupancy snapshot."""

    def __init__(self, admission: Path, admission_sha256: str, *,
                 cpl_remote: str | None = None, timeout: int = 120,
                 cpl_remote_sha256: str | None = None,
                 invoke: Callable = subprocess.run):
        self.admission = admission.resolve()
        self.admission_sha256 = admission_sha256
        self.cpl_remote = cpl_remote or str(
            Path.home() / ".agents" / "skills" / "remote-access" / "scripts" / "cpl-remote"
        )
        if (cpl_remote_sha256 is not None
                and (not Path(self.cpl_remote).is_file()
                     or file_sha256(Path(self.cpl_remote)) != cpl_remote_sha256)):
            raise ProductionError("global cpl-remote hash does not match pinned input")
        self.timeout = timeout
        self.invoke = invoke

    def admit(self) -> list[dict]:
        document = _read_pinned(
            self.admission, self.admission_sha256, "required placement-provider"
        )
        if document.get("schema") != ADMISSION_SCHEMA:
            raise ProductionError(f"admission input requires schema {ADMISSION_SCHEMA}")
        slots = document.get("slots")
        if not isinstance(slots, list):
            raise ProductionError("placement-provider slots must be an array")
        available_targets = set()
        for target in TARGETS:
            for operation in ("capabilities", "preflight"):
                result = self.invoke(
                    [self.cpl_remote, operation, target], text=True,
                    capture_output=True, timeout=self.timeout, check=False,
                )
                if result.returncode:
                    break
            else:
                available_targets.add(target)
        admitted, seen = [], set()
        for slot in slots:
            if (not isinstance(slot, dict) or set(slot) !=
                    {"target", "device", "healthy", "idle"}):
                raise ProductionError("placement-provider slot has an invalid shape")
            identity = (slot["target"], slot["device"])
            if (slot["target"] not in TARGETS or type(slot["device"]) is not int
                    or slot["device"] < 0 or type(slot["healthy"]) is not bool
                    or type(slot["idle"]) is not bool or identity in seen):
                raise ProductionError("placement-provider slot identity is invalid")
            seen.add(identity)
            if slot["target"] in available_targets:
                admitted.append(dict(slot))
        return admitted


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ProductionError(f"isolated skill destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise ProductionError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


class ProductionCellLauncher:
    """Create or resume one isolated four-round branch and its controller."""

    def __init__(self, config: dict, *, infrastructure_failure_type: type[Exception],
                 invoker_factory=None,
                 controller_factory=None, runner_factory=None,
                 verifier_invoke: Callable = subprocess.run):
        if config.get("schema", RUNTIME_SCHEMA) != RUNTIME_SCHEMA:
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
        self.verifier_invoke = verifier_invoke
        self.run_root = Path(config["run_root"]).resolve()
        if "cpl_remote" in config:
            raise ProductionError("runtime config cannot override global cpl-remote")
        self.cpl_remote = str(
            Path.home() / ".agents" / "skills" / "remote-access" / "scripts" / "cpl-remote"
        )
        if (not Path(self.cpl_remote).is_file()
                or file_sha256(Path(self.cpl_remote)) != config.get("cpl_remote_sha256")):
            raise ProductionError("global cpl-remote hash does not match pinned input")
        if config.get("runtime_mode") != "docker":
            raise ProductionError("production agents require an isolated Docker runtime")
        if "adapter_command" in config or "remote_command" in config:
            raise ProductionError("runtime config cannot supply an adapter or remote command")
        self._validate_files()
        if invoker_factory is None or controller_factory is None or runner_factory is None:
            sys.path.insert(0, str(self.scripts))
            try:
                from audited_lifecycle import AuditedExperimentRunner
                from audited_runtime import CodexInvoker, CommandController
            finally:
                sys.path.pop(0)
            invoker_factory = invoker_factory or CodexInvoker
            controller_factory = controller_factory or CommandController
            runner_factory = runner_factory or AuditedExperimentRunner
        self.invoker_factory = invoker_factory
        self.controller_factory = controller_factory
        self.runner_factory = runner_factory

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
        if runtime.get("sha256") != provenance.get("controller_sha256"):
            raise ProductionError("controller provenance does not match runtime closure")
        if self.config.get("runtime_image_digest") != provenance.get("runtime_image_digest"):
            raise ProductionError("runtime image provenance does not match Docker input")
        if provenance.get("model") != {
            "name": self.config.get("model"),
            "reasoning_effort": self.config.get("reasoning_effort"),
        }:
            raise ProductionError("model provenance does not match runtime inputs")
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

    def _prepare_repo(self, cell: dict) -> tuple[Path, bool]:
        root = self.run_root / cell["cell_id"]
        repo = root / "repo"
        source = self.config["source_repositories"].get(cell["task"])
        if not isinstance(source, dict):
            raise ProductionError(f"source repository is missing for {cell['task']}")
        source_path, revision = Path(source["path"]).resolve(), source.get("revision")
        if (not isinstance(revision, str) or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)):
            raise ProductionError("source repository revision must be a pinned commit")
        identity = {"cell_id": cell["cell_id"], "task": cell["task"],
                    "treatment": cell["treatment"], "source_revision": revision}
        identity_path = root / "state" / "cell.json"
        if repo.exists():
            if (not identity_path.is_file()
                    or json.loads(identity_path.read_text()) != identity):
                raise ProductionError("existing isolated repository has a different identity")
            return repo, (repo / ".experiment" / "seed.json").is_file()
        root.mkdir(parents=True, exist_ok=False)
        result = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source_path), str(repo)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise ProductionError(f"isolated clone failed: {result.stderr.strip()}")
        _git(repo, "checkout", "--quiet", "--detach", revision)
        if _git(repo, "rev-parse", "HEAD") != revision:
            raise ProductionError("isolated repository revision does not match pin")
        exclusions = repo / ".git" / "info" / "exclude"
        with exclusions.open("a") as stream:
            stream.write("\n.agents/\nAGENTS.md\n")
        skills = repo / ".agents" / "skills"
        skills.mkdir(parents=True)
        expected = tuple(TREATMENT_SKILLS[cell["treatment"]])
        if tuple(cell["skills"]) != expected:
            raise ProductionError("cell treatment allowlist does not match campaign policy")
        for name in expected:
            binding = self.config["skill_sources"].get(name)
            if not isinstance(binding, dict):
                raise ProductionError(f"pinned skill source is missing: {name}")
            _copy_tree(Path(binding["path"]), skills / name)
        (repo / "AGENTS.md").write_text(
            "Use only the repository-local skills under .agents/skills. "
            "Do not inspect host or global skills.\n"
        )
        if _git(repo, "status", "--porcelain"):
            raise ProductionError("isolated experiment repository is not clean")
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
        return repo, False

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
            "--cpl-remote-sha256", self.config["cpl_remote_sha256"],
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
        return self.controller_factory(command, repo, timeout=self.config.get("timeout", 900))

    def _verify(self, repo: Path, cell: dict) -> dict:
        command = [
            sys.executable, str(self.scripts / "validate_audited_experiment.py"),
            str(repo), "--base", self.config["provenance"]["source_revision"],
        ]
        try:
            result = self.verifier_invoke(
                command, text=True, capture_output=True,
                timeout=self.config.get("timeout", 900), check=False,
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
        if existing and checkpoint is None:
            try:
                return self._receipt(repo, self._verify(repo, cell))
            except ProductionError as error:
                raise self.infrastructure_failure_type(
                    f"existing branch has no valid blocked checkpoint and is not complete: {error}"
                ) from error
        resume = checkpoint is not None
        invoker = self.invoker_factory(
            repo, timeout=self.config.get("timeout", 900),
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
            runner = self.runner_factory(
                repo, Path(self.config["prompt"]["path"]),
                Path(self.config["tasks"][cell["task"]]["path"]),
                invoker, controller, round_count=4,
            )
            try:
                runner.run(
                    self.config["run_id"], cell["cell_id"], resume=resume
                )
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
    args = parser.parse_args(argv)
    config = _read_pinned(args.runtime_config, args.runtime_config_sha256, "runtime config")
    if config.get("schema") != RUNTIME_SCHEMA:
        parser.error(f"runtime config requires schema {RUNTIME_SCHEMA}")
    manifest = json.loads(args.manifest.read_text())
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
        CplRemoteResourcePool(
            args.admission, args.admission_sha256,
            cpl_remote_sha256=config["cpl_remote_sha256"],
        ),
        ProductionCellLauncher(
            config, infrastructure_failure_type=InfrastructureFailure,
        ), resume=args.resume,
    )
    print(json.dumps({"status": ledger["status"], "ledger": str(args.ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
