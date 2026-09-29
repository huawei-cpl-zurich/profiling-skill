#!/usr/bin/env python3
"""Reproducible, isolated campaign preparation and session scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol


BENCHMARK_DEVICE = {"gdn": 0, "bsa": 1, "matmul": 2}
DEVELOPMENT_CASES = {
    "gdn": [40, 49, 47, 46, 45],
    "bsa": [47, 46, 49, 44, 43],
    "matmul": [7, 8, 9],
}
ALL_CASES = {"gdn": list(range(50)), "bsa": list(range(50)), "matmul": list(range(10))}
WAVES = (
    (("gdn", "cannbot"), ("bsa", "cannbot"),
     ("matmul", "cannbot"), ("bsa", "project-guarded")),
    (("gdn", "project-cannbot"), ("bsa", "project-cannbot"),
     ("matmul", "project-cannbot"), ("matmul", "project-guarded")),
    (("gdn", "project-guarded"),),
)
CELL_DEVICE = {
    ("gdn", "cannbot"): 0,
    ("gdn", "project-cannbot"): 1,
    ("gdn", "project-guarded"): 2,
    ("bsa", "cannbot"): 1,
    ("bsa", "project-cannbot"): 2,
    ("bsa", "project-guarded"): 3,
    ("matmul", "cannbot"): 2,
    ("matmul", "project-cannbot"): 3,
    ("matmul", "project-guarded"): 0,
}
CANNBOT_TRITON_SKILLS = (
    "triton-task-extractor",
    "triton-op-designer",
    "triton-op-coding",
    "triton-op-verifier",
    "triton-latency-optimizer",
    "triton-simulator-optimizer",
)
CANNBOT_DEPENDENCIES = ("npu-arch",)
CANNBOT_SKILL_SOURCES = {
    name: f"ops/{name}"
    for name in (*CANNBOT_TRITON_SKILLS, *CANNBOT_DEPENDENCIES, "ops-profiling")
}
CANNBOT_SUPPORT_SOURCE = "plugins-official/triton-op-generator"
TREATMENT_SKILLS = {
    "cannbot": (*CANNBOT_TRITON_SKILLS, *CANNBOT_DEPENDENCIES, "ops-profiling"),
    "project-cannbot": (
        *CANNBOT_TRITON_SKILLS, *CANNBOT_DEPENDENCIES, "ascend-profiling"
    ),
    "project-guarded": ("ascend-profiling", "triton-guarded-kernel"),
}
CONTROLLER_SCRIPTS = (
    "experimentctl.py",
    "benchmark_backend.py",
    "gz_a3_job_client.py",
    "a3_benchmark_runner.py",
    "profile_a3.py",
)
CONTROLLER_BENCHMARK_ASSETS = tuple(
    f"{benchmark}/{name}"
    for benchmark in ("gdn", "bsa", "matmul")
    for name in ("baseline.py", "baseline.json", "cases.jsonl")
)
CONTROLLER_EXTRA_ASSETS = ("streaming_matmul_add.py", "matmul/calibration.py")


class CampaignError(RuntimeError):
    pass


CANDIDATE_OUTCOMES = {"compile_error", "runtime_error", "correctness_error"}


def classify_outcome(result: dict) -> str:
    """Map launcher details onto the experiment's disjoint outcome taxonomy."""
    status = result.get("status")
    if status == "complete":
        return "success"
    if status == "infrastructure_error":
        return "infra_discarded"
    if status == "candidate_error":
        failure = result.get("failure_type")
        if failure in CANDIDATE_OUTCOMES:
            return failure
        terminal = result.get("terminal_evidence", {})
        for gate in ("check", "profile"):
            document = terminal.get(gate, {}).get("result", {})
            if document.get("failure_type") in CANDIDATE_OUTCOMES:
                return document["failure_type"]
        return "runtime_error"
    if status == "dry_run":
        return "dry_run"
    return "infra_discarded"


def normalize_performance(result: dict) -> dict | None:
    """Normalize a profile geomean by bracketing calibration measurements."""
    try:
        calibration = result["calibration"]
        device = str(result["device"])
        local = calibration["devices"][device]
        canonical = calibration["devices"]["0"]
        before = float(local["before"]["latency_us"])
        after = float(local["after"]["latency_us"])
        canonical_before = float(canonical["before"]["latency_us"])
        canonical_after = float(canonical["after"]["latency_us"])
        profile = float(result["terminal_evidence"]["profile"]["result"]["geomean_us"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in
               (before, after, canonical_before, canonical_after, profile)):
        return None
    reference = math.sqrt(before * after)
    canonical_reference = math.sqrt(canonical_before * canonical_after)
    return {
        "method": "raw_latency_times_device0_over_local_bracketing_geomean",
        "before_us": before,
        "after_us": after,
        "reference_us": reference,
        "profile_geomean_us": profile,
        "canonical_reference_us": canonical_reference,
        "normalized_latency_us": profile * canonical_reference / reference,
    }


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_identity(executable: str) -> dict:
    resolved = shutil.which(executable)
    if not resolved:
        raise CampaignError(f"controller runtime is not executable: {executable}")
    result = subprocess.run(
        [resolved, "-c", "import platform; print(platform.python_implementation()); print(platform.python_version())"],
        check=True, text=True, capture_output=True,
    )
    implementation, version = result.stdout.splitlines()
    return {"implementation": implementation, "version": version,
            "executable_sha256": digest_file(Path(resolved))}


def _parse_controller_command(command: list[str], controller_config: Path) -> tuple[str, Path]:
    if not command or not all(isinstance(argument, str) and argument for argument in command):
        raise CampaignError("controller command must be a nonempty JSON string array")
    expected_tail = ["--config", str(controller_config.resolve()), "--cell", "{cell_id}"]
    if len(command) != 6 or command[2:] != expected_tail:
        raise CampaignError("controller command must be PYTHON experimentctl.py --config CONFIG --cell {cell_id}")
    script = Path(command[1]).resolve()
    if script.name != "experimentctl.py" or not script.is_file():
        raise CampaignError("controller command must reference experimentctl.py")
    return command[0], script


def freeze_controller_bundle(manifest_path: Path, controller_config: Path,
                             command: list[str]) -> dict:
    executable, script = _parse_controller_command(command, controller_config)
    scripts = script.parent
    repository = scripts.parent
    required_scripts = {name: scripts / name for name in CONTROLLER_SCRIPTS}
    benchmark_root = repository / "benchmarks"
    required_assets = {name: benchmark_root / name for name in
                       (*CONTROLLER_BENCHMARK_ASSETS, *CONTROLLER_EXTRA_ASSETS)}
    missing = [name for name, path in {**required_scripts, **required_assets}.items()
               if not path.is_file() or path.is_symlink()]
    if missing:
        detail = ", ".join(missing)
        raise CampaignError(f"controller runtime closure is incomplete: {detail}")
    bundle_name = f"{manifest_path.stem}.controller"
    destination = manifest_path.parent / bundle_name
    if destination.exists():
        raise CampaignError(f"controller bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_name}.", dir=manifest_path.parent))
    try:
        bundled_scripts = temporary / "scripts"
        bundled_scripts.mkdir()
        for name, source in required_scripts.items():
            shutil.copy2(source, bundled_scripts / name)
        for name, source in required_assets.items():
            target = temporary / "benchmarks" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        try:
            config = json.loads(controller_config.read_text())
            cells_document = config["cells"]
            for cell in cells_document.values():
                backend_command = cell["backend"]["command"]
                if (len(backend_command) < 2
                        or Path(backend_command[1]).resolve() != required_scripts["benchmark_backend.py"].resolve()):
                    raise CampaignError("controller config must invoke the bundled benchmark_backend.py")
                try:
                    client_index = backend_command.index("--job-client-json") + 1
                    client_command = json.loads(backend_command[client_index])
                except (ValueError, IndexError, json.JSONDecodeError) as error:
                    raise CampaignError("controller backend requires a JSON job-client command") from error
                expected_client = required_scripts["gz_a3_job_client.py"].resolve()
                if (not isinstance(client_command, list) or len(client_command) < 2
                        or not all(isinstance(value, str) and value for value in client_command)
                        or Path(client_command[1]).resolve() != expected_client):
                    raise CampaignError("controller config must invoke gz_a3_job_client.py")
                client_command[0:2] = ["{python}", "{bundle}/scripts/gz_a3_job_client.py"]
                backend_command[client_index] = json.dumps(client_command, separators=(",", ":"))
                backend_command[0:2] = ["{python}", "{bundle}/scripts/benchmark_backend.py"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
            raise CampaignError(f"invalid controller config: {error}") from error
        (temporary / "controller.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        files = _regular_file_hashes(temporary)
        template = ["{python}", "{bundle}/scripts/experimentctl.py", "--config",
                    "{bundle}/controller.json", "--cell", "{cell_id}"]
        encoded = json.dumps(template, separators=(",", ":")).encode()
        binding = {"bundle": bundle_name, "files": files,
                   "command_argv": template,
                   "command_sha256": hashlib.sha256(encoded).hexdigest(),
                   "runtime": _runtime_identity(executable)}
        os.replace(temporary, destination)
        return binding
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def digest_tree(root: Path, exclude: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    paths = (
        path for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] not in exclude
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(f"{stat.S_IMODE(path.stat().st_mode) & 0o111:o}".encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_regular_tree(source: Path, destination: Path) -> None:
    """Copy a tree while rejecting symlinks and special files."""
    if not source.is_dir() or source.is_symlink():
        raise CampaignError(f"skill source is not a regular directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if item.is_symlink():
            raise CampaignError(f"symlink forbidden in skill bundle: {relative}")
        if item.is_dir():
            (destination / relative).mkdir(exist_ok=True)
        elif item.is_file():
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination / relative)
        else:
            raise CampaignError(f"special file forbidden in skill bundle: {relative}")


def _regular_file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise CampaignError(f"controller bundle is not a regular directory: {root}")
    files: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise CampaignError(f"symlink forbidden in controller bundle: {relative}")
        if item.is_file():
            files[relative] = digest_file(item)
        elif not item.is_dir():
            raise CampaignError(f"special file forbidden in controller bundle: {relative}")
    return files


def resolve_master(repo_url: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repo_url, "refs/heads/master"],
        check=True,
        text=True,
        capture_output=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) != 2 or len(fields[0]) != 40:
        raise CampaignError("could not resolve CANNBot master")
    return fields[0]


def freeze_cannbot(repo_url: str, destination: Path) -> dict:
    """Freeze the exact Triton plugin skill and support layout from master."""
    commit = resolve_master(repo_url)
    with tempfile.TemporaryDirectory(prefix="cannbot-freeze-") as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(["git", "clone", "--no-checkout", repo_url, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
        destination.mkdir(parents=True, exist_ok=False)
        skills = destination / "skills"
        skills.mkdir()
        hashes = {}
        for name, relative in CANNBOT_SKILL_SOURCES.items():
            copy_regular_tree(checkout / relative, skills / name)
            hashes[name] = digest_tree(skills / name)
        support = destination / "support" / "triton-op-generator"
        support.parent.mkdir()
        copy_regular_tree(checkout / CANNBOT_SUPPORT_SOURCE, support)
    record = {
        "repository": repo_url,
        "commit": commit,
        "skills": hashes,
        "support": {"triton-op-generator": digest_tree(support)},
    }
    (destination / "freeze.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def validate_freeze_record(record: dict) -> None:
    if set(record.get("skills", {})) != set(CANNBOT_SKILL_SOURCES):
        raise CampaignError("CANNBot freeze does not contain the complete skill bundle")
    if set(record.get("support", {})) != {"triton-op-generator"}:
        raise CampaignError("CANNBot freeze does not contain plugin support")
    commit = record.get("commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise CampaignError("CANNBot freeze has an invalid commit")


@dataclass(frozen=True)
class Cell:
    cell_id: str
    benchmark: str
    treatment: str
    device: int
    wave: int
    rounds: int
    request_budget: int
    development_cases: list[int]
    all_cases: list[int]


def cells(rounds: int = 3, request_budget: int = 12) -> list[Cell]:
    result = []
    for wave, pairs in enumerate(WAVES, 1):
        for benchmark, treatment in pairs:
            result.append(
                Cell(
                    f"{benchmark}-{treatment}", benchmark, treatment,
                    CELL_DEVICE[(benchmark, treatment)], wave, rounds, request_budget,
                    DEVELOPMENT_CASES[benchmark], ALL_CASES[benchmark],
                )
            )
    return result


def write_manifest(path: Path, prompt: Path, baselines: dict[str, Path],
                   project_skill: Path, guarded_skill: Path, cannbot_freeze: Path,
                   controller_config: Path, controller_command: list[str],
                   rounds: int = 3, request_budget: int = 12,
                   guarded_revision: str = "", calibration_max_drift: float = 0.10) -> dict:
    if rounds < 1 or request_budget < 1:
        raise CampaignError("rounds and request budget must be positive")
    if (len(guarded_revision) != 40
            or any(character not in "0123456789abcdef" for character in guarded_revision)):
        raise CampaignError("guarded skill revision must be a full lowercase Git commit")
    if (not isinstance(calibration_max_drift, (int, float))
            or isinstance(calibration_max_drift, bool)
            or not math.isfinite(calibration_max_drift)
            or calibration_max_drift <= 0):
        raise CampaignError("calibration max drift must be positive and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = {name for name in BENCHMARK_DEVICE if name not in baselines}
    if missing:
        raise CampaignError(f"missing baselines: {', '.join(sorted(missing))}")
    freeze_file = cannbot_freeze / "freeze.json"
    freeze = json.loads(freeze_file.read_text())
    validate_freeze_record(freeze)
    document = {
        "version": 2,
        "prompt": {"path": str(prompt.resolve()), "sha256": digest_file(prompt)},
        "baselines": {
            name: {"path": str(value.resolve()), "sha256": digest_tree(value)}
            for name, value in sorted(baselines.items())
        },
        "skill_sources": {
            "ascend-profiling": {
                "path": str(project_skill.resolve()),
                "sha256": digest_tree(project_skill),
            },
            "triton-guarded-kernel": {
                "path": str(guarded_skill.resolve()),
                "sha256": digest_tree(guarded_skill),
                "repository": "https://github.com/huawei-cpl-zurich/cpl-skills.git",
                "revision": guarded_revision,
            },
            "cannbot": {
                "path": str(cannbot_freeze.resolve()),
                "freeze": freeze,
                "freeze_sha256": digest_file(freeze_file),
            },
        },
        "controller": freeze_controller_bundle(path, controller_config, controller_command),
        "cells": [asdict(cell) for cell in cells(rounds, request_budget)],
        "max_parallel": 4,
        "calibration": {"devices": [0, 1, 2, 3], "case": 7,
                        "selector": "streaming_matmul_add_kernel_mix_aic",
                        "max_drift_fraction": calibration_max_drift},
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def _skill_source(manifest: dict, skill: str) -> Path:
    if skill in {"ascend-profiling", "triton-guarded-kernel"}:
        return Path(manifest["skill_sources"][skill]["path"])
    return Path(manifest["skill_sources"]["cannbot"]["path"]) / "skills" / skill


def verify_controller_schema(manifest: dict, allow_unbound: bool = False) -> dict | None:
    if manifest.get("version") != 2 or not isinstance(manifest.get("controller"), dict):
        if allow_unbound:
            return None
        raise CampaignError("production requires a version 2 controller-bound manifest")
    controller = manifest["controller"]
    required = {"bundle", "files", "command_argv", "command_sha256", "runtime"}
    if set(controller) != required:
        raise CampaignError("controller binding schema is incomplete")
    if Path(controller["bundle"]).name != controller["bundle"]:
        raise CampaignError("controller bundle must be a relative directory name")
    encoded = json.dumps(controller["command_argv"], separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != controller["command_sha256"]:
        raise CampaignError("controller command hash does not match its template")
    return controller


def _verify_bundle(root: Path, binding: dict) -> None:
    try:
        actual = _regular_file_hashes(root)
    except (OSError, CampaignError) as error:
        raise CampaignError(f"frozen controller bundle is invalid: {error}") from error
    if set(actual) != set(binding["files"]):
        raise CampaignError("frozen controller bundle file set drifted")
    for name, expected in binding["files"].items():
        if actual[name] != expected:
            raise CampaignError(f"frozen controller bundle drifted: {name}")


def _make_tree_read_only(root: Path) -> None:
    for item in sorted(root.rglob("*"), reverse=True):
        item.chmod(0o555 if item.is_dir() else 0o444)
    root.chmod(0o555)


def stage_controller(manifest: dict, manifest_path: Path, campaign_root: Path,
                     executable: str, resume: bool = False) -> tuple[list[str], dict]:
    """Copy the frozen bundle into private run evidence and return its concrete argv."""
    binding = verify_controller_schema(manifest)
    assert binding is not None
    runtime = _runtime_identity(executable)
    if runtime != binding["runtime"]:
        raise CampaignError("controller runtime does not match frozen identity")
    source = manifest_path.resolve().parent / binding["bundle"]
    evidence = campaign_root / "controller"
    if resume:
        if not evidence.is_dir():
            raise CampaignError("resumed campaign has no private controller evidence")
    else:
        if evidence.exists():
            ledger_path = campaign_root / "ledger.json"
            try:
                previous_status = json.loads(ledger_path.read_text()).get("status")
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                raise CampaignError("private controller evidence already exists") from error
            if previous_status != "dry_run" or not evidence.is_dir():
                raise CampaignError("private controller evidence already exists")
        else:
            copy_regular_tree(source, evidence)
            _make_tree_read_only(evidence)
    _verify_bundle(evidence, binding)
    command = [argument.replace("{python}", shutil.which(executable) or executable)
               .replace("{bundle}", str(evidence.resolve()))
               for argument in binding["command_argv"]]
    ledger_binding = {"files": binding["files"], "command_argv": binding["command_argv"],
                      "command_sha256": binding["command_sha256"], "runtime": runtime}
    return command, ledger_binding


def verify_frozen_sources(manifest: dict) -> None:
    for name in ("ascend-profiling", "triton-guarded-kernel"):
        project = manifest["skill_sources"][name]
        if digest_tree(Path(project["path"])) != project["sha256"]:
            raise CampaignError(f"project skill drifted after manifest freeze: {name}")
    frozen = manifest["skill_sources"]["cannbot"]
    root = Path(frozen["path"])
    freeze_file = root / "freeze.json"
    if digest_file(freeze_file) != frozen["freeze_sha256"]:
        raise CampaignError("CANNBot freeze record drifted after manifest freeze")
    record = json.loads(freeze_file.read_text())
    if record != frozen["freeze"]:
        raise CampaignError("CANNBot freeze record does not match manifest")
    validate_freeze_record(record)
    for name, expected in record["skills"].items():
        if digest_tree(root / "skills" / name) != expected:
            raise CampaignError(f"frozen CANNBot skill drifted: {name}")
    support = root / "support" / "triton-op-generator"
    if digest_tree(support) != record["support"]["triton-op-generator"]:
        raise CampaignError("frozen CANNBot plugin support drifted")


def prepare_cell(manifest: dict, cell: dict, campaigns_root: Path) -> Path:
    verify_frozen_sources(manifest)
    sandbox = campaigns_root / cell["cell_id"]
    if sandbox.exists():
        raise CampaignError(f"fresh sandbox required: {sandbox}")
    sandbox.mkdir(parents=True)
    workspace = sandbox / "workspace"
    shutil.copyfile(manifest["prompt"]["path"], sandbox / "PROMPT.md")
    copy_regular_tree(
        Path(manifest["baselines"][cell["benchmark"]]["path"]),
        workspace,
    )
    skill_root = workspace / ".agents" / "skills"
    skill_root.mkdir(parents=True)
    skill_hashes = {}
    for skill in TREATMENT_SKILLS[cell["treatment"]]:
        target = skill_root / skill
        copy_regular_tree(_skill_source(manifest, skill), target)
        skill_hashes[skill] = digest_tree(target)
    if cell["treatment"] != "project-guarded":
        support_source = (Path(manifest["skill_sources"]["cannbot"]["path"])
                          / "support" / "triton-op-generator")
        copy_regular_tree(support_source,
                          workspace / ".agents" / "plugins-official" / "triton-op-generator")
        for name in ("AGENTS.md", "config.json"):
            if (workspace / name).exists():
                raise CampaignError(f"baseline conflicts with CANNBot workspace file: {name}")
            shutil.copy2(support_source / name, workspace / name)
    metadata = {
        "cell": cell,
        "prompt_sha256": digest_file(sandbox / "PROMPT.md"),
        "baseline_sha256": manifest["baselines"][cell["benchmark"]]["sha256"],
        "skills": skill_hashes,
    }
    (sandbox / "cell.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    preflight(manifest, sandbox)
    return sandbox


def preflight(manifest: dict, sandbox: Path) -> dict:
    metadata = json.loads((sandbox / "cell.json").read_text())
    cell = metadata["cell"]
    expected = set(TREATMENT_SKILLS[cell["treatment"]])
    workspace = sandbox / "workspace"
    skill_root = workspace / ".agents" / "skills"
    actual = {path.name for path in skill_root.iterdir()}
    if actual != expected:
        raise CampaignError(f"skill isolation mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    if any(path.is_symlink() for path in sandbox.rglob("*")):
        raise CampaignError("sandbox contains a symlink")
    if digest_file(sandbox / "PROMPT.md") != manifest["prompt"]["sha256"]:
        raise CampaignError("prompt hash mismatch")
    expected_baseline = manifest["baselines"][cell["benchmark"]]["sha256"]
    injected = ((".agents", "AGENTS.md", "config.json")
                if cell["treatment"] != "project-guarded" else (".agents",))
    if digest_tree(workspace, injected) != expected_baseline:
        raise CampaignError("baseline hash mismatch")
    support = workspace / ".agents" / "plugins-official" / "triton-op-generator"
    if (cell["treatment"] == "project-guarded") == support.exists():
        raise CampaignError("CANNBot support isolation mismatch")
    for forbidden in ("siblings", "orchestration", "global-skills"):
        if (sandbox / forbidden).exists():
            raise CampaignError(f"forbidden root visible: {forbidden}")
    for skill, expected_hash in metadata["skills"].items():
        if digest_tree(skill_root / skill) != expected_hash:
            raise CampaignError(f"skill hash mismatch: {skill}")
    return metadata


class Launcher(Protocol):
    def launch(self, sandbox: Path, cell: dict) -> dict: ...

    def calibrate(self, sandbox: Path, cell: dict, phase: str, wave: int,
                  attempt_id: str) -> dict: ...


class CommandLauncher:
    """Launcher boundary; command must provide the outer isolation mechanism."""

    def __init__(self, command: list[str]):
        if not command:
            raise CampaignError("launcher command is required")
        self.command = command

    def launch(self, sandbox: Path, cell: dict) -> dict:
        environment = os.environ.copy()
        environment.update({
            "CAMPAIGN_CELL": cell["cell_id"],
            "CAMPAIGN_DEVICE": str(cell["device"]),
            "CAMPAIGN_ROUNDS": str(cell["rounds"]),
            "CAMPAIGN_REQUEST_BUDGET": str(cell["request_budget"]),
        })
        result = subprocess.run(
            self.command, cwd=sandbox / "workspace", env=environment,
            input=(sandbox / "PROMPT.md").read_text(), text=True, capture_output=True,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_campaign(manifest: dict, root: Path, launcher: Launcher,
                 on_wave: Callable[[int], None] | None = None,
                 resume: bool = False, controller_evidence: dict | None = None,
                 controller_bundle: Path | None = None,
                 allow_unbound: bool = False) -> dict:
    """Run fixed waves; launcher implementations may execute each pair concurrently."""
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.json"
    dry_run = bool(getattr(launcher, "dry_run", False))
    binding = verify_controller_schema(manifest, allow_unbound)
    if binding is not None and (controller_evidence is None or controller_bundle is None):
        raise CampaignError("private controller evidence is required")
    if binding is not None:
        _verify_bundle(controller_bundle, binding)
    if ledger_path.exists() and resume:
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("controller") != controller_evidence:
            raise CampaignError("controller binding does not match existing ledger")
        target_ids = set(ledger.get("reschedule", []))
        if not target_ids:
            raise CampaignError("ledger has no infrastructure cells to reschedule")
        ledger.pop("failure", None)
        ledger["reschedule"] = []
        ledger["status"] = "running"
    elif ledger_path.exists() and json.loads(ledger_path.read_text()).get("status") != "dry_run":
        raise CampaignError("campaign ledger already exists; use --resume")
    else:
        ledger = {"version": 1, "status": "running", "cells": []}
        if controller_evidence is not None:
            ledger["controller"] = controller_evidence
        target_ids = {cell["cell_id"] for cell in manifest["cells"]}

    def checkpoint() -> None:
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        temporary.replace(ledger_path)

    checkpoint()
    wave_ids = sorted({cell["wave"] for cell in manifest["cells"]})
    by_wave = {wave: [] for wave in wave_ids}
    for cell in manifest["cells"]:
        if cell["cell_id"] in target_ids:
            by_wave[cell["wave"]].append(cell)

    def calibrate_wave(sandbox: Path, wave: int, phase: str,
                       attempt_id: str) -> dict[str, dict]:
        representatives = {
            device: next(cell for cell in manifest["cells"] if cell["device"] == device)
            for device in manifest["calibration"]["devices"]
        }
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    launcher.calibrate, sandbox, cell, phase, wave, attempt_id
                ): device
                for device, cell in representatives.items()
            }
            records = {}
            for future in as_completed(futures):
                device = futures[future]
                try:
                    records[str(device)] = future.result()
                except BaseException as error:
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    records[str(device)] = {
                        "status": "infrastructure_error",
                        "diagnostics": f"{type(error).__name__}: {error}",
                    }
            return records

    def calibration_summary(before: dict, after: dict) -> tuple[dict, str | None]:
        devices, invalid = {}, []
        threshold = manifest["calibration"]["max_drift_fraction"]
        for device in manifest["calibration"]["devices"]:
            key = str(device)
            phases = {"before": before[key], "after": after[key]}
            compact = {}
            values = []
            phase_handles = []
            for phase, evidence in phases.items():
                result = evidence.get("result", {})
                latency = result.get("latency_us")
                handles = result.get("handles", [])
                compact[phase] = {
                    "status": evidence.get("status"), "latency_us": latency,
                    "handles": handles,
                    "timestamp": evidence.get("timestamp"),
                    "selector": result.get("selector"),
                    "evidence_path": evidence.get("evidence_path"),
                }
                if (evidence.get("status") != "complete"
                        or not isinstance(latency, (int, float))
                        or isinstance(latency, bool) or not math.isfinite(latency)
                        or latency <= 0):
                    invalid.append(f"device {device} {phase} calibration failed")
                else:
                    values.append(float(latency))
                phase_handles.append(set(handles) if isinstance(handles, list) else set())
            if len(phase_handles) == 2 and phase_handles[0] & phase_handles[1]:
                invalid.append(f"device {device} calibration reused a durable handle")
            if len(values) == 2:
                drift = abs(values[1] / values[0] - 1.0)
                compact["drift_fraction"] = drift
                if drift > threshold:
                    invalid.append(f"device {device} calibration drift {drift:.6f} exceeds {threshold:.6f}")
            devices[key] = compact
        return {"devices": devices, "max_drift_fraction": threshold}, "; ".join(invalid) or None
    try:
        saw_dry_run = False
        for wave in wave_ids:
            if binding is not None:
                _verify_bundle(controller_bundle, binding)
            if on_wave:
                on_wave(wave)
            if not by_wave[wave]:
                continue
            attempt_root = (Path(tempfile.mkdtemp(prefix="campaign-dry-run-")) if dry_run
                            else root / "attempts" / f"wave-{wave}-{uuid.uuid4().hex}")
            calibration_attempt_id = attempt_root.name
            prepared = [(cell, prepare_cell(manifest, cell, attempt_root))
                        for cell in by_wave[wave]]
            if len(prepared) > manifest["max_parallel"]:
                raise CampaignError("wave exceeds max_parallel")
            calibration_before = None
            if not dry_run and hasattr(launcher, "calibrate"):
                calibration_before = calibrate_wave(
                    prepared[0][1], wave, "before", calibration_attempt_id
                )
            wave_results = []
            before_failed = (calibration_before is not None and any(
                evidence.get("status") != "complete"
                for evidence in calibration_before.values()
            ))
            if before_failed:
                wave_results = [(cell, {"status": "infrastructure_error",
                                        "failure_type": "calibration_failure"})
                                for cell, _ in prepared]
            else:
                with ThreadPoolExecutor(max_workers=manifest["max_parallel"]) as executor:
                    futures = {
                        executor.submit(launcher.launch, sandbox, cell): cell
                        for cell, sandbox in prepared
                    }
                    for future in as_completed(futures):
                        cell = futures[future]
                        try:
                            result = future.result()
                        except BaseException as error:
                            if isinstance(error, KeyboardInterrupt):
                                raise
                            result = {
                                "status": "infrastructure_error",
                                "failure_type": "launcher_exception",
                                "diagnostics": f"{type(error).__name__}: {error}",
                            }
                            wave_results.append((cell, result))
                        else:
                            status = result.get("status")
                            if status is None:
                                status = ("complete" if result.get("exit_code") == 0
                                          and result.get("rounds_completed") == 3
                                          else "infrastructure_error")
                                result["status"] = status
                            wave_results.append((cell, result))
            calibration = None
            calibration_error = None
            if calibration_before is not None:
                after = calibrate_wave(
                    prepared[0][1], wave, "after", calibration_attempt_id
                )
                calibration, calibration_error = calibration_summary(calibration_before, after)
                ledger.setdefault("calibrations", {})[str(wave)] = calibration
            for cell, result in wave_results:
                if calibration is not None:
                    result["device"] = cell["device"]
                    result["calibration"] = calibration
                if calibration_error:
                    result = {
                        "status": "infrastructure_error",
                        "failure_type": "calibration_failure",
                        "diagnostics": calibration_error,
                        "discarded_result": result,
                        "device": cell["device"], "calibration": calibration,
                    }
                status = result["status"]
                entry = {"cell": cell, "result": result,
                         "outcome": classify_outcome(result)}
                normalized = normalize_performance(result)
                if normalized is not None:
                    entry["normalized_performance"] = normalized
                ledger["cells"].append(entry)
                if status == "dry_run":
                    saw_dry_run = True
                elif status == "infrastructure_error":
                    ledger.setdefault("reschedule", []).append(cell["cell_id"])
                checkpoint()
            if dry_run:
                shutil.rmtree(attempt_root)
        if ledger.get("reschedule"):
            ledger["status"] = "needs_reschedule"
        elif saw_dry_run:
            ledger["status"] = "dry_run"
        elif any(entry["result"].get("status") == "candidate_error"
                 for entry in ledger["cells"]):
            ledger["status"] = "completed_with_candidate_failures"
        else:
            ledger["status"] = "complete"
        checkpoint()
    except BaseException as error:
        ledger["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        ledger["failure"] = {"type": type(error).__name__, "message": str(error)}
        checkpoint()
        raise
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-cannbot")
    freeze.add_argument("--repository", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate-manifest", help="freeze all campaign inputs in a manifest")
    generate.add_argument("--prompt", type=Path, required=True)
    generate.add_argument("--gdn-baseline", type=Path, required=True)
    generate.add_argument("--bsa-baseline", type=Path, required=True)
    generate.add_argument("--matmul-baseline", type=Path, required=True)
    generate.add_argument("--project-skill", type=Path, required=True)
    generate.add_argument("--guarded-skill", type=Path, required=True)
    generate.add_argument("--guarded-skill-revision", required=True)
    generate.add_argument("--cannbot-freeze", type=Path, required=True)
    generate.add_argument("--controller-config", type=Path, required=True)
    generate.add_argument("--controller-json", required=True,
                          help="frozen JSON argv; must contain the resolved controller config")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--rounds", type=int, default=3)
    generate.add_argument("--request-budget", type=int, default=12)
    generate.add_argument("--calibration-max-drift", type=float, default=0.10)
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--sandbox", type=Path, required=True)
    run = sub.add_parser("run", help="run the frozen campaign in outer-isolated Codex sessions")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--python", default=sys.executable,
                     help="Python runtime; identity must match the frozen controller")
    run.add_argument("--codex", default="codex")
    run.add_argument("--forbid", type=Path, action="append", default=[])
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true",
                     help="run only infrastructure cells listed for reschedule")
    args = parser.parse_args()
    if args.command == "freeze-cannbot":
        print(json.dumps(freeze_cannbot(args.repository, args.output), sort_keys=True))
    elif args.command == "generate-manifest":
        try:
            controller = json.loads(args.controller_json)
            if (not isinstance(controller, list) or not controller
                    or not all(isinstance(value, str) and value for value in controller)):
                raise ValueError("expected a nonempty JSON array of nonempty strings")
        except (json.JSONDecodeError, ValueError) as error:
            parser.error(f"--controller-json must be a JSON string array: {error}")
        document = write_manifest(
            args.output, args.prompt,
            {"gdn": args.gdn_baseline, "bsa": args.bsa_baseline,
             "matmul": args.matmul_baseline},
            args.project_skill, args.guarded_skill, args.cannbot_freeze,
            args.controller_config, controller,
            args.rounds, args.request_budget, args.guarded_skill_revision,
            args.calibration_max_drift,
        )
        print(json.dumps(document, sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(preflight(json.loads(args.manifest.read_text()), args.sandbox), sort_keys=True))
    else:
        from production_launcher import ProductionLauncher
        manifest = json.loads(args.manifest.read_text())
        args.output.mkdir(parents=True, exist_ok=True)
        controller, evidence = stage_controller(
            manifest, args.manifest, args.output, args.python, resume=args.resume,
        )
        launcher = ProductionLauncher(controller, codex=args.codex,
                                      forbidden_paths=args.forbid, dry_run=args.dry_run)
        print(json.dumps(run_campaign(
            manifest, args.output, launcher, resume=args.resume,
            controller_evidence=evidence, controller_bundle=args.output / "controller",
        ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
