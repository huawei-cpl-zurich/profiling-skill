#!/usr/bin/env python3
"""Reproducible, isolated campaign preparation and session scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol


BENCHMARK_DEVICE = {"gdn": 0, "bsa": 1}
WAVES = (
    (("gdn", "cannbot"), ("bsa", "project-cannbot")),
    (("gdn", "project-cannbot"), ("bsa", "project-only")),
    (("gdn", "project-only"), ("bsa", "cannbot")),
)
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
    "project-only": ("ascend-profiling",),
}


class CampaignError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def cells(rounds: int = 3, request_budget: int = 12) -> list[Cell]:
    result = []
    for wave, pairs in enumerate(WAVES, 1):
        for benchmark, treatment in pairs:
            result.append(
                Cell(
                    f"{benchmark}-{treatment}", benchmark, treatment,
                    BENCHMARK_DEVICE[benchmark], wave, rounds, request_budget,
                )
            )
    return result


def write_manifest(path: Path, prompt: Path, baselines: dict[str, Path],
                   project_skill: Path, cannbot_freeze: Path,
                   rounds: int = 3, request_budget: int = 12) -> dict:
    missing = {name for name in BENCHMARK_DEVICE if name not in baselines}
    if missing:
        raise CampaignError(f"missing baselines: {', '.join(sorted(missing))}")
    freeze_file = cannbot_freeze / "freeze.json"
    freeze = json.loads(freeze_file.read_text())
    validate_freeze_record(freeze)
    document = {
        "version": 1,
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
            "cannbot": {
                "path": str(cannbot_freeze.resolve()),
                "freeze": freeze,
                "freeze_sha256": digest_file(freeze_file),
            },
        },
        "cells": [asdict(cell) for cell in cells(rounds, request_budget)],
        "max_parallel": 2,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def _skill_source(manifest: dict, skill: str) -> Path:
    if skill == "ascend-profiling":
        return Path(manifest["skill_sources"][skill]["path"])
    return Path(manifest["skill_sources"]["cannbot"]["path"]) / "skills" / skill


def verify_frozen_sources(manifest: dict) -> None:
    project = manifest["skill_sources"]["ascend-profiling"]
    if digest_tree(Path(project["path"])) != project["sha256"]:
        raise CampaignError("project profiling skill drifted after manifest freeze")
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
    if cell["treatment"] != "project-only":
        copy_regular_tree(
            Path(manifest["skill_sources"]["cannbot"]["path"])
            / "support" / "triton-op-generator",
            workspace / ".agents" / "plugins-official" / "triton-op-generator",
        )
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
    if digest_tree(workspace, (".agents",)) != expected_baseline:
        raise CampaignError("baseline hash mismatch")
    support = workspace / ".agents" / "plugins-official" / "triton-op-generator"
    if (cell["treatment"] == "project-only") == support.exists():
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
                 on_wave: Callable[[int], None] | None = None) -> dict:
    """Run fixed waves; launcher implementations may execute each pair concurrently."""
    root.mkdir(parents=True, exist_ok=True)
    ledger = {"version": 1, "status": "running", "cells": []}
    ledger_path = root / "ledger.json"

    def checkpoint() -> None:
        temporary = ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        temporary.replace(ledger_path)

    checkpoint()
    by_wave = {wave: [] for wave in range(1, 4)}
    for cell in manifest["cells"]:
        by_wave[cell["wave"]].append(cell)
    try:
        for wave in range(1, 4):
            if on_wave:
                on_wave(wave)
            prepared = [(cell, prepare_cell(manifest, cell, root)) for cell in by_wave[wave]]
            if len(prepared) > manifest["max_parallel"]:
                raise CampaignError("wave exceeds max_parallel")
            failures = []
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
                        failures.append(error)
                    else:
                        ledger["cells"].append({"cell": cell, "result": result})
                        checkpoint()
            if failures:
                raise failures[0]
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
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--sandbox", type=Path, required=True)
    run = sub.add_parser("run", help="run the frozen campaign in outer-isolated Codex sessions")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--controller", nargs="+", required=True)
    run.add_argument("--codex", default="codex")
    run.add_argument("--forbid", type=Path, action="append", default=[])
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze-cannbot":
        print(json.dumps(freeze_cannbot(args.repository, args.output), sort_keys=True))
    elif args.command == "preflight":
        print(json.dumps(preflight(json.loads(args.manifest.read_text()), args.sandbox), sort_keys=True))
    else:
        from production_launcher import ProductionLauncher
        launcher = ProductionLauncher(
            args.controller, codex=args.codex, forbidden_paths=args.forbid,
            dry_run=args.dry_run,
        )
        print(json.dumps(run_campaign(
            json.loads(args.manifest.read_text()), args.output, launcher,
        ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
