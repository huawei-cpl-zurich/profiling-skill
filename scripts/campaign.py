#!/usr/bin/env python3
"""Reproducible, isolated campaign preparation and session scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


BENCHMARK_DEVICE = {"gdn": 0, "bsa": 1}
WAVES = (
    (("gdn", "cannbot"), ("bsa", "project-cannbot")),
    (("gdn", "project-cannbot"), ("bsa", "project-only")),
    (("gdn", "project-only"), ("bsa", "cannbot")),
)
TREATMENT_SKILLS = {
    "cannbot": ("cannbot-triton", "ops-profiling"),
    "project-cannbot": ("cannbot-triton", "ascend-profiling"),
    "project-only": ("ascend-profiling",),
}


class CampaignError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
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
            shutil.copyfile(item, destination / relative)
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


def freeze_cannbot(repo_url: str, destination: Path, skill_paths: Iterable[str]) -> dict:
    """Resolve master, checkout exactly that commit, and freeze selected trees."""
    commit = resolve_master(repo_url)
    with tempfile.TemporaryDirectory(prefix="cannbot-freeze-") as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(["git", "clone", "--no-checkout", repo_url, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
        destination.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for relative in skill_paths:
            source = checkout / relative
            name = Path(relative).name
            copy_regular_tree(source, destination / name)
            hashes[name] = digest_tree(destination / name)
    record = {"repository": repo_url, "commit": commit, "skills": hashes}
    (destination / "freeze.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


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
    document = {
        "version": 1,
        "prompt": {"path": str(prompt.resolve()), "sha256": digest_file(prompt)},
        "baselines": {
            name: {"path": str(value.resolve()), "sha256": digest_tree(value)}
            for name, value in sorted(baselines.items())
        },
        "skill_sources": {
            "ascend-profiling": str(project_skill.resolve()),
            "cannbot": str(cannbot_freeze.resolve()),
        },
        "cells": [asdict(cell) for cell in cells(rounds, request_budget)],
        "max_parallel": 2,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def _skill_source(manifest: dict, skill: str) -> Path:
    if skill == "ascend-profiling":
        return Path(manifest["skill_sources"][skill])
    return Path(manifest["skill_sources"]["cannbot"]) / skill


def prepare_cell(manifest: dict, cell: dict, campaigns_root: Path) -> Path:
    sandbox = campaigns_root / cell["cell_id"]
    if sandbox.exists():
        raise CampaignError(f"fresh sandbox required: {sandbox}")
    (sandbox / ".agents" / "skills").mkdir(parents=True)
    shutil.copyfile(manifest["prompt"]["path"], sandbox / "PROMPT.md")
    copy_regular_tree(
        Path(manifest["baselines"][cell["benchmark"]]["path"]),
        sandbox / "workspace",
    )
    skill_hashes = {}
    for skill in TREATMENT_SKILLS[cell["treatment"]]:
        target = sandbox / ".agents" / "skills" / skill
        copy_regular_tree(_skill_source(manifest, skill), target)
        skill_hashes[skill] = digest_tree(target)
    metadata = {
        "cell": cell,
        "prompt_sha256": digest_file(sandbox / "PROMPT.md"),
        "baseline_sha256": digest_tree(sandbox / "workspace"),
        "skills": skill_hashes,
    }
    (sandbox / "cell.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    preflight(manifest, sandbox)
    return sandbox


def preflight(manifest: dict, sandbox: Path) -> dict:
    metadata = json.loads((sandbox / "cell.json").read_text())
    cell = metadata["cell"]
    expected = set(TREATMENT_SKILLS[cell["treatment"]])
    skill_root = sandbox / ".agents" / "skills"
    actual = {path.name for path in skill_root.iterdir()}
    if actual != expected:
        raise CampaignError(f"skill isolation mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    if any(path.is_symlink() for path in sandbox.rglob("*")):
        raise CampaignError("sandbox contains a symlink")
    if digest_file(sandbox / "PROMPT.md") != manifest["prompt"]["sha256"]:
        raise CampaignError("prompt hash mismatch")
    expected_baseline = manifest["baselines"][cell["benchmark"]]["sha256"]
    if digest_tree(sandbox / "workspace") != expected_baseline:
        raise CampaignError("baseline hash mismatch")
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
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "CAMPAIGN_CELL": cell["cell_id"],
            "CAMPAIGN_DEVICE": str(cell["device"]),
            "CAMPAIGN_ROUNDS": str(cell["rounds"]),
            "CAMPAIGN_REQUEST_BUDGET": str(cell["request_budget"]),
            "CODEX_HOME": str((sandbox / ".agents").resolve()),
        }
        result = subprocess.run(
            self.command, cwd=sandbox / "workspace", env=environment,
            text=True, capture_output=True,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_campaign(manifest: dict, root: Path, launcher: Launcher,
                 on_wave: Callable[[int], None] | None = None) -> dict:
    """Run fixed waves; launcher implementations may execute each pair concurrently."""
    ledger = {"version": 1, "status": "running", "cells": []}
    by_wave = {wave: [] for wave in range(1, 4)}
    for cell in manifest["cells"]:
        by_wave[cell["wave"]].append(cell)
    for wave in range(1, 4):
        if on_wave:
            on_wave(wave)
        prepared = [(cell, prepare_cell(manifest, cell, root)) for cell in by_wave[wave]]
        if len(prepared) > manifest["max_parallel"]:
            raise CampaignError("wave exceeds max_parallel")
        with ThreadPoolExecutor(max_workers=manifest["max_parallel"]) as executor:
            futures = [executor.submit(launcher.launch, sandbox, cell) for cell, sandbox in prepared]
        for (cell, _), future in zip(prepared, futures):
            result = future.result()
            ledger["cells"].append({"cell": cell, "result": result})
    ledger["status"] = "complete"
    (root / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-cannbot")
    freeze.add_argument("--repository", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--skill", action="append", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--sandbox", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-cannbot":
        print(json.dumps(freeze_cannbot(args.repository, args.output, args.skill), sort_keys=True))
    else:
        print(json.dumps(preflight(json.loads(args.manifest.read_text()), args.sandbox), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
