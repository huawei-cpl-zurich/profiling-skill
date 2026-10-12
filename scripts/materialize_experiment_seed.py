#!/usr/bin/env python3
"""Create a deterministic Git seed and simple-runner cell configuration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


class SeedError(RuntimeError):
    pass


def checked_file(binding: object, label: str) -> tuple[Path, bytes]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise SeedError(f"{label} binding is invalid")
    configured = Path(binding["path"])
    if configured.is_symlink():
        raise SeedError(f"{label} source is unavailable")
    path = configured.resolve()
    if not path.is_file():
        raise SeedError(f"{label} source is unavailable")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != binding["sha256"]:
        raise SeedError(f"{label} hash mismatch")
    return path, content


def destination(value: object) -> Path:
    if not isinstance(value, str):
        raise SeedError("seed destination must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", "..", ".git"} for part in pure.parts):
        raise SeedError("seed destination must stay inside the repository")
    return Path(*pure.parts)


def git(repo: Path, *arguments: str, env: dict | None = None) -> str:
    result = subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false",
                             "-c", "core.hooksPath=/dev/null", *arguments], text=True,
                            capture_output=True, env=env, check=False)
    if result.returncode:
        raise SeedError(result.stderr.strip() or "git failed")
    return result.stdout.strip()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(mode)):
            raise SeedError(f"seed contains unsupported entry: {relative}")
        if path.is_file():
            digest.update(relative.encode() + b"\0")
            digest.update(oct(stat.S_IMODE(mode)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def materialize(spec: dict) -> dict:
    if spec.get("schema") != "profiling-skill/simple-seed-spec/v1":
        raise SeedError("unsupported seed schema")
    if not isinstance(spec.get("output"), str) or not spec["output"]:
        raise SeedError("seed output is required")
    output = Path(spec["output"]).resolve()
    if output.exists():
        raise SeedError("seed output already exists")
    _, prompt = checked_file(spec.get("prompt"), "prompt")
    _, task = checked_file(spec.get("task"), "task")
    try:
        prompt_text, _ = prompt.decode(), task.decode()
    except UnicodeDecodeError as exc:
        raise SeedError("prompt and task must be UTF-8") from exc
    files = spec.get("files")
    if not isinstance(files, list):
        raise SeedError("files must be a list")
    resolved = []
    seen = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict) or "destination" not in item:
            raise SeedError(f"files[{index}] is invalid")
        target = destination(item["destination"])
        if target.as_posix() in seen:
            raise SeedError("seed destinations must be unique")
        seen.add(target.as_posix())
        _, content = checked_file({"path": item.get("path"), "sha256": item.get("sha256")},
                                  f"files[{index}]")
        resolved.append((target, content))
    config = spec.get("config")
    if not isinstance(config, dict) or {"source_repo", "prompt", "task_path", "base_ref"} & config.keys():
        raise SeedError("config contains generated or invalid fields")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        repo = temporary / "repo"
        repo.mkdir()
        git(repo, "init", "--object-format=sha1", "-b", "main")
        for target, content in resolved:
            path = repo / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        git(repo, "add", ".")
        environment = {**os.environ, "GIT_AUTHOR_NAME": "Experiment Seed",
                       "GIT_AUTHOR_EMAIL": "experiment-seed@example.invalid",
                       "GIT_COMMITTER_NAME": "Experiment Seed",
                       "GIT_COMMITTER_EMAIL": "experiment-seed@example.invalid",
                       "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
                       "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"}
        git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m",
            "experiment: immutable seed", env=environment)
        (temporary / "TASK.md").write_bytes(task)
        emitted = {**config, "source_repo": str(output / "repo"), "base_ref": "main",
                   "prompt": prompt_text, "task_path": str(output / "TASK.md")}
        (temporary / "cell.json").write_text(json.dumps(emitted, indent=2, sort_keys=True) + "\n")
        receipt = {"schema": "profiling-skill/simple-seed-receipt/v1",
                   "commit": git(repo, "rev-parse", "HEAD"),
                   "seed_tree_sha256": tree_sha256(repo),
                   "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                   "task_sha256": hashlib.sha256(task).hexdigest()}
        (temporary / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: materialize_experiment_seed.py SPEC.json", file=sys.stderr)
        return 2
    try:
        spec = json.loads(Path(sys.argv[1]).read_text())
        print(json.dumps(materialize(spec), sort_keys=True))
    except (SeedError, json.JSONDecodeError, OSError) as exc:
        print(f"materialize-experiment-seed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
