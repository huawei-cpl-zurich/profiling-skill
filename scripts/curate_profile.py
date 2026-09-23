#!/usr/bin/env python3
"""Create a compact, authenticated archive from retained A5 profiler evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


class InvalidEvidence(ValueError):
    pass


def safe_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def relative(path: Path, root: Path) -> str:
    value = PurePosixPath(path.relative_to(root).as_posix())
    if value.is_absolute() or ".." in value.parts:
        raise InvalidEvidence(f"unsafe path: {path}")
    return value.as_posix()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_csvs(root: Path, output: Path, kernels: set[str]) -> list[dict]:
    inventory = []
    for path in safe_files(root):
        if path.suffix.lower() != ".csv":
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames or []
        name_field = next(
            (
                x
                for x in ("Op Name", "OpName", "Kernel Name", "kernel_name")
                if x in fields
            ),
            None,
        )
        if kernels:
            if not name_field:
                continue
            rows = [row for row in rows if row.get(name_field) in kernels]
        if not rows:
            continue
        target = output / "metrics" / f"{relative(path, root)}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
        inventory.append(
            {
                "source": relative(path, root),
                "rows": len(rows),
                "artifact": relative(target, output),
            }
        )
    return inventory


def copy_summaries(
    root: Path, output: Path, mode: str, kernels: set[str]
) -> list[str]:
    if kernels:
        return []
    names = {
        "sample-pmu-summary.json",
        "sample-pmu-summary.md",
        "timeline-summary.json",
    }
    if mode == "analysis":
        names |= {"sample-pmu-rows.csv.gz", "timeline-events.csv.gz"}
    copied = []
    for path in safe_files(root):
        if path.name not in names:
            continue
        target = output / "derived" / relative(path, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(relative(target, output))
    return copied


def deterministic_archive(root: Path, archive: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in safe_files(root):
            info = tar.gettarinfo(str(path), arcname=relative(path, root))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(buffer.getvalue())


def curate(source: Path, destination: Path, kernels: set[str], mode: str) -> Path:
    if not source.is_dir():
        raise InvalidEvidence(f"input is not a directory: {source}")
    source = source.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise InvalidEvidence("output must not be inside input evidence")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        compact = Path(temp) / "profile"
        compact.mkdir()
        csvs = selected_csvs(source, compact, kernels)
        derived = copy_summaries(source, compact, mode, kernels)
        if not csvs and not derived:
            raise InvalidEvidence("no useful metric or derived evidence found")
        sources = [
            {
                "path": relative(path, source),
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in safe_files(source)
        ]
        ignored_symlinks = [
            relative(path, source)
            for path in sorted(source.rglob("*"))
            if path.is_symlink()
        ]
        manifest = {
            "schema_version": 1,
            "target": "Ascend950-A5",
            "mode": mode,
            "kernel_names": sorted(kernels),
            "metric_exports": csvs,
            "derived_artifacts": derived,
            "source_inventory": sources,
            "ignored_symlinks": ignored_symlinks,
        }
        (compact / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        archive = destination / f"ascend-profile-{mode}.tar.gz"
        deterministic_archive(compact, archive)
        (destination / f"ascend-profile-{mode}.sha256").write_text(
            f"{digest(archive)}  {archive.name}\n"
        )
        return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name", action="append", default=[])
    parser.add_argument("--mode", choices=("summary", "analysis"), default="analysis")
    args = parser.parse_args()
    print(curate(args.input, args.output, set(args.kernel_name), args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
