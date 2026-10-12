#!/usr/bin/env python3
"""Build the deterministic, reusable A3 terminal-gate harness bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


FILES = (
    "scripts/a3_terminal_gate.py", "scripts/a3_benchmark_runner.py",
    "scripts/fully_fused_contract.py", "scripts/profile_a3.py",
    "scripts/batch_profile_a3.py",
    "scripts/normalize_a3_pipe_evidence.py",
    "references/a3-pipe-activity-model.json",
    "benchmarks/gdn/baseline.py", "benchmarks/gdn/cases.jsonl",
    "benchmarks/gdn/baseline.json", "benchmarks/bsa/baseline.py",
    "benchmarks/bsa/cases.jsonl", "benchmarks/bsa/baseline.json",
    "benchmarks/matmul/baseline.py", "benchmarks/matmul/cases.jsonl",
    "benchmarks/matmul/baseline.json",
)


def build(source: Path, output: Path) -> dict:
    contents = {}
    for name in FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular bundle input is missing: {name}")
        contents[name] = path.read_bytes()
    manifest = {
        "schema": "profiling-skill/a3-terminal-gate-bundle/v1",
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(contents.items())},
    }
    contents["bundle-manifest.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return {**manifest, "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
