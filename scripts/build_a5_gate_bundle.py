#!/usr/bin/env python3
"""Build the deterministic A5 gate bundle and machine-local config fragments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


from a5_terminal_gate import SPECS


FILES = (
    "scripts/a5_terminal_gate.py", "scripts/a3_benchmark_runner.py",
    "scripts/fully_fused_contract.py",
    "benchmarks/gdn/baseline.py", "benchmarks/gdn/cases.jsonl",
    "benchmarks/gdn/baseline.json", "benchmarks/bsa/baseline.py",
    "benchmarks/bsa/cases.jsonl", "benchmarks/bsa/baseline.json",
    "benchmarks/matmul/baseline.py", "benchmarks/matmul/cases.jsonl",
    "benchmarks/matmul/baseline.json",
)
BUNDLE_SCHEMA = "profiling-skill/a5-terminal-gate-bundle/v1"


def manifest(task: str) -> dict:
    spec = SPECS[task]
    return {
        "schema": "profiling-skill/candidate-kernel/v2",
        "kernel_name": spec["kernel_name"], "entrypoint": spec["entrypoint"],
        "fusion": {"schema_version": 1, "mode": "single-logical-launch",
                   "complete_operator": True},
    }


def build(source: Path, output: Path) -> dict:
    contents = {}
    for name in FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular bundle input is missing: {name}")
        contents[name] = path.read_bytes()
    document = {
        "schema": BUNDLE_SCHEMA,
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(contents.items())},
    }
    contents["bundle-manifest.json"] = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return {**document, "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def gate_script(task: str, archive_sha256: str) -> str:
    if task not in SPECS or len(archive_sha256) != 64:
        raise ValueError("invalid task or archive digest")
    return f'''#!/usr/bin/env bash
set -euo pipefail
bundle="${{TERMINAL_GATE_BUNDLE_PATH:?host did not stage terminal gate bundle}}"
manifest="${{CANDIDATE_MANIFEST_PATH:?host did not stage candidate manifest}}"
root="$experiment_dir/a5-gate"
expected={archive_sha256}
actual=$(sha256sum "$bundle" | awk '{{print $1}}')
if [[ "$actual" != "$expected" ]]; then
  printf '%s\n' '{{"schema":"simple-experiment-result/v1","outcome":"infrastructure_failure","diagnostics":"terminal gate bundle digest mismatch"}}'
  exit 1
fi
mkdir -p "$root"
if ! tar -xf "$bundle" -C "$root" --no-same-owner --no-same-permissions; then
  printf '%s\n' '{{"schema":"simple-experiment-result/v1","outcome":"infrastructure_failure","diagnostics":"terminal gate bundle extraction failed"}}'
  exit 1
fi
exec python "$root/scripts/a5_terminal_gate.py" --root "$root" \\
  --candidate "$CANDIDATE_PATH" --manifest "$manifest" \\
  --benchmark {task} --repeats 3
'''


def generate(source: Path, output: Path, *, wrapper_root: Path,
             state_root: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "a5-terminal-gates.tar"
    identity = build(source, bundle)
    tasks = {}
    for task, spec in SPECS.items():
        declaration = output / f"{task}.manifest.json"
        gate = output / f"{task}.sh"
        declaration.write_text(json.dumps(manifest(task), sort_keys=True) + "\n")
        gate.write_text(gate_script(task, identity["archive_sha256"]))
        gate.chmod(0o700)
        tasks[task] = {
            "candidate_manifest_path": str(declaration.resolve()),
            "terminal_gate_path": str(gate.resolve()),
            "remote_transport": {
                "kind": "bz-a5-wrapper", "wrapper_root": str(wrapper_root.resolve()),
                "state_root": str((state_root / task).resolve()),
                "bundle_path": str(bundle.resolve()),
                "bundle_sha256": identity["archive_sha256"],
                "warm_up": len(spec["cases"]),
                "launch_count": len(spec["ranked"]) * 3,
                "ranked_cases": spec["ranked"], "repeats": 3,
            },
        }
    document = {"schema": BUNDLE_SCHEMA, "bundle": identity, "tasks": tasks}
    (output / "config-fragments.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.source.resolve(), args.output.resolve(),
                      wrapper_root=args.wrapper_root, state_root=args.state_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
