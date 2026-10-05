#!/usr/bin/env python3
"""Production JSON client for retained GZ-A3 workloads via global cpl-remote."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


VALID = {"ok", "compile_error", "runtime_error", "correctness_error", "infrastructure_error"}


class ClientError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def digest_request(job: dict, files: list[Path]) -> str:
    digest = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":")).encode())
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def call(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClientError(f"managed adapter unavailable: {exc}") from exc


def safe_extract(archive: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r") as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or any(x in ("", ".", "..") for x in path.parts):
                    raise ClientError("result archive contains an unsafe member")
            stream.extractall(temporary, filter="data")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def identity(job: dict) -> dict:
    result = {key: job[key] for key in ("benchmark", "action", "device")}
    if job["action"] == "check":
        result.update(cases=job["cases"], scope=job["scope"])
    elif job["action"] == "profile":
        result.update(cases=job["cases"], repeats=job["repeats"])
    else:
        result["case"] = job["case"]
    if job["action"] == "measure":
        result["phase"] = job["phase"]
    if job["action"] == "profile":
        result.update(round=job["round"], kernel_name=job["profiling"]["kernel_name"])
    return result


def prepare(job: dict, root: Path, runner: Path, profiler: Path,
            batch_profiler: Path | None = None) -> tuple[Path, str]:
    source = Path(job["candidate"])
    baseline = Path(job["baseline"])
    cases = Path(job["case_spec"])
    profile_files = [profiler, batch_profiler] if job["action"] == "profile" else []
    files = [source, baseline, cases, runner] + [path for path in profile_files if path is not None]
    if any(not path.is_file() for path in files):
        raise ClientError("a required candidate or frozen benchmark file is missing")
    request = json.loads(json.dumps(job))
    request.update(candidate="candidate.py", baseline="baseline.py", case_spec="baseline.json")
    if request["action"] == "profile":
        request["profiling"]["driver"] = "profile_a3.py"
    key = digest_request(request, files)
    stage = root / key / "payload"
    if not stage.exists():
        temporary = stage.with_name(".payload.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for path, name in ((source, "candidate.py"), (baseline, "baseline.py"),
                           (cases, "baseline.json"), (runner, "runner.py")):
            shutil.copyfile(path, temporary / name)
        if job["action"] == "profile":
            shutil.copyfile(profiler, temporary / "profile_a3.py")
            if batch_profiler is not None:
                shutil.copyfile(batch_profiler, temporary / "batch_profile_a3.py")
        atomic_json(temporary / "job.json", request)
        os.replace(temporary, stage)
    return stage, key


def attach_profile_evidence(result: dict, evidence_path: Path) -> None:
    """Validate and attach the compact evidence emitted by the batch driver."""
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("status") != "success":
        raise ClientError(f"msprof op did not produce valid evidence: {evidence}")
    # ``cases`` is the immutable input identity on the internal wire;
    # ``profile_cases`` contains the rows exposed as ``cases`` by experimentctl.
    if result.get("profile_cases") != evidence.get("cases"):
        raise ClientError("remote response does not match compact profiling evidence")
    result["profile"] = evidence


def execute(job: dict, args: argparse.Namespace) -> dict:
    from bz_a3_job_client import BzA3JobClient, RemoteTransport

    if job.get("action") == "profile":
        job = json.loads(json.dumps(job))
        job.setdefault("profiling", {})["tool"] = "msprof op"
    here = Path(__file__).resolve().parent
    device = job.get("device")
    placements = {str(device): {"profile": "gz-a3", "device": device}}
    return BzA3JobClient(
        RemoteTransport(args.remote, args.runtime_activate), args.state_dir, placements,
        runner=here / "a3_benchmark_runner.py", profiler=here / "profile_a3.py",
        batch_profiler=here / "batch_profile_a3.py", remote_root=args.remote_root,
        allowed_profiles={"gz-a3"},
    ).run(job, args.timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-json", default='["cpl-remote"]')
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--runtime-activate", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        args.remote = json.loads(args.remote_json)
        if (not isinstance(args.remote, list) or not args.remote
                or not all(isinstance(x, str) and x for x in args.remote)):
            raise ClientError("--remote-json must be a non-empty JSON string array")
        job = json.load(sys.stdin)
        result = execute(job, args)
    except (ClientError, OSError, ValueError, json.JSONDecodeError) as exc:
        bound = identity(job) if "job" in locals() and isinstance(job, dict) else {}
        result = {"status": "infrastructure_error", "diagnostics": str(exc), **bound}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
