#!/usr/bin/env python3
"""Production JSON client for profile-managed, mutable GZ-A3 workloads."""

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
    else:
        result["case"] = job["case"]
    if job["action"] == "measure":
        result["phase"] = job["phase"]
    if job["action"] == "profile":
        result.update(round=job["round"], kernel_name=job["profiling"]["kernel_name"])
    return result


def prepare(job: dict, root: Path, runner: Path, profiler: Path) -> tuple[Path, str]:
    source = Path(job["candidate"])
    baseline = Path(job["baseline"])
    cases = Path(job["case_spec"])
    files = [source, baseline, cases, runner] + ([profiler] if job["action"] == "profile" else [])
    if any(not path.is_file() for path in files):
        raise ClientError("a required candidate or frozen benchmark file is missing")
    request = json.loads(json.dumps(job))
    request.update(candidate="candidate.py", baseline="baseline.py", case_spec="cases.jsonl")
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
        atomic_json(temporary / "job.json", request)
        os.replace(temporary, stage)
    return stage, key


def execute(job: dict, args: argparse.Namespace) -> dict:
    here = Path(__file__).resolve().parent
    stage, key = prepare(job, args.state_dir, here / "a3_benchmark_runner.py", here / "profile_a3.py")
    state = stage.parent
    upload = state / "upload.json"
    run_receipt = state / "run.json"
    transfer = state / "download.json"
    result_tar = state / "result.tar"
    result_dir = state / "result"
    includes = ["candidate.py", "baseline.py", "baseline.json", "runner.py", "job.json"]
    if job["action"] == "profile":
        includes.append("profile_a3.py")
    stage_cmd = args.adapter + ["--profile", "gz-a3", "--operation", f"experiment-stage-{key[:16]}",
                                "stage", "--source-root", str(stage), "--receipt", str(upload)]
    for name in includes:
        stage_cmd += ["--include", name]
    staged = call(stage_cmd, args.timeout)
    if staged.returncode:
        raise ClientError(f"bundle staging failed\n{staged.stdout}{staged.stderr}")
    command = "python3 runner.py --job job.json --output \"$A3_BUNDLE_OUTPUT_DIR/response.json\""
    results = ["response.json"]
    if job["action"] == "profile":
        kernel = job["profiling"]["kernel_name"]
        command = (f"python3 profile_a3.py --output \"$A3_BUNDLE_OUTPUT_DIR/profile\" "
                   f"--kernel-name {shlex.quote(kernel)} -- python3 runner.py --job job.json "
                   f"--output \"$A3_BUNDLE_OUTPUT_DIR/response.json\" || "
                   f"test -f \"$A3_BUNDLE_OUTPUT_DIR/response.json\"")
        results.extend(("profile/evidence.json", "profile/msprof.log"))
    run_cmd = args.adapter + ["--profile", "gz-a3", "--operation", f"experiment-run-{key[:16]}",
                              "run-bundle", "--receipt", str(upload), "--run-receipt", str(run_receipt),
                              "--run-id", key[:24], "--device", str(job["device"]),
                              "--runtime", "py311-torch", "--timeout", str(args.timeout)]
    for name in results:
        run_cmd += ["--result", name]
    run = call(run_cmd + ["--", "bash", "-c", command], args.timeout + 30)
    receipt = json.loads(run_receipt.read_text()) if run_receipt.exists() else {}
    handle = f"gz-a3:{receipt['command_handle']}" if receipt.get("command_handle") else None
    if run.returncode or receipt.get("state") != "succeeded":
        raise ClientError(f"managed command failed or observation was interrupted; handle={handle}\n{run.stdout}{run.stderr}")
    expected = receipt.get("result_sha256")
    if not result_dir.exists():
        if result_tar.exists() and hashlib.sha256(result_tar.read_bytes()).hexdigest() != expected:
            raise ClientError("retained result archive does not match the run receipt")
        if not result_tar.exists():
            fetch = call(args.adapter + ["--profile", "gz-a3", "--operation", f"experiment-fetch-{key[:16]}",
                         "fetch-bundle-result", "--run-receipt", str(run_receipt), "--transfer-receipt", str(transfer),
                         "--expected-sha256", str(expected), "--output", str(result_tar), "--timeout", str(args.timeout)], args.timeout)
            if fetch.returncode:
                raise ClientError(f"result fetch failed; handle={handle}\n{fetch.stdout}{fetch.stderr}")
        safe_extract(result_tar, result_dir)
    result = json.loads((result_dir / "response.json").read_text())
    if result.get("status") not in VALID:
        raise ClientError("remote harness returned an invalid status")
    result["handle"] = handle
    result["artifacts"] = {"request_digest": key, "result_sha256": expected,
                           "profile": str(result_dir / "profile/evidence.json") if job["action"] == "profile" else None,
                           "msprof_log": str(result_dir / "profile/msprof.log") if job["action"] == "profile" else None}
    if job["action"] == "profile" and result["status"] == "ok":
        evidence = json.loads((result_dir / "profile/evidence.json").read_text())
        if evidence.get("status") != "success":
            raise ClientError(f"msprof op did not produce valid evidence: {evidence}")
        result["profile"] = evidence
        result["latency_us"] = evidence["kernels"][0]["duration_us"]["median"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-json", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        args.adapter = json.loads(args.adapter_json)
        if not isinstance(args.adapter, list) or not args.adapter or not all(isinstance(x, str) and x for x in args.adapter):
            raise ClientError("--adapter-json must be a non-empty JSON string array")
        job = json.load(sys.stdin)
        result = execute(job, args)
    except (ClientError, OSError, ValueError, json.JSONDecodeError) as exc:
        bound = identity(job) if "job" in locals() and isinstance(job, dict) else {}
        result = {"status": "infrastructure_error", "diagnostics": str(exc), **bound}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
