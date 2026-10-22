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


VALID = {"ok", "submission_error", "compile_error", "runtime_error",
         "correctness_error", "infrastructure_error"}


class ClientError(RuntimeError):
    pass


def _valid_selector_mapping(declared: object, resolved: object) -> bool:
    return (isinstance(declared, str) and bool(declared)
            and isinstance(resolved, str) and bool(resolved)
            and (resolved == declared
                 or any(declared == resolved + suffix
                        for suffix in ("_mix_aic", "_mix_aiv"))))


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
        result.update(round=job["round"], kernel_name=job["profiling"]["kernel_name"],
                      replay_mode=job["profiling"].get("replay_mode", "kernel"))
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
    declared = result.get("kernel_name")
    explicit_names = ("declared_kernel_name" in evidence
                      or "resolved_kernel_name" in evidence)
    evidence_declared = evidence.get("declared_kernel_name") if explicit_names else declared
    resolved = evidence.get("resolved_kernel_name") if explicit_names else declared
    captures = evidence.get("captures")
    if (not isinstance(declared, str) or not declared
            or evidence.get("kernel_name") != declared
            or evidence_declared != declared
            or not _valid_selector_mapping(declared, resolved)
            or result.get("declared_kernel_name", declared) != declared
            or result.get("resolved_kernel_name", resolved) != resolved
            or not isinstance(captures, list)
            or any(not isinstance(capture, dict)
                   or capture.get("kernel_name") != resolved
                   or (explicit_names and
                       (capture.get("declared_kernel_name") != declared
                        or capture.get("resolved_kernel_name") != resolved))
                   for capture in captures)):
        raise ClientError("remote response does not match compact profiling selector identity")
    # ``cases`` is the immutable input identity on the internal wire;
    # ``profile_cases`` contains the rows exposed as ``cases`` by experimentctl.
    if result.get("profile_cases") != evidence.get("cases"):
        raise ClientError("remote response does not match compact profiling evidence")
    captures = evidence.get("captures")
    if (not isinstance(captures, list) or not captures
            or not all(isinstance(capture, dict) for capture in captures)
            or result.get("replay_mode", "kernel") != evidence.get("replay_mode", "kernel")
            or any(capture.get("replay_mode", "kernel") != evidence.get("replay_mode", "kernel")
                   for capture in captures)):
        raise ClientError("remote response does not match compact profiling replay mode")
    result["profile"] = evidence


def execute(job: dict, args: argparse.Namespace) -> dict:
    from bz_a3_job_client import BzA3JobClient, GlobalCplRemoteTransport

    class GzA3JobClient(BzA3JobClient):
        TARGETS = {"gz-a3"}

        def _remote_script(self, archive, archive_sha, run_root, timeout, job):
            # Upload paths use the registry namespace, but run's cwd is already
            # our staging child. Materialize absolute paths before cd'ing into
            # the extracted worker directory so its script paths remain valid.
            prefix = self.remote_root + "/"
            if not archive.startswith(prefix) or not run_root.startswith(prefix):
                raise ClientError("GZ worker paths must stay below the staging root")
            script = super()._remote_script(
                archive[len(prefix):], archive_sha, run_root[len(prefix):], timeout, job)
            assignment, rest = script.split("\ntest ", 1)
            return (assignment + '\narchive="$PWD/$archive"; run_root="$PWD/$run_root"'
                    + "\ntest " + rest)

    if not isinstance(job, dict):
        raise ClientError("job must be a JSON object")
    request = json.loads(json.dumps(job))
    if request.get("action") == "profile":
        if not isinstance(request.get("profiling", {}), dict):
            raise ClientError("profiling must be a JSON object")
        request.setdefault("profiling", {}).setdefault("tool", "msprof op")
    device = request.get("device")
    placements = {str(device): {"target": "gz-a3", "device": device}}
    here = Path(__file__).resolve().parent
    transport = GlobalCplRemoteTransport(
        args.cpl_remote_sha256, args.remote_root, targets={"gz-a3"})
    return GzA3JobClient(
        transport, args.state_dir, placements,
        runner=here / "a3_benchmark_runner.py", profiler=here / "profile_a3.py",
        batch_profiler=here / "batch_profile_a3.py", remote_root=args.remote_root,
    ).run(
        request, args.timeout,
        operation_mode=os.environ.get("PROFILING_SKILL_CONTROLLER_MODE", "submit"),
        controller_request_sha256=os.environ.get(
            "PROFILING_SKILL_CONTROLLER_REQUEST_SHA256"),
    )


def main() -> int:
    from bz_a3_job_client import JobError

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cpl-remote-sha256", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        job = json.load(sys.stdin)
        result = execute(job, args)
    except (ClientError, JobError, OSError, ValueError, TypeError) as exc:
        result = {
            "status": "infrastructure_error",
            "failure_type": exc.failure_type if isinstance(exc, JobError) else "request_error",
            "diagnostics": str(exc), "handle": None,
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
