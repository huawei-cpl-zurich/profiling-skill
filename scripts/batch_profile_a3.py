#!/usr/bin/env python3
"""Run an ordered profile matrix inside one managed A3 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


MAX_LOG_BYTES = 64 * 1024


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(job: dict) -> dict:
    return {
        "benchmark": job["benchmark"],
        "action": "profile",
        "device": job["device"],
        "cases": job["cases"],
        "repeats": job["repeats"],
        "round": job["round"],
        "kernel_name": job["profiling"]["kernel_name"],
        "replay_mode": job["profiling"].get("replay_mode", "kernel"),
    }


def bounded_log(parts: list[str]) -> str:
    joined = "\n".join(parts)
    data = joined.encode(errors="replace")
    if len(data) <= MAX_LOG_BYTES:
        return joined
    marker = b"\n... earlier batch log truncated ...\n"
    return (marker + data[-(MAX_LOG_BYTES - len(marker)):]).decode(errors="replace")


def failure(job: dict, status: str, diagnostics: str, captures: list[dict]) -> dict:
    return {
        "status": status,
        "diagnostics": diagnostics,
        **identity(job),
        "completed_captures": captures,
    }


def emit_failure(output: Path, response_path: Path, result: dict,
                 captures: list[dict], logs: list[str]) -> int:
    write_json(response_path, result)
    evidence = {
        "schema_version": 1,
        "status": "failure",
        "failure": {"kind": result["status"], "message": result["diagnostics"]},
        "captures": captures,
    }
    if "replay_mode" in result:
        evidence["replay_mode"] = result["replay_mode"]
    write_json(output / "evidence.json", evidence)
    (output / "msprof.log").write_text(bounded_log(logs))
    return 1


def execute(job: dict, *, runner: Path, profiler: Path, output: Path,
            response_path: Path, kernel_name: str) -> int:
    bound = identity(job)
    replay_mode = bound["replay_mode"]
    if replay_mode not in {"kernel", "application"}:
        result = failure(job, "infrastructure_error",
                         f"unsupported replay mode: {replay_mode!r}", [])
        return emit_failure(output, response_path, result, [], [])
    if kernel_name != bound["kernel_name"]:
        result = failure(job, "infrastructure_error",
                         "batch kernel selector does not match job", [])
        return emit_failure(output, response_path, result, [], [])
    captures: list[dict] = []
    rows: list[dict] = []
    logs: list[str] = []
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    for case in job["cases"]:
        samples: list[float] = []
        for iteration in range(job["repeats"]):
            capture_id = f"case-{case}-repeat-{iteration}"
            capture_dir = work / capture_id
            item_job = {
                **job,
                "case": case,
                "iteration": iteration,
            }
            item_job.pop("cases", None)
            item_job.pop("repeats", None)
            item_job_path = work / f"{capture_id}.json"
            item_response = work / f"{capture_id}-response.json"
            write_json(item_job_path, item_job)
            command = [
                sys.executable, str(profiler), "--output", str(capture_dir),
                "--kernel-name", kernel_name, "--replay-mode", replay_mode,
                "--", sys.executable, str(runner),
                "--job", str(item_job_path), "--output", str(item_response),
            ]
            run = subprocess.run(command, text=True, capture_output=True, check=False)
            if (capture_dir / "msprof.log").is_file():
                logs.append(f"[{capture_id}]\n" + (capture_dir / "msprof.log").read_text(errors="replace"))
            try:
                remote = json.loads(item_response.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                result = failure(job, "infrastructure_error",
                                 f"{capture_id} produced no valid runner response: {exc}", captures)
                return emit_failure(output, response_path, result, captures, logs)
            if remote.get("status") != "ok":
                status = remote.get("status", "infrastructure_error")
                if status not in {"compile_error", "runtime_error", "correctness_error"}:
                    status = "infrastructure_error"
                result = failure(job, status, str(remote.get("diagnostics", "")), captures)
                return emit_failure(output, response_path, result, captures, logs)
            evidence_path = capture_dir / "evidence.json"
            try:
                evidence = json.loads(evidence_path.read_text())
                kernels = evidence["kernels"]
                if (run.returncode or evidence.get("status") != "success" or len(kernels) != 1
                        or kernels[0].get("name") != kernel_name
                        or evidence.get("protocol", {}).get("replay_mode") != replay_mode):
                    raise ValueError(evidence.get("failure", "invalid selected kernel evidence"))
                latency = float(kernels[0]["duration_us"]["median"])
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError("latency must be positive and finite")
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                result = failure(job, "infrastructure_error",
                                 f"{capture_id} msprof evidence is invalid: {exc}", captures)
                return emit_failure(output, response_path, result, captures, logs)
            captures.append({
                "case": case,
                "iteration": iteration,
                "duration_us": latency,
                "kernel_name": kernel_name,
                "replay_mode": replay_mode,
                "evidence_sha256": digest(evidence_path),
                "msprof_log_sha256": evidence["msprof_log_sha256"],
            })
            samples.append(latency)
        rows.append({"case": case, "median_us": statistics.median(samples),
                     "samples_us": samples})
    geomean = math.exp(sum(math.log(row["median_us"]) for row in rows) / len(rows))
    evidence = {
        "schema_version": 1,
        "status": "success",
        "profiler": "msprof-op",
        "timing_scope": "device-task",
        "kernel_name": kernel_name,
        "replay_mode": replay_mode,
        "captures": captures,
        "cases": rows,
        "repeats": job["repeats"],
        "geomean_us": geomean,
    }
    response = {"status": "ok", "diagnostics": "", **bound, "profile_cases": rows,
                "geomean_us": geomean, "passed": True}
    write_json(output / "evidence.json", evidence)
    (output / "msprof.log").write_text(bounded_log(logs))
    write_json(response_path, response)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--profiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--kernel-name", required=True)
    args = parser.parse_args()
    try:
        job = json.loads(args.job.read_text())
        return execute(job, runner=args.runner, profiler=args.profiler,
                       output=args.output, response_path=args.response,
                       kernel_name=args.kernel_name)
    except BaseException as exc:
        result = {"status": "infrastructure_error", "diagnostics": f"batch driver failed: {exc}"}
        if "job" in locals() and isinstance(job, dict):
            try:
                result.update(identity(job))
            except (KeyError, TypeError):
                pass
        return emit_failure(args.output, args.response, result, [], [])


if __name__ == "__main__":
    raise SystemExit(main())
