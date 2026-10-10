#!/usr/bin/env python3
"""Run an ordered profile matrix inside one managed A3 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
    }


def bounded_log(parts: list[str]) -> str:
    joined = "\n".join(parts)
    data = joined.encode(errors="replace")
    if len(data) <= MAX_LOG_BYTES:
        return joined
    marker = b"\n... earlier batch log truncated ...\n"
    return (marker + data[-(MAX_LOG_BYTES - len(marker)):]).decode(errors="replace")


def failure(job: dict, status: str, diagnostics: str, captures: list[dict],
            resolved_kernel_name: str | None = None) -> dict:
    result = {
        "status": status,
        "diagnostics": diagnostics,
        **identity(job),
        "declared_kernel_name": job["profiling"]["kernel_name"],
        "completed_captures": captures,
    }
    if resolved_kernel_name is not None:
        result["resolved_kernel_name"] = resolved_kernel_name
    return result


def emit_failure(output: Path, response_path: Path, result: dict,
                 captures: list[dict], logs: list[str]) -> int:
    write_json(response_path, result)
    write_json(output / "evidence.json", {
        "schema_version": 1,
        "status": "failure",
        "failure": {"kind": result["status"], "message": result["diagnostics"]},
        "declared_kernel_name": result.get("declared_kernel_name"),
        "resolved_kernel_name": result.get("resolved_kernel_name"),
        "captures": captures,
    })
    (output / "msprof.log").write_text(bounded_log(logs))
    return 1


def fallback_selector(selector: str) -> str | None:
    fallback = re.sub(r"_mix_ai[cv]$", "", selector)
    return fallback if fallback != selector and fallback else None


def read_capture(evidence_path: Path, run: subprocess.CompletedProcess[str],
                 selector: str) -> tuple[str, float | str]:
    """Return (success, latency), (selector_miss, message), or (error, message)."""
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "error", f"msprof evidence is not valid JSON: {exc}"
    if not isinstance(evidence, dict):
        return "error", "msprof evidence is not an object"
    if evidence.get("status") != "success":
        failure = evidence.get("failure")
        if not isinstance(failure, dict):
            return "error", "msprof failure evidence is malformed"
        message = str(failure.get("message", "msprof capture failed"))
        clean_miss = (
            failure.get("kind") == "profiling"
            and failure.get("reason") == "selector_miss"
            and failure.get("kernel_selector") == selector
            and failure.get("msprof_returncode") == 0
        )
        return ("selector_miss" if clean_miss else "error"), message
    if run.returncode:
        return "error", f"profiler exited with status {run.returncode} despite success evidence"
    try:
        kernels = evidence["kernels"]
        if len(kernels) != 1 or kernels[0].get("name") != selector:
            raise ValueError("selected kernel identity mismatch")
        latency = float(kernels[0]["duration_us"]["median"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("latency must be positive and finite")
        if not isinstance(evidence.get("msprof_log_sha256"), str):
            raise ValueError("msprof log digest is missing")
    except (KeyError, TypeError, ValueError) as exc:
        return "error", f"msprof success evidence is invalid: {exc}"
    return "success", latency


def execute(job: dict, *, runner: Path, profiler: Path, output: Path,
            response_path: Path, kernel_name: str) -> int:
    bound = identity(job)
    if kernel_name != bound["kernel_name"]:
        result = failure(job, "infrastructure_error",
                         "batch kernel selector does not match job", [])
        return emit_failure(output, response_path, result, [], [])
    captures: list[dict] = []
    rows: list[dict] = []
    logs: list[str] = []
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    declared_kernel_name = kernel_name
    resolved_kernel_name: str | None = None
    for case in job["cases"]:
        samples: list[float] = []
        for iteration in range(job["repeats"]):
            capture_id = f"case-{case}-repeat-{iteration}"
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
            selectors = [resolved_kernel_name or declared_kernel_name]
            fallback = fallback_selector(declared_kernel_name) if resolved_kernel_name is None else None
            if fallback:
                selectors.append(fallback)
            for selector_index, selector in enumerate(selectors):
                suffix = "" if selector_index == 0 else "-selector-fallback"
                capture_dir = work / f"{capture_id}{suffix}"
                command = [
                    sys.executable, str(profiler), "--output", str(capture_dir),
                    "--kernel-name", selector, "--target-family",
                    ("Ascend-A5" if job.get("product") == "a5" else "Ascend-A2-A3"),
                    "--", sys.executable, str(runner),
                    "--job", str(item_job_path), "--output", str(item_response),
                ]
                run = subprocess.run(command, text=True, capture_output=True, check=False)
                if (capture_dir / "msprof.log").is_file():
                    logs.append(f"[{capture_id} selector={selector}]\n" +
                                (capture_dir / "msprof.log").read_text(errors="replace"))
                try:
                    remote = json.loads(item_response.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    result = failure(
                        job, "infrastructure_error",
                        f"{capture_id} produced no valid runner response: {exc}", captures,
                        resolved_kernel_name)
                    return emit_failure(output, response_path, result, captures, logs)
                if remote.get("status") != "ok":
                    status = remote.get("status", "infrastructure_error")
                    if status not in {"compile_error", "runtime_error", "correctness_error"}:
                        status = "infrastructure_error"
                    result = failure(job, status, str(remote.get("diagnostics", "")), captures,
                                     resolved_kernel_name)
                    return emit_failure(output, response_path, result, captures, logs)
                capture_status, detail = read_capture(capture_dir / "evidence.json", run, selector)
                if capture_status == "success":
                    latency = float(detail)
                    resolved_kernel_name = selector
                    evidence_path = capture_dir / "evidence.json"
                    evidence = json.loads(evidence_path.read_text())
                    break
                if capture_status == "selector_miss" and selector_index + 1 < len(selectors):
                    continue
                status = ("submission_error" if capture_status == "selector_miss"
                          else "infrastructure_error")
                result = failure(
                    job, status, f"{capture_id} msprof selector {selector!r}: {detail}",
                    captures, resolved_kernel_name)
                return emit_failure(output, response_path, result, captures, logs)
            captures.append({
                "case": case,
                "iteration": iteration,
                "duration_us": latency,
                "kernel_name": resolved_kernel_name,
                "declared_kernel_name": declared_kernel_name,
                "resolved_kernel_name": resolved_kernel_name,
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
        "target_family": ("Ascend-A5" if job.get("product") == "a5"
                          else "Ascend-A2-A3"),
        "profiler": "msprof-op",
        "timing_scope": "device-task",
        "kernel_name": declared_kernel_name,
        "declared_kernel_name": declared_kernel_name,
        "resolved_kernel_name": resolved_kernel_name,
        "captures": captures,
        "cases": rows,
        "repeats": job["repeats"],
        "geomean_us": geomean,
    }
    response = {"status": "ok", "diagnostics": "", **bound,
                "declared_kernel_name": declared_kernel_name,
                "resolved_kernel_name": resolved_kernel_name,
                "profile_cases": rows, "geomean_us": geomean, "passed": True}
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
