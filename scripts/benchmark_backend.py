#!/usr/bin/env python3
"""Translate experiment-controller requests into pinned benchmark jobs.

The injected job client accepts one JSON object on stdin and returns one JSON
object on stdout.  A production client is responsible for submitting that job
through the checked-in gz-a3 execution profile; this adapter deliberately has
no SSH, container, or device-discovery fallback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PINNED_REVISION = "a42c54b916189500e2f7cb47640980f230f2eb65"
BENCHMARKS = {
    "gdn": {
        "device": 0,
        "development_cases": [40, 49, 47, 46, 45],
        "asset": "benchmarks/gdn/baseline.py",
        "cases": "benchmarks/gdn/cases.jsonl",
    },
    "bsa": {
        "device": 1,
        "development_cases": [47, 46, 49, 44, 43],
        "asset": "benchmarks/bsa/baseline.py",
        "cases": "benchmarks/bsa/cases.jsonl",
    },
}
ALL_CASES = list(range(50))
VALID_ACTIONS = {"measure", "check", "profile"}
VALID_RESULTS = {"ok", "compile_error", "runtime_error", "correctness_error", "infrastructure_error"}
MANIFEST_SCHEMA = "profiling-skill/candidate-kernel/v1"


def response(status: str, diagnostics: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "diagnostics": diagnostics, **values}


def validate_request(raw: Any, benchmark: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    spec = BENCHMARKS[benchmark]
    if not isinstance(raw, dict):
        return None, response("infrastructure_error", "controller request is not an object")
    if raw.get("protocol_version") != 1 or raw.get("action") not in VALID_ACTIONS:
        return None, response("infrastructure_error", "unsupported controller protocol or action")
    if raw.get("benchmark") != benchmark:
        return None, response("infrastructure_error", "request benchmark does not match adapter")
    if raw.get("device") != spec["device"]:
        return None, response("infrastructure_error", "request violates benchmark device binding")
    action = raw["action"]
    if action in {"measure", "profile"}:
        case = raw.get("case")
        allowed = ALL_CASES if action == "measure" else spec["development_cases"]
        if isinstance(case, bool) or case not in allowed:
            return None, response("infrastructure_error", f"invalid {action} case")
        if action == "measure" and raw.get("phase") not in {"warmup", "sample"}:
            return None, response("infrastructure_error", "measure phase must be warmup or sample")
        if action == "profile" and (
            isinstance(raw.get("round"), bool)
            or not isinstance(raw.get("round"), int)
            or raw["round"] < 1
        ):
            return None, response("infrastructure_error", "profile round must be a positive integer")
    else:
        expected = ALL_CASES if raw.get("scope") == "full" else spec["development_cases"]
        if raw.get("cases") != expected:
            return None, response("infrastructure_error", "check cases are not the exact configured sequence")
    return raw, None


def load_kernel_selector(path: Path) -> tuple[str | None, dict[str, Any] | None]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, response("compile_error", f"cannot load candidate kernel manifest {path}: {error}")
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        return None, response("compile_error", f"candidate kernel manifest requires schema {MANIFEST_SCHEMA!r}")
    kernel_name = manifest.get("kernel_name")
    if not isinstance(kernel_name, str) or not kernel_name.strip() or kernel_name != kernel_name.strip():
        return None, response("compile_error", "candidate kernel manifest requires a non-empty trimmed kernel_name")
    return kernel_name, None


def make_job(request: dict[str, Any], benchmark: str, candidate: Path, root: Path,
             kernel_name: str | None = None) -> dict[str, Any]:
    spec = BENCHMARKS[benchmark]
    action = request["action"]
    job = {
        "protocol_version": 1,
        "profile": "gz-a3",
        "runtime": "py311-torch",
        "benchmark": benchmark,
        "device": spec["device"],
        "logical_device": 0,
        "action": action,
        "candidate": str(candidate.resolve()),
        "baseline": str((root / spec["asset"]).resolve()),
        "case_spec": str((root / spec["cases"]).resolve()),
        "reference_revision": PINNED_REVISION,
    }
    if action == "check":
        job.update(cases=request["cases"], scope=request.get("scope"), round=request.get("round"))
    else:
        job.update(case=request["case"], iteration=request.get("iteration"))
        if action == "measure":
            job["phase"] = request["phase"]
        if action == "profile":
            job["round"] = request["round"]
            if kernel_name is None:
                raise ValueError("profile job requires a kernel selector")
            job["profiling"] = {
                "driver": str((root / "scripts/profile_a3.py").resolve()),
                "tool": "msprof op",
                "captures": 1,
                "aic_metrics": "BasicInfo",
                "warm_up": 3,
                "launch_count": 1,
                "replay_mode": "kernel",
                "kernel_name": kernel_name,
                "driver_arguments": ["--kernel-name", kernel_name],
            }
    return job


def identity_fields(job: dict[str, Any]) -> dict[str, Any]:
    fields = {name: job[name] for name in ("benchmark", "action", "device")}
    if job["action"] == "check":
        fields.update(cases=job["cases"], scope=job["scope"])
    else:
        fields["case"] = job["case"]
    if job["action"] == "measure":
        fields["phase"] = job["phase"]
    if job["action"] == "profile":
        fields["round"] = job["round"]
        fields["kernel_name"] = job["profiling"]["kernel_name"]
    return fields


def invoke(command: list[str], job: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        run = subprocess.run(command, input=json.dumps(job), text=True, capture_output=True,
                             timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        detail = f"job client timed out after {timeout} seconds"
        if error.stderr:
            detail += f"\n{error.stderr}"
        return response("infrastructure_error", detail)
    except OSError as error:
        return response("infrastructure_error", f"job client could not start: {error}")
    try:
        result = json.loads(run.stdout)
        if not isinstance(result, dict):
            raise ValueError("response is not an object")
    except (json.JSONDecodeError, ValueError) as error:
        return response("infrastructure_error", f"invalid job client response: {error}; stdout={run.stdout!r}; stderr={run.stderr}")
    diagnostics = str(result.get("diagnostics", ""))
    if run.stderr:
        diagnostics = f"{diagnostics}\n{run.stderr}".strip()
    status = result.get("status")
    if status not in VALID_RESULTS:
        return response("infrastructure_error", f"job client returned invalid status {status!r}\n{diagnostics}".strip(),
                        handle=result.get("handle"))
    result["diagnostics"] = diagnostics
    if run.returncode and status == "ok":
        return response("infrastructure_error", f"job client exited {run.returncode} after success\n{diagnostics}".strip(),
                        handle=result.get("handle"))
    mismatches = [name for name, expected in identity_fields(job).items() if result.get(name) != expected]
    if mismatches:
        return response(
            "infrastructure_error",
            f"job result identity mismatch for {', '.join(mismatches)}\n{diagnostics}".strip(),
            handle=result.get("handle"),
            evidence=result,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--candidate", type=Path, default=Path("candidate.py"))
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--job-client-json", required=True,
                        help="JSON string array containing the executable and exact arguments")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def parse_command(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid --job-client-json: {error}") from error
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("--job-client-json must be a non-empty JSON string array")
    return command


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        command = parse_command(args.job_client_json)
    except ValueError as error:
        result = response("infrastructure_error", str(error))
    else:
        try:
            raw = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            result = response("infrastructure_error", f"invalid controller JSON: {error}")
        else:
            request, error = validate_request(raw, args.benchmark)
            if error:
                result = error
            elif not args.candidate.is_file():
                result = response("compile_error", f"candidate source does not exist: {args.candidate}")
            else:
                kernel_name = None
                if request["action"] == "profile":
                    manifest = args.candidate_manifest or args.candidate.with_suffix(".manifest.json")
                    kernel_name, result = load_kernel_selector(manifest)
                if kernel_name is not None or request["action"] != "profile":
                    result = invoke(
                        command,
                        make_job(request, args.benchmark, args.candidate, root, kernel_name),
                        args.timeout,
                    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
