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
    else:
        expected = ALL_CASES if raw.get("scope") == "full" else spec["development_cases"]
        if raw.get("cases") != expected:
            return None, response("infrastructure_error", "check cases are not the exact configured sequence")
    return raw, None


def make_job(request: dict[str, Any], benchmark: str, candidate: Path, root: Path) -> dict[str, Any]:
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
        if action == "profile":
            job["profiling"] = {
                "driver": str((root / "scripts/profile_a3.py").resolve()),
                "tool": "msprof op",
                "captures": 1,
                "aic_metrics": "BasicInfo",
                "warm_up": 3,
                "launch_count": 1,
                "replay_mode": "kernel",
                "kernel_selector_source": "candidate-manifest",
            }
    return job


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
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--candidate", type=Path, default=Path("candidate.py"))
    parser.add_argument("--job-client", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
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
            result = invoke(args.job_client, make_job(request, args.benchmark, args.candidate, root), args.timeout)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
