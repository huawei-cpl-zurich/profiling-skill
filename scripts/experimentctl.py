#!/usr/bin/env python3
"""Deterministic controller for isolated kernel experiments.

Backends receive one JSON request on stdin and must emit one JSON object on
stdout.  The controller owns the physical device field; callers cannot
override it.  Backend stderr is retained as diagnostic evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


EXIT = {"ok": 0, "candidate_error": 2, "infrastructure_error": 3, "config_error": 4}
CANDIDATE_STATUSES = {"compile_error", "runtime_error", "correctness_error", "candidate_error"}
VALID_STATUSES = CANDIDATE_STATUSES | {"ok", "infrastructure_error"}


class ConfigError(Exception):
    pass


def failure(kind: str, diagnostics: str, **extra: Any) -> dict[str, Any]:
    return {"status": kind, "diagnostics": diagnostics, "handles": [], **extra}


def read_cell(path: Path, cell_id: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
        cell = document["cells"][cell_id]
        device = cell["device"]
        command = cell["backend"]["command"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ConfigError(f"cannot load cell {cell_id!r}: {error}") from error
    if not isinstance(device, int) or device < 0:
        raise ConfigError("cell device must be a non-negative integer")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ConfigError("backend.command must be a non-empty string array")
    for field in ("development_cases", "all_cases"):
        values = cell.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(x, int) and x >= 0 for x in values):
            raise ConfigError(f"{field} must be a non-empty array of case indices")
    return {**cell, "id": cell_id}


def invoke(cell: dict[str, Any], action: str, **payload: Any) -> dict[str, Any]:
    request = {
        "protocol_version": 1,
        "action": action,
        "cell": cell["id"],
        "benchmark": cell.get("benchmark"),
        "device": cell["device"],
        **payload,
    }
    timeout = cell.get("backend", {}).get("timeout_seconds", 900)
    try:
        run = subprocess.run(
            cell["backend"]["command"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return failure("infrastructure_error", f"backend transport failed: {error}")
    try:
        response = json.loads(run.stdout)
        if not isinstance(response, dict):
            raise ValueError("response is not an object")
    except (json.JSONDecodeError, ValueError) as error:
        detail = f"invalid backend response: {error}; stdout={run.stdout!r}"
        if run.stderr:
            detail += f"; stderr={run.stderr}"
        return failure("infrastructure_error", detail)
    status = response.get("status")
    if status not in VALID_STATUSES:
        return failure("infrastructure_error", f"backend returned invalid status {status!r}")
    response["status"] = "candidate_error" if status in CANDIDATE_STATUSES else status
    response["failure_type"] = status if status in CANDIDATE_STATUSES else None
    diagnostics = str(response.get("diagnostics", ""))
    if run.stderr:
        diagnostics = f"{diagnostics}\n{run.stderr}".strip()
    response["diagnostics"] = diagnostics
    response["handles"] = [str(response["handle"])] if response.get("handle") else []
    if run.returncode and response["status"] == "ok":
        return failure(
            "infrastructure_error",
            f"backend exited {run.returncode} after reporting success\n{diagnostics}".strip(),
        )
    if response.get("device", cell["device"]) != cell["device"]:
        return failure("infrastructure_error", "backend result did not match hard-bound device")
    return response


def merge_failure(result: dict[str, Any], operation: str, cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": operation,
        "cell": cell["id"],
        "benchmark": cell.get("benchmark"),
        "device": cell["device"],
        **result,
    }


def rank(cell: dict[str, Any], warmups: int, repeats: int) -> dict[str, Any]:
    rows, handles = [], []
    for case in cell["all_cases"]:
        for index in range(warmups):
            result = invoke(cell, "measure", case=case, phase="warmup", iteration=index)
            handles += result["handles"]
            if result["status"] != "ok":
                result["handles"] = handles
                return merge_failure(result, "rank", cell)
        samples = []
        for index in range(repeats):
            result = invoke(cell, "measure", case=case, phase="sample", iteration=index)
            handles += result["handles"]
            if result["status"] != "ok":
                result["handles"] = handles
                return merge_failure(result, "rank", cell)
            try:
                latency = float(result["latency_us"])
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return merge_failure(
                    failure("infrastructure_error", "backend returned invalid latency", handles=handles),
                    "rank",
                    cell,
                )
            samples.append(latency)
        rows.append({"case": case, "median_us": statistics.median(samples), "samples_us": samples})
    rows.sort(key=lambda row: (-row["median_us"], row["case"]))
    return merge_failure(
        {"status": "ok", "diagnostics": "", "handles": handles, "warmups": warmups, "repeats": repeats, "ranking": rows},
        "rank",
        cell,
    )


def check(cell: dict[str, Any], scope: str, round_number: int | None) -> dict[str, Any]:
    cases = cell["all_cases" if scope == "full" else "development_cases"]
    result = invoke(cell, "check", cases=cases, scope=scope, round=round_number)
    result.setdefault("cases", cases)
    return merge_failure(result, "check", cell)


def profile(cell: dict[str, Any], repeats: int, round_number: int | None) -> dict[str, Any]:
    rows, handles = [], []
    for case in cell["development_cases"]:
        samples = []
        for index in range(repeats):
            result = invoke(cell, "profile", case=case, iteration=index, round=round_number)
            handles += result["handles"]
            if result["status"] != "ok":
                result["handles"] = handles
                return merge_failure(result, "profile", cell)
            try:
                latency = float(result["latency_us"])
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return merge_failure(
                    failure("infrastructure_error", "backend returned invalid latency", handles=handles),
                    "profile",
                    cell,
                )
            samples.append(latency)
        rows.append({"case": case, "median_us": statistics.median(samples), "samples_us": samples})
    score = math.exp(sum(math.log(row["median_us"]) for row in rows) / len(rows))
    return merge_failure(
        {"status": "ok", "diagnostics": "", "handles": handles, "repeats": repeats, "cases": rows, "geomean_us": score},
        "profile",
        cell,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", required=True, type=Path)
    root.add_argument("--cell", required=True)
    commands = root.add_subparsers(dest="command", required=True)
    rank_parser = commands.add_parser("rank")
    rank_parser.add_argument("--benchmark")
    rank_parser.add_argument("--warmups", type=int, default=3)
    rank_parser.add_argument("--repeats", type=int, default=7)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--scope", choices=("development", "full"), default="development")
    check_parser.add_argument("--round", type=int)
    profile_parser = commands.add_parser("profile")
    profile_parser.add_argument("--repeats", type=int, default=3)
    profile_parser.add_argument("--round", type=int)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        cell = read_cell(args.config, args.cell)
        if getattr(args, "benchmark", None) and args.benchmark != cell.get("benchmark"):
            raise ConfigError("requested benchmark does not match the cell")
        repeats = getattr(args, "repeats", 1)
        if repeats < 1 or getattr(args, "warmups", 0) < 0:
            raise ConfigError("repeat counts must be positive and warmups non-negative")
        if args.command == "rank":
            output = rank(cell, args.warmups, args.repeats)
        elif args.command == "check":
            output = check(cell, args.scope, args.round)
        else:
            output = profile(cell, args.repeats, args.round)
    except ConfigError as error:
        output = failure("config_error", str(error))
    print(json.dumps(output, sort_keys=True))
    return EXIT[output["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
