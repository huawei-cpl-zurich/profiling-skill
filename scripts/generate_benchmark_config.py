#!/usr/bin/env python3
"""Generate controller cells for the two pinned A3 benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_backend import ALL_CASES, BENCHMARKS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-client", nargs="+", required=True)
    parser.add_argument("--candidate", default="candidate.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backend = Path(__file__).resolve().with_name("benchmark_backend.py")
    cells = {}
    for benchmark, spec in BENCHMARKS.items():
        cells[benchmark] = {
            "benchmark": benchmark,
            "device": spec["device"],
            "development_cases": spec["development_cases"],
            "all_cases": ALL_CASES,
            "backend": {
                "command": [sys.executable, str(backend), "--benchmark", benchmark,
                            "--candidate", args.candidate, "--job-client", *args.job_client],
                "timeout_seconds": 3700,
            },
        }
    rendered = json.dumps({"schema_version": 1, "cells": cells}, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
