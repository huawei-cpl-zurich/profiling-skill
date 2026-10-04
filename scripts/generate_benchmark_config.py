#!/usr/bin/env python3
"""Generate controller cells for the nine-cell A3 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_backend import BENCHMARKS


TREATMENTS = ("cannbot", "project-cannbot", "project-guarded")
CELL_DEVICE = {
    ("gdn", "cannbot"): 0,
    ("gdn", "project-cannbot"): 1,
    ("gdn", "project-guarded"): 2,
    ("bsa", "cannbot"): 1,
    ("bsa", "project-cannbot"): 2,
    ("bsa", "project-guarded"): 3,
    ("matmul", "cannbot"): 2,
    ("matmul", "project-cannbot"): 3,
    ("matmul", "project-guarded"): 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-client-json", required=True,
                        help="JSON string array containing the executable and exact arguments")
    parser.add_argument("--candidate", default="candidate.py")
    parser.add_argument("--candidate-manifest", default="candidate.manifest.json")
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(BENCHMARKS),
                        default=list(BENCHMARKS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backend = Path(__file__).resolve().with_name("benchmark_backend.py")
    cells = {}
    single_benchmark = len(args.benchmarks) == 1
    for benchmark in args.benchmarks:
        spec = BENCHMARKS[benchmark]
        for treatment_index, treatment in enumerate(TREATMENTS):
            cell_id = f"{benchmark}-{treatment}"
            cells[cell_id] = {
                "benchmark": benchmark,
                "treatment": treatment,
                "device": (treatment_index if single_benchmark
                           else CELL_DEVICE[(benchmark, treatment)]),
                "development_cases": spec["development_cases"],
                "all_cases": spec["all_cases"],
                "tolerances": spec["tolerances"],
                "backend": {
                    "command": [sys.executable, str(backend), "--benchmark", benchmark,
                                "--candidate", args.candidate,
                                "--candidate-manifest", args.candidate_manifest,
                                "--job-client-json", args.job_client_json],
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
