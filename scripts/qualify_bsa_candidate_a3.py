#!/usr/bin/env python3
"""Fail-fast GZ-A3 qualification for the BSA candidate profiler."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bsa_candidate_a3_executor import (  # noqa: E402
    CplRemoteTransport,
    Executor,
    ExecutorConfig,
    ExecutorError,
    QUALIFICATION_SCHEMA,
    atomic_json,
    qualify,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--campaign-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    here = Path(__file__).resolve().parents[1]
    try:
        target = os.environ.get("BSA_A3_TARGET", "")
        remote_root = os.environ.get("BSA_A3_REMOTE_ROOT", "")
        if target != "gz-a3" or not remote_root:
            raise ExecutorError(
                "infrastructure_error",
                "qualification requires BSA_A3_TARGET=gz-a3 and BSA_A3_REMOTE_ROOT",
            )
        minimum = int(os.environ.get(
            "BSA_A3_MIN_FREE_DEVICE_BYTES", str(32 * 1024**3)))
        state = Path(os.environ.get(
            "BSA_A3_STATE_DIR", str(args.output.parent / ".bsa-a3-qualification-state")))
        config = ExecutorConfig(
            target, remote_root, state,
            here / "benchmarks/bsa/baseline.py",
            here / "benchmarks/bsa/cases.jsonl",
            here / "scripts/a3_benchmark_runner.py", minimum,
            args.campaign_revision, None,
        )
        executor = Executor(config, CplRemoteTransport(), enforce_qualification=False)
        receipt = qualify(
            executor, config, args.candidate, args.kernel_name, args.campaign_revision)
    except (OSError, ValueError, ExecutorError) as exc:
        receipt = {"schema": QUALIFICATION_SCHEMA,
                   "passed": False, "failure": str(exc)}
    atomic_json(args.output, receipt)
    return 0 if receipt.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
