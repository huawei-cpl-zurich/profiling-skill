#!/usr/bin/env python3
"""Run one A2/A3 msprof-op capture and emit compact latency evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
from pathlib import Path


SUCCESS_LINE = "Profiling running finished. All task success."
NAME_FIELDS = ("Op Name", "OpName", "Kernel Name", "kernel_name")
DURATION_FIELDS = ("Task Duration(us)", "Task Duration (us)", "task_duration_us")


class InvalidCapture(RuntimeError):
    pass


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def field(fields: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in fields), None)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(
    root: Path,
    kernel: str | None,
    *,
    warm_up: int,
    launch_count: int,
) -> dict:
    files = sorted(root.rglob("OpBasicInfo*.csv"))
    if not files:
        raise InvalidCapture(f"no OpBasicInfo CSV under {root}")

    rows: list[dict] = []
    sources: list[dict] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            name_field = field(fields, NAME_FIELDS)
            duration_field = field(fields, DURATION_FIELDS)
            if not name_field or not duration_field:
                continue
            accepted = 0
            for raw in reader:
                name = (raw.get(name_field) or "").strip()
                if kernel and name != kernel:
                    continue
                try:
                    duration = float((raw.get(duration_field) or "").strip())
                except ValueError as exc:
                    raise InvalidCapture(
                        f"non-numeric {duration_field!r} in {path}: {raw.get(duration_field)!r}"
                    ) from exc
                if duration < 0:
                    raise InvalidCapture(f"negative duration in {path}: {duration}")
                rows.append({"kernel_name": name, "duration_us": duration})
                accepted += 1
            sources.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "matched_rows": accepted,
                }
            )
    if not rows:
        selection = f" matching {kernel!r}" if kernel else ""
        raise InvalidCapture(f"no numeric operator rows{selection}")

    by_kernel: dict[str, list[float]] = {}
    for row in rows:
        by_kernel.setdefault(row["kernel_name"], []).append(row["duration_us"])
    kernels = []
    for name, values in sorted(by_kernel.items()):
        kernels.append(
            {
                "name": name,
                "samples": len(values),
                "duration_us": {
                    "min": min(values),
                    "median": statistics.median(values),
                    "p90": percentile(values, 0.9),
                    "max": max(values),
                },
                "sample_values_us": values,
            }
        )
    return {
        "schema_version": 1,
        "target_family": "Ascend-A2-A3",
        "profiler": "msprof-op",
        "metric": "BasicInfo",
        "timing_scope": "device-task",
        "units": {"duration": "us"},
        "protocol": {
            "warm_up": warm_up,
            "launch_count": launch_count,
            "replay_mode": "kernel",
            "kernel_selector": kernel,
        },
        "kernels": kernels,
        "sources": sources,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name")
    parser.add_argument("--warm-up", type=int, default=3)
    parser.add_argument("--launch-count", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--msprof", default="msprof", help=argparse.SUPPRESS)
    parser.add_argument("application", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.warm_up < 0:
        parser.error("--warm-up must be non-negative")
    if args.application[:1] == ["--"]:
        args.application = args.application[1:]
    if not args.application:
        parser.error("provide an application after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        print(f"error: output directory is not empty: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    log = output / "msprof.log"
    evidence_path = output / "evidence.json"

    application = shlex.join(args.application)
    command = [
        args.msprof,
        "op",
        f"--application={application}",
        f"--output={raw}",
        "--aic-metrics=BasicInfo",
        f"--warm-up={args.warm_up}",
        f"--launch-count={args.launch_count}",
        "--replay-mode=kernel",
    ]
    if args.kernel_name:
        command.append(f"--kernel-name={args.kernel_name}")

    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        transcript, _ = process.communicate(timeout=args.timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            remainder, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            remainder, _ = process.communicate()
        transcript = (exc.stdout or "") + (remainder or "") + "\nprofile timed out\n"
        returncode = 124
    log.write_text(transcript)

    try:
        if returncode != 0:
            raise InvalidCapture(f"msprof exited with status {returncode}")
        if SUCCESS_LINE not in transcript:
            raise InvalidCapture("msprof terminal success marker is missing")
        evidence = summarize(
            raw,
            args.kernel_name,
            warm_up=args.warm_up,
            launch_count=args.launch_count,
        )
        evidence["status"] = "success"
        evidence["application"] = args.application
        evidence["msprof_log_sha256"] = sha256(log)
    except InvalidCapture as exc:
        evidence = {
            "schema_version": 1,
            "target_family": "Ascend-A2-A3",
            "profiler": "msprof-op",
            "status": "failure",
            "failure": {
                "kind": "profiling",
                "message": str(exc),
                "msprof_returncode": returncode,
            },
            "application": args.application,
            "msprof_log": str(log),
        }
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(json.dumps(evidence, sort_keys=True))
        return 1

    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
