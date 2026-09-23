#!/usr/bin/env python3
"""Summarize A5 SAMPLE_PMU_TIMELINE evidence without moving its database."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import io
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED = {
    "deviceId",
    "timestampNs",
    "totalCycle",
    "usage",
    "freq",
    "coreId",
    "coreType",
}


class InvalidProfile(ValueError):
    pass


def database(root: Path) -> tuple[Path, sqlite3.Connection]:
    matches = []
    for path in sorted(root.glob("**/msprof*.db")):
        with sqlite3.connect(path) as conn:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='SAMPLE_PMU_TIMELINE'"
            ).fetchone():
                matches.append(path)
    if not matches:
        raise InvalidProfile(f"no database with SAMPLE_PMU_TIMELINE under {root}")
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(root)) for path in matches)
        raise InvalidProfile(f"multiple sampled-PMU databases under {root}: {names}")
    return matches[0], sqlite3.connect(matches[0])


def strings(conn: sqlite3.Connection) -> dict[int, str]:
    try:
        return {
            int(key): str(value)
            for key, value in conn.execute("SELECT id,value FROM STRING_IDS")
        }
    except sqlite3.Error:
        return {}


def task_window(conn: sqlite3.Connection, names: dict[int, str], pattern: str | None):
    if not pattern:
        return None
    matcher = pattern
    try:
        raw = list(conn.execute(
            "SELECT t.startNs,t.endNs,c.name FROM TASK t "
            "JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId"
        ))
    except sqlite3.Error as exc:
        raise InvalidProfile(
            f"cannot resolve requested task filter {pattern!r} from profiler metadata"
        ) from exc
    matched = [
        (int(start), int(end))
        for start, end, name in raw
        if fnmatch.fnmatchcase(names.get(int(name), str(name)), pattern)
    ]
    if not matched:
        contained = f"*{pattern.strip('*')}*"
        matcher = contained
        matched = [
            (int(start), int(end))
            for start, end, name in raw
            if fnmatch.fnmatchcase(names.get(int(name), str(name)), contained)
        ]
    if not matched:
        raise InvalidProfile(f"no task matches {pattern!r}")
    envelope = min(x[0] for x in matched), max(x[1] for x in matched)
    for start, end, name in raw:
        task_name = names.get(int(name), str(name))
        if fnmatch.fnmatchcase(task_name, matcher):
            continue
        if max(int(start), envelope[0]) < min(int(end), envelope[1]):
            raise InvalidProfile(
                f"task filter {pattern!r} encloses unrelated task {task_name!r}"
            )
    return envelope[0], envelope[1], len(matched)


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def domain(rows: list[dict]) -> dict:
    active = [row for row in rows if row["usage"] > 0]
    if not active:
        return {"state": "inactive", "samples": len(rows), "active_samples": 0}
    first, last = (
        min(x["timestamp_ns"] for x in active),
        max(x["timestamp_ns"] for x in active),
    )
    bounded = [x for x in rows if first <= x["timestamp_ns"] <= last]
    usage = [x["usage"] for x in bounded]
    freq = [x["frequency_mhz"] for x in bounded]
    by_core: dict[int, int] = defaultdict(int)
    for row in active:
        by_core[row["core_id"]] += row["cycles"]
    core_values = list(by_core.values())
    cv = (
        statistics.pstdev(core_values) / statistics.mean(core_values)
        if len(core_values) > 1 and statistics.mean(core_values)
        else 0.0
    )
    span = max(1, last - first + 1)
    bins = []
    for index in range(32):
        low, high = first + span * index // 32, first + span * (index + 1) // 32
        values = [x["usage"] for x in bounded if low <= x["timestamp_ns"] < high]
        bins.append(statistics.mean(values) if values else 0.0)
    return {
        "state": "ok",
        "samples": len(rows),
        "active_samples": len(active),
        "active_window_ns": last - first,
        "active_cores": len(by_core),
        "cycles": sum(x["cycles"] for x in active),
        "utilization": {
            "mean": statistics.mean(usage),
            "median": statistics.median(usage),
            "p95": percentile(usage, 0.95),
            "idle_fraction": sum(x == 0 for x in usage) / len(usage),
        },
        "frequency_mhz": {
            "min": min(freq),
            "median": statistics.median(freq),
            "max": max(freq),
        },
        "core_cycle_cv": cv,
        "cycles_by_core": dict(sorted(by_core.items())),
        "utilization_bins": bins,
    }


def summarize(root: Path, pattern: str | None) -> tuple[dict, list[dict]]:
    db, conn = database(root)
    try:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(SAMPLE_PMU_TIMELINE)")
        }
        if missing := REQUIRED - columns:
            raise InvalidProfile(
                f"SAMPLE_PMU_TIMELINE missing columns: {sorted(missing)}"
            )
        names = strings(conn)
        window = task_window(conn, names, pattern)
        raw = conn.execute(
            "SELECT deviceId,timestampNs,totalCycle,usage,freq,coreId,coreType "
            "FROM SAMPLE_PMU_TIMELINE ORDER BY timestampNs,coreType,coreId"
        ).fetchall()
        rows = [
            {
                "device_id": int(x[0]),
                "timestamp_ns": int(x[1]),
                "cycles": int(x[2]),
                "usage": float(x[3]),
                "frequency_mhz": float(x[4]),
                "core_id": int(x[5]),
                "core_type": names.get(int(x[6]), str(x[6])),
            }
            for x in raw
        ]
        if window:
            rows = [x for x in rows if window[0] <= x["timestamp_ns"] < window[1]]
        if not rows:
            raise InvalidProfile("no sampled rows remain")
        if len({row["device_id"] for row in rows}) > 1:
            raise InvalidProfile("sampled-PMU capture contains multiple devices")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["core_type"]].append(row)
        summary_rows = []
        if not pattern:
            try:
                for device, metric, value, core, kind in conn.execute(
                    "SELECT deviceId,metric,value,coreId,coreType FROM SAMPLE_PMU_SUMMARY"
                ):
                    summary_rows.append(
                        {
                            "device_id": int(device),
                            "metric": names.get(int(metric), str(metric)),
                            "value": float(value),
                            "core_id": int(core),
                            "core_type": names.get(int(kind), str(kind)),
                        }
                    )
            except sqlite3.Error:
                pass
        return {
            "schema_version": 1,
            "target": "Ascend950-A5",
            "database": str(db.relative_to(root)),
            "task_pattern": pattern,
            "task_window": window,
            "domains": {key: domain(value) for key, value in sorted(grouped.items())},
            "sample_pmu_summary": summary_rows,
        }, rows
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-pattern")
    args = parser.parse_args()
    result, rows = summarize(args.input, args.task_pattern)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sample-pmu-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# A5 sampled PMU summary", ""]
    for name, values in result["domains"].items():
        lines.append(
            f"- {name}: {values['state']}, {values['samples']} samples, {values.get('active_cores', 0)} active cores"
        )
    (args.output / "sample-pmu-summary.md").write_text("\n".join(lines) + "\n")
    with (args.output / "sample-pmu-rows.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
