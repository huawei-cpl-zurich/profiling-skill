#!/usr/bin/env python3
"""Normalize A5 PipeTimeline and per-pipe InstrTimeline JSON traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class InvalidTimeline(ValueError):
    pass


TIME_UNITS_NS = {"ns": Decimal(1), "us": Decimal(1000), "ms": Decimal(1_000_000)}


def trace_payload(path: Path) -> tuple[list[Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidTimeline(f"cannot read timeline {path}: {error}") from error
    raw = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise InvalidTimeline(f"trace event list expected in {path}")
    metadata = payload if isinstance(payload, dict) else {}
    return raw, metadata


def exact_ns(value: Any, unit: str, location: str) -> int:
    try:
        converted = Decimal(str(value)) * TIME_UNITS_NS[unit]
    except (InvalidOperation, ValueError) as error:
        raise InvalidTimeline(f"invalid {location}: {value!r}") from error
    if not converted.is_finite() or converted != converted.to_integral_value():
        raise InvalidTimeline(f"{location} does not resolve to an integer nanosecond")
    result = int(converted)
    if result < 0:
        raise InvalidTimeline(f"{location} must be non-negative")
    return result


def events(path: Path, source: str, forced_pipe: str | None = None) -> list[dict]:
    raw, metadata = trace_payload(path)
    unit = str(metadata.get("displayTimeUnit", "us")).lower()
    if unit not in TIME_UNITS_NS:
        raise InvalidTimeline(f"unsupported displayTimeUnit {unit!r} in {path}")
    out = []
    for event in raw:
        if not isinstance(event, dict) or event.get("ph") not in (None, "X"):
            continue
        if "ts" not in event or "dur" not in event:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        pipe = (
            forced_pipe
            or str(
                args.get("pipe")
                or event.get("cat")
                or event.get("tid")
                or event.get("name")
                or "unknown"
            ).lower()
        )
        start_ns = exact_ns(event["ts"], unit, f"event timestamp in {path}")
        duration_ns = exact_ns(event["dur"], unit, f"event duration in {path}")
        if duration_ns == 0:
            continue
        out.append(
            {
                "source": source,
                "pipe": pipe,
                "core": str(event.get("pid", args.get("core_id", ""))),
                "sub_core": str(event.get("tid", args.get("sub_core_id", ""))),
                "name": str(event.get("name", "")),
                "pc": str(args.get("pc_addr", args.get("pc", ""))),
                "start": float(event["ts"]),
                "duration": float(event["dur"]),
                "start_ns": start_ns,
                "end_ns": start_ns + duration_ns,
            }
        )
    if not out:
        raise InvalidTimeline(f"no complete-duration events in {path}")
    return out


def common_clock_phases(rows: list[dict]) -> list[dict[str, Any]]:
    """Partition the observed window whenever its active-pipe set changes."""
    boundaries = sorted(
        {point for row in rows for point in (row["start_ns"], row["end_ns"])}
    )
    phases = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = sorted(
            {
                row["pipe"]
                for row in rows
                if row["start_ns"] < end and row["end_ns"] > start
            }
        )
        phases.append(
            {
                "phase_id": f"phase-{len(phases):04d}",
                "start_ns": start,
                "end_ns": end,
                "metrics": {},
                "activity": {"pipes": active},
                "capacity_join": {
                    "state": "unavailable",
                    "reason": "no exact-window capacity evidence",
                },
            }
        )
    return phases


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidTimeline(f"{location} must be an object")
    return value


def join_capacity(
    phases: list[dict[str, Any]], capacity_path: Path, timeline_hash: str, complete: bool
) -> tuple[str, dict[str, Any]]:
    try:
        capacity = require_object(json.loads(capacity_path.read_text()), "capacity evidence")
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidTimeline(
            f"cannot read capacity evidence {capacity_path}: {error}"
        ) from error
    if capacity.get("schema_version") != 1:
        raise InvalidTimeline("capacity evidence schema_version must be integer 1")
    if (
        capacity.get("product") != "a5"
        or capacity.get("target_product") != "Ascend950/V6"
    ):
        raise InvalidTimeline("capacity evidence product must be a5 / Ascend950/V6")
    if capacity.get("clock") != "pipe-timeline-common-clock-ns":
        raise InvalidTimeline("capacity evidence clock is incompatible with PipeTimeline")
    provenance = require_object(capacity.get("provenance"), "capacity evidence provenance")
    capture_id = provenance.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise InvalidTimeline("capacity evidence provenance.capture_id must be non-empty")
    if provenance.get("timeline_source_sha256") != timeline_hash:
        raise InvalidTimeline("capacity evidence timeline_source_sha256 does not match capture")
    raw_windows = capacity.get("phases")
    if not isinstance(raw_windows, list):
        raise InvalidTimeline("capacity evidence phases must be an array")
    windows: dict[tuple[int, int], dict[str, Any]] = {}
    for index, raw in enumerate(raw_windows):
        window = require_object(raw, f"capacity evidence phases[{index}]")
        start, end = window.get("start_ns"), window.get("end_ns")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise InvalidTimeline(f"capacity evidence phases[{index}] has invalid boundaries")
        key = (start, end)
        if key in windows:
            raise InvalidTimeline(f"duplicate capacity evidence window {start}:{end}")
        windows[key] = require_object(
            window.get("metrics"), f"capacity evidence phases[{index}].metrics"
        )

    for phase in phases:
        if not complete:
            phase["capacity_join"] = {
                "state": "incompatible",
                "reason": "common-clock timeline is marked truncated",
            }
            continue
        matched = windows.get((phase["start_ns"], phase["end_ns"]))
        if matched is not None:
            phase["metrics"] = matched
            phase["capacity_join"] = {"state": "matched-exact-window"}
    return capture_id, provenance


def phase_evidence(path: Path, rows: list[dict], capacity_path: Path | None) -> dict:
    _, metadata = trace_payload(path)
    trace_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    trace_metadata = metadata.get("metadata")
    truncated = (
        bool(trace_metadata.get("truncated", False))
        if isinstance(trace_metadata, dict)
        else False
    )
    phases = common_clock_phases(rows)
    capture_id = f"pipe-timeline:{trace_hash[:16]}"
    capacity_provenance = None
    if capacity_path is not None:
        capture_id, capacity_provenance = join_capacity(
            phases, capacity_path, trace_hash, not truncated
        )
    return {
        "schema_version": 1,
        "product": "a5",
        "target_product": "Ascend950/V6",
        "clock": "pipe-timeline-common-clock-ns",
        "timeline_complete": not truncated,
        "provenance": {"capture_id": capture_id, "source_sha256": trace_hash},
        "capacity_provenance": capacity_provenance,
        "phases": phases,
    }


def overlaps(rows: list[dict]) -> dict[str, float]:
    by_pipe: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_pipe[row["pipe"]].append((row["start"], row["start"] + row["duration"]))
    merged = {}
    for pipe, intervals in by_pipe.items():
        merged[pipe] = []
        for start, end in sorted(intervals):
            if merged[pipe] and start <= merged[pipe][-1][1]:
                merged[pipe][-1] = (merged[pipe][-1][0], max(end, merged[pipe][-1][1]))
            else:
                merged[pipe].append((start, end))

    result = {}
    names = sorted(by_pipe)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_intervals, right_intervals = merged[left], merged[right]
            left_index = right_index = 0
            value = 0.0
            while left_index < len(left_intervals) and right_index < len(right_intervals):
                a0, a1 = left_intervals[left_index]
                b0, b1 = right_intervals[right_index]
                value += max(0.0, min(a1, b1) - max(a0, b0))
                if a1 <= b1:
                    left_index += 1
                else:
                    right_index += 1
            result[f"{left}:{right}"] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe-timeline", type=Path)
    parser.add_argument("--instr", action="append", default=[], metavar="PIPE=TRACE")
    parser.add_argument("--capacity-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        pipe_rows = (
            events(args.pipe_timeline, "pipe-timeline") if args.pipe_timeline else []
        )
    except InvalidTimeline as error:
        parser.error(str(error))
    if pipe_rows and {x["pipe"] for x in pipe_rows} <= {"scalar", "unknown"}:
        raise InvalidTimeline("PipeTimeline has no non-scalar pipeline tracks")
    instruction_rows = []
    for item in args.instr:
        if "=" not in item:
            parser.error("--instr requires PIPE=TRACE")
        pipe, path = item.split("=", 1)
        try:
            trace_rows = events(Path(path), f"instr-{pipe}", pipe.lower())
        except InvalidTimeline as error:
            parser.error(str(error))
        if len(trace_rows) >= 1024:
            raise InvalidTimeline(
                f"instruction trace {path} reaches the known 1024-event cap"
            )
        instruction_rows.extend(trace_rows)
    rows = pipe_rows + instruction_rows
    if not rows:
        raise InvalidTimeline("at least one timeline is required")
    counts = Counter((x["source"], x["pipe"], x["name"]) for x in rows)
    summary = {
        "schema_version": 1,
        "target": "Ascend950-A5",
        "pipe_timeline": {
            "events": len(pipe_rows),
            "pipes": sorted({x["pipe"] for x in pipe_rows}),
            "overlap_duration": overlaps(pipe_rows) if pipe_rows else {},
        },
        "instruction_timelines": {
            "events": len(instruction_rows),
            "simultaneous": False,
            "alignment": "profiler-task-relative-estimated-overlay",
        },
        "event_counts": [
            {"source": key[0], "pipe": key[1], "name": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "timeline-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if args.pipe_timeline:
        try:
            evidence = phase_evidence(
                args.pipe_timeline, pipe_rows, args.capacity_evidence
            )
        except InvalidTimeline as error:
            parser.error(str(error))
        (args.output / "phase-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
    with (args.output / "timeline-events.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
