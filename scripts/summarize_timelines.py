#!/usr/bin/env python3
"""Normalize A5 PipeTimeline and per-pipe InstrTimeline JSON traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


class InvalidTimeline(ValueError):
    pass


def events(path: Path, source: str, forced_pipe: str | None = None) -> list[dict]:
    payload = json.loads(path.read_text())
    raw = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise InvalidTimeline(f"trace event list expected in {path}")
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
            }
        )
    if not out:
        raise InvalidTimeline(f"no complete-duration events in {path}")
    return out


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipe_rows = (
        events(args.pipe_timeline, "pipe-timeline") if args.pipe_timeline else []
    )
    if pipe_rows and {x["pipe"] for x in pipe_rows} <= {"scalar", "unknown"}:
        raise InvalidTimeline("PipeTimeline has no non-scalar pipeline tracks")
    instruction_rows = []
    for item in args.instr:
        if "=" not in item:
            parser.error("--instr requires PIPE=TRACE")
        pipe, path = item.split("=", 1)
        trace_rows = events(Path(path), f"instr-{pipe}", pipe.lower())
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
    with (args.output / "timeline-events.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
