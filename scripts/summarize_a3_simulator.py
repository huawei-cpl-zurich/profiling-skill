#!/usr/bin/env python3
"""Compact vendor simulator instruction traces, never device measurements."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CORE = re.compile(r"core\d+\.(?:vec|cube)core\d+\Z")
PIPES = {"SCALAR", "SCALARLDST", "VECTOR", "CUBE", "MTE1", "MTE2", "MTE3", "FIXP", "FLOWCTRL", "ALL"}
CONTROL = re.compile(r"(?:BAR|WAIT.*|SET_FLAG.*|SETFLAG.*|.*EVENT.*|JUMP.*|CALL.*|RET)\Z")
KERNEL = re.compile(r"<ProfInit>\s*Start profiling on kernel:\s*(\S+)")


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(left, right):
    """Intersect sorted disjoint interval unions in linear time."""
    total, i, j = 0.0, 0, 0
    while i < len(left) and j < len(right):
        a0, a1 = left[i]
        b0, b1 = right[j]
        total += max(0, min(a1, b1) - max(a0, b0))
        if a1 <= b1:
            i += 1
        else:
            j += 1
    return total


def parse_trace(payload, path):
    """Resolve lanes from identifiers or Chrome metadata, never args.detail."""
    raw = payload.get("traceEvents", []) if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return [], Counter({"invalid_event_list": 1})
    processes, threads = {}, {}
    for item in raw:
        if not isinstance(item, dict) or item.get("ph") != "M":
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            continue
        key = str(item.get("pid"))
        if item.get("name") == "process_name":
            processes[key] = str(args.get("name", ""))
        elif item.get("name") == "thread_name":
            threads[key, str(item.get("tid"))] = str(args.get("name", ""))
    rows, skipped = [], Counter()
    file_core = path.parent.name if CORE.fullmatch(path.parent.name) else None
    for item in raw:
        if not isinstance(item, dict) or item.get("ph") != "X":
            continue
        pid, tid = str(item.get("pid", "")), str(item.get("tid", ""))
        core = pid if CORE.fullmatch(pid) else file_core
        process, thread = processes.get(pid, ""), threads.get((pid, tid), "")
        if CORE.fullmatch(pid) and tid in PIPES:
            pipe, provenance = tid, "core-pid/pipeline-tid"
        elif core and process in PIPES:
            pipe, provenance = process, "process_name"
        elif core and thread in PIPES:
            pipe, provenance = thread, "thread_name"
        else:
            reason = "unsupported_lane" if core and (CORE.fullmatch(pid) or process or thread) else "missing_core_or_pipeline_metadata"
            skipped[reason] += 1
            continue
        try:
            start, duration = float(item["ts"]), float(item["dur"])
            if not all(math.isfinite(v) and v >= 0 for v in (start, duration)):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            skipped["invalid_timestamp"] += 1
            continue
        name = str(item.get("name", ""))
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        kind = "control" if pipe in {"FLOWCTRL", "ALL"} or CONTROL.fullmatch(name.upper()) else "instruction"
        rows.append({"core": core, "pipeline": pipe, "lane_provenance": provenance,
                     "thread": thread or tid, "name": name, "kind": kind,
                     "start_us": start, "duration_us": duration,
                     "pc": str(args.get("pc_addr", "")),
                     "instruction_detail": str(args.get("detail", ""))[:512],
                     "process_bytes": str(args.get("process_bytes", ""))})
    return rows, skipped


def summarize(report: Path, selector: str, *, acquisition_status="unknown", acquisition_failure=None):
    """Require a single retained report and its sibling profiler stdout binding."""
    if acquisition_failure and acquisition_status != "partial":
        raise ValueError("acquisition failure requires partial capture status")
    report = report.resolve()
    candidates = sorted(report.glob("OPPROF_*/simulator"))
    diagnostics, sources, rows = Counter(), [], []
    log = report.parent / "profiler.stdout.log"
    log_bytes = log.read_bytes() if log.is_file() else b""
    observed = sorted(set(KERNEL.findall(log_bytes.decode(errors="replace"))))
    bound = len(candidates) == 1 and observed == [selector]
    for simulator in candidates:
        aggregate = simulator / "trace.json"
        per_core = sorted(simulator.glob("core*.*/trace.json"))
        # Never combine aggregate and per-core views of one capture.
        paths = [aggregate] if aggregate.is_file() else per_core
        diagnostics["duplicate_views_ignored"] += len(per_core) if aggregate.is_file() else 0
        for path in paths:
            raw = path.read_bytes()
            sources.append({"name": str(path.relative_to(report)), "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()})
            try:
                parsed, skipped = parse_trace(json.loads(raw), path)
            except (ValueError, UnicodeError):
                diagnostics["invalid_json"] += 1
                continue
            diagnostics.update(skipped)
            rows.extend(parsed)
    # Different report identities are not a shared clock or selected invocation.
    interpretable = bool(bound and rows)
    by_core = defaultdict(lambda: defaultdict(list))
    if interpretable:
        for item in rows:
            if item["kind"] == "instruction" and item["duration_us"] > 0:
                by_core[item["core"]][item["pipeline"]].append(
                    (item["start_us"], item["start_us"] + item["duration_us"]))
    cores = {}
    for core, pipes in sorted(by_core.items()):
        intervals = {pipe: union(values) for pipe, values in pipes.items()}
        cores[core] = {
            "pipeline_active_us": {p: sum(b-a for a, b in values) for p, values in intervals.items()},
            "pipeline_overlap_us": {f"{a}:{b}": overlap(intervals[a], intervals[b])
                                    for a, b in itertools.combinations(sorted(intervals), 2)},
        }
    return {
        "schema_version": 1, "evidence_kind": "simulation-instruction-timeline",
        "product": "a3", "clock": "simulator-trace-microseconds",
        "acquisition_status": acquisition_status,
        "partial_capture": acquisition_status == "partial",
        "acquisition_failure": acquisition_failure,
        "clean_acquisition_gate_passed": acquisition_status == "clean" and interpretable,
        "raw_trace_obtained": bool(sources), "interpretable": interpretable,
        "binding": {"state": "bound-single-report" if bound else "unavailable",
                    "expected_kernel": selector, "observed_kernel_names": observed,
                    "log_sha256": hashlib.sha256(log_bytes).hexdigest() if log_bytes else None},
        "diagnostics": dict(diagnostics), "sources": sources,
        "event_count": len(rows), "events": rows[:256] if interpretable else [],
        "events_truncated": len(rows) > 256, "cores": cores,
        "source_phase_interpretation": "unavailable: instruction lanes are not source phases",
        "device_timing": "unavailable", "saturation": "unavailable",
        "limitations": ["simulated timestamps are not hardware latency or utilization",
                        "pipeline activity/overlap does not establish capacity saturation",
                        "no source-phase attribution without independently validated mapping"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--acquisition-status", choices=("unknown", "clean", "partial"), default="unknown")
    parser.add_argument("--acquisition-failure", choices=("application_failed", "teardown_failed", "profiler_failed"))
    args = parser.parse_args()
    if args.acquisition_failure and args.acquisition_status != "partial":
        parser.error("--acquisition-failure requires --acquisition-status partial")
    result = summarize(args.report, args.kernel_name, acquisition_status=args.acquisition_status,
                       acquisition_failure=args.acquisition_failure)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["interpretable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
