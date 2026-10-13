#!/usr/bin/env python3
"""Resumable, shardable A3 microbenchmark profile campaign coordinator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "a3-profile-hypotheses.json"
ACQUIRE = ROOT / "scripts" / "acquire_profile.py"
INFRA_FAILURES = {"transport_error", "host_environment", "device_unavailable"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def enumerate_cells(manifest: dict) -> list[dict]:
    cells = []
    for kind, key in (("hypothesis", "hypotheses"), ("timeline", "timeline_probes")):
        for experiment in manifest[key]:
            for sweep in experiment["sweeps"]:
                for variant in ("before", "after"):
                    cells.append({
                        "id": f"{experiment['id']}/{sweep['id']}/{variant}",
                        "experiment": experiment["id"], "sweep": sweep["id"],
                        "variant": variant, "kind": kind,
                        "kernel_names": experiment["kernel_names"][variant],
                        "expected_evidence": experiment["expected_evidence"],
                        "falsifier": experiment["falsifier"],
                    })
    return cells


def select_shard(cells: list[dict], index: int, count: int) -> list[dict]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index must be in [0, shard count)")
    return [cell for position, cell in enumerate(cells) if position % count == index]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def _receipt_handle(path: Path) -> str | None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    handle = value.get("handle")
    return handle if isinstance(handle, str) else None


def invoke_acquisition(
    *, phase: str, target: str, dispatch_key: str, bundle: Path, entrypoint: str,
    workload_args: list[str], evidence_path: Path, selector: str | None,
    basic_evidence: Path | None, cpl_remote: Path | None, launch_count: int = 1,
) -> dict:
    """Run or resume one acquire_profile pass and return compact evidence."""
    receipt = Path(str(evidence_path) + ".dispatch.json")
    if evidence_path.exists():
        return {"ok": True, "handle": _receipt_handle(receipt),
                "evidence": json.loads(evidence_path.read_bytes())}
    command = [
        sys.executable, str(ACQUIRE), phase, "--product", "a3", "--target", target,
        "--dispatch-key", dispatch_key, "--bundle", str(bundle),
        "--entrypoint", entrypoint, "--launch-count", str(launch_count),
        "--evidence", str(evidence_path), "--dispatch-receipt", str(receipt),
    ]
    for argument in workload_args:
        command.append("--workload-arg=" + argument)
    if cpl_remote:
        command.extend(("--cpl-remote", str(cpl_remote)))
    if selector:
        command.extend(("--kernel-name", selector))
    if basic_evidence:
        command.extend(("--basic-evidence", str(basic_evidence)))
    handle = _receipt_handle(receipt)
    if handle:
        command.extend(("--resume-handle", handle))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        match = re.search(
            r"classification=([^ ]+) phase=([^ ]+)(?: handle=([^: ]+:[^ ]+))?",
            result.stderr,
        )
        return {
            "ok": False,
            "classification": match.group(1) if match else "controller_failure",
            "phase": match.group(2) if match else phase,
            "handle": handle,
            "message": result.stderr[-2000:],
        }
    return {"ok": True, "handle": _receipt_handle(receipt),
            "evidence": json.loads(evidence_path.read_bytes())}


def _numeric_rows(pipe_results: list[dict]) -> tuple[float, dict[str, list[float]]]:
    selector_times = []
    signals: dict[str, list[float]] = {}
    for result in pipe_results:
        rows = result["evidence"]["capture"]["rows"]
        times = []
        for row in rows:
            for key, raw in row.items():
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if key in {"aic_time(us)", "aiv_time(us)"}:
                    times.append(value)
                if key.endswith("_ratio"):
                    signals.setdefault(key, []).append(value)
        if not times:
            raise ValueError("PipeUtilization evidence lacks aic/aiv reported time")
        selector_times.append(max(times))
    return sum(selector_times), signals


def _summary(samples: list[float], required: int, threshold: float) -> dict:
    median = statistics.median(samples) if samples else None
    relative_range = ((max(samples) - min(samples)) / median
                      if samples and median and len(samples) >= required else None)
    stable = bool(len(samples) >= required and
                  (required == 1 or relative_range <= threshold))
    return {
        "metric": "PipeUtilization reported core-time proxy",
        "unit": "us", "samples": samples, "median": median,
        "relative_range": relative_range, "stable": stable,
        "required_samples": required, "maximum_relative_range": threshold,
    }


def run_cell(
    cell: dict, cell_dir: Path, acquire, *, samples: int, max_attempts: int,
    max_relative_range: float,
) -> dict:
    state_path = cell_dir / "result.json"
    if state_path.exists():
        prior = json.loads(state_path.read_bytes())
        if prior.get("status") in {
            "success", "candidate_failure", "profiling_failure", "inconclusive"
        }:
            return prior
    values, signal_values, attempts, infra = [], {}, [], 0
    handles, sources, devices = [], [], []
    for attempt in range(max_attempts):
        basic = acquire("basic", cell, attempt, None)
        handles.extend(x for x in [basic.get("handle")] if x)
        if not basic["ok"]:
            record = {k: basic.get(k) for k in ("classification", "phase", "message", "handle")}
            attempts.append(record)
            if basic.get("classification") in INFRA_FAILURES:
                infra += 1
                continue
            result = _failure_result(cell, attempts, handles, infra, basic)
            _atomic_json(state_path, result)
            return result
        observed = basic["evidence"]["capture"]["observed_kernel_names"]
        if not all(selector in observed for selector in cell["kernel_names"]):
            failure = {"classification": "evidence_failure", "phase": "basic",
                       "message": f"expected {cell['kernel_names']!r}, observed {observed!r}"}
            result = _failure_result(cell, attempts, handles, infra, failure)
            _atomic_json(state_path, result)
            return result
        pipes = []
        for selector in cell["kernel_names"]:
            pipe = acquire("pipe", cell, attempt, selector)
            handles.extend(x for x in [pipe.get("handle")] if x)
            if not pipe["ok"]:
                attempts.append({k: pipe.get(k) for k in
                                 ("classification", "phase", "message", "handle")})
                if pipe.get("classification") in INFRA_FAILURES:
                    infra += 1
                    pipes = []
                    break
                result = _failure_result(cell, attempts, handles, infra, pipe)
                _atomic_json(state_path, result)
                return result
            pipes.append(pipe)
        if not pipes:
            continue
        try:
            duration, current_signals = _numeric_rows(pipes)
        except ValueError as exc:
            failure = {"classification": "evidence_failure", "phase": "pipe",
                       "message": str(exc)}
            result = _failure_result(cell, attempts, handles, infra, failure)
            _atomic_json(state_path, result)
            return result
        values.append(duration)
        for name, observations in current_signals.items():
            signal_values.setdefault(name, []).extend(observations)
        for item in [basic, *pipes]:
            evidence = item["evidence"]
            devices.append(evidence.get("device"))
            sources.extend(evidence["capture"].get("sources", []))
        attempts.append({"classification": "valid", "attempt": attempt,
                         "selectors": cell["kernel_names"], "duration_proxy_us": duration})
        measurements = _summary(values, samples, max_relative_range)
        partial = _result(cell, "success" if measurements["stable"] else "running",
                          measurements, signal_values, attempts, handles, sources,
                          devices, infra)
        _atomic_json(state_path, partial)
        if measurements["stable"]:
            return partial
    measurements = _summary(values, samples, max_relative_range)
    status = "inconclusive" if values else "infrastructure_exhausted"
    result = _result(cell, status, measurements, signal_values, attempts, handles,
                     sources, devices, infra)
    _atomic_json(state_path, result)
    return result


def _signals(values: dict[str, list[float]]) -> dict:
    return {name: {"mean": statistics.fmean(items), "samples": len(items),
                   "values": items} for name, items in sorted(values.items())}


def _result(cell, status, measurements, signals, attempts, handles, sources,
            devices, infra):
    value = {
        **cell, "status": status, "measurements": measurements,
        "signals": _signals(signals), "attempts": attempts,
        "discarded_infrastructure_attempts": infra,
        "provenance": {"handles": handles, "devices": devices, "sources": sources},
    }
    if cell["kind"] == "timeline":
        value["timeline"] = {
            "status": "unavailable",
            "reason": "A3 msprof-op acquisition emitted no TimelineDetail evidence",
        }
    return value


def _failure_result(cell, attempts, handles, infra, failure):
    status = ("candidate_failure" if failure.get("classification") == "workload_failure"
              else "profiling_failure")
    value = _result(cell, status, _summary([], 1, 0.05), {},
                    attempts, handles, [], [], infra)
    value["failure"] = {k: failure.get(k) for k in
                        ("classification", "phase", "message") if failure.get(k)}
    return value


def _direction(before: float, after: float, expected: dict, *, absolute: bool) -> str:
    direction = expected["direction"]
    threshold = expected.get(
        "minimum_absolute_change" if absolute else "minimum_relative_change", 0.0
    )
    change = before - after if absolute else (before - after) / before
    if direction == "decrease":
        return "supports" if change >= threshold else (
            "refutes" if change <= -threshold else "inconclusive"
        )
    if direction == "increase":
        return "supports" if change <= -threshold else (
            "refutes" if change >= threshold else "inconclusive"
        )
    return "supports" if abs(change) <= threshold else "refutes"


def _measurement_status(before: dict | None, after: dict | None) -> str:
    statuses = {item.get("status") for item in (before, after) if item}
    if "candidate_failure" in statuses:
        return "candidate_failure"
    if "infrastructure_exhausted" in statuses:
        return "infrastructure_failure"
    if "profiling_failure" in statuses:
        return "profiling_failure"
    return "unavailable"


def summarize_hypotheses(raw_cells: list[dict], manifest: dict) -> list[dict]:
    """Pair variant cells into build_profile_reverse_report's input contract."""
    by_key = {
        (cell.get("experiment"), cell.get("sweep"), cell.get("variant")): cell
        for cell in raw_cells
    }
    results = []
    for key in ("hypotheses", "timeline_probes"):
        for spec in manifest[key]:
            measurements, assessments = [], {item["name"]: [] for item in spec["signal_predictions"]}
            for sweep in spec["sweeps"]:
                sweep_id = sweep["id"]
                before = by_key.get((spec["id"], sweep_id, "before"))
                after = by_key.get((spec["id"], sweep_id, "after"))
                valid = (
                    before and after and before.get("status") == after.get("status") == "success"
                    and before.get("measurements", {}).get("median")
                    and after.get("measurements", {}).get("median")
                )
                if valid:
                    before_value = float(before["measurements"]["median"])
                    after_value = float(after["measurements"]["median"])
                    measurement = {
                        "status": "valid", "sweep": sweep_id,
                        "before": before_value, "after": after_value,
                        "metric": before["measurements"]["metric"],
                        "unit": before["measurements"]["unit"],
                        "before_cell_id": before["id"], "after_cell_id": after["id"],
                    }
                    measurements.append(measurement)
                    outcome = _direction(before_value, after_value, spec["expected"], absolute=False)
                else:
                    measurements.append({
                        "status": _measurement_status(before, after), "sweep": sweep_id,
                        "before_status": before.get("status") if before else "missing",
                        "after_status": after.get("status") if after else "missing",
                    })
                    outcome = "unavailable"
                for prediction in spec["signal_predictions"]:
                    name = prediction["name"]
                    if prediction.get("availability") == "required":
                        assessments[name].append((sweep_id, "not_exported", None, None))
                        continue
                    left = (before or {}).get("signals", {}).get(name, {}).get("mean")
                    right = (after or {}).get("signals", {}).get(name, {}).get("mean")
                    if left is None or right is None:
                        assessments[name].append((sweep_id, "not_exported", left, right))
                        continue
                    signal = _direction(float(left), float(right), prediction, absolute=True)
                    if outcome == "unavailable" or "inconclusive" in {outcome, signal}:
                        relationship = "context_only"
                    elif outcome == signal:
                        relationship = "tracks_outcome"
                    else:
                        relationship = "contradicts_outcome"
                    assessments[name].append((sweep_id, relationship, float(left), float(right)))
            signals = []
            for prediction in spec["signal_predictions"]:
                name, rows = prediction["name"], assessments[prediction["name"]]
                relationships = {row[1] for row in rows}
                relationship = next(iter(relationships)) if len(relationships) == 1 else "context_only"
                if relationship == "not_exported" and key == "timeline_probes":
                    summary = "TimelineDetail was unavailable for every controlled capture"
                else:
                    details = []
                    for sweep_id, relation, left, right in rows:
                        if left is None or right is None:
                            details.append(f"{sweep_id}: {relation}")
                        else:
                            details.append(
                                f"{sweep_id}: before={left:g}, after={right:g}; "
                                + ({
                                    "tracks_outcome": "signal and latency track the declared outcome",
                                    "contradicts_outcome": "signal contradicts the observed latency outcome",
                                    "context_only": "signal is contextual or below its declared threshold",
                                    "not_exported": "signal was not exported",
                                }[relation])
                            )
                    summary = "; ".join(details)
                signals.append({"name": name, "relationship": relationship, "summary": summary})
            results.append({
                "hypothesis_id": spec["id"], "title": spec.get("question", spec["id"]),
                "expected": spec["expected"], "falsification": spec["falsifier"],
                "measurements": measurements, "signals": signals,
            })
    return results


def validate_report_hypotheses(document: dict) -> bool:
    """Validate the subset consumed by build_profile_reverse_report.py."""
    if document.get("schema_version") != 1 or not isinstance(document.get("results"), list):
        raise ValueError("report hypotheses require schema_version 1 and results")
    relationships = {"tracks_outcome", "context_only", "contradicts_outcome", "not_exported"}
    for item in document["results"]:
        if not isinstance(item.get("hypothesis_id"), str) or not item["hypothesis_id"]:
            raise ValueError("hypothesis_id is required")
        expected = item.get("expected", {})
        if expected.get("direction") not in {"decrease", "increase", "no_change"}:
            raise ValueError("invalid expected direction")
        threshold = expected.get("minimum_relative_change")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold < 1:
            raise ValueError("invalid expected threshold")
        if not isinstance(item.get("measurements"), list) or not isinstance(item.get("signals"), list):
            raise ValueError("measurements and signals must be arrays")
        for row in item["measurements"]:
            if row.get("status") == "valid" and not (
                float(row["before"]) > 0 and float(row["after"]) > 0
            ):
                raise ValueError("valid paired measurements must be positive")
        for signal in item["signals"]:
            if not signal.get("name") or signal.get("relationship") not in relationships:
                raise ValueError("invalid signal relationship")
    return True


def merge_shards(shards: list[dict], manifest: dict) -> dict:
    if not shards:
        raise ValueError("at least one shard is required")
    digests = {item["manifest_sha256"] for item in shards}
    if len(digests) != 1:
        raise ValueError("manifest hashes differ")
    results, ids = [], set()
    for shard in shards:
        for result in shard["raw_cells"]:
            if result["id"] in ids:
                raise ValueError(f"duplicate cell {result['id']}")
            ids.add(result["id"])
            results.append(result)
    raw_cells = sorted(results, key=lambda item: item["id"])
    value = {"schema_version": 1, "manifest_sha256": digests.pop(),
             "raw_cells": raw_cells,
             "results": summarize_hypotheses(raw_cells, manifest)}
    validate_report_hypotheses(value)
    return value


def _bundle(output: Path, manifest: Path) -> Path:
    bundle = output / "bundle"
    (bundle / "benchmarks").mkdir(parents=True, exist_ok=True)
    (bundle / "experiments").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "benchmarks/a3_profile_microbench.py",
                 bundle / "benchmarks/a3_profile_microbench.py")
    shutil.copy2(manifest, bundle / "experiments/a3-profile-hypotheses.json")
    return bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    run = sub.add_parser("run")
    merge = sub.add_parser("merge")
    for command in (plan, run):
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--shard-count", type=int, default=1)
    plan.add_argument("--output", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--target", choices=("bz-a3-1", "bz-a3-2"), required=True)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--max-attempts", type=int, default=7)
    run.add_argument("--max-relative-range", type=float, default=0.05)
    run.add_argument("--cell-limit", type=int)
    run.add_argument("--cpl-remote", type=Path)
    merge.add_argument("--input", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "merge":
        manifest = json.loads(args.manifest.read_bytes())
        value = merge_shards([json.loads(path.read_bytes()) for path in args.input], manifest)
        _atomic_json(args.output, value)
        return 0
    manifest = json.loads(args.manifest.read_bytes())
    cells = select_shard(enumerate_cells(manifest), args.shard_index, args.shard_count)
    if args.command == "plan":
        value = {"schema_version": 1, "shard": {"index": args.shard_index,
                 "count": args.shard_count}, "cells": cells}
        if args.output:
            _atomic_json(args.output, value)
        else:
            print(json.dumps(value, sort_keys=True))
        return 0
    if args.samples < 1 or args.max_attempts < args.samples:
        raise SystemExit("samples must be positive and no greater than max-attempts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _bundle(args.output_dir, args.manifest)
    cells = cells[:args.cell_limit] if args.cell_limit else cells
    results = []
    for cell in cells:
        cell_dir = args.output_dir / "cells" / cell["id"].replace("/", "__")

        def acquire(phase, current, attempt, selector):
            base = cell_dir / f"attempt-{attempt}" / phase
            suffix = "" if selector is None else "-" + selector
            basic_path = cell_dir / f"attempt-{attempt}" / "basic.json"
            return invoke_acquisition(
                phase=phase, target=args.target,
                dispatch_key=hashlib.sha256(
                    f"a3-profile-microbench/v1/{current['id']}/{attempt}/{phase}/{selector}".encode()
                ).hexdigest(), bundle=bundle,
                entrypoint="benchmarks/a3_profile_microbench.py",
                workload_args=["--experiment", current["experiment"], "--sweep",
                               current["sweep"], "--variant", current["variant"],
                               "--launches", "1"],
                evidence_path=Path(str(base) + suffix + ".json"), selector=selector,
                basic_evidence=basic_path if phase == "pipe" else None,
                cpl_remote=args.cpl_remote,
                launch_count=(len(current["kernel_names"]) if phase == "basic" else 1),
            )

        results.append(run_cell(cell, cell_dir, acquire, samples=args.samples,
                                max_attempts=args.max_attempts,
                                max_relative_range=args.max_relative_range))
    value = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "target": args.target,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "raw_cells": results,
        "results": summarize_hypotheses(results, manifest),
    }
    validate_report_hypotheses(value)
    _atomic_json(args.output_dir / "shard-result.json", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
