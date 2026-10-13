#!/usr/bin/env python3
"""Join A3 candidate, capture, and microbenchmark evidence into a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class InvalidReportInput(ValueError):
    pass


VALID_CAPTURE_STATES = {"valid", "candidate_failure", "infrastructure_failure"}
RELATIONSHIP_CLASSES = {
    "tracks_outcome": "useful",
    "context_only": "contextual",
    "contradicts_outcome": "misleading",
    "not_exported": "unavailable",
}


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidReportInput(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise InvalidReportInput(f"{label} must be a schema-version 1 object")
    return value


def objects(value: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise InvalidReportInput(f"{location} must be an array")
    if not all(isinstance(item, dict) for item in value):
        raise InvalidReportInput(f"every {location} item must be an object")
    return value


def identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidReportInput(f"{location} must be a non-empty string")
    return value


def finite_positive(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise InvalidReportInput(f"{location} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidReportInput(f"{location} must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise InvalidReportInput(f"{location} must be a positive finite number")
    return number


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summaries_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    samples: dict[str, list[float]] = defaultdict(list)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row.get("status") == "valid":
            case_id = identifier(row.get("case_id"), f"captures[{index}].case_id")
            samples[case_id].append(
                finite_positive(row.get("latency_us"), f"captures[{index}].latency_us")
            )
            profile = row.get("profile")
            raw_metrics = profile.get("metrics") if isinstance(profile, dict) else None
            if raw_metrics is not None:
                if not isinstance(raw_metrics, dict):
                    raise InvalidReportInput(f"captures[{index}].profile.metrics must be an object")
                for name, value in raw_metrics.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    number = float(value)
                    if math.isfinite(number):
                        metrics[case_id][str(name)].append(number)
    result = {}
    for case, values in sorted(samples.items()):
        median = statistics.median(values)
        result[case] = {
            "median_latency_us": median,
            "latency_samples_us": sorted(values),
            "relative_range": (max(values) - min(values)) / median,
            "profile_metrics": {
                name: statistics.median(metric_values)
                for name, metric_values in sorted(metrics[case].items())
            },
        }
    return result


def effective_candidate(
    candidate_id: str, candidates: dict[str, dict[str, Any]]
) -> str:
    seen = set()
    current = candidate_id
    while candidates[current].get("duplicate_of") is not None:
        if current in seen:
            raise InvalidReportInput(f"duplicate cycle includes candidate {current!r}")
        seen.add(current)
        current = identifier(
            candidates[current]["duplicate_of"], f"candidate {current!r}.duplicate_of"
        )
        if current not in candidates:
            raise InvalidReportInput(f"candidate duplicate references unknown candidate {current!r}")
    return current


def validate_saturation(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        profile = row.get("profile")
        if not isinstance(profile, dict):
            continue
        saturation = profile.get("saturation")
        if not isinstance(saturation, dict):
            continue
        state = saturation.get("state", "unknown")
        if state != "unknown":
            raise InvalidReportInput(
                f"captures[{index}] makes a saturation claim; no validated capacity evidence "
                "is accepted by this A3 report"
            )


def classify_hypothesis(item: dict[str, Any]) -> tuple[str, str]:
    measurements = objects(item.get("measurements", []), "hypothesis.measurements")
    valid = [row for row in measurements if row.get("status") == "valid"]
    if not valid:
        return "unavailable", "no valid controlled measurement"
    expected = item.get("expected")
    if not isinstance(expected, dict):
        raise InvalidReportInput("hypothesis.expected must be an object")
    direction = expected.get("direction")
    if direction not in {"decrease", "increase", "no_change"}:
        raise InvalidReportInput("hypothesis expected direction must be decrease, increase, or no_change")
    threshold = float(expected.get("minimum_relative_change", 0.0))
    if not 0 <= threshold < 1:
        raise InvalidReportInput("minimum_relative_change must be in [0, 1)")
    effects = []
    for row in valid:
        before = finite_positive(row.get("before"), "hypothesis measurement before")
        after = finite_positive(row.get("after"), "hypothesis measurement after")
        effects.append((before - after) / before)
    effect = statistics.median(effects)
    if direction == "decrease":
        if effect >= threshold:
            classification = "supported"
        elif effect <= -threshold:
            classification = "refuted"
        else:
            classification = "inconclusive"
    elif direction == "increase":
        if effect <= -threshold:
            classification = "supported"
        elif effect >= threshold:
            classification = "refuted"
        else:
            classification = "inconclusive"
    else:
        classification = "supported" if abs(effect) <= threshold else "refuted"
    return classification, f"median relative before-to-after reduction={effect:.6f}"


def compile_report(
    manifest: dict[str, Any], capture_doc: dict[str, Any], hypothesis_doc: dict[str, Any]
) -> dict[str, Any]:
    raw_candidates = objects(manifest.get("candidates"), "manifest.candidates")
    candidates: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(raw_candidates):
        candidate_id = identifier(candidate.get("candidate_id"), f"candidates[{index}].candidate_id")
        if candidate_id in candidates:
            raise InvalidReportInput(f"duplicate candidate_id {candidate_id!r}")
        lineage_state = candidate.get("lineage_state", "complete")
        if lineage_state not in {"complete", "incomplete", "unrecoverable"}:
            raise InvalidReportInput(f"candidate {candidate_id!r} has invalid lineage_state")
        if lineage_state != "complete":
            identifier(candidate.get("lineage_note"), f"candidate {candidate_id!r}.lineage_note")
        candidates[candidate_id] = candidate
    starter_id = identifier(manifest.get("starter_candidate_id"), "starter_candidate_id")
    if starter_id not in candidates:
        raise InvalidReportInput("starter_candidate_id references unknown candidate")

    captures = objects(capture_doc.get("captures"), "captures")
    validate_saturation(captures)
    captures_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, capture in enumerate(captures):
        candidate_id = identifier(capture.get("candidate_id"), f"captures[{index}].candidate_id")
        if candidate_id not in candidates:
            raise InvalidReportInput(f"capture references unknown candidate {candidate_id!r}")
        if capture.get("status") not in VALID_CAPTURE_STATES:
            raise InvalidReportInput(f"captures[{index}] has invalid status")
        captures_by_candidate[candidate_id].append(capture)

    summaries = {key: summaries_by_case(value) for key, value in captures_by_candidate.items()}
    starter_summaries = summaries.get(effective_candidate(starter_id, candidates), {})
    compiled_candidates = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        evidence_id = effective_candidate(candidate_id, candidates)
        candidate_summaries = summaries.get(evidence_id, {})
        parent_id = candidate.get("parent_candidate_id")
        if parent_id is not None:
            parent_id = identifier(parent_id, f"candidate {candidate_id!r}.parent_candidate_id")
            if parent_id not in candidates:
                raise InvalidReportInput(f"candidate parent references unknown candidate {parent_id!r}")
            parent_summaries = summaries.get(effective_candidate(parent_id, candidates), {})
        else:
            parent_summaries = {}
        case_results: dict[str, dict[str, Any]] = {}
        starter_ratios, parent_ratios = [], []
        for case_id, summary in candidate_summaries.items():
            latency = summary["median_latency_us"]
            starter = starter_summaries.get(case_id)
            parent = parent_summaries.get(case_id)
            starter_speedup = starter["median_latency_us"] / latency if starter else None
            parent_speedup = parent["median_latency_us"] / latency if parent else None
            if starter_speedup is not None:
                starter_ratios.append(starter_speedup)
            if parent_speedup is not None:
                parent_ratios.append(parent_speedup)
            case_results[case_id] = {
                **summary,
                "starter_speedup": starter_speedup,
                "parent_speedup": parent_speedup,
            }
        failures = []
        for row in captures_by_candidate.get(candidate_id, []):
            if row.get("status") == "valid":
                continue
            failure = row.get("failure") if isinstance(row.get("failure"), dict) else {}
            failures.append({
                "case_id": str(row.get("case_id", "unknown")),
                "status": row["status"],
                "kind": str(failure.get("kind", "unknown")),
                "message": str(failure.get("message", "")),
            })
        compiled_candidates.append({
            "candidate_id": candidate_id,
            "treatment": candidate.get("treatment"),
            "round": candidate.get("round"),
            "option": candidate.get("option"),
            "source_hash": candidate.get("source_hash"),
            "parent_candidate_id": parent_id,
            "duplicate_of": candidate.get("duplicate_of"),
            "lineage_state": candidate.get("lineage_state", "complete"),
            "lineage_note": candidate.get("lineage_note"),
            "evidence_source_candidate_id": evidence_id,
            "source_features": candidate.get("source_features", {}),
            "code_delta": candidate.get("code_delta", []),
            "cases": case_results,
            "geometric_mean": {
                "latency_us": geometric_mean([
                    item["median_latency_us"] for item in candidate_summaries.values()
                ]),
                "starter_speedup": geometric_mean(starter_ratios),
                "parent_speedup": geometric_mean(parent_ratios),
            },
            "failures": failures,
            "saturation": {
                "state": "unknown",
                "reason": "no validated capacity evidence",
            },
        })

    compiled_hypotheses, raw_signal_rows = [], []
    for item in objects(hypothesis_doc.get("results"), "hypotheses.results"):
        hypothesis_id = identifier(item.get("hypothesis_id"), "hypothesis_id")
        classification, reason = classify_hypothesis(item)
        compiled_hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "title": item.get("title", hypothesis_id),
            "classification": classification,
            "reason": reason,
        })
        for signal in objects(item.get("signals", []), f"hypothesis {hypothesis_id!r}.signals"):
            relationship = signal.get("relationship")
            if relationship not in RELATIONSHIP_CLASSES:
                raise InvalidReportInput(f"unknown signal relationship {relationship!r}")
            raw_signal_rows.append({
                "signal": identifier(signal.get("name"), "signal.name"),
                "classification": RELATIONSHIP_CLASSES[relationship],
                "hypothesis_id": hypothesis_id,
                "summary": str(signal.get("summary", "")),
            })

    grouped_signals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_signal_rows:
        grouped_signals[row["signal"]].append(row)
    signal_rows = []
    for name, assessments in sorted(grouped_signals.items()):
        classes = {row["classification"] for row in assessments}
        classification = next(iter(classes)) if len(classes) == 1 else "contextual"
        signal_rows.append({
            "signal": name,
            "classification": classification,
            "summary": "; ".join(row["summary"] for row in assessments if row["summary"]),
            "assessments": assessments,
        })

    return {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_id"),
        "report_state": "reproducible_scaffold",
        "references": manifest.get("references", []),
        "candidates": compiled_candidates,
        "hypotheses": sorted(compiled_hypotheses, key=lambda row: row["hypothesis_id"]),
        "signals": signal_rows,
    }


def fmt(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6g}"


def speedup(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6g}x"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A3 profiling reverse-engineering report",
        "",
        "> **Reproducible scaffold.** Replace fixture inputs with retained campaign outputs; missing evidence remains explicit.",
        "",
        f"Campaign: `{report.get('campaign_id')}`",
        "",
        "## Candidate results",
        "",
        "| Candidate | Treatment | Parent | Duplicate of | Geomean µs | vs starter | vs parent |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for item in report["candidates"]:
        mean = item["geometric_mean"]
        lines.append(
            f"| `{item['candidate_id']}` | {item.get('treatment')} | {item.get('parent_candidate_id') or '—'} | "
            f"{item.get('duplicate_of') or '—'} | {fmt(mean['latency_us'])} | {speedup(mean['starter_speedup'])} | {speedup(mean['parent_speedup'])} |"
        )
    lines.extend(["", "### Candidate deltas and failures", ""])
    for item in report["candidates"]:
        if item["lineage_state"] != "complete":
            lines.append(
                f"- Lineage for `{item['candidate_id']}` is **{item['lineage_state']}**: "
                f"{item.get('lineage_note') or 'no additional detail supplied'}."
            )
        deltas = item.get("code_delta") or []
        if deltas:
            lines.append(f"- Code delta for `{item['candidate_id']}`: " + "; ".join(map(str, deltas)) + ".")
        if item["failures"]:
            for failure in item["failures"]:
                lines.append(
                    f"- Failure `{item['candidate_id']}` case `{failure['case_id']}`: "
                    f"{failure['status']}/{failure['kind']} — {failure['message']}"
                )
    lines.extend(["", "Saturation: unknown — no validated capacity evidence.", "", "## Hypotheses", ""])
    for item in report["hypotheses"]:
        lines.append(f"- **{item['classification']}** — {item['title']} (`{item['hypothesis_id']}`): {item['reason']}")
    lines.extend(["", "## Profiler signals", ""])
    for item in report["signals"]:
        lines.append(f"- `{item['signal']}`: **{item['classification']}** — {item['summary']}")
    lines.extend(["", "## References", ""])
    for reference in report.get("references", []):
        if isinstance(reference, dict):
            lines.append(f"- `{reference.get('citation')}` (query `{reference.get('query_id')}`)")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    fields = ["candidate_id", "treatment", "round", "option", "source_hash", "parent_candidate_id", "duplicate_of", "lineage_state", "lineage_note", "evidence_source_candidate_id", "geomean_latency_us", "starter_speedup", "parent_speedup", "failure_count"]
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in report["candidates"]:
            mean = item["geometric_mean"]
            writer.writerow({
                "candidate_id": item["candidate_id"], "treatment": item["treatment"],
                "round": item["round"], "option": item["option"], "source_hash": item["source_hash"],
                "parent_candidate_id": item["parent_candidate_id"], "duplicate_of": item["duplicate_of"],
                "lineage_state": item["lineage_state"], "lineage_note": item["lineage_note"],
                "evidence_source_candidate_id": item["evidence_source_candidate_id"],
                "geomean_latency_us": mean["latency_us"], "starter_speedup": mean["starter_speedup"],
                "parent_speedup": mean["parent_speedup"], "failure_count": len(item["failures"]),
            })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--captures", required=True, type=Path)
    result.add_argument("--hypotheses", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = compile_report(
            read_object(args.manifest, "manifest"),
            read_object(args.captures, "captures"),
            read_object(args.hypotheses, "hypotheses"),
        )
        write_outputs(report, args.output_dir)
    except InvalidReportInput as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
