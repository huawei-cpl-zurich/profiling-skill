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
    if not isinstance(value, dict):
        raise InvalidReportInput(f"{label} must be an object")
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


def pipe_profile(pipe: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not isinstance(pipe, dict):
        return {}, []
    if pipe.get("metric") == "PipeUtilization" and isinstance(pipe.get("rows"), list):
        selector = identifier(pipe.get("selector"), "PipeUtilization selector")
        components = []
        for index, raw in enumerate(pipe["rows"]):
            if not isinstance(raw, dict):
                raise InvalidReportInput(f"PipeUtilization rows[{index}] must be an object")
            block = identifier(raw.get("block_id"), f"PipeUtilization rows[{index}].block_id")
            sub_block = identifier(
                raw.get("sub_block_id"), f"PipeUtilization rows[{index}].sub_block_id"
            )
            values = {}
            for name, value in raw.items():
                if name in {"block_id", "sub_block_id"} or isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    values[str(name)] = number
            components.append({
                "metric": "PipeUtilization", "selector": selector,
                "block_id": block, "sub_block_id": sub_block,
                "metrics": values,
            })
        return {}, components
    return {
        str(name): float(value)
        for name, value in pipe.items()
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    }, []


def sample_profile(sample: dict[str, Any]) -> dict[str, Any]:
    nested = sample.get("components")
    if not isinstance(nested, dict):
        pipe = sample.get("pipe_utilization")
        numeric, components = pipe_profile(pipe)
        return {
            "metrics": numeric, "components": components, "raw": pipe,
            "evidence_sha256": sample.get("evidence_sha256"),
            "provenance": sample.get("provenance"),
        }
    parsed_components = []
    evidence, provenance = {}, {}
    for component_selector, component in sorted(nested.items()):
        selector = identifier(component_selector, "sample component selector")
        if not isinstance(component, dict):
            raise InvalidReportInput(f"sample component {selector!r} must be an object")
        _, rows = pipe_profile(component.get("pipe_utilization"))
        for row in rows:
            parsed_components.append({"component_selector": selector, **row})
        evidence[selector] = component.get("evidence_sha256")
        provenance[selector] = component.get("provenance")
    return {
        "metrics": {},
        "components": parsed_components,
        "raw": {
            "duration_aggregation": sample.get("duration_aggregation"),
            "components": nested,
        },
        "evidence_sha256": {"components": evidence},
        "provenance": {"components": provenance},
    }


def summaries_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    samples: dict[str, list[float]] = defaultdict(list)
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    evidence_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_metrics: dict[
        str, dict[tuple[str | None, str, str, str, str], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        if row.get("status") == "valid":
            case_id = identifier(row.get("case_id"), f"captures[{index}].case_id")
            latency = finite_positive(row.get("latency_us"), f"captures[{index}].latency_us")
            profile = row.get("profile")
            raw_metrics = profile.get("metrics") if isinstance(profile, dict) else None
            evidence_samples[case_id].append({
                "latency_us": latency,
                "profile_metrics": raw_metrics or {},
                "profile_evidence": profile.get("raw") if isinstance(profile, dict) else None,
                "profile_components": profile.get("components", []) if isinstance(profile, dict) else [],
                "evidence_sha256": row.get("evidence_sha256"),
                "provenance": row.get("provenance"),
                "accepted": row.get("accepted"),
            })
            if row.get("accepted") is False:
                continue
            samples[case_id].append(latency)
            if raw_metrics is not None:
                if not isinstance(raw_metrics, dict):
                    raise InvalidReportInput(f"captures[{index}].profile.metrics must be an object")
                for name, value in raw_metrics.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    number = float(value)
                    if math.isfinite(number):
                        metrics[case_id][str(name)].append(number)
            for component in profile.get("components", []) if isinstance(profile, dict) else []:
                key = (
                    component.get("component_selector"),
                    component["metric"], component["selector"],
                    component["block_id"], component["sub_block_id"],
                )
                for name, number in component["metrics"].items():
                    component_metrics[case_id][key][name].append(number)
    result = {}
    for case in sorted(evidence_samples):
        values = samples[case]
        median = statistics.median(values) if values else None
        components = []
        for key, metric_values in sorted(
            component_metrics[case].items(),
            key=lambda item: tuple("" if value is None else value for value in item[0]),
        ):
            component_selector, metric, selector, block, sub_block = key
            output_component = {
                "metric": metric, "selector": selector,
                "block_id": block, "sub_block_id": sub_block,
                "metrics": {
                    name: statistics.median(observations)
                    for name, observations in sorted(metric_values.items())
                },
            }
            if component_selector is not None:
                output_component["component_selector"] = component_selector
            components.append(output_component)
        result[case] = {
            "median_latency_us": median,
            "latency_samples_us": sorted(values),
            "relative_range": ((max(values) - min(values)) / median if values else None),
            "profile_metrics": {
                name: statistics.median(metric_values)
                for name, metric_values in sorted(metrics[case].items())
            },
            "profile_components": components,
            "samples": evidence_samples[case],
        }
    return result


def normalize_producer_inputs(
    manifest: dict[str, Any], results: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_schema = results.get("schema")
    if result_schema not in {
        "profiling-skill/bsa-candidate-results/v1",
        "profiling-skill/bsa-candidate-merged-results/v1",
    }:
        raise InvalidReportInput("producer results schema mismatch")
    occurrences = objects(manifest.get("occurrences"), "manifest.occurrences")
    occurrence_ids = {
        identifier(item.get("occurrence_id"), "manifest occurrence_id")
        for item in occurrences
    }
    declared_starter = manifest.get("starter_occurrence_id")
    role_starters = [
        item["occurrence_id"] for item in occurrences
        if item.get("lineage_role") == "starter"
    ]
    legacy_starters = [
        item["occurrence_id"] for item in occurrences if item.get("is_starter") is True
    ]
    if declared_starter is not None:
        starter_id = identifier(declared_starter, "manifest.starter_occurrence_id")
        if starter_id not in occurrence_ids or role_starters != [starter_id]:
            raise InvalidReportInput(
                "starter_occurrence_id must reference the sole lineage_role starter"
            )
        if legacy_starters and legacy_starters != [starter_id]:
            raise InvalidReportInput("legacy starter alias conflicts with starter_occurrence_id")
    else:
        candidates_for_starter = role_starters or legacy_starters
        if len(candidates_for_starter) > 1:
            raise InvalidReportInput("producer manifest identifies more than one starter occurrence")
        if role_starters and legacy_starters and role_starters != legacy_starters:
            raise InvalidReportInput("producer starter annotations conflict")
        starter_id = candidates_for_starter[0] if candidates_for_starter else None
    executions = {
        identifier(item.get("execution_id"), "manifest execution_id"): item
        for item in objects(manifest.get("executions"), "manifest.executions")
    }
    result_executions = results.get("executions")
    if not isinstance(result_executions, dict):
        raise InvalidReportInput("producer results.executions must be an object")

    first_occurrence: dict[str, str] = {}
    for occurrence in occurrences:
        execution_id = occurrence.get("execution_id")
        if isinstance(execution_id, str) and execution_id:
            first_occurrence.setdefault(execution_id, identifier(
                occurrence.get("occurrence_id"), "manifest occurrence_id"
            ))

    candidates = []
    for occurrence in occurrences:
        occurrence_id = identifier(occurrence.get("occurrence_id"), "manifest occurrence_id")
        execution_id = occurrence.get("execution_id")
        recoverability = occurrence.get("recoverability", "recoverable")
        execution = executions.get(execution_id, {}) if isinstance(execution_id, str) else {}
        result_state = (
            result_executions.get(execution_id, {})
            if isinstance(execution_id, str) else {}
        )
        recoverable = recoverability == "recoverable" and bool(execution)
        parent_id = occurrence.get("parent_occurrence_id", occurrence.get("parent_candidate_id"))
        if parent_id is not None:
            parent_id = identifier(parent_id, f"occurrence {occurrence_id!r} parent")
        is_starter = occurrence_id == starter_id
        note = occurrence.get("lineage_note") or occurrence.get("artifact_diagnostics")
        if not recoverable and not note:
            note = f"candidate is not executable: {recoverability}"
        if recoverable and not parent_id and not is_starter and not note:
            note = "producer supplied no evidence-backed parent"
        if not recoverable:
            lineage_state = "unrecoverable"
        elif is_starter:
            lineage_state = "starter"
        elif parent_id:
            lineage_state = "linked"
        else:
            lineage_state = "unlinked"
        deltas = occurrence.get("parent_feature_deltas", {})
        if not isinstance(deltas, dict):
            raise InvalidReportInput(
                f"occurrence {occurrence_id!r}.parent_feature_deltas must be an object"
            )
        features = execution.get("features", {})
        if not isinstance(features, dict):
            raise InvalidReportInput(f"execution {execution_id!r}.features must be an object")
        candidates.append({
            "candidate_id": occurrence_id,
            "treatment": occurrence.get("treatment"),
            "round": occurrence.get("round"),
            "option": occurrence.get("option_id", occurrence.get("position")),
            "source_hash": occurrence.get("source_sha256", execution.get("source_sha256")),
            "parent_candidate_id": parent_id,
            "duplicate_of": (
                first_occurrence.get(execution_id)
                if recoverable and first_occurrence.get(execution_id) != occurrence_id
                else None
            ),
            "lineage_state": lineage_state,
            "lineage_note": str(note) if note else None,
            "source_features": features,
            "code_delta": deltas,
            "hypothesis": occurrence.get("hypothesis"),
            "execution_id": execution_id,
            "execution_status": result_state.get("status") if isinstance(result_state, dict) else None,
            "kernel_name": result_state.get("kernel_name") if isinstance(result_state, dict) else None,
            "basic_info": result_state.get("basic_info") if isinstance(result_state, dict) else None,
        })
    captures = []
    for execution_id, state in result_executions.items():
        if execution_id not in first_occurrence:
            raise InvalidReportInput(f"results reference unknown execution {execution_id!r}")
        if not isinstance(state, dict):
            raise InvalidReportInput(f"result execution {execution_id!r} must be an object")
        candidate_id = first_occurrence[execution_id]
        cases = state.get("cases", {})
        if not isinstance(cases, dict):
            raise InvalidReportInput(f"result execution {execution_id!r}.cases must be an object")
        for case_id, case in cases.items():
            if not isinstance(case, dict):
                raise InvalidReportInput(f"result case {case_id!r} must be an object")
            # Captures can exist before the producer finds a stable window.
            # Missing selection is not implicit approval, even for a terminal case.
            accepted = set(case.get("accepted_sample_indexes", []))
            for sample_index, sample in enumerate(objects(case.get("samples", []), "result samples")):
                profile = sample_profile(sample)
                captures.append({
                    "candidate_id": candidate_id,
                    "case_id": str(case_id),
                    "status": "valid",
                    "latency_us": sample.get("duration_us"),
                    "profile": {
                        "metrics": profile["metrics"],
                        "components": profile["components"],
                        "raw": profile["raw"],
                    },
                    "evidence_sha256": profile["evidence_sha256"],
                    "provenance": profile["provenance"],
                    "accepted": sample_index in accepted,
                })
        failure = state.get("failure")
        if isinstance(failure, dict):
            category = failure.get("category")
            captures.append({
                "candidate_id": candidate_id,
                "case_id": "__execution__",
                "status": "infrastructure_failure" if category == "infrastructure" else "candidate_failure",
                "failure": {**failure, "kind": failure.get("kind", "unknown"), "message": failure.get("diagnostics", "")},
            })
        for failure in objects(state.get("infrastructure_failures", []), "infrastructure_failures"):
            captures.append({
                "candidate_id": candidate_id,
                "case_id": "__execution__",
                "status": "infrastructure_failure",
                "failure": {**failure, "kind": failure.get("kind", "unknown"), "message": failure.get("diagnostics", "")},
            })
    normalized_manifest = {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_root", "bsa-candidate-campaign"),
        "starter_candidate_id": starter_id,
        "required_case_ids": [
            str(case) for case in (
                results.get("run_contract", {}).get("cases")
                if isinstance(results.get("run_contract"), dict)
                else results.get("cases", [])
            )
        ],
        "campaign_evidence": {
            "result_schema": result_schema,
            "run_contract": results.get("run_contract"),
            "shard_count": results.get("shard_count"),
            "present_shards": results.get("present_shards", []),
            "missing_shards": results.get("missing_shards", []),
            "missing_execution_ids": results.get("missing_execution_ids", []),
            "sources": results.get("sources", []),
        },
        "references": manifest.get("references", []),
        "candidates": candidates,
    }
    for candidate in normalized_manifest["candidates"]:
        if candidate.get("execution_id") in normalized_manifest["campaign_evidence"]["missing_execution_ids"]:
            candidate["execution_status"] = "missing"
    return normalized_manifest, {"schema_version": 1, "captures": captures}


def normalize_inputs(
    manifest: dict[str, Any], results: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest.get("schema") == "profiling-skill/bsa-candidate-manifest/v1":
        return normalize_producer_inputs(manifest, results)
    if manifest.get("schema_version") != 1 or results.get("schema_version") != 1:
        raise InvalidReportInput("manifest and captures must use compatible version 1 schemas")
    return manifest, results


def normalize_hypothesis_inputs(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise InvalidReportInput("hypotheses must be a schema-version 1 object")
    results = objects(document.get("results"), "hypotheses.results")
    if any("hypothesis_id" not in item and "variant" in item for item in results):
        raise InvalidReportInput(
            "raw unpaired microcampaign cells are not valid hypothesis evidence"
        )
    report_ready = "raw_cells" in document
    raw_by_id: dict[str, dict[str, Any]] = {}
    if report_ready:
        for cell in objects(document.get("raw_cells"), "hypotheses.raw_cells"):
            cell_id = identifier(cell.get("id"), "raw cell id")
            if cell_id in raw_by_id:
                raise InvalidReportInput(f"duplicate raw microcampaign cell {cell_id!r}")
            raw_by_id[cell_id] = cell
        identifier(document.get("manifest_sha256"), "hypotheses.manifest_sha256")
    allowed = {
        "valid", "candidate_failure", "infrastructure_failure",
        "profiling_failure", "unavailable", "inconclusive",
    }
    for item in results:
        identifier(item.get("hypothesis_id"), "hypothesis_id")
        measurements = objects(item.get("measurements", []), "hypothesis.measurements")
        for measurement in measurements:
            status = measurement.get("status")
            if status not in allowed:
                raise InvalidReportInput(f"invalid hypothesis measurement status {status!r}")
            if status == "valid":
                finite_positive(measurement.get("before"), "hypothesis measurement before")
                finite_positive(measurement.get("after"), "hypothesis measurement after")
                if report_ready:
                    before_id = identifier(
                        measurement.get("before_cell_id"), "measurement.before_cell_id"
                    )
                    after_id = identifier(
                        measurement.get("after_cell_id"), "measurement.after_cell_id"
                    )
                    if (
                        before_id not in raw_by_id
                        or after_id not in raw_by_id
                        or raw_by_id[before_id].get("variant") != "before"
                        or raw_by_id[after_id].get("variant") != "after"
                    ):
                        raise InvalidReportInput(
                            "paired hypothesis measurement must reference retained before/after cells"
                        )
    return document


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
    # The producer's timeline controls currently report scalar core-time proxies,
    # not semantic order/duration/overlap observations. Their timing changes are
    # retained below but cannot validate (or refute) a timeline interpretation.
    if any(
        str(signal.get("name", "")).startswith("timeline_")
        for signal in objects(item.get("signals", []), "hypothesis.signals")
    ):
        return "unavailable", "scalar timing controls do not establish semantic timeline evidence"
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
    hypothesis_doc = normalize_hypothesis_inputs(hypothesis_doc)
    raw_candidates = objects(manifest.get("candidates"), "manifest.candidates")
    candidates: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(raw_candidates):
        candidate_id = identifier(candidate.get("candidate_id"), f"candidates[{index}].candidate_id")
        if candidate_id in candidates:
            raise InvalidReportInput(f"duplicate candidate_id {candidate_id!r}")
        lineage_state = candidate.get("lineage_state", "complete")
        if lineage_state not in {
            "complete", "incomplete", "unrecoverable", "starter", "linked", "unlinked",
        }:
            raise InvalidReportInput(f"candidate {candidate_id!r} has invalid lineage_state")
        if lineage_state in {"incomplete", "unrecoverable", "unlinked"}:
            identifier(candidate.get("lineage_note"), f"candidate {candidate_id!r}.lineage_note")
        candidates[candidate_id] = candidate
    starter_id = manifest.get("starter_candidate_id")
    if starter_id is not None:
        starter_id = identifier(starter_id, "starter_candidate_id")
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
    declared_cases = manifest.get("required_case_ids")
    if declared_cases:
        if not isinstance(declared_cases, list):
            raise InvalidReportInput("required_case_ids must be an array")
        required_cases = {str(case) for case in declared_cases}
        if len(required_cases) != len(declared_cases):
            raise InvalidReportInput("required_case_ids must be unique")
    else:
        required_cases = None

    def aggregate_coverage(case_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
        available = {
            case_id for case_id, summary in case_summaries.items()
            if summary["median_latency_us"] is not None
        }
        expected = required_cases if required_cases is not None else set(case_summaries)
        missing = expected - available
        return {
            "required_case_ids": sorted(expected),
            "available_case_ids": sorted(available),
            "missing_case_ids": sorted(missing),
            "complete": bool(expected) and not missing,
        }

    starter_summaries = (
        summaries.get(effective_candidate(starter_id, candidates), {})
        if starter_id is not None else {}
    )
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
        coverage = aggregate_coverage(candidate_summaries)
        starter_coverage = aggregate_coverage(starter_summaries)
        parent_coverage = aggregate_coverage(parent_summaries)
        case_results: dict[str, dict[str, Any]] = {}
        starter_ratios, parent_ratios = [], []
        for case_id, summary in candidate_summaries.items():
            latency = summary["median_latency_us"]
            starter = starter_summaries.get(case_id)
            parent = parent_summaries.get(case_id)
            starter_speedup = (
                starter["median_latency_us"] / latency
                if latency is not None and starter and starter["median_latency_us"] is not None
                else None
            )
            parent_speedup = (
                parent["median_latency_us"] / latency
                if latency is not None and parent and parent["median_latency_us"] is not None
                else None
            )
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
        failure_rows = list(captures_by_candidate.get(evidence_id, []))
        if evidence_id != candidate_id:
            failure_rows.extend(captures_by_candidate.get(candidate_id, []))
        for row in failure_rows:
            if row.get("status") == "valid":
                continue
            failure = row.get("failure") if isinstance(row.get("failure"), dict) else {}
            failures.append({
                **failure,
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
            "execution_id": candidate.get("execution_id"),
            "execution_status": candidate.get("execution_status"),
            "kernel_name": candidate.get("kernel_name"),
            "basic_info": candidate.get("basic_info"),
            "evidence_source_candidate_id": evidence_id,
            "source_features": candidate.get("source_features", {}),
            "code_delta": candidate.get("code_delta", []),
            "hypothesis": candidate.get("hypothesis"),
            "cases": case_results,
            "aggregate_coverage": coverage,
            "geometric_mean": {
                "latency_us": geometric_mean([
                    candidate_summaries[case]["median_latency_us"]
                    for case in coverage["required_case_ids"]
                ]) if coverage["complete"] else None,
                "starter_speedup": (
                    geometric_mean(starter_ratios)
                    if coverage["complete"] and starter_coverage["complete"]
                    and len(starter_ratios) == len(coverage["required_case_ids"])
                    else None
                ),
                "parent_speedup": (
                    geometric_mean(parent_ratios)
                    if parent_id is not None and coverage["complete"]
                    and parent_coverage["complete"]
                    and len(parent_ratios) == len(coverage["required_case_ids"])
                    else None
                ),
            },
            "failures": failures,
            "infrastructure_failures": [
                failure for failure in failures
                if failure["status"] == "infrastructure_failure"
            ],
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
            "expected": item.get("expected"),
            "falsification": item.get("falsification"),
            "measurements": item.get("measurements", []),
            "signals": item.get("signals", []),
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

    terminal_statuses = {"completed", "candidate_failure", "measurement_unstable"}
    terminal = all(
        item["lineage_state"] == "unrecoverable"
        or (
            item["execution_status"] in terminal_statuses
            if item["execution_status"] is not None
            else item["aggregate_coverage"]["complete"]
            or any(failure["status"] == "candidate_failure" for failure in item["failures"])
        )
        for item in compiled_candidates
    )
    valid_evidence = any(item["cases"] for item in compiled_candidates)
    hypotheses_complete = bool(compiled_hypotheses) and all(
        item["classification"] != "unavailable" for item in compiled_hypotheses
    )
    has_evidence = valid_evidence or any(item["failures"] for item in compiled_candidates)
    campaign_evidence = manifest.get("campaign_evidence", {})
    campaign_complete = not (
        campaign_evidence.get("missing_shards")
        or campaign_evidence.get("missing_execution_ids")
    )
    if not has_evidence:
        report_state = "reproducible_scaffold"
    elif terminal and hypotheses_complete and campaign_complete:
        report_state = "complete"
    else:
        report_state = "partial"
    return {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_id"),
        "report_state": report_state,
        "references": manifest.get("references", []),
        "campaign_evidence": campaign_evidence,
        "hypothesis_evidence": {
            "campaign_identity": hypothesis_doc.get("campaign_identity"),
            "reporting_revision": hypothesis_doc.get("reporting_revision"),
            "manifest_sha256": hypothesis_doc.get("manifest_sha256"),
            "raw_cell_count": len(hypothesis_doc.get("raw_cells", [])),
            "paired_result_count": len(compiled_hypotheses),
            "raw_cells": hypothesis_doc.get("raw_cells", []),
        },
        "candidates": compiled_candidates,
        "hypotheses": sorted(compiled_hypotheses, key=lambda row: row["hypothesis_id"]),
        "signals": signal_rows,
    }


def fmt(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6g}"


def speedup(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.6g}x"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def render_markdown(report: dict[str, Any], *, compact_evidence: bool = False) -> str:
    state = report["report_state"]
    if state == "complete":
        banner = "**Complete retained evidence.** All declared candidates and controlled hypotheses reached terminal evidence states."
    elif state == "partial":
        banner = "**Partial evidence.** Missing, retryable, or unavailable evidence remains explicit below."
    else:
        banner = "**Reproducible scaffold.** Replace fixture inputs with retained campaign outputs; missing evidence remains explicit."
    lines = [
        "# A3 profiling reverse-engineering report",
        "",
        f"> {banner}",
        "",
        f"Campaign: `{report.get('campaign_id')}`",
        "",
    ]
    campaign = report.get("campaign_evidence", {})
    if campaign:
        lines.extend([
            f"Result schema: `{campaign.get('result_schema')}`",
            f"Missing shards: `{', '.join(map(str, campaign.get('missing_shards', []))) or 'none'}`",
            f"Missing executions: `{', '.join(campaign.get('missing_execution_ids', [])) or 'none'}`",
            "",
        ])
    lines.extend([
        "## Candidate results", "",
        "| Candidate | Treatment | Parent | Duplicate of | Geomean µs | vs starter | vs parent |",
        "|---|---|---|---|---:|---:|---:|",
    ])
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
            rendered = (
                json.dumps(deltas, sort_keys=True, separators=(",", ":"))
                if isinstance(deltas, dict)
                else "; ".join(map(str, deltas))
            )
            lines.append(f"- Code delta for `{item['candidate_id']}`: {rendered}.")
        if item.get("hypothesis"):
            lines.append(f"- Candidate hypothesis for `{item['candidate_id']}`: {item['hypothesis']}.")
        if item["failures"]:
            for failure in item["failures"]:
                message = failure["message"]
                if compact_evidence and len(message) > 2000:
                    message = message[:2000] + "… [full diagnostic in report.json]"
                lines.append(
                    f"- Failure `{item['candidate_id']}` case `{failure['case_id']}`: "
                    f"{failure['status']}/{failure['kind']} — {message}"
                )
    lines.extend(["", "### Auditable candidate evidence", ""])
    if compact_evidence:
        lines.append("Full samples, per-component profiles and failure provenance are retained in [report.json](report.json).")
    for item in ([] if compact_evidence else report["candidates"]):
        candidate_id = item["candidate_id"]
        coverage = item["aggregate_coverage"]
        lines.extend([
            f"#### `{candidate_id}`", "",
            "- Aggregate coverage: "
            f"complete={str(coverage['complete']).lower()}; "
            f"required={compact_json(coverage['required_case_ids'])}; "
            f"available={compact_json(coverage['available_case_ids'])}; "
            f"missing={compact_json(coverage['missing_case_ids'])}.",
        ])
        if item.get("basic_info") is not None:
            lines.extend([
                f"- BasicInfo for `{candidate_id}`:", "",
                "```json", json.dumps(item["basic_info"], indent=2, sort_keys=True), "```",
            ])
        else:
            lines.append(f"- BasicInfo for `{candidate_id}`: unavailable.")
        if not item["cases"]:
            lines.extend(["- Per-case retained evidence: unavailable.", ""])
            continue
        for case_id, case in sorted(item["cases"].items()):
            lines.extend([
                f"##### Case `{case_id}`", "",
                f"- Median latency: {fmt(case['median_latency_us'])} µs; "
                f"relative range: {fmt(case['relative_range'])}; "
                f"accepted samples: {len(case['latency_samples_us'])}; "
                f"retained samples: {len(case['samples'])}.",
            ])
            for index, sample in enumerate(case["samples"]):
                accepted = sample.get("accepted") is not False
                evidence = sample.get("evidence_sha256") or "unavailable"
                rendered_evidence = (
                    evidence if isinstance(evidence, str) else compact_json(evidence)
                )
                lines.append(
                    f"- Sample {index}: latency={fmt(sample.get('latency_us'))} µs; "
                    f"accepted={str(accepted).lower()}; "
                    f"evidence_sha256=`{rendered_evidence}`; "
                    f"provenance={compact_json(sample.get('provenance'))}."
                )
                if sample.get("profile_evidence") is not None:
                    lines.extend([
                        f"  Raw profile evidence for sample {index}:", "",
                        "```json",
                        json.dumps(sample["profile_evidence"], indent=2, sort_keys=True),
                        "```",
                    ])
            if case["profile_metrics"]:
                lines.append(
                    f"- Comparable scalar profile metrics: {compact_json(case['profile_metrics'])}."
                )
            for component in case["profile_components"]:
                component_boundary = (
                    f"component=`{component['component_selector']}` "
                    if component.get("component_selector") is not None else ""
                )
                lines.append(
                    f"- Pipe component `{component['metric']}` "
                    f"{component_boundary}"
                    f"selector=`{component['selector']}` block=`{component['block_id']}` "
                    f"sub-block=`{component['sub_block_id']}`: "
                    f"{compact_json(component['metrics'])}."
                )
            lines.append("")
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


def write_outputs(report: dict[str, Any], output_dir: Path, *, compact_markdown: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report, compact_evidence=compact_markdown), encoding="utf-8")
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
    evidence = result.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--results", type=Path, help="producer results or normalized captures")
    evidence.add_argument("--captures", type=Path, help="legacy alias for --results")
    result.add_argument("--hypotheses", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--compact-markdown", action="store_true", help="keep verbose sample evidence in JSON only and bound Markdown diagnostics")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        manifest, captures = normalize_inputs(
            read_object(args.manifest, "manifest"),
            read_object(args.results or args.captures, "results"),
        )
        report = compile_report(manifest, captures, read_object(args.hypotheses, "hypotheses"))
        write_outputs(report, args.output_dir, compact_markdown=args.compact_markdown)
    except InvalidReportInput as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
