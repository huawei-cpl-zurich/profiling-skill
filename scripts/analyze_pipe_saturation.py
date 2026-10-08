#!/usr/bin/env python3
"""Classify phase-local resource saturation from a validated product model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


UNKNOWN_MARKERS = {"", "na", "n/a", "nan", "none", "null", "unknown"}


class InvalidSaturationInput(ValueError):
    """The model or evidence cannot be analyzed safely."""


def require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidSaturationInput(f"{location} must be an object")
    return value


def require_provenance(value: Any, location: str) -> dict[str, Any]:
    provenance = require_mapping(value, location)
    if not provenance:
        raise InvalidSaturationInput(f"{location} must not be empty")
    return provenance


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSaturationInput(f"{location} must be a non-empty string")
    return value


def available_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value.strip().lower() in UNKNOWN_MARKERS:
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_model(model: dict[str, Any], product: str) -> list[dict[str, Any]]:
    if model.get("schema_version") != 1:
        raise InvalidSaturationInput("model schema_version must be 1")
    model_product = require_string(model.get("product"), "model.product")
    if model_product != product:
        raise InvalidSaturationInput(
            f"requested product {product!r} does not match model product {model_product!r}"
        )
    require_string(model.get("model_id"), "model.model_id")
    require_provenance(model.get("provenance"), "model.provenance")
    resources = model.get("resources")
    if not isinstance(resources, list) or not resources:
        raise InvalidSaturationInput("model.resources must be a non-empty array")

    seen = set()
    checked = []
    for index, raw in enumerate(resources):
        resource = require_mapping(raw, f"model.resources[{index}]")
        name = require_string(resource.get("resource"), f"model.resources[{index}].resource")
        if name in seen:
            raise InvalidSaturationInput(f"duplicate model resource {name!r}")
        seen.add(name)
        require_string(
            resource.get("numerator_metric"),
            f"model.resources[{index}].numerator_metric",
        )
        require_string(
            resource.get("denominator_metric"),
            f"model.resources[{index}].denominator_metric",
        )
        threshold = available_number(resource.get("saturation_threshold"))
        if threshold is None or not 0 < threshold <= 1:
            raise InvalidSaturationInput(
                f"model resource {name!r} saturation_threshold must be in (0, 1]"
            )
        validation = require_mapping(
            resource.get("capacity_validation"),
            f"model.resources[{index}].capacity_validation",
        )
        state = require_string(
            validation.get("state"),
            f"model.resources[{index}].capacity_validation.state",
        )
        if state == "validated":
            require_string(
                validation.get("source"),
                f"model.resources[{index}].capacity_validation.source",
            )
        checked.append(resource)
    return sorted(checked, key=lambda item: item["resource"])


def classify_resource(resource: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    name = resource["resource"]
    numerator_name = resource["numerator_metric"]
    denominator_name = resource["denominator_metric"]
    threshold = float(resource["saturation_threshold"])
    result = {
        "resource": name,
        "state": "unknown",
        "ratio": None,
        "saturation_threshold": threshold,
        "numerator_metric": numerator_name,
        "denominator_metric": denominator_name,
        "capacity_source": resource["capacity_validation"].get("source"),
    }

    if resource["capacity_validation"]["state"] != "validated":
        result["reason"] = "capacity denominator is not validated for this product model"
        return result

    numerator = available_number(metrics.get(numerator_name))
    denominator = available_number(metrics.get(denominator_name))
    if numerator is None or denominator is None:
        result["reason"] = "capacity numerator or denominator metric is missing"
        return result
    if denominator <= 0:
        result["reason"] = "capacity denominator must be positive"
        return result
    if numerator < 0:
        result["reason"] = "capacity numerator must be non-negative"
        return result

    ratio = numerator / denominator
    result["ratio"] = ratio
    result["state"] = "saturated" if ratio >= threshold else "unsaturated"
    result["reason"] = "ratio meets threshold" if ratio >= threshold else "ratio is below threshold"
    return result


def analyze(model: dict[str, Any], evidence: dict[str, Any], product: str) -> dict[str, Any]:
    resources = validate_model(model, product)
    if evidence.get("schema_version") != 1:
        raise InvalidSaturationInput("evidence schema_version must be 1")
    evidence_product = require_string(evidence.get("product"), "evidence.product")
    if evidence_product != product:
        raise InvalidSaturationInput(
            f"requested product {product!r} does not match evidence product {evidence_product!r}"
        )
    provenance = require_provenance(evidence.get("provenance"), "evidence.provenance")
    phases = evidence.get("phases")
    if not isinstance(phases, list) or not phases:
        raise InvalidSaturationInput("evidence.phases must be a non-empty array")

    phase_results = []
    seen_phases = set()
    for index, raw in enumerate(phases):
        phase = require_mapping(raw, f"evidence.phases[{index}]")
        phase_id = require_string(phase.get("phase_id"), f"evidence.phases[{index}].phase_id")
        if phase_id in seen_phases:
            raise InvalidSaturationInput(f"duplicate evidence phase_id {phase_id!r}")
        seen_phases.add(phase_id)
        start = available_number(phase.get("start_ns"))
        end = available_number(phase.get("end_ns"))
        if start is None or start < 0:
            raise InvalidSaturationInput(
                f"evidence phase {phase_id!r} start_ns must be finite and non-negative"
            )
        if end is None or end <= start:
            raise InvalidSaturationInput(
                f"evidence phase {phase_id!r} end_ns must be greater than start_ns"
            )
        metrics = require_mapping(phase.get("metrics"), f"evidence.phases[{index}].metrics")
        classified = [classify_resource(resource, metrics) for resource in resources]
        saturated = [item["resource"] for item in classified if item["state"] == "saturated"]
        states = {item["state"] for item in classified}
        if saturated:
            state = "saturated"
        elif states == {"unsaturated"}:
            state = "unsaturated"
        else:
            state = "unknown"
        phase_results.append(
            {
                "phase_id": phase_id,
                "interval_ns": {"start": start, "end": end},
                "state": state,
                "saturated_resources": saturated,
                "resources": classified,
            }
        )

    return {
        "schema_version": 1,
        "product": product,
        "model": {
            "model_id": model["model_id"],
            "provenance": model["provenance"],
        },
        "evidence_provenance": provenance,
        "phases": phase_results,
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return require_mapping(json.loads(path.read_text()), label)
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidSaturationInput(f"cannot read {label} {path}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify phase-local saturation using a validated product model."
    )
    parser.add_argument("--product", required=True, choices=("a3", "a5"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(
            read_json(args.model, "model"),
            read_json(args.evidence, "evidence"),
            args.product,
        )
    except InvalidSaturationInput as error:
        parser.error(str(error))

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
