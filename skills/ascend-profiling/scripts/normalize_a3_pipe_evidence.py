#!/usr/bin/env python3
"""Normalize single-launch A2/A3 pipe rows as non-temporal observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any


NAME_FIELDS = ("Op Name", "OpName", "Kernel Name", "kernel_name")
MEMORY_FIELDS = {
    "gm_to_l1": "GM_to_L1_datas(KB)",
    "l0c_to_l1": "L0C_to_L1_datas(KB)",
    "l0c_to_gm": "L0C_to_GM_datas(KB)",
    "gm_to_ub": "GM_to_UB_datas(KB)",
    "ub_to_gm": "UB_to_GM_datas(KB)",
}
UNAVAILABLE = {"", "na", "n/a", "nan", "none", "null", "unknown"}


class InvalidEvidence(ValueError):
    pass


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            return fields, list(reader)
    except OSError as error:
        raise InvalidEvidence(f"cannot read {path}: {error}") from error


def read_model(path: Path) -> dict[str, Any]:
    try:
        model = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidEvidence(f"cannot read activity model {path}: {error}") from error
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise InvalidEvidence("activity model must be a schema-version 1 object")
    if (
        model.get("product") != "a3"
        or model.get("evidence_kind") != "per-block-pipe-activity"
    ):
        raise InvalidEvidence("activity model must describe A3 per-block pipe activity")
    for field in ("model_id", "provenance", "saturation"):
        if not model.get(field):
            raise InvalidEvidence(f"activity model is missing {field}")
    saturation = model["saturation"]
    if (
        not isinstance(saturation, dict)
        or saturation.get("state") != "unknown"
        or not saturation.get("reason")
    ):
        raise InvalidEvidence("activity model saturation must be unknown with a reason")
    resources = model.get("resources")
    if not isinstance(resources, list) or not resources:
        raise InvalidEvidence("activity model resources must be a non-empty array")
    names, fields = set(), set()
    for resource in resources:
        if not isinstance(resource, dict) or resource.get("evidence_kind") != "activity_only":
            raise InvalidEvidence("every activity model resource must be activity_only")
        name, export = resource.get("resource"), resource.get("export_field")
        if not isinstance(name, str) or not name or name in names:
            raise InvalidEvidence("activity model resource names must be unique strings")
        if not isinstance(export, str) or not export or export in fields:
            raise InvalidEvidence("activity model export fields must be unique strings")
        names.add(name)
        fields.add(export)
    return model


def select_field(fields: list[str], choices: tuple[str, ...], label: str) -> str:
    selected = next((name for name in choices if name in fields), None)
    if selected is None:
        raise InvalidEvidence(f"missing {label} field; expected one of {choices}")
    return selected


def validate_single_launch(path: Path, kernel: str, label: str) -> None:
    fields, rows = read_rows(path)
    name_field = select_field(fields, NAME_FIELDS, f"{label} operator name")
    if len(rows) != 1:
        raise InvalidEvidence(
            f"{label} BasicInfo must contain exactly one total row for the single-launch contract"
        )
    observed = rows[0].get(name_field, "").strip()
    if observed != kernel:
        raise InvalidEvidence(
            f"{label} BasicInfo row {observed!r} does not match exact selector {kernel!r}"
        )


def number(value: Any, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise InvalidEvidence(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(parsed):
        raise InvalidEvidence(f"{label} must be finite, got {value!r}")
    return parsed


def optional_number(value: Any, label: str) -> float | None:
    if value is None or str(value).strip().lower() in UNAVAILABLE:
        return None
    return number(value, label)


def duration_ns(value: str) -> int:
    try:
        nanoseconds = Decimal(value.strip()) * 1000
    except InvalidOperation as error:
        raise InvalidEvidence(f"task duration must be numeric, got {value!r}") from error
    result = int(nanoseconds.to_integral_value(rounding=ROUND_CEILING))
    if result <= 0:
        raise InvalidEvidence("task duration must be positive")
    return result


def topology_rows(
    path: Path, label: str
) -> tuple[list[str], list[dict[str, str]], set[tuple[str, str]]]:
    fields, rows = read_rows(path)
    for required in ("block_id", "sub_block_id"):
        if required not in fields:
            raise InvalidEvidence(f"{label} evidence is missing {required!r}")
    if not rows:
        raise InvalidEvidence(f"{label} evidence has no rows")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["block_id"].strip(), row["sub_block_id"].strip())
        if not all(key):
            raise InvalidEvidence(f"{label} topology keys must not be empty")
        if key in keys:
            raise InvalidEvidence(f"duplicate {label} topology key {key!r}")
        keys.add(key)
    return fields, rows, keys


def memory_values(
    fields: list[str], rows: list[dict[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        values = {}
        for name, column in MEMORY_FIELDS.items():
            if column in fields:
                values[name] = {
                    "raw_value": row.get(column, "").strip(),
                    "unit": None,
                    "source_column_label": column,
                    "scaling": "unvalidated",
                }
        result[(row["block_id"].strip(), row["sub_block_id"].strip())] = values
    return result


def memory_bound(row: dict[str, str], prefix: str) -> dict[str, object] | None:
    compute_field = "aic_cube_ratio" if prefix == "aic" else "aiv_vec_ratio"
    movement = optional_number(row.get(f"{prefix}_mte2_ratio"), f"{prefix}_mte2_ratio")
    compute = optional_number(row.get(compute_field), compute_field)
    if movement is None or compute is None or compute == 0:
        return None
    ratio = movement / compute
    meaning = (
        "no_memory_bottleneck" if ratio < 1 else
        "memory_activity_dominates" if ratio > 1 else
        "unclassified_boundary"
    )
    return {"value": ratio, "interpretation": meaning}


def source_digest(sources: list[tuple[str, Path]]) -> tuple[str, list[dict[str, str]]]:
    combined = hashlib.sha256()
    records = []
    for role, path in sources:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        combined.update(role.encode())
        combined.update(b"\0")
        combined.update(path.name.encode())
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
        records.append({"role": role, "name": path.name, "sha256": digest})
    return combined.hexdigest(), records


def require_same_report(basic: Path, data: Path, label: str) -> None:
    if basic.resolve().parent != data.resolve().parent:
        raise InvalidEvidence(
            f"{label} BasicInfo and data must come from the same report directory"
        )


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    model = read_model(args.activity_model)
    require_same_report(args.basic_info, args.pipe_utilization, "pipe")
    validate_single_launch(args.basic_info, args.kernel_name, "pipe")
    pipe_fields, pipe_rows, pipe_keys = topology_rows(args.pipe_utilization, "pipe")

    memory_requested = any(
        value is not None
        for value in (args.memory_access, args.memory_basic_info, args.memory_capture_id)
    )
    memory: dict[tuple[str, str], dict[str, Any]] = {}
    sources = [
        ("pipe_basic_info", args.basic_info),
        ("pipe_utilization", args.pipe_utilization),
    ]
    if memory_requested:
        if not all((args.memory_access, args.memory_basic_info, args.memory_capture_id)):
            raise InvalidEvidence(
                "memory replay requires --memory-access, --memory-basic-info, "
                "and --memory-capture-id"
            )
        if args.memory_capture_id == args.capture_id:
            raise InvalidEvidence("memory capture ID must be distinct from pipe capture ID")
        require_same_report(args.memory_basic_info, args.memory_access, "memory")
        validate_single_launch(args.memory_basic_info, args.kernel_name, "memory")
        memory_fields, memory_rows, memory_keys = topology_rows(args.memory_access, "memory")
        if memory_keys != pipe_keys:
            missing = sorted(pipe_keys - memory_keys)
            extra = sorted(memory_keys - pipe_keys)
            raise InvalidEvidence(
                "memory topology key set must exactly equal pipe topology; "
                f"missing={missing}, extra={extra}"
            )
        memory = memory_values(memory_fields, memory_rows)
        sources.extend([
            ("memory_basic_info", args.memory_basic_info),
            ("memory_access", args.memory_access),
        ])

    observations = []
    resources = [
        (item["resource"], item["export_field"]) for item in model["resources"]
    ]
    for row in pipe_rows:
        block, sub_block = row["block_id"].strip(), row["sub_block_id"].strip()
        aic_time = optional_number(row.get("aic_time(us)"), "aic_time(us)")
        aiv_time = optional_number(row.get("aiv_time(us)"), "aiv_time(us)")
        if (aic_time is None) == (aiv_time is None):
            raise InvalidEvidence("each pipe row must contain exactly one AIC or AIV duration")
        prefix = "aic" if aic_time is not None else "aiv"
        duration_field = f"{prefix}_time(us)"
        activity, metrics = {}, {}
        for resource, field in resources:
            if field not in pipe_fields:
                continue
            ratio = optional_number(row.get(field), field)
            if ratio is None:
                continue
            if not 0 <= ratio <= 1:
                raise InvalidEvidence(f"{field} must be a fraction between 0 and 1")
            activity[resource] = ratio
            metrics[f"{resource}_activity_fraction"] = ratio
        cycles_field = f"{prefix}_total_cycles"
        cycles = optional_number(row.get(cycles_field), cycles_field)
        if cycles is not None:
            if cycles < 0 or not cycles.is_integer():
                raise InvalidEvidence(f"{cycles_field} must be a non-negative integer")
            metrics[cycles_field] = int(cycles)
        observation: dict[str, Any] = {
            "observation_id": f"{args.kernel_name}:block-{block}:{sub_block}",
            "topology": {"block_id": block, "sub_block_id": sub_block},
            "engine": prefix,
            "duration_ns": duration_ns(row[duration_field]),
            "duration_provenance": {
                "reported_duration_us": row[duration_field].strip(),
                "resolution_ns": 1,
                "rounding": "ceiling",
            },
            "activity": activity,
            "metrics": metrics,
            "saturation": model["saturation"],
        }
        diagnostic = memory_bound(row, prefix)
        if diagnostic is not None:
            observation["diagnostics"] = {"memory_bound": diagnostic}
        if memory:
            observation["memory_access"] = memory[(block, sub_block)]
        observations.append(observation)

    digest, source_records = source_digest(sources)
    provenance: dict[str, Any] = {
        "capture_id": args.capture_id,
        "kernel_selector": args.kernel_name,
        "launch_count": 1,
        "capture_contract": "single-launch exact-selector",
        "source_sha256": digest,
        "sources": source_records,
    }
    if memory:
        provenance["memory_capture_id"] = args.memory_capture_id
        provenance["memory_join"] = "exact block_id/sub_block_id key-set equality"
    return {
        "schema_version": 1,
        "product": "a3",
        "evidence_kind": "per-block-pipe-activity",
        "activity_model": {
            "model_id": model["model_id"],
            "provenance": model["provenance"],
        },
        "provenance": provenance,
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, choices=("a3",))
    parser.add_argument("--activity-model", type=Path, required=True)
    parser.add_argument("--basic-info", type=Path, required=True)
    parser.add_argument("--pipe-utilization", type=Path, required=True)
    parser.add_argument("--memory-access", type=Path)
    parser.add_argument("--memory-basic-info", type=Path)
    parser.add_argument("--memory-capture-id")
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--launch-count", type=int, choices=(1,), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if not args.capture_id.strip():
            raise InvalidEvidence("capture ID must not be empty")
        result = normalize(args)
    except InvalidEvidence as error:
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
