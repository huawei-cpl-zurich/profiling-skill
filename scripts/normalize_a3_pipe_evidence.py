#!/usr/bin/env python3
"""Normalize compact A2/A3 msprof-op rows without inventing capacity."""

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
PIPE_FIELDS = {
    "aic_cube": "aic_cube_ratio",
    "aic_mte1": "aic_mte1_ratio",
    "aic_mte2": "aic_mte2_ratio",
    "aic_mte3": "aic_mte3_ratio",
    "aic_fixpipe": "aic_fixpipe_ratio",
    "aic_scalar": "aic_scalar_ratio",
    "aiv_vector": "aiv_vec_ratio",
    "aiv_mte2": "aiv_mte2_ratio",
    "aiv_mte3": "aiv_mte3_ratio",
    "aiv_scalar": "aiv_scalar_ratio",
}
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


def select_field(fields: list[str], choices: tuple[str, ...], label: str) -> str:
    selected = next((name for name in choices if name in fields), None)
    if selected is None:
        raise InvalidEvidence(f"missing {label} field; expected one of {choices}")
    return selected


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
    integral = nanoseconds.to_integral_value(rounding=ROUND_CEILING)
    result = int(integral)
    if result <= 0:
        raise InvalidEvidence("task duration must be positive")
    return result


def source_digest(paths: list[Path]) -> tuple[str, list[dict[str, str]]]:
    combined = hashlib.sha256()
    sources = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        combined.update(path.name.encode())
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
        sources.append({"name": path.name, "sha256": digest})
    return combined.hexdigest(), sources


def memory_values(path: Path | None) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    if path is None:
        return {}
    fields, rows = read_rows(path)
    for required in ("block_id", "sub_block_id"):
        if required not in fields:
            raise InvalidEvidence(f"memory evidence is missing {required!r}")
    result = {}
    for row in rows:
        values = {}
        for key, field in MEMORY_FIELDS.items():
            if field in fields:
                values[key] = {
                    "raw_value": row.get(field, "").strip(),
                    "reported_unit": "KB",
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
    if ratio < 1:
        meaning = "no_memory_bottleneck"
    elif ratio > 1:
        meaning = "memory_activity_dominates"
    else:
        meaning = "unclassified_boundary"
    return {"value": ratio, "interpretation": meaning}


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.basic_info.resolve().parent != args.pipe_utilization.resolve().parent:
        raise InvalidEvidence("basic-info and pipe-utilization must come from the same report directory")
    basic_fields, basic_rows = read_rows(args.basic_info)
    basic_name = select_field(basic_fields, NAME_FIELDS, "operator name")
    selected_basic = [
        row for row in basic_rows
        if args.kernel_name is None or row.get(basic_name, "").strip() == args.kernel_name
    ]
    if len(selected_basic) != 1:
        raise InvalidEvidence(
            f"expected exactly one basic-info row for kernel {args.kernel_name!r}, "
            f"found {len(selected_basic)}"
        )
    kernel_name = selected_basic[0][basic_name].strip()
    if args.memory_access is not None:
        if args.memory_basic_info is None:
            raise InvalidEvidence("--memory-basic-info is required with --memory-access")
        if args.memory_basic_info.resolve().parent != args.memory_access.resolve().parent:
            raise InvalidEvidence("memory-basic-info and memory-access must come from the same report directory")
        memory_fields, memory_rows = read_rows(args.memory_basic_info)
        memory_name = select_field(memory_fields, NAME_FIELDS, "memory operator name")
        matching_memory = [row for row in memory_rows if row.get(memory_name, "").strip() == args.kernel_name]
        if len(matching_memory) != 1:
            raise InvalidEvidence(
                f"expected exactly one memory basic-info row for kernel {args.kernel_name!r}, "
                f"found {len(matching_memory)}"
            )
    fields, rows = read_rows(args.pipe_utilization)
    for required in ("block_id", "sub_block_id"):
        if required not in fields:
            raise InvalidEvidence(f"pipe evidence is missing {required!r}")
    if not rows:
        raise InvalidEvidence(f"no pipe rows match kernel {args.kernel_name!r}")
    memory = memory_values(args.memory_access)
    phases = []
    for row in rows:
        activity = {}
        metrics: dict[str, int | float] = {}
        for pipe, field in PIPE_FIELDS.items():
            if field not in fields:
                continue
            ratio = optional_number(row.get(field), field)
            if ratio is None:
                continue
            if not 0 <= ratio <= 1:
                raise InvalidEvidence(f"{field} must be a fraction between 0 and 1")
            activity[pipe] = ratio
            metrics[f"{pipe}_activity_fraction"] = ratio
        prefix = "aic" if optional_number(row.get("aic_time(us)"), "aic_time(us)") is not None else "aiv"
        cycles_field = f"{prefix}_total_cycles"
        cycles = optional_number(row.get(cycles_field), cycles_field)
        if cycles is not None:
            if cycles < 0 or not cycles.is_integer():
                raise InvalidEvidence(f"{cycles_field} must be a non-negative integer")
            metrics[cycles_field] = int(cycles)
        diagnostic = memory_bound(row, prefix)
        block = row["block_id"].strip()
        sub_block = row["sub_block_id"].strip()
        phase: dict[str, Any] = {
            "phase_id": f"{kernel_name}:block-{block}:{sub_block}",
            "start_ns": 0,
            "end_ns": duration_ns(row[f"{prefix}_time(us)"]),
            "metrics": metrics,
            "activity": activity,
            "interval_provenance": {
                "reported_duration_us": row[f"{prefix}_time(us)"].strip(),
                "resolution_ns": 1,
                "rounding": "ceiling",
                "scope": "per-export-row-task-relative",
                "cross_row_alignable": False,
            },
        }
        if diagnostic is not None:
            phase["diagnostics"] = {"memory_bound": diagnostic}
        if (block, sub_block) in memory:
            phase["memory_access"] = memory[(block, sub_block)]
        phases.append(phase)
    paths = [args.basic_info, args.pipe_utilization]
    if args.memory_access:
        paths.extend((args.memory_basic_info, args.memory_access))
    digest, sources = source_digest(paths)
    return {
        "schema_version": 1,
        "product": "a3",
        "provenance": {
            "capture_id": args.capture_id,
            "kernel_selector": args.kernel_name,
            "row_association": "exact selector plus same report directory",
            "source_sha256": digest,
            "sources": sources,
        },
        "phases": phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, choices=("a3",))
    parser.add_argument("--basic-info", type=Path, required=True)
    parser.add_argument("--pipe-utilization", type=Path, required=True)
    parser.add_argument("--memory-access", type=Path)
    parser.add_argument("--memory-basic-info", type=Path)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--kernel-name", required=True)
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
