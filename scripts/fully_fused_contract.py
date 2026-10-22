#!/usr/bin/env python3
"""Validate the fully-fused candidate declaration and trusted runtime evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA = "profiling-skill/candidate-kernel/v2"
EVIDENCE_SCHEMA = "profiling-skill/fusion-evidence/v1"
FUSION_DECLARATION = {
    "schema_version": 1,
    "mode": "single-logical-launch",
    "complete_operator": True,
}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.$:/-]*")


class FusionContractError(ValueError):
    """Candidate declaration or trusted runtime evidence violates the contract."""


def _name(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or _NAME.fullmatch(value) is None):
        raise FusionContractError(f"{label} must be a nonempty exact exported name")
    return value


def _belongs_to_entrypoint(kernel: str, entrypoint: str) -> bool:
    return kernel == entrypoint or kernel in {
        f"{entrypoint}_mix_aic", f"{entrypoint}_mix_aiv"
    }


def validate_manifest(value: object) -> dict[str, Any]:
    """Return a normalized v2 manifest or fail closed."""
    if not isinstance(value, dict):
        raise FusionContractError("candidate manifest must be an object")
    if set(value) != {"schema", "kernel_name", "entrypoint", "fusion"}:
        raise FusionContractError("candidate manifest has missing or undeclared fields")
    if value.get("schema") != MANIFEST_SCHEMA:
        raise FusionContractError(f"candidate manifest requires schema {MANIFEST_SCHEMA}")
    kernel = _name(value.get("kernel_name"), "kernel_name")
    entrypoint = _name(value.get("entrypoint"), "entrypoint")
    if not _belongs_to_entrypoint(kernel, entrypoint):
        raise FusionContractError("kernel_name does not belong to the declared entrypoint")
    if value.get("fusion") != FUSION_DECLARATION:
        raise FusionContractError("candidate manifest requires the exact v1 fusion declaration")
    return {
        "schema": MANIFEST_SCHEMA,
        "kernel_name": kernel,
        "entrypoint": entrypoint,
        "fusion": dict(FUSION_DECLARATION),
    }


def _case_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FusionContractError("fusion evidence case must be a non-negative integer")
    return value


def validate_fusion_evidence(
    manifest: object, evidence: object, *, expected_cases: Iterable[int]
) -> dict[str, Any]:
    """Validate trusted evidence from one isolated candidate forward per case.

    ``operators`` is the complete compute-op stream captured for the forward;
    launch identifiers group compiler-emitted AIC/AIV components from the same
    logical Triton invocation. The trusted runner identifies which launch
    produced the returned output through ``output_launch_id``.
    """
    declaration = validate_manifest(manifest)
    expected = [_case_id(case) for case in expected_cases]
    if len(set(expected)) != len(expected):
        raise FusionContractError("expected case inventory contains duplicates")
    if not isinstance(evidence, list) or not all(isinstance(row, dict) for row in evidence):
        raise FusionContractError("fusion evidence must be an array of case objects")
    observed = [_case_id(row.get("case")) for row in evidence]
    if observed != expected or len(set(observed)) != len(observed):
        raise FusionContractError("fusion evidence case coverage is not exact and ordered")

    for row in evidence:
        if row.get("schema") != EVIDENCE_SCHEMA:
            raise FusionContractError("fusion evidence schema is unsupported")
        operators = row.get("operators")
        if not isinstance(operators, list):
            raise FusionContractError("fusion evidence operators must be an array")
        forbidden = [op for op in operators if isinstance(op, dict)
                     and op.get("origin") in {"torch", "acl"}]
        if forbidden:
            raise FusionContractError("isolated forward contains Torch/ACL compute")
        if any(not isinstance(op, dict) or op.get("origin") != "triton"
               for op in operators):
            raise FusionContractError("isolated forward contains unclassified compute")
        launch_ids = {op.get("launch_id") for op in operators
                      if isinstance(op.get("launch_id"), str) and op["launch_id"]}
        if len(operators) == 0 or len(launch_ids) != 1 or any(
                op.get("launch_id") not in launch_ids for op in operators):
            raise FusionContractError("case must execute exactly one logical Triton launch")
        if any(op.get("entrypoint") != declaration["entrypoint"] for op in operators):
            raise FusionContractError("Triton launch uses an unrelated entrypoint")
        if any(op.get("component") not in {"aic", "aiv"} for op in operators):
            raise FusionContractError("Triton launch has an invalid compiler component")
        names = [op.get("name") for op in operators]
        if (declaration["kernel_name"] not in names
                or any(not isinstance(name, str)
                       or not _belongs_to_entrypoint(name, declaration["entrypoint"])
                       for name in names)):
            raise FusionContractError("profile selector does not belong to the declared entrypoint")
        launch_id = next(iter(launch_ids))
        if row.get("output_launch_id") != launch_id:
            raise FusionContractError("candidate output is not produced by the Triton launch")

    return {
        "entrypoint": declaration["entrypoint"],
        "kernel_name": declaration["kernel_name"],
        "cases": expected,
        "logical_launches_per_case": 1,
    }


def _cases(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cases must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one case is required")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected-cases", type=_cases, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        evidence = json.loads(args.evidence.read_text())
        result = validate_fusion_evidence(
            manifest, evidence.get("cases") if isinstance(evidence, dict) else None,
            expected_cases=args.expected_cases,
        )
    except (OSError, json.JSONDecodeError, FusionContractError) as exc:
        print(json.dumps({"status": "candidate_error", "diagnostics": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "ok", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
