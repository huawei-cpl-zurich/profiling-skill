#!/usr/bin/env python3
"""Validate and query the reviewed Triton-to-pipe attribution inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class InvalidInventory(ValueError):
    pass


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInventory(f"{location} must be a non-empty string")
    return value


def require_strings(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise InvalidInventory(f"{location} must be a non-empty array")
    return [require_string(item, f"{location}[]") for item in value]


def validate_inventory(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidInventory("inventory must be an object")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise InvalidInventory("schema_version must be integer 1")
    revisions = raw.get("compiler_revisions")
    if not isinstance(revisions, dict):
        raise InvalidInventory("compiler_revisions must be an object")
    for compiler in ("triton_ascend", "ascendnpuir"):
        revision = require_string(revisions.get(compiler), f"compiler_revisions.{compiler}")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise InvalidInventory(f"compiler_revisions.{compiler} must be a 40-hex commit")

    lowering = raw.get("lowering_path")
    if not isinstance(lowering, list) or not lowering:
        raise InvalidInventory("lowering_path must be a non-empty array")
    for index, step in enumerate(lowering):
        if not isinstance(step, dict):
            raise InvalidInventory(f"lowering_path[{index}] must be an object")
        require_string(step.get("stage"), f"lowering_path[{index}].stage")
        if step.get("status") != "direct":
            raise InvalidInventory(f"lowering_path[{index}].status must be direct")
        require_string(step.get("summary"), f"lowering_path[{index}].summary")
        citation = require_string(step.get("citation"), f"lowering_path[{index}].citation")
        if not citation.startswith("ref://"):
            raise InvalidInventory(f"lowering_path[{index}].citation must be ref://")

    mappings = raw.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise InvalidInventory("mappings must be a non-empty array")
    seen = set()
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise InvalidInventory(f"mappings[{index}] must be an object")
        construct = require_string(mapping.get("construct"), f"mappings[{index}].construct")
        if construct in seen:
            raise InvalidInventory(f"duplicate construct {construct!r}")
        seen.add(construct)
        status = mapping.get("status")
        if status not in {"direct", "inferred", "unknown"}:
            raise InvalidInventory(f"mapping {construct!r} has invalid status")
        products = require_strings(mapping.get("products"), f"mapping {construct!r}.products")
        if any(product not in {"a2", "a3", "a5"} for product in products):
            raise InvalidInventory(f"mapping {construct!r} has invalid product")
        citations = require_strings(mapping.get("citations"), f"mapping {construct!r}.citations")
        if any(not citation.startswith("ref://") for citation in citations):
            raise InvalidInventory(f"mapping {construct!r} citations must be ref://")
        if status == "unknown":
            if mapping.get("compiler_pipe") is not None or mapping.get("profiler_pipe") is not None:
                raise InvalidInventory(f"mapping {construct!r}: unknown mapping must not name a pipe")
            require_string(mapping.get("why_unknown"), f"mapping {construct!r}.why_unknown")
            require_string(mapping.get("next_evidence"), f"mapping {construct!r}.next_evidence")
        else:
            pipe = require_string(mapping.get("compiler_pipe"), f"mapping {construct!r}.compiler_pipe")
            if not pipe.startswith("PIPE_"):
                raise InvalidInventory(f"mapping {construct!r}.compiler_pipe must start PIPE_")
            require_string(mapping.get("reason"), f"mapping {construct!r}.reason")
            if mapping.get("profiler_pipe") is not None:
                require_string(mapping.get("profiler_pipe"), f"mapping {construct!r}.profiler_pipe")
                if mapping.get("profiler_pipe_status") not in {"direct", "inferred"}:
                    raise InvalidInventory(
                        f"mapping {construct!r}.profiler_pipe_status must be direct or inferred"
                    )
    return raw


def load_inventory(path: Path) -> dict[str, Any]:
    try:
        return validate_inventory(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidInventory(f"cannot read inventory {path}: {error}") from error


def query(inventory: dict[str, Any], construct: str, product: str | None) -> dict[str, Any] | None:
    mapping = next(
        (item for item in inventory["mappings"] if item["construct"] == construct), None
    )
    if mapping is None:
        return None
    if product is not None and product not in mapping["products"]:
        return {
            "construct": construct,
            "product": product,
            "status": "unknown",
            "compiler_pipe": None,
            "profiler_pipe": None,
            "why_unknown": "mapping has not been validated for the requested product",
            "source_mapping_status": mapping["status"],
            "citations": mapping["citations"],
            "compiler_revisions": inventory["compiler_revisions"],
        }
    result = dict(mapping)
    result["compiler_revisions"] = inventory["compiler_revisions"]
    if result["status"] == "unknown":
        result["lowering_path"] = inventory["lowering_path"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).parents[1] / "references" / "triton-pipe-attribution.json",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--construct")
    action.add_argument("--list", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--product", choices=("a2", "a3", "a5"))
    args = parser.parse_args(argv)
    if args.construct is not None and args.product is None:
        parser.error("--product is required with --construct")
    try:
        inventory = load_inventory(args.mapping)
    except InvalidInventory as error:
        parser.error(str(error))

    if args.validate:
        payload = {"status": "valid", "inventory_id": inventory.get("inventory_id")}
    elif args.list:
        mappings = inventory["mappings"]
        payload = {
            "schema_version": inventory["schema_version"],
            "inventory_id": inventory.get("inventory_id"),
            "compiler_revisions": inventory["compiler_revisions"],
            "constructs": sorted(item["construct"] for item in mappings),
            "counts": {
                "direct": sum(item["status"] == "direct" for item in mappings),
                "inferred": sum(
                    item["status"] == "inferred"
                    or item.get("profiler_pipe_status") == "inferred"
                    for item in mappings
                ),
                "unknown": sum(item["status"] == "unknown" for item in mappings),
            },
        }
    else:
        payload = query(inventory, args.construct, args.product)
        if payload is None:
            print(json.dumps({
                "construct": args.construct,
                "status": "unknown",
                "why_unknown": "construct is not present in the reviewed attribution inventory",
            }, sort_keys=True))
            return 3
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
