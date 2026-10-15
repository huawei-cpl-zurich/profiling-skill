#!/usr/bin/env python3
"""Recover characterization reports and selectively retry a frozen campaign."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import a3_profile_microbench_campaign as reporting


def load_producer(root: Path):
    path = root / "scripts/a3_profile_microbench_campaign.py"
    spec = importlib.util.spec_from_file_location("frozen_characterization_producer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    frozen_functions = {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(path.read_bytes()).body
                        if isinstance(node, ast.FunctionDef)}
    current_functions = {node.name: ast.dump(node, include_attributes=False)
                         for node in ast.parse(Path(reporting.__file__).read_bytes()).body
                         if isinstance(node, ast.FunctionDef)}
    for name in ("_numeric_rows", "_summary", "_selected_signal_values", "_signals",
                 "_result", "_failure_result"):
        if frozen_functions[name] != current_functions[name]:
            raise ValueError(f"recovery measurement helper changed: {name}")
    return module


def reporting_revision() -> dict:
    return {"adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "coordinator_sha256": hashlib.sha256(Path(reporting.__file__).read_bytes()).hexdigest()}


def validate_campaign(campaign: Path, producer, qualification: Path, count: int) -> tuple:
    """Recompute every original identity before reporting or dispatching anything."""
    manifest = json.loads(producer.DEFAULT_MANIFEST.read_bytes())
    receipt = json.loads(qualification.read_bytes())
    target = receipt["identity"]["target"]
    identity = producer.campaign_identity(target, manifest)
    producer.validate_qualification(qualification, identity)
    # Recovery is a reporting/lifecycle patch, never a measurement migration.
    current = reporting.campaign_identity(target, manifest)
    for key in ("runtime", "benchmark_sha256", "acquisition_sha256", "manifest_sha256"):
        if current[key] != identity[key]:
            raise ValueError(f"recovery measurement source mismatch: {key}")
    shards, seen, policies = [], set(), set()
    expected_cells = producer.enumerate_cells(manifest)
    for index in range(count):
        expected = producer.select_shard(expected_cells, index, count)
        paths = list((campaign / f"shard-{index}" / "cells").glob("*/result.json"))
        records = []
        for path in paths:
            record = json.loads(path.read_bytes())
            cell = next((cell for cell in expected if cell["id"] == record.get("id")), None)
            if cell is None or record["id"] in seen:
                raise ValueError(f"duplicate or unexpected cell: {record.get('id')}")
            seen.add(record["id"])
            contract = record["run_contract"]
            options = {name: contract[name] for name in
                       ("samples", "max_attempts", "max_relative_range")}
            policies.add(producer.canonical(options))
            cell_identity = {"schema_version": 1, "cell": cell, **identity, **options}
            expected_contract = {"schema_version": 1, "cell": cell, **options,
                                 "campaign_identity_sha256": hashlib.sha256(
                                     producer.canonical(cell_identity)).hexdigest()}
            if (any(record.get(key) != value for key, value in cell.items()) or
                    contract != expected_contract or record.get("campaign_identity_sha256") !=
                    hashlib.sha256(producer.canonical(expected_contract)).hexdigest()):
                raise ValueError(f"cell measurement identity mismatch: {record['id']}")
            records.append(record)
        if {item["id"] for item in records} != {cell["id"] for cell in expected}:
            raise ValueError(f"missing cell coverage in shard {index}")
        shards.append({"schema_version": 1, "manifest_sha256": identity["manifest_sha256"],
                       "campaign_identity": identity, "target": target,
                       "shard": {"index": index, "count": count}, "raw_cells": records})
    if len(policies) != 1:
        raise ValueError("mixed original sampling policies")
    return manifest, identity, shards


def recover_report(campaign: Path, output: Path, producer, qualification: Path, count: int):
    separate_output(campaign, output)
    manifest, _, shards = validate_campaign(campaign, producer, qualification, count)
    producer.summarize_hypotheses = reporting.summarize_hypotheses
    # merge_shards still checks the exact ORIGINAL producer sources and identity.
    merged = producer.merge_shards(shards, manifest)
    revision = reporting_revision()
    for shard in shards:
        shard["results"] = reporting.summarize_hypotheses(shard["raw_cells"], manifest)
        shard["reporting_revision"] = revision
        producer._atomic_json(output / f"shard-{shard['shard']['index']}" / "shard-result.json", shard)
    merged["reporting_revision"] = revision
    producer._atomic_json(output / "merged.json", merged)
    lines = ["# Characterization recovery report", "", "| Cell outcome | Count |",
             "|---|---:|"]
    lines.extend(f"| {status} | {number} |" for status, number in
                 sorted(Counter(cell["status"] for cell in merged["raw_cells"]).items()))
    for result in merged["results"]:
        lines.extend(["", f"## {result['hypothesis_id']}", ""])
        for measurement in result["measurements"]:
            if measurement["status"] == "valid":
                lines.append(f"- {measurement['sweep']}: {measurement['before']:g} → "
                             f"{measurement['after']:g} µs (core-time proxy).")
            else:
                lines.append(f"- {measurement['sweep']}: {measurement['status']}.")
        lines.extend(f"- {signal['name']}: {signal['relationship']}; {signal['summary']}"
                     for signal in result["signals"])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return merged


def separate_output(campaign: Path, output: Path):
    if (campaign.resolve() == output.resolve() or
            campaign.resolve() in output.resolve().parents or
            output.resolve() in campaign.resolve().parents):
        raise ValueError("recovery output must be separate from original campaign")


def verify_terminal(client: Path, record: dict) -> None:
    """Observe every last retained failure handle; no new identity while live."""
    handle = next((item.get("handle") for item in reversed(record.get("attempts", []))
                   if item.get("handle")), None)
    if not handle:
        raise ValueError(f"retry requires a retained terminal handle: {record['id']}")
    try:
        result = subprocess.run([str(client), "--json", "observe", handle],
                                capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot observe retained handle {handle}: {exc}") from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot observe retained handle {handle}") from exc
    if (result.returncode or value.get("handle") != handle or
            value.get("state") not in {"failed", "cancelled"}):
        raise ValueError(f"retained handle has not proved terminal failure: {handle}")


def retry_shard(campaign: Path, output: Path, producer, qualification: Path,
                count: int, index: int, client: Path) -> int:
    _, identity, shards = validate_campaign(campaign, producer, qualification, count)
    if not 0 <= index < count:
        raise ValueError("invalid shard index")
    separate_output(campaign, output)
    source, destination = campaign / f"shard-{index}", output / f"shard-{index}"
    if not destination.exists():
        shutil.copytree(source, destination)
    # Validate the working copy against the same frozen identity before mutation.
    expected = {item["id"]: item for item in shards[index]["raw_cells"]}
    selected = set()
    retained = list((destination / "cells").glob("*/result.json"))
    if len(retained) != len(expected):
        raise ValueError("retry copy cell coverage differs from original")
    seen = set()
    for path in retained:
        value = json.loads(path.read_bytes())
        original = expected.get(value.get("id"))
        if (not original or value["id"] in seen or
                value.get("run_contract") != original["run_contract"] or
                value.get("campaign_identity_sha256") != original["campaign_identity_sha256"]):
            raise ValueError("retry copy measurement identity differs from original")
        seen.add(value["id"])
        if not reporting.recovery_eligible(original) and value != original:
            raise ValueError("retry changed an unselected original cell")
        if value["attempts"][:len(original["attempts"])] != original["attempts"]:
            raise ValueError("retry did not preserve historical attempts")
        if reporting.recovery_eligible(value) and not value.get("recovery"):
            verify_terminal(client, value)
            selected.add(value["id"])
        elif value.get("recovery"):
            selected.add(value["id"])
    original_run_cell = producer.run_cell
    policy = next(iter(expected.values()))["run_contract"]

    def resume_cell(cell, cell_dir, acquire, **kwargs):
        if cell["id"] not in selected:
            return original_run_cell(cell, cell_dir, acquire, **kwargs)
        return reporting.run_cell(cell, cell_dir, acquire,
                                  recovery_budget=policy["max_attempts"], **kwargs)

    producer.run_cell = resume_cell
    producer.summarize_hypotheses = reporting.summarize_hypotheses
    original_acquire = producer.invoke_acquisition

    def acquire_with_diagnostics(**kwargs):
        result = original_acquire(**kwargs)
        if (result.get("classification") == "remote_failure" and
                "remote job failed without one valid failure receipt" in result.get("message", "")):
            try:
                verify_terminal(client, {"id": "new acquisition", "attempts": [result]})
            except ValueError as exc:
                return {**result, "classification": "transport_error", "message": str(exc)}
            return {**result, "classification": "host_environment"}
        return result

    producer.invoke_acquisition = acquire_with_diagnostics
    code = producer.main(["run", "--target", identity["target"], "--qualification", str(qualification),
                          "--shard-index", str(index), "--shard-count", str(count),
                          "--samples", str(policy["samples"]),
                          "--max-attempts", str(policy["max_attempts"]),
                          "--max-relative-range", str(policy["max_relative_range"]),
                          "--output-dir", str(destination), "--cpl-remote", str(client)])
    path = destination / "shard-result.json"
    value = json.loads(path.read_bytes())
    value["reporting_revision"] = reporting_revision()
    producer._atomic_json(path, value)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "retry"))
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cpl-remote", type=Path)
    args = parser.parse_args(argv)
    producer = load_producer(args.producer_root)
    if args.command == "report":
        recover_report(args.campaign_dir, args.output_dir, producer, args.qualification, args.shard_count)
        return 0
    if not args.cpl_remote:
        parser.error("retry requires --cpl-remote")
    return retry_shard(args.campaign_dir, args.output_dir, producer, args.qualification,
                       args.shard_count, args.shard_index, args.cpl_remote)


if __name__ == "__main__":
    raise SystemExit(main())
